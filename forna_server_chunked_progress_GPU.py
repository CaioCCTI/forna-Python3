
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forna_server_chunked_progress.py — forna server with chunked layout + cross-pair sanitization
and detailed progress logging to stdout.

Changes vs. original:
1) Fix double '>' in chunk FASTA headers.
2) Preserve output format (list of [x, y] pairs OR dict {x:[], y:[]}) matching first valid chunk.
3) Optional JSON field "chunk_size" to tune chunking (default: 3000).
4) Clear HTTP 4xx on invalid input / parsing errors instead of returning empty arrays.
5) Compatible import: from werkzeug.utils import secure_filename.
6) Dev mode (-s) serves ./htdocs; prod mode via gunicorn uses `app` instance.
7) NEW: Sanitize cross-boundary pairings when a global dot-bracket `struct` is provided:
   any bracket whose mate falls outside the current chunk is replaced by '.' to avoid
   "Too many opening/closing brackets!" from downstream layout.
8) NEW: Progress logging per chunk with elapsed time and ETA.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple, Union

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

# --- GPU (RAPIDS) opcional: cuDF + cuGraph ---
try:
    import cudf  # type: ignore
    import cugraph  # type: ignore
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False


# Project-local dependency: module `forna` must provide:
#   - fasta_to_positions(fasta_text: str) -> Union[List[List[float]], Dict[str, List[float]]]
#   - pdb_to_graph(path: str) -> Any
#   - mmcif_to_graph(path: str) -> Any
try:
    import forna  # type: ignore
except Exception as e:  # pragma: no cover
    print("[ERROR] Failed to import 'forna':", e, file=sys.stderr)
    raise

# Optional: ViennaRNA for folding when no struct is provided
try:
    import RNA  # type: ignore
except Exception as e:  # pragma: no cover
    RNA = None
    print("[WARN] ViennaRNA (RNA) not available; JSON must include 'struct' when using chunked mode.", file=sys.stderr)

HERE = os.path.abspath(os.path.dirname(__file__))
STATIC_ROOT = os.path.join(HERE, "htdocs")

# Basic validation regexes
SEQ_RE = re.compile(r"^[ACGTUacgtunN]+$")
# Allow ., (), [], {}, <>, and common separators tolerated by forna parsers
STRUCT_RE = re.compile(r"^[().<>\[\]\{\}/\\|_~:,;'+=*#^&%$!?\-\s.]+$")

JsonLike = Union[Dict[str, Any], List[Any]]

# Bracket pairing maps for dot-bracket style notations
BR_OPEN_TO_CLOSE: Dict[str, str] = {"(": ")", "[": "]", "{": "}", "<": ">"}
BR_CLOSE_TO_OPEN: Dict[str, str] = {v: k for k, v in BR_OPEN_TO_CLOSE.items()}


def _norm_header(h: str) -> str:
    """Strip leading '>' and whitespace from header; default to 'sequence'."""
    return h.lstrip('>').strip() if h else "sequence"


def _make_chunk_fasta(header: str, seq_chunk: str, struct_chunk: str) -> str:
    """Build a minimal FASTA-with-structure entry for a single chunk."""
    return f">{header}\n{seq_chunk}\n{struct_chunk}"


def _pairmap_from_dotbracket(struct: str) -> Dict[int, int | None]:
    """Return dict: index -> paired_index (or None) for (), [], {}, <> in dot-bracket string."""
    stacks: Dict[str, List[int]] = {k: [] for k in BR_OPEN_TO_CLOSE}
    pair: Dict[int, int | None] = {}
    for i, ch in enumerate(struct):
        if ch in BR_OPEN_TO_CLOSE:  # opening
            stacks[ch].append(i)
        elif ch in BR_CLOSE_TO_OPEN:  # closing
            opener = BR_CLOSE_TO_OPEN[ch]
            if stacks[opener]:
                j = stacks[opener].pop()
                pair[i] = j
                pair[j] = i
            else:
                # unmatched closing
                pair[i] = None
        else:
            # '.' or other symbols are ignored
            pass
    # unmatched openings
    for opener, stk in stacks.items():
        for j in stk:
            pair[j] = None
    return pair

def gpu_layout_positions(seq: str, struct: str, max_iter: int = 500) -> Dict[str, list]:
    """
    Layout 2D global na GPU (ForceAtlas2/cuGraph).
    Nós = nucleotídeos (0..N-1)
    Arestas = backbone (i,i+1) + pareamentos do dot-bracket.
    Retorna {"x":[...], "y":[...]} no índice natural (ordenado).
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("RAPIDS (cuDF/cuGraph) não disponível no ambiente.")
    if not struct or len(struct) != len(seq):
        raise ValueError("GPU layout requer 'struct' com mesmo comprimento da sequência.")

    # 1) Edgelist backbone
    n = len(seq)
    src = list(range(0, n - 1))
    dst = list(range(1, n))

    # 2) Edgelist pareamentos (dot-bracket)
    pair = _pairmap_from_dotbracket(struct)
    for i, j in pair.items():
        if j is not None and i < j:
            src.append(i)
            dst.append(j)

    # 3) DataFrame na GPU
    gdf = cudf.DataFrame({"src": src, "dst": dst})

    # 4) Grafo não-direcionado
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(gdf, source="src", destination="dst", renumber=False)

    # 5) ForceAtlas2 (GPU)
    pos = cugraph.force_atlas2(G, max_iter=int(max_iter))
    # pos: ["vertex", "fa2_x", "fa2_y"]

    pos = pos.sort_values("vertex")
    xs = pos["fa2_x"].to_pandas().tolist()
    ys = pos["fa2_y"].to_pandas().tolist()
    return {"x": xs, "y": ys}


def _fold_or_slice_struct(
    seq_chunk: str,
    orig_struct: str | None,
    pairmap: Dict[int, int | None] | None,
    i: int,
    chunk_size: int,
) -> str:
    """
    If `orig_struct` is provided, slice the corresponding region and sanitize:
    any bracket whose mate lies outside the [start, end) of the chunk is replaced with '.'.
    Otherwise, fold the chunk with ViennaRNA (if available).
    """
    if orig_struct is not None:
        start = i * chunk_size
        end = start + len(seq_chunk)  # exclusive
        chars = list(orig_struct[start:end])
        if pairmap:
            for local_idx, ch in enumerate(chars):
                if ch in BR_OPEN_TO_CLOSE or ch in BR_CLOSE_TO_OPEN:
                    g = start + local_idx
                    mate = pairmap.get(g)
                    if mate is None or mate < start or mate >= end:
                        chars[local_idx] = '.'
        return ''.join(chars)

    if RNA is None:
        raise RuntimeError("ViennaRNA unavailable and no 'struct' provided.")
    struct_chunk, _mfe = RNA.fold(seq_chunk)
    return struct_chunk


def _extend_agg(
    agg_mode: str,
    agg_xy: List[Tuple[float, float]],
    agg_dict_x: List[float],
    agg_dict_y: List[float],
    res: JsonLike,
    offset_x: float,
) -> str:
    """
    Accumulate results while preserving the format of the first valid chunk:
    - If first chunk returns dict {x:[], y:[]}, keep dict.
    - If first chunk returns list [[x,y],...], keep list.
    """
    if isinstance(res, dict) and 'x' in res and 'y' in res:
        cur_mode = 'dict'
        xs = list(res.get('x', []))
        ys = list(res.get('y', []))
        if len(xs) != len(ys):
            raise ValueError("Invalid format from layout: x and y lengths differ.")
        agg_dict_x.extend([float(x) + offset_x for x in xs])
        agg_dict_y.extend([float(y) for y in ys])
    else:
        cur_mode = 'list'
        points: List[Tuple[float, float]] = []
        try:
            for p in res:  # type: ignore[assignment]
                x, y = float(p[0]), float(p[1])
                points.append((x + offset_x, y))
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Invalid list format from layout: {e}")
        agg_xy.extend(points)

    return cur_mode if not agg_mode else agg_mode


def fasta_to_positions_chunked(
    fasta_text: str,
    chunk_size: int = 3000,
    x_spacing: float = 150.0,
) -> JsonLike:
    """
    Split a long sequence into chunks, layout each chunk, and concatenate X
    coordinates with an offset so chunks are placed side-by-side.
    """
    lines = [ln.strip() for ln in fasta_text.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty FASTA text")

    header = _norm_header(lines[0])
    if len(lines) < 2:
        raise ValueError("FASTA missing sequence line")
    seq = lines[1]
    if not SEQ_RE.match(seq):
        raise ValueError("Sequence contains invalid characters")

    orig_struct: str | None = None
    pairmap: Dict[int, int | None] | None = None
    if len(lines) >= 3 and lines[2]:
        if not STRUCT_RE.match(lines[2]):
            raise ValueError("Structure contains invalid characters")
        if len(lines[2]) != len(seq):
            raise ValueError("Sequence and structure lengths differ")
        orig_struct = lines[2]
        pairmap = _pairmap_from_dotbracket(orig_struct)

    chunks: List[str] = [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]
    total = len(chunks)
    print(f"[chunked] total_length={len(seq)} nt | chunk_size={chunk_size} | chunks={total}", file=sys.stderr)

    agg_mode = ''  # 'list' or 'dict', chosen by first successful chunk
    agg_xy: List[Tuple[float, float]] = []
    agg_dict_x: List[float] = []
    agg_dict_y: List[float] = []

    last_err: Exception | None = None
    offset_x = 0.0
    t0 = time.perf_counter()
    processed = 0
    first_ok_time = None

    for i, chunk_seq in enumerate(chunks):
        t_start = time.perf_counter()
        try:
            struct_chunk = _fold_or_slice_struct(chunk_seq, orig_struct, pairmap, i, chunk_size)
            chunk_fasta = _make_chunk_fasta(f"{header}_chunk{i}", chunk_seq, struct_chunk)
            res = forna.fasta_to_positions(chunk_fasta)
            agg_mode = _extend_agg(agg_mode, agg_xy, agg_dict_x, agg_dict_y, res, offset_x)
            processed += 1
            dt = time.perf_counter() - t_start
            if first_ok_time is None:
                first_ok_time = dt
            avg = (time.perf_counter() - t0) / max(1, processed)
            eta = max(0.0, (total - processed) * avg)
            print(f"[chunked] processed {processed}/{total} | last={dt:.2f}s | avg={avg:.2f}s | ETA≈{eta:.2f}s", file=sys.stderr)
            offset_x += x_spacing
        except Exception as e:
            last_err = e
            print(f"[WARN] chunk {i+1}/{total} failed: {e}", file=sys.stderr)
            # continue to next chunk
            continue

    if not agg_xy and not agg_dict_x:
        raise RuntimeError(f"Chunked layout failed: {last_err}")

    total_time = time.perf_counter() - t0
    shape = f"dict(len={len(agg_dict_x)})" if agg_mode == 'dict' else f"list(n_points={len(agg_xy)})"
    print(f"[chunked] DONE in {total_time:.2f}s | format={agg_mode} {shape}", file=sys.stderr)

    if agg_mode == 'dict':
        return {'x': agg_dict_x, 'y': agg_dict_y}
    else:
        return [[x, y] for (x, y) in agg_xy]


def create_app(serve_static: bool = True) -> Flask:
    app = Flask(__name__, static_folder=None)

    # Dev static routes
    if serve_static:
        @app.route('/')
        def index() -> Any:
            return send_from_directory(STATIC_ROOT, 'index.html')

        @app.route('/js/<path:filename>')
        def js(filename: str) -> Any:
            return send_from_directory(os.path.join(STATIC_ROOT, 'js'), filename)

        @app.route('/css/<path:filename>')
        def css(filename: str) -> Any:
            return send_from_directory(os.path.join(STATIC_ROOT, 'css'), filename)

        @app.route('/assets/<path:filename>')
        def assets(filename: str) -> Any:
            return send_from_directory(os.path.join(STATIC_ROOT, 'assets'), filename)

        @app.route('/static/<path:filename>')
        def static_files(filename: str) -> Any:
            return send_from_directory(STATIC_ROOT, filename)
    else:
        @app.route('/')
        def index_prod() -> Any:
            return send_from_directory(STATIC_ROOT, 'index.html')

    @app.get('/healthz')
    def healthz() -> Any:
        return jsonify({'ok': True})

    @app.post('/struct_positions')
    def struct_positions() -> Any:
        if not request.is_json:
            # Use 403 in dev to make it obvious in browser that JSON is required
            abort(403, description='Requisição deve ser JSON (Content-Type: application/json)')
        data = request.get_json(force=True)

        header = data.get('header') or 'sequence'
        seq = (data.get('seq') or '').strip()
        struct = (data.get('struct') or '').strip() or None
        chunk_size = int(data.get('chunk_size') or 3000)
        
        use_gpu = bool(data.get("use_gpu_layout", False))
	gpu_max_iter = int(data.get("gpu_max_iter", 500))

        if not seq or not SEQ_RE.match(seq):
            abort(400, description="Campo 'seq' ausente ou inválido")
        if struct is not None and not STRUCT_RE.match(struct):
            abort(400, description="Campo 'struct' inválido")
        if struct is not None and len(struct) != len(seq):
            abort(400, description="'seq' e 'struct' têm comprimentos distintos")

        fasta_text = f">{_norm_header(header)}\n{seq}\n{struct if struct else ''}"

	try:
	    if use_gpu:
		if not _GPU_AVAILABLE:
		    abort(400, description="GPU layout solicitado, mas RAPIDS/cuGraph não está disponível no servidor.")
		if struct is None:
		    abort(400, description="GPU layout exige 'struct' (dot-bracket). Envie 'struct' no JSON.")
		# Layout 2D global na GPU (sem chunking de layout)
		res = gpu_layout_positions(seq, struct, max_iter=gpu_max_iter)

	    else:
		# Caminho original (CPU). Se sequência gigante, particiona layout por chunks.
		if len(seq) > chunk_size:
		    res = fasta_to_positions_chunked(f">{_norm_header(header)}\n{seq}\n{struct if struct else ''}",
		                                     chunk_size=chunk_size)
		else:
		    if struct is None:
		        if 'RNA' in globals() and RNA is None:
		            abort(400, description="'struct' ausente e ViennaRNA indisponível")
		        folded, _ = RNA.fold(seq)  # type: ignore[union-attr]
		        fasta_text = f">{_norm_header(header)}\n{seq}\n{folded}"
		    else:
		        fasta_text = f">{_norm_header(header)}\n{seq}\n{struct}"
		    res = forna.fasta_to_positions(fasta_text)

except Exception as e:
    abort(400, description=f"Secondary structure parsing error: {e}")


        return jsonify(res)

    @app.post('/pdb_to_graph')
    def pdb_to_graph() -> Any:
        if 'file' not in request.files:
            abort(400, description="Arquivo PDB ausente em 'file'")
        file = request.files['file']
        if not file or file.filename == '':
            abort(400, description='Arquivo inválido')
        fname = secure_filename(file.filename)
        tmp_path = os.path.join('/tmp', fname)
        file.save(tmp_path)
        try:
            res = forna.pdb_to_graph(tmp_path)
        except Exception as e:
            abort(400, description=f"Erro ao processar PDB: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return jsonify(res)

    @app.post('/mmcif_to_graph')
    def mmcif_to_graph() -> Any:
        if 'file' not in request.files:
            abort(400, description="Arquivo mmCIF ausente em 'file'")
        file = request.files['file']
        if not file or file.filename == '':
            abort(400, description='Arquivo inválido')
        fname = secure_filename(file.filename)
        tmp_path = os.path.join('/tmp', fname)
        file.save(tmp_path)
        try:
            res = forna.mmcif_to_graph(tmp_path)
        except Exception as e:
            abort(400, description=f"Erro ao processar mmCIF: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return jsonify(res)

    return app


# Global app for gunicorn: `gunicorn -w 2 -b 0.0.0.0:8008 'forna_server_chunked_progress:app'`
app = create_app(serve_static=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='forna server (chunked + progress)')
    parser.add_argument('-s', '--static', action='store_true', help='Serve ./htdocs (dev mode)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable Flask debug/reload')
    parser.add_argument('-o', '--host', default='127.0.0.1')
    parser.add_argument('-p', '--port', default=8008, type=int)
    args = parser.parse_args(argv)

    if args.static:
        dev_app = create_app(serve_static=True)
        dev_app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
