
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forna_server_gpu.py — Flask server for RNA 2D layout with two modes:
  (A) CPU mode (default): chunked layout using forna.fasta_to_positions, with
      cross-boundary bracket sanitization and progress logging.
  (B) GPU mode (optional): global 2D layout on NVIDIA GPU using RAPIDS cuGraph
      ForceAtlas2 (requires cuDF/cuGraph installed).

Endpoints
---------
POST /struct_positions
  JSON fields:
    header: str (optional)
    seq: str (required, A/C/G/U/T... length N)
    struct: str (optional; dot-bracket with (), [], {}, <>; required if use_gpu_layout=true)
    chunk_size: int (optional; default 3000) — CPU mode only
    use_gpu_layout: bool (optional; default false)
    gpu_max_iter: int (optional; default 500)

  Returns: {"x":[...], "y":[...]}

GET /healthz -> {"ok": true}

Run (dev):    python3 forna_server_gpu.py -s -o 0.0.0.0 -p 8008
Run (prod):   gunicorn -w 2 -b 0.0.0.0:8008 'forna_server_gpu:app'
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple, Union

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

# ---- Optional RAPIDS (GPU) ----
try:
    import cudf  # type: ignore
    import cugraph  # type: ignore
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

# ---- Optional ViennaRNA for folding when 'struct' is not provided ----
try:
    import RNA  # type: ignore
except Exception:
    RNA = None
    print("[WARN] ViennaRNA (RNA) not available; folding requires 'RNA' or explicit 'struct'.", file=sys.stderr)

import importlib

forna = None  # será carregado quando (e se) o caminho CPU for usado

def _ensure_forna():
    """Importa 'forna' apenas quando o caminho CPU é acionado."""
    global forna
    if forna is None:
        try:
            forna = importlib.import_module('forna')
        except Exception as e:
            raise RuntimeError(
                "CPU layout requer o módulo 'forna' e dependências (p.ex. 'forgi' e 'RNA/ViennaRNA'). "
                "Instale com: conda install -c bioconda forgi viennarna  (ou pip install forgi; e ViennaRNA via conda)."
            ) from e

HERE = os.path.abspath(os.path.dirname(__file__))
STATIC_ROOT = os.path.join(HERE, "htdocs")

# ---- Basic validation regexes ----
SEQ_RE = re.compile(r"^[ACGTUacgtunN]+$")
# Allow ., (), [], {}, <>, and common separators tolerated by forna parsers
STRUCT_RE = re.compile(r"^[().<>\[\]\{\}/\\|_~:,;'+=*#^&%$!?\-\s.]+$")

JsonLike = Union[Dict[str, Any], List[Any]]

# ---- Bracket pairing for dot-bracket style notation ----
BR_OPEN_TO_CLOSE: Dict[str, str] = {"(": ")", "[": "]", "{": "}", "<": ">"}
BR_CLOSE_TO_OPEN: Dict[str, str] = {v: k for k, v in BR_OPEN_TO_CLOSE.items()}


def _norm_header(h: str) -> str:
    return h.lstrip('>').strip() if h else "sequence"


def _make_chunk_fasta(header: str, seq_chunk: str, struct_chunk: str) -> str:
    return f">{header}\n{seq_chunk}\n{struct_chunk}"


def _pairmap_from_dotbracket(struct: str) -> Dict[int, int | None]:
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
                pair[i] = None  # unmatched closing
        # '.' or others are ignored
    # unmatched openings
    for opener, stk in stacks.items():
        for j in stk:
            pair[j] = None
    return pair


def _fold_or_slice_struct(
    seq_chunk: str,
    orig_struct: str | None,
    pairmap: Dict[int, int | None] | None,
    i: int,
    chunk_size: int,
) -> str:
    """
    If 'orig_struct' is provided, slice the structure for this chunk and
    replace any bracket whose mate lies outside the chunk by '.'.
    Otherwise, fold the chunk using ViennaRNA if available.
    """
    if orig_struct is not None:
        start = i * chunk_size
        end = start + len(seq_chunk)
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


def _extend_agg_to_dict(
    agg_dict_x: List[float],
    agg_dict_y: List[float],
    res: JsonLike,
    offset_x: float,
) -> None:
    """
    Accumulate chunk result into dict {x:[], y:[]}, converting list-form chunks if needed.
    We always emit dict {x,y} to be frontend-friendly.
    """
    if isinstance(res, dict) and 'x' in res and 'y' in res:
        xs = list(res.get('x', []))
        ys = list(res.get('y', []))
        if len(xs) != len(ys):
            raise ValueError("Invalid layout format: x and y lengths differ.")
        agg_dict_x.extend([float(x) + offset_x for x in xs])
        agg_dict_y.extend([float(y) for y in ys])
    else:
        # assume list of [x,y] pairs
        try:
            for p in res:  # type: ignore[assignment]
                x, y = float(p[0]), float(p[1])
                agg_dict_x.extend([x + offset_x])
                agg_dict_y.extend([y])
        except Exception as e:
            raise ValueError("Invalid list format from layout: %s" % (e,))


def fasta_to_positions_chunked(
    fasta_text: str,
    chunk_size: int = 3000,
    x_spacing: float = 150.0,
) -> Dict[str, List[float]]:
    """
    Chunked CPU layout using forna.fasta_to_positions for each chunk, with
    cross-boundary bracket sanitization. Always returns dict {"x":[...], "y":[...]}.
    """
    
    _ensure_forna()
    
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

    agg_dict_x: List[float] = []
    agg_dict_y: List[float] = []
    last_err: Exception | None = None
    offset_x = 0.0
    t0 = time.perf_counter()
    processed = 0

    for i, chunk_seq in enumerate(chunks):
        t_start = time.perf_counter()
        try:
            struct_chunk = _fold_or_slice_struct(chunk_seq, orig_struct, pairmap, i, chunk_size)
            chunk_fasta = _make_chunk_fasta(f"{header}_chunk{i}", chunk_seq, struct_chunk)
            _ensure_forna()
            res = forna.fasta_to_positions(chunk_fasta)
            _extend_agg_to_dict(agg_dict_x, agg_dict_y, res, offset_x)
            processed += 1
            dt = time.perf_counter() - t_start
            avg = (time.perf_counter() - t0) / max(1, processed)
            eta = max(0.0, (total - processed) * avg)
            print(f"[chunked] processed {processed}/{total} | last={dt:.2f}s | avg={avg:.2f}s | ETA≈{eta:.2f}s", file=sys.stderr)
            offset_x += x_spacing
        except Exception as e:
            last_err = e
            print(f"[WARN] chunk {i+1}/{total} failed: {e}", file=sys.stderr)
            continue

    if not agg_dict_x:
        raise RuntimeError(f"Chunked layout failed: {last_err}")

    total_time = time.perf_counter() - t0
    print(f"[chunked] DONE in {total_time:.2f}s | points={len(agg_dict_x)}", file=sys.stderr)
    return {"x": agg_dict_x, "y": agg_dict_y}


def gpu_layout_positions(seq: str, struct: str, max_iter: int = 500) -> Dict[str, List[float]]:
    """
    Global 2D layout on GPU using cuGraph ForceAtlas2.
    Nodes = nucleotides (0..N-1)
    Edges = backbone (i,i+1) + base-pairing from dot-bracket.
    Returns {"x":[...], "y":[...]}
    """
    if not _GPU_AVAILABLE:
        raise RuntimeError("RAPIDS (cuDF/cuGraph) not available.")
    if not struct or len(struct) != len(seq):
        raise ValueError("GPU layout requires 'struct' with same length as 'seq'.")

    n = len(seq)
    # Backbone edges
    src = list(range(0, n - 1))
    dst = list(range(1, n))

    # Pairing edges
    pair = _pairmap_from_dotbracket(struct)
    for i, j in pair.items():
        if j is not None and i < j:
            src.append(i)
            dst.append(j)

    gdf = cudf.DataFrame({"src": src, "dst": dst})
    G = cugraph.Graph(directed=False)
    # renumber=False to preserve original vertex ids (0..n-1)
    G.from_cudf_edgelist(gdf, source="src", destination="dst", renumber=False)

    print(f"[gpu] building FA2 layout for N={n} nodes, E={len(src)} edges, iters={max_iter}", file=sys.stderr)
    pos = cugraph.force_atlas2(G, max_iter=int(max_iter))
    # pos columns: ["vertex", "fa2_x", "fa2_y"]
    pos = pos.sort_values("vertex")
    xs = pos["fa2_x"].to_pandas().tolist()
    ys = pos["fa2_y"].to_pandas().tolist()
    print(f"[gpu] DONE | points={len(xs)}", file=sys.stderr)
    return {"x": xs, "y": ys}


# ----------------------------- Flask App Factory -----------------------------
def create_app(serve_static: bool = True) -> Flask:
    app = Flask(__name__, static_folder=None)

    # Dev: serve ./htdocs if available
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
        return jsonify({'ok': True, 'gpu_available': _GPU_AVAILABLE})

    @app.post('/struct_positions')
    def struct_positions() -> Any:
        if not request.is_json:
            abort(403, description="Requisição deve ser JSON (Content-Type: application/json)")
        data = request.get_json(force=True)

        header = data.get('header') or 'sequence'
        seq = (data.get('seq') or '').strip()
        struct = (data.get('struct') or '').strip() or None
        chunk_size = int(data.get('chunk_size') or 3000)
        use_gpu = bool(data.get('use_gpu_layout', False))
        gpu_max_iter = int(data.get('gpu_max_iter', 500))

        if not seq or not SEQ_RE.match(seq):
            abort(400, description="Campo 'seq' ausente ou inválido")
        if struct is not None and not STRUCT_RE.match(struct):
            abort(400, description="Campo 'struct' inválido")
        if struct is not None and len(struct) != len(seq):
            abort(400, description="'seq' e 'struct' têm comprimentos distintos")

        try:
            if use_gpu:
                if not _GPU_AVAILABLE:
                    abort(400, description="GPU layout solicitado, mas RAPIDS/cuGraph não está disponível.")
                if struct is None:
                    abort(400, description="GPU layout exige 'struct' (dot-bracket).")
                res = gpu_layout_positions(seq, struct, max_iter=gpu_max_iter)
            else:
                # CPU path: chunked or direct
                if len(seq) > chunk_size:
                    fasta_text = f">{_norm_header(header)}\n{seq}\n{struct if struct else ''}"
                    _ensure_forna()
                    res = fasta_to_positions_chunked(fasta_text, chunk_size=chunk_size)
                    out = forna.fasta_to_positions(fasta_text)
                else:
                    if struct is None:
                        if RNA is None:
                            abort(400, description="'struct' ausente e ViennaRNA indisponível")
                        folded, _ = RNA.fold(seq)  # type: ignore[union-attr]
                        fasta_text = f">{_norm_header(header)}\n{seq}\n{folded}"
                    else:
                        fasta_text = f">{_norm_header(header)}\n{seq}\n{struct}"
                    _ensure_forna()
                    out = forna.fasta_to_positions(fasta_text)
                    # Normalize to dict {x,y} for frontend
                    if isinstance(out, dict) and 'x' in out and 'y' in out:
                        res = {'x': list(out['x']), 'y': list(out['y'])}
                    else:
                        xs, ys = [], []
                        for p in out:  # type: ignore[assignment]
                            xs.append(float(p[0]))
                            ys.append(float(p[1]))
                        res = {'x': xs, 'y': ys}
        except Exception as e:
            abort(400, description=f"Secondary structure parsing error: {e}")

        return jsonify(res)

    # Optional upload endpoints
    @app.post('/pdb_to_graph')
    def pdb_to_graph() -> Any:
        if 'file' not in request.files:
            abort(400, description="Arquivo PDB ausente em 'file'")
        file = request.files['file']
        if not file or file.filename == '':
            abort(400, description="Arquivo inválido")
        fname = secure_filename(file.filename)
        tmp_path = os.path.join('/tmp', fname)
        file.save(tmp_path)
        try: 
            _ensure_forna()
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
            abort(400, description="Arquivo inválido")
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


# ---- Global app for gunicorn ----
app = create_app(serve_static=False)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="forna server (CPU chunked + optional GPU layout)")
    parser.add_argument("-s", "--static", action="store_true", help="Serve ./htdocs (dev mode)")
    parser.add_argument("-o", "--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", default=8008, type=int)
    parser.add_argument("-d", "--debug", action="store_true", help="Enable Flask debug/reload")
    args = parser.parse_args(argv)

    if args.static:
        dev_app = create_app(serve_static=True)
        dev_app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
