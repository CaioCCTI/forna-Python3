#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
forna_server.py — versão corrigida para execução particionada (chunked)

Principais mudanças:
1) Corrige cabeçalho duplicado ('>>header') na montagem de FASTA por chunk.
2) Suporta de forma robusta os dois formatos de retorno de forna.fasta_to_positions: lista de pares
   ou dict {"x": [...], "y": [...]}. A resposta final do modo chunked imita o formato do 1º chunk.
3) Exposição do parâmetro opcional "chunk_size" no JSON do POST /struct_positions.
4) Mensagens de erro claras (HTTP 400) ao invés de retornar resposta vazia (que vira "Undefined" no front).
5) Import compatível do secure_filename: from werkzeug.utils import secure_filename.
6) Mantém modo dev (python forna_server.py -s) e modo produção (gunicorn 'forna_server:app').
"""

from __future__ import annotations
import os
import sys
import json
import re
import argparse
from typing import Any, Dict, List, Tuple, Union

from flask import Flask, request, abort, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# Dependências esperadas do projeto (presentes no seu ambiente)
# - Módulo local "forna" com funções fasta_to_positions, pdb_to_graph, mmcif_to_graph
# - ViennaRNA (RNA) para dobrar estrutura quando não for fornecida
try:
    import forna  # type: ignore
except Exception as e:  # pragma: no cover
    print("[ERRO] Falha ao importar o módulo 'forna':", e, file=sys.stderr)
    raise

try:
    import RNA  # ViennaRNA
except Exception as e:  # pragma: no cover
    RNA = None
    print("[AVISO] ViennaRNA (RNA) não disponível; 'struct' original será obrigatório quando chunking.", file=sys.stderr)

HERE = os.path.abspath(os.path.dirname(__file__))
STATIC_ROOT = os.path.join(HERE, "htdocs")

# Regex de validação básica
SEQ_RE = re.compile(r"^[ACGTUacgtunN]+$")
STRUCT_RE = re.compile(r"^[().<>\[\]{}\\/\\|_\-~:,;'+=\*#\^&%$!?\s\.]+$")  # aceita '.' e caracteres usuais

JsonLike = Union[Dict[str, Any], List[Any]]


def _norm_header(h: str) -> str:
    """Remove prefixo '>' se existir e tira espaços extras."""
    return h.lstrip('>').strip() if h else "sequence"


def _make_chunk_fasta(header: str, seq_chunk: str, struct_chunk: str) -> str:
    return f">{header}\n{seq_chunk}\n{struct_chunk}"


def _fold_or_slice_struct(seq_chunk: str, orig_struct: str | None, i: int, chunk_size: int) -> str:
    if orig_struct is not None:
        return orig_struct[i * chunk_size:(i + 1) * chunk_size]
    if RNA is None:
        raise RuntimeError("ViennaRNA indisponível e 'struct' não fornecida.")
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
    """Acumula resultados de um chunk em 'agg', respeitando formato.
    Retorna o modo final ("list" ou "dict").
    """
    # Descobre formato do retorno
    if isinstance(res, dict) and 'x' in res and 'y' in res:
        cur_mode = 'dict'
        xs = list(res.get('x', []))
        ys = list(res.get('y', []))
        if len(xs) != len(ys):
            raise ValueError("Formato inválido: tamanhos de x e y diferentes")
        agg_dict_x.extend([float(x) + offset_x for x in xs])
        agg_dict_y.extend([float(y) for y in ys])
    else:
        cur_mode = 'list'
        # esperado: lista de pares [ [x,y], [x,y], ... ] ou lista de tuples
        pts: List[Tuple[float, float]] = []
        try:
            for p in res:  # type: ignore
                x, y = float(p[0]), float(p[1])
                pts.append((x + offset_x, y))
        except Exception as e:
            raise ValueError(f"Formato de lista inválido no chunk: {e}")
        agg_xy.extend(pts)

    if agg_mode == '':
        return cur_mode
    # Se já havia um modo definido, mantém o primeiro
    return agg_mode


def fasta_to_positions_chunked(
    fasta_text: str,
    chunk_size: int = 3000,
    x_spacing: float = 150.0,
) -> JsonLike:
    """Divide uma sequência longa em chunks e agrega as posições 2D.

    Mantém o formato do primeiro chunk válido: se forna.fasta_to_positions retornar
    dict {x,y}, a resposta final será dict; se retornar lista de pares, a resposta final será lista.
    """
    lines = [ln.strip() for ln in fasta_text.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("FASTA vazio")

    # Header e sequência obrigatórios
    header = _norm_header(lines[0])
    if len(lines) < 2:
        raise ValueError("FASTA sem sequência")
    seq = lines[1]
    if not SEQ_RE.match(seq):
        raise ValueError("Sequência contém caracteres inválidos")

    # Terceira linha opcional: estrutura
    orig_struct = None
    if len(lines) >= 3 and lines[2]:
        if not STRUCT_RE.match(lines[2]):
            raise ValueError("Estrutura contém caracteres inválidos")
        if len(lines[2]) != len(seq):
            raise ValueError("Comprimentos de 'seq' e 'struct' não batem")
        orig_struct = lines[2]

    chunks: List[str] = [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size)]

    agg_mode = ''  # 'list' ou 'dict'
    agg_xy: List[Tuple[float, float]] = []
    agg_dict_x: List[float] = []
    agg_dict_y: List[float] = []

    last_err: Exception | None = None
    offset_x = 0.0

    for i, chunk_seq in enumerate(chunks):
        try:
            struct_chunk = _fold_or_slice_struct(chunk_seq, orig_struct, i, chunk_size)
            chunk_fasta = _make_chunk_fasta(f"{header}_chunk{i}", chunk_seq, struct_chunk)
            res = forna.fasta_to_positions(chunk_fasta)
            agg_mode = _extend_agg(agg_mode, agg_xy, agg_dict_x, agg_dict_y, res, offset_x)
            offset_x += x_spacing
        except Exception as e:
            last_err = e
            print(f"[WARN] Falha no bloco {i}: {e}", file=sys.stderr)
            continue

    # Se nada deu certo, informe claramente
    if (not agg_xy) and (not agg_dict_x):
        raise RuntimeError(f"Chunked layout failed: {last_err}")

    if agg_mode == 'dict':
        return {"x": agg_dict_x, "y": agg_dict_y}
    else:
        return [[x, y] for (x, y) in agg_xy]


# ----------------------------- Flask App Factory -----------------------------

def create_app(serve_static: bool = True) -> Flask:
    app = Flask(__name__, static_folder=None)

    # ---------------------------- Rotas estáticas ----------------------------
    if serve_static:
        @app.route('/')
        def index():
            return send_from_directory(STATIC_ROOT, 'index.html')

        @app.route('/js/<path:filename>')
        def js(filename: str):
            return send_from_directory(os.path.join(STATIC_ROOT, 'js'), filename)

        @app.route('/css/<path:filename>')
        def css(filename: str):
            return send_from_directory(os.path.join(STATIC_ROOT, 'css'), filename)

        @app.route('/assets/<path:filename>')
        def assets(filename: str):
            return send_from_directory(os.path.join(STATIC_ROOT, 'assets'), filename)

        @app.route('/static/<path:filename>')
        def static_files(filename: str):
            return send_from_directory(STATIC_ROOT, filename)

    else:
        # Mesmo sem static=True, exponha '/' para servir o index.html (ex.: gunicorn)
        @app.route('/')
        def index_prod():
            return send_from_directory(STATIC_ROOT, 'index.html')

    # --------------------------- Endpoint principal --------------------------
    @app.post('/struct_positions')
    def struct_positions():
        if not request.is_json:
            abort(400, description="Requisição deve ser JSON")
        data = request.get_json(force=True)

        header = data.get('header') or 'sequence'
        seq = (data.get('seq') or '').strip()
        struct = (data.get('struct') or '').strip() or None
        chunk_size = int(data.get('chunk_size') or 3000)

        if not seq or not SEQ_RE.match(seq):
            abort(400, description="Campo 'seq' ausente ou inválido")
        if struct is not None and not STRUCT_RE.match(struct):
            abort(400, description="Campo 'struct' inválido")
        if struct is not None and len(struct) != len(seq):
            abort(400, description="'seq' e 'struct' têm comprimentos distintos")

        # Monta FASTA completo
        fasta_text = f">{_norm_header(header)}\n{seq}\n{struct if struct else ''}"

        try:
            # Decide pelo modo chunked com base no tamanho
            if len(seq) > chunk_size:
                res = fasta_to_positions_chunked(fasta_text, chunk_size=chunk_size)
            else:
                # Modo direto
                if struct is None:
                    if RNA is None:
                        abort(400, description="'struct' ausente e ViennaRNA indisponível")
                    struct_fold, _ = RNA.fold(seq)
                    fasta_text = f">{_norm_header(header)}\n{seq}\n{struct_fold}"
                res = forna.fasta_to_positions(fasta_text)
        except Exception as e:
            abort(400, description=f"Secondary structure parsing error: {e}")

        # Devolve exatamente o formato que montamos/recebemos
        return jsonify(res)

    # --------------------------- Upload de estruturas ------------------------
    @app.post('/pdb_to_graph')
    def pdb_to_graph():
        if 'file' not in request.files:
            abort(400, description="Arquivo PDB ausente em 'file'")
        file = request.files['file']
        if not file or file.filename == '':
            abort(400, description="Arquivo inválido")
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
    def mmcif_to_graph():
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


# ----------------------------------------------------------------------------
# App global para produção (gunicorn: 'forna_server:app')
# ----------------------------------------------------------------------------
app = create_app(serve_static=False)


# ----------------------------------------------------------------------------
# Execução local / desenvolvimento
# ----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="forna server (chunked)")
    parser.add_argument('-s', '--static', action='store_true', help='Servir arquivos estáticos (htdocs)')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug/auto-reload')
    parser.add_argument('-o', '--host', default='127.0.0.1')
    parser.add_argument('-p', '--port', default=8008, type=int)
    args = parser.parse_args(argv)

    if args.static:
        dev_app = create_app(serve_static=True)
        dev_app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        # Executa o app global (sem estáticos). Útil somente para testes rápidos.
        app.run(host=args.host, port=args.port, debug=args.debug)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

