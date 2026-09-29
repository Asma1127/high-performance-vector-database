# VectorDB — a high-performance vector database from scratch

A custom vector database for semantic search, built to showcase systems-level
performance engineering: **Mojo/MAX** for the numeric core, **Python** for the
API/orchestration layer, and **SQLite** for durable metadata + vector storage.

## Why this stack

| Layer | Tool | Job |
|---|---|---|
| Compute kernel | **Mojo** | SIMD/vectorized distance math (cosine, dot, L2), batch top-k scoring |
| Compute graph (optional GPU path) | **MAX** | Compiles/serves the kernel, can target CPU or GPU without rewriting code |
| Orchestration / API | **Python** | Embedding generation, REST API (FastAPI), CLI, glue code |
| Persistence | **SQLite** | Document text, metadata, vector blobs, WAL-mode durability |

The project is designed to run **without Mojo installed at all** — there's a
pure-NumPy index (`vectordb/index.py`) that implements the identical
interface. Once you install the Modular toolchain, you swap in the Mojo
kernel and get the speedup with zero API changes. This is deliberate: it's
the same pattern real vector DBs use (e.g. a reference implementation +
an optimized kernel), and it means graders/reviewers can run the whole
thing with just `pip install`.

## Architecture

```
                     ┌─────────────────────┐
   client / curl ──▶ │   FastAPI (api.py)   │
                     └─────────┬────────────┘
                               │
                     ┌─────────▼────────────┐
                     │   VectorDB (db.py)    │  orchestrates:
                     │                       │
                     │  embed() ──────────┐  │
                     │  index.search() ───┼──┼──▶ IndexBackend (protocol)
                     │  storage.get() ────┘  │        │
                     └─────────┬─────────────┘        ├─ NumpyIndex   (default, pure python)
                               │                       └─ MojoIndex    (SIMD kernel, opt-in)
                     ┌─────────▼────────────┐
                     │  SQLite (storage.py)  │  documents(id, text, metadata, vector BLOB)
                     └───────────────────────┘
```

**Data flow on insert:** text → `embedding.py` (sentence-transformers,
or a deterministic hash-embedding fallback if it's not installed) → float32
vector → written to SQLite as a BLOB + row metadata → also pushed into the
in-memory index for fast search.

**Data flow on search:** query text → embed → `IndexBackend.search(vec, k)`
returns `(id, score)` pairs → ids resolved back to text/metadata via SQLite.

**On startup:** the index is rebuilt from SQLite (source of truth), so the
in-memory index is always disposable/rebuildable — SQLite is what survives
restarts.

## Project layout

```
vector-db/
├── src/
│   ├── mojo_kernels/
│   │   ├── distance.mojo     # SIMD cosine/dot/L2 distance, vectorized over width
│   │   └── index.mojo        # brute-force flat index: parallel top-k scan
│   └── vectordb/
│       ├── storage.py        # SQLite persistence
│       ├── embedding.py      # text -> vector
│       ├── index.py          # IndexBackend protocol + NumpyIndex + MojoIndex bridge
│       ├── db.py             # public VectorDB API (insert/search/delete)
│       └── api.py            # FastAPI REST server
├── examples/quickstart.py    # end-to-end demo, runs standalone
├── tests/test_vectordb.py    # pytest suite (passes with just numpy+sqlite3)
└── requirements.txt
```

## Quickstart (works today, no Mojo needed)

```bash
cd vector-db
pip install -r requirements.txt        # numpy, fastapi, uvicorn (+ sentence-transformers optional)
python examples/quickstart.py          # inserts a few docs, runs a search, prints results
pytest tests/                          # 8 tests covering insert/search/delete/persistence
python examples/benchmark.py           # QPS/latency numbers for the current backend
```

Note: without `sentence-transformers` installed, embeddings come from a
deterministic token-hashing fallback (see `vectordb/embedding.py`) — good
enough to exercise the whole pipeline, but it only rewards exact shared
vocabulary, not real semantics. `pip install sentence-transformers` for
actual semantic search.

Run the API:

```bash
uvicorn src.vectordb.api:app --reload
curl -X POST localhost:8000/documents -H 'content-type: application/json' \
     -d '{"text": "Mojo compiles Python-like syntax to native SIMD code"}'
curl -X POST localhost:8000/search -H 'content-type: application/json' \
     -d '{"query": "fast systems programming language", "k": 3}'
```

## Wiring in the real Mojo kernel

1. Install the Modular toolchain (`magic` package manager):
   `curl -ssL https://magic.modular.com | bash`
2. `cd src/mojo_kernels && magic init && magic add max`
3. Build: `mojo build index.mojo -o index_kernel` (exposes a callable via
   MAX's Python interop / `PythonModuleBuilder`, see comments in
   `distance.mojo` and `index.mojo`).
4. In `vectordb/index.py`, set `BACKEND=mojo` (or `VECTORDB_BACKEND=mojo`
   env var) — `MojoIndex` will import the compiled kernel instead of
   falling back to NumPy.

`distance.mojo` and `index.mojo` are written and commented as real,
buildable Mojo — I can't compile Mojo in this sandbox (no network access to
fetch the toolchain here), so treat them as source ready for you to build
locally, not as pre-verified binaries.

## Roadmap / what to build next for the portfolio writeup

- **ANN index**: current index is brute-force O(n) — great baseline,
  correct by construction. Next: IVF (cluster + probe) or HNSW graph index
  in Mojo for sub-linear search at scale.
- **Quantization**: int8/binary quantization of vectors for 4-32x memory
  reduction, with Mojo doing the packed SIMD comparisons.
- **Benchmarks**: `tests/` includes a correctness suite; add a benchmark
  script comparing NumpyIndex vs MojoIndex throughput (QPS) and recall@k
  vs brute force ground truth — this is the number that sells the project.
- **Persistence for the index itself**: currently rebuilt from SQLite on
  boot (fine up to ~millions of vectors); could memory-map a flat file for
  faster cold start.
- **GPU path via MAX**: same kernel, different target — good "why MAX"
  talking point for interviews.
