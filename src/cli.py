"""CLI: ingest a folder of .txt files, then query interactively.

Usage:
    python src/cli.py ingest ./my_docs --db mydb.sqlite3
    python src/cli.py search "some query" --db mydb.sqlite3 --k 5
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from vectordb.db import VectorDB  # noqa: E402


def cmd_ingest(args):
    db = VectorDB(path=args.db)
    files = sorted(Path(args.folder).glob("**/*.txt"))
    if not files:
        print(f"No .txt files found under {args.folder}")
        return
    texts, metas = [], []
    for f in files:
        texts.append(f.read_text(encoding="utf-8", errors="ignore"))
        metas.append({"source": str(f)})
    ids = db.insert_batch(texts, metas)
    print(f"Ingested {len(ids)} files into {args.db}")


def cmd_search(args):
    db = VectorDB(path=args.db)
    results = db.search(args.query, k=args.k)
    if not results:
        print("No results (is the database empty?)")
        return
    for r in results:
        source = r["metadata"].get("source", "")
        print(f"[{r['score']:.3f}] {source}")
        print(f"  {r['text'][:200]!r}")


def main():
    parser = argparse.ArgumentParser(description="VectorDB CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a folder of .txt files")
    p_ingest.add_argument("folder")
    p_ingest.add_argument("--db", default="vectordb.sqlite3")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="Run a search query")
    p_search.add_argument("query")
    p_search.add_argument("--db", default="vectordb.sqlite3")
    p_search.add_argument("--k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
