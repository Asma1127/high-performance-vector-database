"""Standalone demo: insert a few docs, run a search, print results.

Run: python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vectordb.db import VectorDB  # noqa: E402

DOCS = [
    "Mojo is a systems programming language that mixes Python syntax with C-like performance",
    "MAX is Modular's platform for compiling and serving AI models across CPUs and GPUs",
    "SQLite is a lightweight, file-based relational database engine",
    "A vector database stores embeddings and supports fast similarity search",
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris",
    "Cosine similarity measures the angle between two vectors, ignoring magnitude",
    "Sourdough bread relies on wild yeast and lactic acid bacteria for its rise and tang",
]


def main():
    db = VectorDB(path="quickstart.sqlite3")
    print(f"Inserting {len(DOCS)} documents...")
    db.insert_batch(DOCS)
    print(f"Index now has {len(db)} documents.\n")

    for query in ["fast compiled programming language", "how does semantic search work", "baking bread"]:
        print(f"Query: {query!r}")
        for r in db.search(query, k=3):
            print(f"  [{r['score']:.3f}] {r['text']}")
        print()


if __name__ == "__main__":
    main()
