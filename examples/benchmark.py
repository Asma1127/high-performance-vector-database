"""Benchmark: insert N synthetic vectors, measure search QPS/latency.

Run: python examples/benchmark.py --n 5000 --queries 200 --k 10

This is the number to put in your portfolio writeup: run once with
VECTORDB_BACKEND=numpy (default) and once with VECTORDB_BACKEND=mojo (after
building the kernel) and compare.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from vectordb.index import make_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="number of vectors to index")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    backend = os.environ.get("VECTORDB_BACKEND", "numpy")
    print(f"Backend: {backend}")
    print(f"Building index with {args.n} random unit vectors (dim={args.dim})...")

    rng = np.random.default_rng(42)
    index = make_index(args.dim)

    t0 = time.perf_counter()
    for i in range(args.n):
        v = rng.normal(size=args.dim).astype(np.float32)
        index.add(i, v)
    build_time = time.perf_counter() - t0
    print(f"Build time: {build_time:.3f}s  ({args.n / build_time:.0f} inserts/sec)")

    queries = rng.normal(size=(args.queries, args.dim)).astype(np.float32)
    t0 = time.perf_counter()
    for q in queries:
        index.search(q, args.k)
    search_time = time.perf_counter() - t0

    qps = args.queries / search_time
    print(f"Search time: {search_time:.3f}s for {args.queries} queries")
    print(f"QPS: {qps:.1f}")
    print(f"Avg latency: {1000 * search_time / args.queries:.3f} ms/query")


if __name__ == "__main__":
    main()
