"""In-memory search index.

Defines an IndexBackend protocol so the rest of the app doesn't care whether
scoring happens in NumPy (portable default) or in a compiled Mojo kernel.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np


class IndexBackend(Protocol):
    def add(self, doc_id: int, vector: np.ndarray) -> None: ...
    def remove(self, doc_id: int) -> None: ...
    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]: ...
    def __len__(self) -> int: ...


class NumpyIndex:
    """Brute-force flat index using NumPy."""

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[int] = []
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def add(self, doc_id: int, vector: np.ndarray) -> None:
        vector = self._normalize(np.asarray(vector, dtype=np.float32))
        self._ids.append(doc_id)
        self._vectors = np.vstack(
            [self._vectors, vector[None, :]]
        )

    def remove(self, doc_id: int) -> None:
        if doc_id not in self._ids:
            return

        i = self._ids.index(doc_id)
        self._ids.pop(i)
        self._vectors = np.delete(
            self._vectors,
            i,
            axis=0
        )

    def search(
        self,
        query: np.ndarray,
        k: int
    ) -> list[tuple[int, float]]:

        if len(self._ids) == 0:
            return []

        q = self._normalize(
            np.asarray(query, dtype=np.float32)
        )

        # Cosine similarity because vectors are normalized.
        scores = self._vectors @ q

        k = min(k, len(self._ids))

        top_idx = np.argpartition(
            -scores,
            k - 1
        )[:k]

        top_idx = top_idx[
            np.argsort(-scores[top_idx])
        ]

        return [
            (self._ids[i], float(scores[i]))
            for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self._ids)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)

        if n > 0:
            return v / n

        return v


class MojoIndex:
    """Mojo-backed vector search using the compiled Mojo kernel."""

    def __init__(self, dim: int):
        self.dim = dim

        import sys
        from pathlib import Path

        kernel_path = (
            Path(__file__).resolve().parents[1]
            / "mojo_kernels"
        )

        if str(kernel_path) not in sys.path:
            sys.path.insert(0, str(kernel_path))

        try:
            import index_kernel
        except ImportError as e:
            raise ImportError(
                "Mojo kernel not built. Build it with:\n"
                "mojo build src/mojo_kernels/index.mojo "
                "--emit shared-lib "
                "-o src/mojo_kernels/index_kernel.so"
            ) from e

        self._kernel = index_kernel

        self._ids: list[int] = []
        self._vectors: list[np.ndarray] = []

    def add(
        self,
        doc_id: int,
        vector: np.ndarray
    ) -> None:

        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        self._ids.append(doc_id)
        self._vectors.append(vector)

    def remove(self, doc_id: int) -> None:
        if doc_id not in self._ids:
            return

        i = self._ids.index(doc_id)

        self._ids.pop(i)
        self._vectors.pop(i)

    def search(
        self,
        query: np.ndarray,
        k: int
    ) -> list[tuple[int, float]]:

        if not self._ids:
            return []

        query = np.asarray(
            query,
            dtype=np.float32
        )

        results = []

        for doc_id, vector in zip(
            self._ids,
            self._vectors
        ):
            score = float(
                self._kernel.cosine_similarity(
                    query,
                    vector
                )
            )

            results.append(
                (doc_id, score)
            )

        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return results[:k]

    def __len__(self) -> int:
        return len(self._ids)


def make_index(dim: int) -> IndexBackend:
    """Select the vector index backend."""

    backend = os.environ.get(
        "VECTORDB_BACKEND",
        "numpy"
    ).lower()

    if backend == "mojo":
        return MojoIndex(dim)

    return NumpyIndex(dim)