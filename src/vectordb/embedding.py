"""Turns text into vectors.

Uses sentence-transformers if it's installed (real semantic embeddings).
Otherwise falls back to a deterministic hashing-trick embedding so the rest
of the system (index, storage, API) is fully exercisable without any extra
downloads. The fallback is NOT semantically meaningful beyond exact/partial
token overlap — swap in real embeddings for real search quality.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

DEFAULT_DIM = 384  # matches all-MiniLM-L6-v2, so swapping backends is a no-op


class Embedder:
    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dim = self._model.get_sentence_embedding_dimension()
        except Exception:
            self._model = None  # fall back to hashing embedding below

    def embed(self, text: str) -> np.ndarray:
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        return self._hash_embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        return np.stack([self._hash_embed(t) for t in texts])

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic bag-of-tokens hashing embedding (no model download).

        Each token deterministically votes on `dim` buckets via md5; the
        result is L2-normalized so cosine similarity behaves sensibly.
        This rewards shared vocabulary between query and document, which is
        a reasonable stand-in for demoing the pipeline end-to-end.
        """
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
