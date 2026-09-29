
"""Public VectorDB API.

Provides document insertion, batch insertion, semantic search,
deletion, retrieval, and rebuilding the in-memory index from SQLite.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .embedding import Embedder
from .index import make_index
from .storage import Storage


class VectorDB:
    """Main API for the semantic vector database."""

    def __init__(
        self,
        path: str = "vectordb.sqlite3",
        dim: Optional[int] = None
    ):
        if dim is not None and dim <= 0:
            raise ValueError("Embedding dimension must be positive.")

        self.embedder = Embedder(dim=dim) if dim is not None else Embedder()
        self.storage = Storage(path)
        self.index = make_index(self.embedder.dim)

        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the in-memory index using SQLite as the source of truth."""
        self.index = make_index(self.embedder.dim)

        for doc in self.storage.all():
            if doc.vector.size != self.embedder.dim:
                raise ValueError(
                    f"Document {doc.id} has dimension {doc.vector.size}, "
                    f"but the current embedder uses {self.embedder.dim}. "
                    "Use the same embedding model and dimension as the "
                    "stored documents, or regenerate their embeddings."
                )

            self.index.add(doc.id, doc.vector)

    def insert(
        self,
        text: str,
        metadata: Optional[dict] = None
    ) -> int:
        """Generate an embedding, persist the document, and index it."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text cannot be empty.")

        vector = self.embedder.embed(text)
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1 or vector.size != self.embedder.dim:
            raise ValueError("Embedder returned an unexpected vector shape.")

        doc_id = self.storage.insert(text, vector, metadata)
        self.index.add(doc_id, vector)

        return doc_id

    def insert_batch(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None
    ) -> list[int]:
        """Insert multiple documents and return their generated IDs."""
        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(texts) != len(metadatas):
            raise ValueError(
                "texts and metadatas must have the same length."
            )

        if any(not isinstance(meta, dict) for meta in metadatas):
            raise TypeError("Each metadata value must be a dictionary.")

        vectors = self.embedder.embed_batch(texts)
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embedder must return a 2D batch of vectors.")

        if vectors.shape != (len(texts), self.embedder.dim):
            raise ValueError(
                "Embedder returned an unexpected batch shape."
            )

        ids = []

        for text, vector, metadata in zip(texts, vectors, metadatas):
            doc_id = self.storage.insert(text, vector, metadata)
            self.index.add(doc_id, vector)
            ids.append(doc_id)

        return ids

    def search(
        self,
        query: str,
        k: int = 5
    ) -> list[dict]:
        """Search for documents most similar to a natural-language query."""
        if not isinstance(query, str) or not query.strip():
            return []

        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer.")

        if k <= 0:
            return []

        if len(self.index) == 0:
            return []

        vector = self.embedder.embed(query)
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1 or vector.size != self.embedder.dim:
            raise ValueError("Query embedding has an unexpected shape.")

        hits = self.index.search(vector, k)
        results = []

        for doc_id, score in hits:
            doc = self.storage.get(doc_id)

            if doc is None:
                continue

            results.append({
                "id": doc.id,
                "text": doc.text,
                "metadata": doc.metadata,
                "score": float(score),
                "created_at": doc.created_at,
            })

        return results

    def delete(self, doc_id: int) -> bool:
        """Delete a document from SQLite and the in-memory index."""
        deleted = self.storage.delete(doc_id)

        if deleted:
            self.index.remove(doc_id)

        return deleted

    def get(self, doc_id: int) -> Optional[dict]:
        """Retrieve a document and its metadata by ID."""
        doc = self.storage.get(doc_id)

        if doc is None:
            return None

        return {
            "id": doc.id,
            "text": doc.text,
            "metadata": doc.metadata,
            "created_at": doc.created_at,
        }

    def count(self) -> int:
        """Return the number of persisted documents."""
        return self.storage.count()

    def close(self) -> None:
        """Close the current thread's SQLite connection."""
        self.storage.close()

    def __len__(self) -> int:
        """Return the number of documents in the in-memory index."""
        return len(self.index)