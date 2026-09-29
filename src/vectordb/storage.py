
"""SQLite-backed persistence for documents and their vectors.

SQLite is the source of truth. The in-memory index is a disposable
performance layer rebuilt from this table on startup.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    vector      BLOB NOT NULL,
    dim         INTEGER NOT NULL CHECK (dim > 0),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_documents_created_at
ON documents(created_at);
"""


@dataclass
class Document:
    id: int
    text: str
    metadata: dict
    vector: np.ndarray
    created_at: Optional[str] = None


class Storage:
    """SQLite persistence with one connection per thread."""

    def __init__(self, path: str = "vectordb.sqlite3"):
        self.path = str(path)

        parent = Path(self.path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._schema_lock = threading.Lock()

        with self._schema_lock:
            conn = self._connect()
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Return this thread's SQLite connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(
                self.path,
                timeout=30,
                check_same_thread=False
            )
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn

        return self._local.conn

    @staticmethod
    def _validate_text(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Document text must be a string.")

        text = text.strip()
        if not text:
            raise ValueError("Document text cannot be empty.")

        return text

    @staticmethod
    def _validate_metadata(metadata: Optional[dict]) -> dict:
        if metadata is None:
            return {}

        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary.")

        # Ensure metadata can be safely stored as JSON.
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Metadata must contain JSON-serializable values."
            ) from exc

        return metadata

    @staticmethod
    def _validate_vector(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1:
            raise ValueError("Vector must be one-dimensional.")

        if vector.size == 0:
            raise ValueError("Vector cannot be empty.")

        if not np.all(np.isfinite(vector)):
            raise ValueError("Vector must contain only finite values.")

        return np.ascontiguousarray(vector)

    def insert(
        self,
        text: str,
        vector: np.ndarray,
        metadata: Optional[dict] = None
    ) -> int:
        """Insert one document and return its generated ID."""
        text = self._validate_text(text)
        vector = self._validate_vector(vector)
        metadata = self._validate_metadata(metadata)

        conn = self._connect()

        cur = conn.execute(
            """
            INSERT INTO documents (text, metadata, vector, dim)
            VALUES (?, ?, ?, ?)
            """,
            (
                text,
                json.dumps(metadata, ensure_ascii=False),
                sqlite3.Binary(vector.tobytes()),
                int(vector.size),
            ),
        )
        conn.commit()

        return int(cur.lastrowid)

    def delete(self, doc_id: int) -> bool:
        """Delete a document. Return True if a row was deleted."""
        conn = self._connect()

        cur = conn.execute(
            "DELETE FROM documents WHERE id = ?",
            (int(doc_id),)
        )
        conn.commit()

        return cur.rowcount > 0

    def get(self, doc_id: int) -> Optional[Document]:
        """Retrieve one document by ID."""
        conn = self._connect()

        row = conn.execute(
            """
            SELECT id, text, metadata, vector, dim, created_at
            FROM documents
            WHERE id = ?
            """,
            (int(doc_id),)
        ).fetchone()

        return self._row_to_doc(row) if row else None

    def all(self) -> Iterator[Document]:
        """Iterate through all stored documents in ID order."""
        conn = self._connect()

        rows = conn.execute(
            """
            SELECT id, text, metadata, vector, dim, created_at
            FROM documents
            ORDER BY id
            """
        )

        for row in rows:
            yield self._row_to_doc(row)

    def count(self) -> int:
        """Return the number of stored documents."""
        conn = self._connect()

        return int(
            conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
        )

    def close(self) -> None:
        """Close the current thread's connection."""
        conn = getattr(self._local, "conn", None)

        if conn is not None:
            conn.close()
            del self._local.conn

    @staticmethod
    def _row_to_doc(row) -> Document:
        """Convert a SQLite row into a Document object."""
        doc_id, text, metadata_json, blob, dim, created_at = row

        vector = np.frombuffer(
            blob,
            dtype=np.float32,
            count=dim
        ).copy()

        if vector.size != dim:
            raise ValueError(
                f"Stored vector dimension mismatch for document {doc_id}."
            )

        metadata = json.loads(metadata_json)

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Invalid metadata format for document {doc_id}."
            )

        return Document(
            id=doc_id,
            text=text,
            metadata=metadata,
            vector=vector,
            created_at=created_at
        )