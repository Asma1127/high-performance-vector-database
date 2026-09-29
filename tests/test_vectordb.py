import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vectordb.db import VectorDB  # noqa: E402


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        yield VectorDB(path=os.path.join(tmp, "test.sqlite3"))


def test_insert_and_count(db):
    doc_id = db.insert("Mojo is a systems programming language")
    assert isinstance(doc_id, int)
    assert len(db) == 1


def test_search_returns_relevant_result_first(db):
    db.insert("The cat sat on the mat")
    db.insert("Rockets launch into orbit using powerful engines")
    db.insert("Cats love to nap in sunny spots")

    # NOTE: with the default hash-embedding fallback (no sentence-transformers
    # installed), relevance is driven by *exact token overlap*, not real
    # semantics. Query on shared vocabulary ("cat"/"nap") rather than
    # synonyms — this test is about the search pipeline being wired
    # correctly, not about embedding quality. Install sentence-transformers
    # for true semantic matching (synonyms, paraphrase, etc.).
    results = db.search("cat nap", k=3)
    assert len(results) == 3
    top_texts = {r["text"] for r in results[:2]}
    assert "The cat sat on the mat" in top_texts
    assert "Cats love to nap in sunny spots" in top_texts


def test_metadata_roundtrip(db):
    doc_id = db.insert("Vector databases enable semantic search", metadata={"source": "test"})
    doc = db.get(doc_id)
    assert doc["metadata"] == {"source": "test"}


def test_delete_removes_from_index_and_storage(db):
    doc_id = db.insert("Temporary document")
    assert len(db) == 1
    assert db.delete(doc_id) is True
    assert len(db) == 0
    assert db.get(doc_id) is None


def test_delete_nonexistent_returns_false(db):
    assert db.delete(9999) is False


def test_insert_batch(db):
    ids = db.insert_batch(["doc one", "doc two", "doc three"])
    assert len(ids) == 3
    assert len(db) == 3


def test_persistence_across_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "persist.sqlite3")
        db1 = VectorDB(path=path)
        doc_id = db1.insert("Persisted document")

        db2 = VectorDB(path=path)  # reopen, index should rebuild from SQLite
        assert len(db2) == 1
        assert db2.get(doc_id)["text"] == "Persisted document"


def test_search_on_empty_db(db):
    assert db.search("anything", k=5) == []
