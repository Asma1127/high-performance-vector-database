"""REST API for VectorDB.

Run with:
    uvicorn src.vectordb.api:app --reload
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .db import VectorDB

app = FastAPI(title="VectorDB", version="0.1.0")
db = VectorDB(path="vectordb.sqlite3")


class InsertRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(db)}


@app.post("/documents")
def insert_document(req: InsertRequest):
    doc_id = db.insert(req.text, req.metadata)
    return {"id": doc_id}


@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    doc = db.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    deleted = db.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id}


@app.post("/search")
def search(req: SearchRequest):
    return {"results": db.search(req.query, req.k)}
