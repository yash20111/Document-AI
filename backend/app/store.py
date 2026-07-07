import json
import os
import threading
from app import config

_lock = threading.Lock()


def _load():
    if not os.path.exists(config.STORE_PATH):
        return {"documents": {}, "chunks": {}}
    with open(config.STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data):
    with open(config.STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_document(document_id, filename, num_pages, num_chunks, content_hash=None):
    with _lock:
        data = _load()
        data["documents"][document_id] = {
            "filename": filename,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "content_hash": content_hash,
        }
        _save(data)


def find_document_by_hash(content_hash):
    """Returns the existing document dict (with document_id) if this exact
    file content was already ingested, else None. Prevents duplicate uploads
    from polluting retrieval with repeated chunks."""
    data = _load()
    for doc_id, info in data["documents"].items():
        if content_hash and info.get("content_hash") == content_hash:
            return {"document_id": doc_id, **info}
    return None


def add_chunks(document_id, filename, chunks_with_ids):
    """chunks_with_ids: list of {"chunk_id", "page", "text"}"""
    with _lock:
        data = _load()
        for c in chunks_with_ids:
            data["chunks"][c["chunk_id"]] = {
                "text": c["text"],
                "page": c["page"],
                "document_id": document_id,
                "filename": filename,
            }
        _save(data)


def list_documents():
    data = _load()
    return [
        {"document_id": doc_id, **info}
        for doc_id, info in data["documents"].items()
    ]


def get_all_chunks(document_id=None):
    data = _load()
    chunks = data["chunks"]
    if document_id:
        return {cid: c for cid, c in chunks.items() if c["document_id"] == document_id}
    return chunks