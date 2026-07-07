import base64
import hashlib
import json
import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
  
from app import config, pdf_utils, chunking, embeddings, vectorstore, store, retrieval, llm

app = FastAPI(title="Document RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sources"],
)


@app.on_event("startup")
def clear_old_documents():
    """Clear old documents, uploads, and chroma db on startup/refresh to prevent duplicates."""
    if os.path.exists(config.STORE_PATH):
        try:
            os.remove(config.STORE_PATH)
        except Exception:
            pass

    if os.path.exists(config.UPLOADS_DIR):
        try:
            shutil.rmtree(config.UPLOADS_DIR)
            os.makedirs(config.UPLOADS_DIR, exist_ok=True)
        except Exception:
            pass

    if os.path.exists(config.CHROMA_DIR):
        try:
            shutil.rmtree(config.CHROMA_DIR)
            os.makedirs(config.CHROMA_DIR, exist_ok=True)
        except Exception:
            pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(400, "Only .pdf and .txt files are supported.")

    file_bytes = await file.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = store.find_document_by_hash(content_hash)
    if existing:
        return {
            "document_id": existing["document_id"],
            "filename": existing["filename"],
            "num_pages": existing["num_pages"],
            "num_chunks": existing["num_chunks"],
            "note": "This exact file was already uploaded — reusing the existing index instead of duplicating it.",
        }

    document_id = str(uuid.uuid4())
    save_path = os.path.join(config.UPLOADS_DIR, f"{document_id}{ext}")
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    try:
        pages = pdf_utils.extract_pages(save_path)
    except pdf_utils.EmptyDocumentError as e:
        raise HTTPException(422, str(e))

    chunks = chunking.chunk_pages(pages)
    if not chunks:
        raise HTTPException(422, "No chunks could be produced from this document.")

    chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
    texts = [c["text"] for c in chunks]
    vectors = embeddings.encode_documents(texts)
    metadatas = [
        {"page": c["page"], "filename": file.filename, "document_id": document_id}
        for c in chunks
    ]

    vectorstore.add_chunks(chunk_ids, vectors, texts, metadatas)

    store.add_document(
        document_id, file.filename, num_pages=len(pages), num_chunks=len(chunks),
        content_hash=content_hash,
    )
    store.add_chunks(
        document_id,
        file.filename,
        [{"chunk_id": cid, "page": c["page"], "text": c["text"]} for cid, c in zip(chunk_ids, chunks)],
    )

    return {
        "document_id": document_id,
        "filename": file.filename,
        "num_pages": len(pages),
        "num_chunks": len(chunks),
    }


@app.get("/documents")
def list_documents():
    return store.list_documents()


class ChatRequest(BaseModel):
    query: str
    document_id: str | None = None  # None = search across all uploaded documents


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    chunks = retrieval.retrieve(req.query, document_id=req.document_id)

    sources = [
        {
            "chunk_id": c["chunk_id"],
            "page": c["metadata"]["page"],
            "filename": c["metadata"]["filename"],
            "text": c["text"][:300],  # snippet for the "view sources" panel
        }
        for c in chunks
    ]
    sources_b64 = base64.b64encode(json.dumps(sources).encode("utf-8")).decode("ascii")

    if not chunks:
        def empty_gen():
            yield "I couldn't find that information in the document."
        return StreamingResponse(
            empty_gen(), media_type="text/plain", headers={"X-Sources": sources_b64}
        )

    def token_stream():
        for piece in llm.stream_answer(req.query, chunks):
            yield piece

    return StreamingResponse(
        token_stream(), media_type="text/plain", headers={"X-Sources": sources_b64}
    )