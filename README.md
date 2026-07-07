# Document-Based AI Assistant (RAG)

A full-stack RAG assistant: upload a PDF/text document, ask questions, and get
answers grounded in the document with page-level citations.

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend framework | FastAPI | Async, typed, streaming-friendly |
| PDF extraction | PyMuPDF (`fitz`) | Fast, accurate, preserves page numbers |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (700/150) | Splits on semantic boundaries, not fixed characters |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) | Open source, fast, strong retrieval accuracy |
| Vector DB | ChromaDB (persistent, local) | Zero-config, no external service needed |
| Keyword search | `rank_bm25` (BM25Okapi) | Catches exact terms/numbers embeddings can miss |
| Fusion | Reciprocal Rank Fusion (RRF) | Combines vector + BM25 rankings without score normalization issues |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder, toggleable) | Precision boost on final candidate set |
| LLM | Gemini 2.5 Flash (streaming) | Fast, cheap, strong instruction-following |
| Frontend | Streamlit | Simple chat UI with streaming + sources panel |

## How Chunking Works

1. **Per-page extraction**: PyMuPDF extracts text page-by-page (not as one big
   joined string). This is what makes citations trustworthy — if we joined
   everything first, a single chunk could straddle two pages and its cited
   page number would be wrong for part of the text.
2. **Per-page splitting**: Each page's text is independently split using
   `RecursiveCharacterTextSplitter` (chunk size 700 characters, overlap 150).
   Because splitting happens per page, every resulting chunk is guaranteed to
   belong to exactly one page.
3. Each chunk is stored with: chunk ID, page number, source filename,
   document ID, and the chunk text — both in ChromaDB (with the embedding)
   and in a local JSON store (`backend/data/store.json`) used for BM25 and
   for hydrating full chunk text/metadata after retrieval.

## How Retrieval Works

For every user query:

1. **Vector search** — the query is embedded with `bge-small-en-v1.5`
   (using BGE's recommended query instruction prefix) and the top 10 most
   similar chunks are retrieved from Chroma.
2. **BM25 keyword search** — the same query is run against a BM25 index
   built from all chunk text (optionally scoped to a single document),
   returning its own top 10 ranked chunks.
3. **Reciprocal Rank Fusion (RRF)** — the two ranked lists are merged by
   rank position (`score = Σ 1/(k + rank)`), avoiding the need to normalize
   cosine similarity against BM25 scores, which live on unrelated scales.
4. **Reranking (optional, on by default)** — the fused top candidates are
   re-scored by a `bge-reranker-base` cross-encoder, which directly compares
   the query against each candidate chunk for a more precise final ranking.
   The top 4 chunks after reranking are kept.
5. **Grounded generation** — the final chunks (each labeled with its page
   number) are inserted into a strict prompt instructing Gemini to answer
   **only** from the provided context, and to say
   *"I couldn't find that information in the document"* if the answer isn't
   present. The response is streamed token-by-token to the frontend.
6. **Citations are not parsed from the LLM's text.** The backend independently
   returns the exact chunks it retrieved (page number, filename, snippet) as
   structured metadata in an `X-Sources` response header. The Streamlit
   "View sources" panel renders this directly — so what's shown as a source
   is guaranteed to be what was actually retrieved and sent to the model,
   not something the model claims after the fact.

## Setup & Run

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GEMINI_API_KEY
uvicorn app.main:app --reload --port 8000
```

First run will download the embedding and reranker models from HuggingFace
(a few hundred MB total) — this only happens once.

### 2. Frontend

In a second terminal:

```bash
cd frontend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### 3. Use it

1. Upload a PDF or `.txt` file in the sidebar and click **Process document**.
2. Pick "All documents" or a specific file under **Chat scope**.
3. Ask a question in the chat box — the answer streams in, and you can
   expand **View sources** to see exactly which pages/chunks were used.

## Configuration

All tunables live in `backend/.env` (see `.env.example`):
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_VECTOR`, `TOP_K_BM25`, `TOP_K_FUSED`,
`TOP_K_FINAL`, `RRF_K`, `USE_RERANKER`.

If deploying to a memory-constrained host, set `USE_RERANKER=false` to skip
loading the cross-encoder model — retrieval will fall back to RRF-fused
results only.

## Notes / Limitations

- Scanned/image-only PDFs are not supported (no OCR) — upload will return a
  clear error in that case rather than silently indexing nothing.
- BM25 index is rebuilt from the JSON store on each query rather than kept
  incrementally in memory — simple and correct at assessment scale, but
  would want to be an incremental/persisted index for larger corpora.
