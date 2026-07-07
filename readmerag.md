# 📄 Document RAG Assistant
> A full-stack Retrieval-Augmented Generation (RAG) system with advanced hybrid retrieval, cross-encoder reranking, and mathematically grounded page-level citations.

## 🚀 Executive Summary
This project implements a highly accurate Document RAG Assistant capable of ingesting PDF and text documents and answering questions based strictly on the provided context. It tackles common RAG issues—such as hallucinated citations and loss of context during chunking—through page-aware chunking, Reciprocal Rank Fusion (RRF), and a deterministic citation pipeline.

---

## 🛠️ Tech Stack & Architecture

The architecture was deliberately chosen for performance, accuracy, and reliability:

| Layer | Technology | Rationale |
|---|---|---|
| **Backend** | `FastAPI` | Highly performant, async-first, and native support for streaming LLM responses. |
| **PDF Processing**| `PyMuPDF (fitz)` | Exceptionally fast extraction that strictly preserves accurate page-number mapping. |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Splits on semantic boundaries (paragraphs, sentences) rather than arbitrary character limits. |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Top-tier open-source embedding model optimized for retrieval tasks. |
| **Vector Database**| `ChromaDB` | Embedded, persistent local vector store requiring zero external infrastructure. |
| **Lexical Search** | `rank_bm25` (BM25Okapi) | Keyword-based retrieval to catch exact terms, acronyms, and numbers that dense embeddings might miss. |
| **Ranking Fusion** | Reciprocal Rank Fusion (RRF) | Merges Vector and BM25 search results gracefully without requiring score normalization. |
| **Reranker** | `BAAI/bge-reranker-base` | Cross-encoder model that directly scores the query against candidates for maximum precision. |
| **LLM Engine** | Gemini 2.5 Flash | Provides rapid, token-by-token streaming and exceptional instruction-following for grounded generation. |
| **Frontend** | `Streamlit` | Provides an interactive, reactive chat UI with a dedicated "View Sources" expandable panel. |

---

## 🧩 Data Pipeline & Chunking Strategy

A major failure point in naive RAG systems is the loss of document structure (like page boundaries) during the chunking phase. This system solves this through **Page-Aware Chunking**:

1. **Strict Per-Page Extraction**: Rather than dumping the entire PDF into a single continuous string, PyMuPDF extracts text page-by-page. 
2. **Isolated Splitting**: Each page's text is independently split using LangChain's `RecursiveCharacterTextSplitter` (configured to 700 characters with a 150-character overlap). 
3. **Dual Storage**: Chunks are embedded and stored in ChromaDB for semantic search, and simultaneously saved to a local JSON store (`backend/data/store.json`) for BM25 indexing and fast metadata hydration.

---

## 🔍 Advanced Hybrid Retrieval

To maximize retrieval recall and precision, the system employs a multi-stage hybrid pipeline for every user query:

1. **Dense Vector Search**: The query is embedded (using BGE's recommended instruction prefix) to retrieve the top 10 semantically similar chunks from ChromaDB.
2. **Sparse Keyword Search (BM25)**: Concurrently, the query is run against a BM25 index built from the document corpus to retrieve 10 chunks based on exact keyword overlap.
3. **Reciprocal Rank Fusion (RRF)**: The two ranked lists are merged using the RRF algorithm (`score = Σ 1/(k + rank)`). This eliminates the notoriously difficult problem of normalizing cosine similarity scores against BM25 scores, which exist on completely different scales.
4. **Cross-Encoder Reranking**: The fused top candidates are passed through a `bge-reranker-base` cross-encoder. Unlike embeddings (which compare two independent vectors), a cross-encoder passes the query and the chunk through the transformer layers *together*, resulting in a highly accurate relevance score. The top 4 chunks are retained.

---

## 📑 Grounding & Deterministic Citations

LLMs are notoriously prone to "hallucinating" citations—inventing page numbers that look plausible but are entirely fake. This project implements a deterministic citation architecture:

1. **Strict Prompting**: The top 4 reranked chunks (labeled with their precise page numbers) are injected into the LLM prompt. The system instructions strictly forbid the use of outside knowledge and mandate a fallback response (*"I couldn't find that information in the document"*) if the context is insufficient.
2. **Out-of-Band Citations**: **We do not rely on the LLM to tell us what sources it used.** Instead, the backend independently serializes the exact retrieved chunks (including filenames, page numbers, and text snippets) and sends them to the frontend via a custom `X-Sources` HTTP header.
3. **Transparent UI Validation**: The Streamlit frontend decodes this header and populates the "View Sources" panel. It parses the LLM's inline citations (e.g., `(p. 3)`) and cross-references them against the `X-Sources` metadata. This allows the UI to distinctly categorize sources as **"Cited in this answer"** versus **"Retrieved but not used"**. 
*Result: What the user sees in the sources panel is guaranteed to be the exact text retrieved from the database, completely bypassing LLM hallucination.*

---
