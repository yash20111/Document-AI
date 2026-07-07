from app import config, store, embeddings, vectorstore, bm25_search, reranker


def retrieve(query: str, document_id: str = None):
    """
    Full hybrid retrieval pipeline for a query, optionally scoped to one document.
    Returns a list of chunk dicts: {"chunk_id", "text", "metadata": {"page", "filename", ...}}
    """
    where = {"document_id": document_id} if document_id else None

    # 1. Vector search
    query_embedding = embeddings.encode_query(query)
    vector_hits = vectorstore.query(query_embedding, top_k=config.TOP_K_VECTOR, where=where)

    # 2. BM25 search (rebuilt fresh from the JSON store each query - fine at assessment scale)
    chunk_pool = store.get_all_chunks(document_id=document_id)
    bm25_index, chunk_ids = bm25_search.build_bm25_index(chunk_pool)
    bm25_hits = bm25_search.bm25_search(bm25_index, chunk_ids, query, top_k=config.TOP_K_BM25)

    # 3. Reciprocal Rank Fusion of the two ranked lists
    fused = bm25_search.reciprocal_rank_fusion([vector_hits, bm25_hits])
    fused = fused[: config.TOP_K_FUSED]

    # 4. Hydrate fused chunk_ids back into full chunk dicts (text + metadata),
    #    skipping any chunk whose text we've already seen. This guards against
    #    duplicate uploads of the same document producing repeated chunks with
    #    different chunk_ids (RRF fusion only dedupes by chunk_id, not content).
    hydrated = []
    seen_text = set()
    for item in fused:
        cid = item["chunk_id"]
        chunk = chunk_pool.get(cid)
        if chunk is None:
            continue
        normalized = " ".join(chunk["text"].split()).lower()
        if normalized in seen_text:
            continue
        seen_text.add(normalized)
        hydrated.append({
            "chunk_id": cid,
            "text": chunk["text"],
            "metadata": {
                "page": chunk["page"],
                "filename": chunk["filename"],
                "document_id": chunk["document_id"],
            },
        })

    # 5. Optional cross-encoder rerank down to the final top-K sent to the LLM
    final = reranker.rerank(query, hydrated, top_k=config.TOP_K_FINAL)
    return final
