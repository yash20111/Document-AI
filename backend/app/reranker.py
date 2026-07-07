from app import config

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(config.RERANKER_MODEL_NAME)
    return _reranker


def rerank(query: str, candidates: list, top_k: int):
    """
    candidates: list of dicts with at least "chunk_id" and "text"
    Returns the top_k candidates re-sorted by cross-encoder relevance score.
    """
    if not config.USE_RERANKER or not candidates:
        return candidates[:top_k]

    model = _get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
