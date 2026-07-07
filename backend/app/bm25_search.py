import re
from rank_bm25 import BM25Okapi
from app import config


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: dict):
    """
    chunks: dict {chunk_id: {"text": ..., ...}}
    Returns (bm25_index, ordered_chunk_ids) or (None, []) if empty.
    """
    if not chunks:
        return None, []
    chunk_ids = list(chunks.keys())
    corpus = [_tokenize(chunks[cid]["text"]) for cid in chunk_ids]
    bm25 = BM25Okapi(corpus)
    return bm25, chunk_ids


def bm25_search(bm25, chunk_ids, query: str, top_k: int):
    if bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"chunk_id": cid, "score": float(score)} for cid, score in ranked if score > 0]


def reciprocal_rank_fusion(rank_lists: list, k: int = None):
    """
    rank_lists: list of ranked lists, each a list of dicts with "chunk_id",
                ordered best-first (e.g. output of vector search or bm25_search).
    Returns a list of {"chunk_id", "rrf_score"} sorted descending by fused score.

    RRF avoids needing to normalize/compare scores from different retrieval
    methods (cosine similarity vs BM25 score are on unrelated scales).
    """
    if k is None:
        k = config.RRF_K
    fused_scores = {}
    for rank_list in rank_lists:
        for rank, item in enumerate(rank_list):
            cid = item["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    fused = [{"chunk_id": cid, "rrf_score": score} for cid, score in fused_scores.items()]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused
