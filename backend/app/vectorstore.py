import chromadb
from app import config

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(chunk_ids, embeddings, texts, metadatas):
    collection = get_collection()
    collection.add(ids=chunk_ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def query(query_embedding, top_k, where=None):
    collection = get_collection()
    kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)

    if not results["ids"] or not results["ids"][0]:
        return []

    out = []
    for i in range(len(results["ids"][0])):
        out.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return out
