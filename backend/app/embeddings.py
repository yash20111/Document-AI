from sentence_transformers import SentenceTransformer
from app import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def encode_documents(texts: list):
    """Embed chunk texts for storage. No instruction prefix needed for bge on the passage side."""
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def encode_query(query: str):
    """Embed a user query. BGE models recommend a specific instruction prefix for queries."""
    model = get_model()
    prefixed = f"{config.BGE_QUERY_INSTRUCTION}{query}"
    embedding = model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)
    return embedding[0].tolist()
