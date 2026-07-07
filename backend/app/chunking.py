from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import config

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages: list):
    """
    Input: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    Output: [{"page": 1, "text": "chunk text ..."}, ...]  (page-tagged chunks)

    Splitting per-page (instead of splitting one giant joined string) guarantees
    every chunk is attributable to exactly one page.
    """
    all_chunks = []
    for p in pages:
        pieces = _splitter.split_text(p["text"])
        for piece in pieces:
            piece = piece.strip()
            if piece:
                all_chunks.append({"page": p["page"], "text": piece})
    return all_chunks
