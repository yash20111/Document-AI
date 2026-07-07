"""
Page-aware text extraction.

Why per-page extraction matters:
If we joined the whole document into one big string before chunking, a single
chunk could straddle two pages, making the eventual "cited page number"
wrong for part of that chunk. Extracting page-by-page and chunking
page-by-page guarantees every chunk maps to exactly one real page.
"""
import os
import fitz  # PyMuPDF


class EmptyDocumentError(Exception):
    pass


def extract_pages(file_path: str):
    """
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    Page numbers are 1-indexed (human-friendly, matches what a user sees in a PDF viewer).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        pages = []
        doc = fitz.open(file_path)
        try:
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"page": i + 1, "text": text})
        finally:
            doc.close()

        if not pages:
            raise EmptyDocumentError(
                "No extractable text found in this PDF. It may be a scanned/image-only "
                "PDF that requires OCR, which this app does not support."
            )
        return pages

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if not content:
            raise EmptyDocumentError("The uploaded .txt file is empty.")
        # Plain text files have no real "page" concept -> treat as a single page.
        return [{"page": 1, "text": content}]

    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .txt are supported.")
