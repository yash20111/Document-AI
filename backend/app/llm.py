from google import genai
from google.genai import types
from app import config

# We initialize the client when streaming is called to pick up the API key correctly
# if it changes, or we can initialize it globally.
client = None

def get_client():
    global client
    if client is None:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
    return client


SYSTEM_INSTRUCTION = (
    "You are a document assistant. Answer ONLY using the provided context below. "
    "Do not use outside knowledge. If the answer is not present in the context, "
    "say exactly: \"I couldn't find that information in the document.\" "
    "When you use information from the context, cite the page it came from "
    "inline like (p. 3). Be concise and direct."
)


def _build_context(chunks: list) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Page {c['metadata']['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def stream_answer(query: str, chunks: list):
    """
    chunks: list of retrieved chunk dicts (each with "text" and "metadata"->"page")
    Yields text pieces as they are generated.
    """
    context = _build_context(chunks)
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )

    gemini_client = get_client()

    try:
        response = gemini_client.models.generate_content_stream(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
            )
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        yield f"⚠️ **Error**: {str(e)}"
