import base64
import json
import os
import re

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Document RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 Document RAG Assistant")
st.caption("Upload a PDF or text file, then ask questions grounded in its content.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}


def fetch_documents():
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


# ---------------- Sidebar: upload + document selector ----------------
with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF or .txt file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("Process document", use_container_width=True):
            with st.spinner("Extracting, chunking, and embedding..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"'{data['filename']}' processed: "
                            f"{data['num_pages']} pages, {data['num_chunks']} chunks."
                        )
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach backend: {e}")

    st.divider()
    st.header("Chat scope")
    docs = fetch_documents()
    doc_options = {"All documents": None}
    for d in docs:
        doc_options[f"{d['filename']}"] = d["document_id"]

    selected_label = st.selectbox("Ask questions about", list(doc_options.keys()))
    selected_document_id = doc_options[selected_label]

    if docs:
        st.caption(f"{len(docs)} document(s) uploaded so far.")
    else:
        st.caption("No documents uploaded yet.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- Chat history ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View sources"):
                cited = [s for s in msg["sources"] if s.get("used", False)]
                uncited = [s for s in msg["sources"] if not s.get("used", False)]
                
                if cited:
                    st.markdown("#### Cited in this answer")
                    for s in cited:
                        st.markdown(f"**{s['filename']} — page {s['page']}**")
                        st.caption(s["text"] + "...")
                        st.divider()
                
                if uncited:
                    st.markdown("#### Retrieved but not used")
                    for s in uncited:
                        st.markdown(f"**{s['filename']} — page {s['page']}**")
                        st.caption(s["text"] + "...")
                        st.divider()


# ---------------- Chat input ----------------
user_query = st.chat_input("Ask a question about the document...")

if user_query:
    if not docs:
        st.warning("Please upload a document first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        sources = []

        try:
            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={"query": user_query, "document_id": selected_document_id},
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()

            sources_header = resp.headers.get("X-Sources")
            if sources_header:
                sources = json.loads(base64.b64decode(sources_header).decode("utf-8"))

            for chunk_bytes in resp.iter_content(chunk_size=None):
                if chunk_bytes:
                    full_response += chunk_bytes.decode("utf-8", errors="ignore")
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except requests.exceptions.RequestException as e:
            full_response = f"Error contacting backend: {e}"
            placeholder.markdown(full_response)

        if sources:
            # Parse cited pages from the LLM response
            cited_pages = set(re.findall(r'\(p\.\s*(\d+)\)', full_response))
            for s in sources:
                s["used"] = str(s["page"]) in cited_pages

            with st.expander("View sources"):
                cited = [s for s in sources if s.get("used", False)]
                uncited = [s for s in sources if not s.get("used", False)]
                
                if cited:
                    st.markdown("#### Cited in this answer")
                    for s in cited:
                        st.markdown(f"**{s['filename']} — page {s['page']}**")
                        st.caption(s["text"] + "...")
                        st.divider()
                
                if uncited:
                    st.markdown("#### Retrieved but not used")
                    for s in uncited:
                        st.markdown(f"**{s['filename']} — page {s['page']}**")
                        st.caption(s["text"] + "...")
                        st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )
