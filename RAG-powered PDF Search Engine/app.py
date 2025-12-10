# app.py
import streamlit as st
from rag_engine import (
    load_pdf,
    chunk_text,
    build_faiss_index,
    answer_question,
    load_faiss_index,
)

import os

st.set_page_config(page_title="📄 RAG PDF Search Engine", layout="wide")

st.title("📄 RAG-powered PDF Search Engine")
st.write("Upload a PDF, ask questions, and get answers with page citations.")

# --- SESSION STATE SETUP ---
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None

# --- 1. PDF UPLOAD (MAIN AREA) ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # If new file uploaded (different from previous one), rebuild index
    if uploaded_file.name != st.session_state.current_pdf_name:
        st.session_state.index_ready = False
        st.session_state.current_pdf_name = uploaded_file.name

        with st.spinner("Reading and indexing PDF..."):
            # Save PDF to a temporary path
            pdf_path = "temp.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            # Build index
            pages = load_pdf(pdf_path)
            chunks = chunk_text(pages)
            build_faiss_index(chunks)

        st.session_state.index_ready = True
        st.success(f"Index built for: {uploaded_file.name}")
    else:
        # Same file as before, index should already exist
        if st.session_state.index_ready:
            st.info(f"Index already built for: {uploaded_file.name}")
        else:
            st.warning("Index is not ready yet. Try re-uploading the PDF.")
else:
    st.info("Please upload a PDF to start.")
    st.stop()

# Safety check: ensure index exists
index, meta = load_faiss_index()
if index is None:
    st.error("No index found. Try re-uploading the PDF.")
    st.stop()

# --- 2. QUESTION + SEARCH BUTTON ---
st.header("Ask a question about this PDF")

col1, col2 = st.columns([1, 1])

with col1:
    question = st.text_input("Enter your question", label_visibility="collapsed", key="user_question")

with col2:
    search_clicked = st.button("Search")

if search_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer, retrieved = answer_question(question, k=5)
            except Exception as e:
                st.error(f"Error while answering: {e}")
                st.stop()

        # --- 3. SHOW ANSWER ---
        st.subheader("Answer")
        st.write(answer)

        # --- 4. SHOW CITATIONS / CONTEXT ---
        st.subheader("Citations & Context Chunks")
        for i, r in enumerate(retrieved, start=1):
            with st.expander(f"[{i}] Page {r['page_num']}  •  score={r['score']:.4f}"):
                st.write(r["text"])
