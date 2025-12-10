# rag_engine.py
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle

def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page_num": i + 1, "text": text})
    return pages


def chunk_text(pages, chunk_size=1500, chunk_overlap=300):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "page_num": page["page_num"],
                "text": chunk_text
            })
            start = end - chunk_overlap
    return chunks

INDEX_DIR = "data/index"

def build_faiss_index(chunks, index_dir=INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)
    model = get_embedding_model()

    texts = [c["text"] for c in chunks]
    print("Encoding embeddings...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    d = embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    # Save metadata (chunks & texts)
    meta = {
        "chunks": chunks,
    }
    with open(os.path.join(index_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return index, meta

def load_faiss_index(index_dir=INDEX_DIR):
    index_path = os.path.join(index_dir, "index.faiss")
    meta_path = os.path.join(index_dir, "meta.pkl")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return None, None
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def search(query, k=5, index_dir=INDEX_DIR):
    index, meta = load_faiss_index(index_dir)
    if index is None:
        raise RuntimeError("FAISS index not built yet.")

    model = get_embedding_model()
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        chunk = meta["chunks"][idx]
        results.append({
            "score": float(score),
            "page_num": chunk["page_num"],
            "text": chunk["text"]
        })
    return results


import requests
import json

def call_llm_ollama(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["response"]

def build_prompt(question, retrieved_chunks):
    context_strs = []
    for i, c in enumerate(retrieved_chunks, start=1):
        context_strs.append(
            f"[{i}] (page {c['page_num']})\n{c['text']}\n"
        )
    context = "\n\n".join(context_strs)

    system_instructions = """
You are a helpful assistant answering questions strictly based on the provided context.
Rules:
- Only use the information in the context.
- If the answer is not in the context, say you don't know.
- At the end, list citations in the form [1], [2], etc., based on the chunk numbers.
"""
    user_prompt = f"""
{system_instructions}

Context:
{context}

Question: {question}

Answer in a clear way and add citations like [1], [2] next to the sentences that use that chunk.
"""
    return user_prompt


def answer_question(question, k=5):
    retrieved = search(question, k=k)
    prompt = build_prompt(question, retrieved)
    answer = call_llm_ollama(prompt)

    # Also return retrieved chunks to show proper citations in UI
    return answer, retrieved


EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"  # strong & free

def get_embedding_model():
    return SentenceTransformer(EMB_MODEL_NAME)