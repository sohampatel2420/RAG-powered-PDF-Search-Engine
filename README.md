# 📄 RAG-powered PDF Search Engine (FAISS + High-Quality Embeddings)

A **high-accuracy, fully free, Retrieval-Augmented Generation (RAG) system** that allows users to:

✅ Upload any PDF  
✅ Automatically build a vector index  
✅ Ask questions in natural language  
✅ Get **precise answers with page-level citations**  

Built using **FAISS, BGE embeddings, Streamlit, and a local LLM via Ollama** — no paid APIs required.

---

## 🚀 Key Features

- ✅ **Automatic PDF indexing** (no manual buttons)
- ✅ **High-accuracy semantic search** using **BGE embeddings**
- ✅ **FAISS vector database** for ultra-fast retrieval
- ✅ **Local LLM inference using Ollama** (100% free)
- ✅ **Page-wise citations for every answer**
- ✅ **Clean, minimal Streamlit UI**
- ✅ Fully offline-capable after setup

---

## 🧠 Project Architecture (RAG Pipeline)

    PDF Upload
    ↓
    Text Extraction (PyPDF)
    ↓
    Chunking with Overlap
    ↓
    Vector Embeddings (BGE)
    ↓
    FAISS Index
    ↓
    User Query
    ↓
    Semantic Retrieval (Top-K Chunks)
    ↓
    LLM Answer Generation (Ollama)
    ↓
    Final Answer + Citations



---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| UI | Streamlit |
| PDF Processing | PyPDF |
| Embeddings | `BAAI/bge-base-en-v1.5` |
| Vector Database | FAISS |
| LLM | Ollama (LLaMA 3 / Qwen / Phi) |
| Language | Python |

All tools are **open-source and free of cost** ✅

---

## 📁 Project Structure

    rag-pdf-search/
    │
    ├── app.py # Streamlit app
    ├── rag_engine.py # RAG logic (PDF, chunks, embeddings, FAISS, LLM)
    ├── requirements.txt
    │
    └── data/
    └── index/ # Stored FAISS index + metadata



---

## ⚙️ Installation & Setup

1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/rag-pdf-search.git
cd rag-pdf-search


2️⃣ Install Python Dependencies

pip install -r requirements.txt


3️⃣ Install & Run Ollama (Local LLM)

Download Ollama from:


https://ollama.com


Then pull a free LLM model:

ollama pull llama3.2
ollama serve


4️⃣ Run the Application

streamlit run app.py

Open your browser at:

http://localhost:8501


✅ How It Works (User Flow)

Upload a PDF file

If you do not have PDF file you will use sample PDF file that is located on data/sample/sample_data.pdf

The system automatically builds the FAISS index

Type your question

Click Search

Get:

✅ AI-generated answer

✅ Exact page-number citations

✅ Retrieved context chunks

🎯 Why This Project is High Accuracy

Uses BGE embeddings (state-of-the-art open-source)

Uses overlapping smart chunking

Uses semantic search instead of keyword matching

Uses retrieval-grounded answer generation

Prevents hallucination by enforcing:

“If the answer is not in the context, say you don’t know.”

📊 Example Use Cases

📚 Study Notes Search

📄 Legal Document Questioning

🏫 Research Paper Assistant

📘 Company Policy Search

📑 Technical Documentation QA

🔒 Privacy & Cost

✅ No cloud APIs
✅ No data leaves your system
✅ No monthly payment
✅ Works fully offline after setup

🧪 Future Improvements (Optional)

✅ Multi-PDF Search

✅ Re-ranking with Cross-Encoder

✅ Chat history

✅ Export answers as PDF

✅ API version using FastAPI

✅ OCR support for scanned PDFs

👨‍💻 Author

Soham Patel
Machine Learning | Deep Learning | GenAI | Computer Vision