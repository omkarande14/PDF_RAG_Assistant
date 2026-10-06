# Enterprise PDF RAG Assistant (Llama 3 8B & ChromaDB)

A privacy-focused, 100% on-premise **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, **ChromaDB**, and **Ollama**.

Upload any PDF document (resumes, project briefs, specifications, reports) to extract structured details—such as candidate names, skills, project milestones, and metrics—and converse directly with your document through natural language Q&A.

---

## 📌 Architecture & Pipeline

```text
[ Upload PDF ] ───> Streamlit File Uploader
       │
       ▼
[ PyPDFLoader ] ───> Extract text across all pages
       │
       ▼
[ RecursiveCharacterTextSplitter ] ───> Chunking (size: 1000, overlap: 200)
       │
       ▼
[ OllamaEmbeddings (nomic-embed-text) ] ───> Vector embeddings
       │
       ▼
[ ChromaDB ] ───> In-memory vector store indexing
       │
       ▼
[ User Query ] ───> Top-k retrieval (k=3) ───> LCEL Prompt ───> OllamaLLM (llama3:8b)
       │
       ▼
[ Streamlit Chat UI ] ───> Real-time conversational response
```

---

## 🚀 Features

- **Document Extraction & Q&A**: Extracts project names, roles, technical skills, and key metrics directly from PDFs.
- **Dual Chat Modes**:
  - **RAG Mode**: Answers grounded strictly in your uploaded PDF content with citation context.
  - **Direct Chat Mode**: General AI assistant mode when no document is uploaded.
- **100% Local & Confidential**: All embeddings (`nomic-embed-text`) and LLM inference (`llama3:8b`) run on your local machine via Ollama. No data is sent to external clouds or APIs.
- **Modern Streamlit Chat Interface**: Built-in `st.chat_message` and `st.chat_input` for a responsive UI.

---

## 🛠️ Quick Start

### 1. Prerequisites (Ollama)

Install [Ollama](https://ollama.com/) and download the required models:

```bash
# Pull the LLM and embedding models
ollama pull llama3:8b
ollama pull nomic-embed-text

# Ensure the Ollama server is running
ollama serve
```

### 2. Environment Setup & Dependencies

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The web dashboard will open automatically in your browser at `http://localhost:8501`.