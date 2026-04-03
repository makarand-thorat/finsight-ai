# 📊 FinSight AI — Production RAG Evaluation Platform

A production-grade Retrieval-Augmented Generation (RAG) system for financial documents, featuring automated LLM-as-judge evaluation, LangGraph orchestration, FastAPI backend, and a Streamlit dashboard.

---

## 🎯 What It Does

Upload any financial PDF (annual reports, earnings transcripts, Central Bank documents) and ask questions. The system retrieves relevant context, generates a grounded answer, and automatically evaluates the response quality using two metrics:

- **Faithfulness** — is the answer grounded in the document? (no hallucination)
- **Answer Relevancy** — does the answer directly address the question?

Every evaluation is logged and displayed on a live dashboard so you can track quality over time.

---

## 🏗 Architecture
```
PDF Upload → Ingestion (PyPDF + chunking) → ChromaDB (vector store)
                                                    ↓
User Question → LangGraph RAG Pipeline → Gemini 2.0 Flash → Answer
                                                    ↓
                                        LLM-as-Judge Evaluator
                                                    ↓
                                        Evaluation Dashboard (Streamlit)
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Gemini 2.0 Flash |
| Embeddings | Gemini Embedding 001 |
| Vector Store | ChromaDB |
| Evaluation | LLM-as-Judge (Faithfulness + Relevancy) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Observability | LangSmith |

---

## 📈 Sample Evaluation Results

| Question | Faithfulness | Answer Relevancy | Average |
|---|---|---|---|
| What was total revenue in 2024? | 0.95 | 0.92 | 0.935 |
| What is the outlook for 2025? | 0.90 | 0.88 | 0.890 |
| How many digital banking users? | 0.93 | 0.91 | 0.920 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker (optional)
- Gemini API key — [Get one free at Google AI Studio](https://aistudio.google.com)
- LangSmith API key — [Get one free at smith.langchain.com](https://smith.langchain.com)

### 1. Clone the repo

    git clone https://github.com/makarand-thorat/finsight-ai.git
    cd finsight-ai

### 2. Create virtual environment

    python -m venv venv

    # Mac/Linux
    source venv/bin/activate

    # Windows
    venv\Scripts\activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Set up environment variables

Create a `.env` file in the root:

    GEMINI_API_KEY=your_gemini_api_key_here
    LANGSMITH_API_KEY=your_langsmith_api_key_here
    LANGSMITH_PROJECT=finsight-ai
    LANGCHAIN_TRACING_V2=true

### 5. Run the app

Open two terminals:

Terminal 1 — FastAPI backend:

    uvicorn app.main:app --reload --port 8000

Terminal 2 — Streamlit frontend:

    streamlit run frontend/app.py

Then open `http://localhost:8501` in your browser.

---

## 🐳 Running with Docker

    docker-compose up --build

Then open `http://localhost:8501`.

---

## 🧪 Running Tests

    pytest tests/ -v

All 7 tests should pass.

---

## 📁 Project Structure

    finsight-ai/
    ├── app/
    │   ├── ingestion.py      # PDF loading, chunking, ChromaDB storage
    │   ├── retriever.py      # Semantic search over ChromaDB
    │   ├── rag_pipeline.py   # LangGraph RAG orchestration
    │   ├── model_router.py   # Gemini model wrapper
    │   ├── evaluator.py      # LLM-as-judge evaluation + result logging
    │   └── main.py           # FastAPI endpoints
    ├── frontend/
    │   └── app.py            # Streamlit dashboard
    ├── tests/
    │   └── test_eval.py      # Pytest test suite
    ├── .github/
    │   └── workflows/
    │       └── ci.yml        # GitHub Actions CI/CD
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/upload` | POST | Upload and ingest a PDF |
| `/ask` | POST | Ask a question with optional evaluation |
| `/results` | GET | Get all evaluation results |
| `/scores` | GET | Get average scores across all evaluations |

---

## 🔮 Roadmap

- [ ] Add Claude and GPT-4o model support for cross-model benchmarking
- [ ] Add AWS S3 for cloud document storage
- [ ] Add context recall metric
- [ ] Add batch evaluation mode for golden dataset testing

---

## 👤 Author

**Makarand Thorat**
- MSc Computer Science (AI) — Dublin City University
- [LinkedIn](https://www.linkedin.com/in/makarand-thorat/)
- [GitHub](https://github.com/makarand-thorat)

---