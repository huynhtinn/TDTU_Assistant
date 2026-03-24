# TDTU Assistant — AI Chatbot for Ton Duc Thang University

**Graduation Thesis (KLTN)**  
An intelligent question-answering system for Ton Duc Thang University (TDTU) built on a Multi-Agent architecture combining RAG and SQL, with user authentication, conversation history, and a feedback loop.

---

## Introduction

**TDTU Assistant** is an intelligent chatbot designed to serve students and staff at Ton Duc Thang University. The system is capable of:

- Retrieving academic regulations, scholarships, discipline rules, training scores, and more
- Looking up student data (GPA, credits, failed subjects) via SQL
- Answering questions about admissions, tuition fees, dormitories, and departmental contacts
- Detecting and handling out-of-scope and greeting messages with PhoBERT
- Supporting multiple LLMs: **LLaMA** (via Groq) and **Gemini** (via Google AI Studio)
- **User accounts** with two roles: `student` and `lecturer`
- **Conversation history** — each chat session is saved and can be renamed, pinned, or deleted
- **Feedback system** — students rate bot answers; lecturers can review and reply
- **Document Manager** — lecturers can upload PDF / plain-text documents into any vector DB directly from the UI


## System Architecture

The system uses pipeline:

```
User Question
      │
      ▼
┌───────────────────────────────┐
│  Layer 1: PhoBERT Classifier  │  ← Fine-tuned PhoBERT
│  IN_SCOPE / GREETING / OOS    │    Intent classification
└───────────┬───────────────────┘
            │ (if IN_SCOPE)
            ▼
┌───────────────────────────────┐
│  Layer 2: LLM Router          │  ← LLaMA-3.1 analyzes & routes
│  → Selects appropriate Agents │
└───────────┬───────────────────┘
            │
            ▼
┌─────────────────────────────────────────────┐
│  Layer 3: Specialist Agents (HybridAgent)   │
│                                             │
│  ACADEMIC  │ FINANCIAL │ ADMISSION          │  ← Each agent has:
│  STUDENT_LIFE │ GENERAL                     │    - RAG tool (ChromaDB)
│                                             │    - SQL tool (SQLite)
└───────────┬─────────────────────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  Synthesizer LLM              │  ← Merges results → final answer
│  (LLaMA or Gemini)            │
└───────────────────────────────┘
```

### Key Technical Features

| Feature | Description |
|---------|-------------|
| **Semantic Cache** | LRU cache with 128 entries; cosine similarity ≥ 0.95 → cache HIT |
| **Parallel Agents** | Multiple agents called concurrently via `ThreadPoolExecutor` |
| **RAG Score Threshold** | Filters docs with relevance score ≥ 0.78 |
| **Context Deduplication** | MD5-based deduplication of retrieved chunks |
| **Follow-up Rewriting** | LLM rewrites follow-up questions into standalone queries |
| **Source Attribution** | Displays reference source with URL and page number |
| **Multi-Provider** | Compare LLaMA vs Gemini responses side-by-side |
| **Streaming** | Supports token streaming for faster perceived response |
| **User Auth** | Role-based login (student / lecturer) backed by SQLite |
| **Chat Persistence** | Conversations, messages, and contexts saved per user |
| **Feedback Loop** | Students submit feedback; lecturers reply via dashboard |
| **Document Manager** | Upload PDF/text into vector DBs without rerunning scripts |


## Project Structure

```
├── .env                          # Environment variables (API keys)
├── .streamlit/                   # Streamlit theme config
├── requirements.txt              # Python dependencies
├── run_web.bat                   # Startup script (Windows)
├── run_web.sh                    # Startup script (Linux/macOS)
│
├── data/
│   ├── raw/                      # JSON data crawled from TDTU website
│   ├── processed/                # ChromaDB vector stores + SQLite DBs
│   │   ├── academic_db/          # Vector DB: Academic / Training regulations
│   │   ├── financial_db/         # Vector DB: Tuition / Scholarships
│   │   ├── admission_db/         # Vector DB: Admissions
│   │   ├── student_life_db/      # Vector DB: Student life
│   │   ├── general_db/           # Vector DB: General information
│   │   ├── student_data.db       # SQLite: Student records (GPA, credits …)
│   │   └── users.db              # SQLite: User accounts, conversations, feedback
│   ├── stdportal/                # JSONL data from the student portal
│   └── training/                 # PhoBERT training data
│
├── models/
│   └── intent_classifier/        # Fine-tuned PhoBERT model
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer files ...
│       └── label_map.json
│
├── src/
│   ├── app/                      # Core application
│   │   ├── app.py                # Streamlit UI (chat, auth, sidebar, settings)
│   │   ├── main.py               # Query processing pipeline (3-layer + cache)
│   │   ├── agents.py             # HybridAgent (RAG + SQL via ReAct)
│   │   ├── auth.py               # Auth + conversation/message/feedback persistence
│   │   ├── doc_manager.py        # Add/delete documents in vector DBs
│   │   ├── intent_classifier.py  # PhoBERT Intent Classifier wrapper
│   │   └── embeddings.py         # E5Embeddings (multilingual-e5-base)
│   │
│   ├── data_collection/          # Data collection
│   │   ├── crawl_public.py       # Crawl TDTU public website
│   │   ├── tdtu_client.py        # Login & fetch student portal data
│   │   ├── download_doc.py       # Download regulation documents (PDF/DOCX)
│   │   ├── tdtu_db.py            # Store data into SQLite
│   │   └── tdtu_main.py          # Entry point for data collection
│   │
│   ├── data_processing/          # Data processing & indexing
│   │   ├── build_specialized_dbs.py  # Build ChromaDB from raw data
│   │   ├── setup_sql.py              # Create SQLite student database
│   │   ├── embed_data.py             # General embedding pipeline
│   │   ├── process_stdportal_jsonl.py # Process student portal data
│   │   └── inspect_db.py             # Inspect database contents
│   │
│   ├── model_training/           # PhoBERT training
│   │   ├── generate_data.py      # Generate synthetic training data
│   │   ├── generate_general.py   # Generate IN_SCOPE examples
│   │   ├── generate_greeting.py  # Generate GREETING examples
│   │   ├── generate_oos.py       # Generate OUT_OF_SCOPE examples
│   │   ├── merge_data.py         # Merge CSV files
│   │   ├── train_classifier.py   # Fine-tune PhoBERT classifier
│   │   ├── test_model.py         # Test model after training
│   │   └── visualize_metrics.py  # Plot accuracy/loss charts
│   │
│   ├── eval_layers.py            # Evaluate Layer 2 routing accuracy
│   ├── ragas_dataset.py          # Generate RAGAS evaluation dataset
│   ├── OCR.ipynb                 # Notebook: OCR for PDF documents
│   └── RAGAS.ipynb               # Notebook: RAGAS evaluation
│
└── evaluate/                     # Evaluation results
    ├── Ragas_gemini/             # RAGAS results using Gemini
    ├── Ragas_llama/              # RAGAS results using LLaMA
    └── layer_evaluation/         # Layer evaluation results
```


## Installation

### 1. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the PhoBERT model

Download the PhoBERT model here: https://drive.google.com/drive/folders/1yYjKeF__Cz2VGkihOj_U5QUif5u2RatL?usp=sharing


## Environment Variables

Create a `.env` file at the project root:

```env
# === REQUIRED — Groq (LLaMA router + agents + synthesizer) ===
API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx       # Groq API Key
LLM_MODEL=llama-3.1-8b-instant                # Groq model name

# === OPTIONAL — Gemini (comparison / alternative synthesizer) ===
GOOGLE_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxx        # Google AI Studio API Key
GEMINI_MODEL=gemini-2.5-flash                 # Gemini model name

# === OPTIONAL — Lecturer access code ===
LECTURER_CODE=TDTU@LECTURER2025               # Secret code for lecturer registration
```

---

## Running the App

**Windows (one-click):**
```bat
.\run_web.bat
```

**Or manually:**
```bash
streamlit run src/app/app.py
```

Open your browser at: **http://localhost:8501**

---

## User Roles

| Role | How to register | Capabilities |
|------|-----------------|--------------|
| **Student** | Open app → Register (no code needed) | Ask questions, view history, submit feedback |
| **Lecturer** | Open app → Register with `LECTURER_CODE` | All student features + Feedback dashboard + Document Manager |

### Feedback System

- After each bot answer, students can rate it (like / dislike) and leave a note.
- Lecturers see all feedback in a dedicated dashboard and can reply to individual entries.
- Students are notified when a lecturer has replied.

### Document Manager (Lecturer only)

- Upload **PDF** or **plain-text** documents directly into any of the 5 vector DBs.
- View all indexed sources (name, chunk count) per DB.
- Delete individual source documents from a DB.
- Chunk size: 800 tokens / overlap: 150 tokens.

---

## Evaluation

Evaluation results are stored in the `evaluate/` directory:

| Folder | Contents |
|--------|----------|
| `evaluate/Ragas_gemini/` | RAGAS results using Gemini as judge |
| `evaluate/Ragas_llama/` | RAGAS results using LLaMA as judge |
| `evaluate/layer_evaluation/` | Layer-level routing evaluation results |

### RAGAS Metrics

- **Faithfulness** — Is the answer faithful to the retrieved context?
- **Answer Relevancy** — Is the answer relevant to the question?
- **Context Precision** — Is the retrieved context precise?
- **Context Recall** — Is the retrieved context complete?

---
