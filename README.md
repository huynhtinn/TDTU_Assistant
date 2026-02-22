# 🎓 TDTU AI Assistant

<div align="center">

**An intelligent virtual assistant for Ton Duc Thang University students**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-green)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## 📖 Overview

**TDTU AI Assistant** is an AI-powered chatbot built as a graduation thesis at Ton Duc Thang University (TDTU). The system leverages a **Dual-Layer Multi-Agent RAG** architecture to answer questions about academic affairs, tuition, admissions, and student life in Vietnamese.

## ✨ Key Features

- 🤖 **AI Chatbot** — Natural language Q&A in Vietnamese about all university-related topics
- 🗂️ **Intent Classification** — Detects greetings and out-of-scope questions using a fine-tuned PhoBERT model
- 🔀 **Smart Routing** — LLM automatically selects the most appropriate agent for each query
- 📚 **Document Database** — Browse and download 30+ official university PDF regulations
- 📞 **Contact Directory** — Quickly look up phone numbers and emails for all departments

## 🏗️ System Architecture

```
User Question
      │
      ▼
┌──────────────────────┐
│  Layer 1: PhoBERT    │  ← Intent Classification (GREETING / OUT_OF_SCOPE / IN_SCOPE)
│  Intent Classifier   │
└──────────┬───────────┘
           │ IN_SCOPE
           ▼
┌──────────────────────┐
│  Layer 2: Groq LLM   │  ← Analyzes query & builds execution plan
│  Router & Planner    │
└──────────┬───────────┘
           │
     ┌─────┴─────┐────────────┐────────────┐────────────┐
     ▼           ▼            ▼            ▼            ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ACADEMIC │ │FINANCIAL │ │ADMISSION │ │STUDENT   │ │ GENERAL  │
│ Agent   │ │  Agent   │ │  Agent   │ │LIFE Agent│ │  Agent   │
│         │ │          │ │          │ │          │ │          │
│SQL+RAG  │ │SQL+RAG   │ │  RAG     │ │  RAG     │ │  RAG     │
└─────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
           │
           ▼
┌──────────────────────┐
│  Groq LLM Synthesizer│  ← Merges agent responses into final answer
└──────────────────────┘
```

### Specialized Agents

| Agent | Handles | Data Source |
|---|---|---|
| **ACADEMIC** | GPA, training points, academic regulations | SQLite + ChromaDB |
| **FINANCIAL** | Tuition fees, scholarships, student debts | SQLite + ChromaDB |
| **ADMISSION** | Entrance exams, admission benchmarks, enrollment | ChromaDB |
| **STUDENT_LIFE** | Dormitory, insurance, student clubs | ChromaDB |
| **GENERAL** | Contact info, general university information | ChromaDB |

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web UI | Streamlit |
| LLM | Groq (Llama / Mixtral) |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Store | ChromaDB |
| Intent Classifier | PhoBERT (fine-tuned) |
| Vietnamese NLP | Underthesea |
| SQL Database | SQLite + SQLAlchemy |
| RAG Framework | LangChain |

## 📁 Project Structure

```
TDTU_Assistant/
├── .streamlit/                 # Streamlit config & logo
│   ├── config.toml
│   └── Logo ĐH Tôn Đức Thắng-TDT.png
├── data/
│   ├── raw/                    # JSON data crawled from TDTU websites
│   ├── processed/              # Processed and embedded data
│   └── stdportal/
│       └── downloads_pdf/      # 30+ official university PDF documents
├── src/
│   ├── app/
│   │   ├── app.py              # Main Streamlit UI
│   │   ├── main.py             # AI pipeline (Router + Synthesizer)
│   │   ├── agents.py           # 5 specialized agents
│   │   ├── rag_engine.py       # RAG pipeline
│   │   └── intent_classifier.py
│   ├── data_collection/        # Web crawling scripts
│   ├── data_processing/        # Data processing & embedding scripts
│   └── model_training/         # PhoBERT fine-tuning scripts
├── requirements.txt
├── run_web.bat                  # Launch script (Windows)
└── run_web.sh                   # Launch script (Linux/macOS)
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- API Key from [Groq](https://console.groq.com)

### Step 1: Clone & install dependencies
```bash
git clone <repo-url>
cd TDTU_Assistant

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### Step 2: Configure environment variables
Create a `.env` file in the project root:
```env
API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.1-8b-instant
```

### Step 3: Run the application
```bash
# Windows
run_web.bat

# Or directly
streamlit run src/app/app.py
```

Open your browser at: **http://localhost:8501**

## 🖥️ Application Pages

| Page | Description |
|---|---|
| **🤖 Chatbot** | Chat with the AI about TDTU information |
| **📚 Database** | Browse and download PDF regulations & policies |
| **📞 Contact** | Contact information for all university departments |

## 📊 Data Sources

- **26+ JSON files** crawled from TDTU websites (admissions, tuition, academics...)
- **30+ PDF files** of official university regulations and policies
- **SQLite database** with student records (grades, tuition, training scores)
- **ChromaDB** vector store with document embeddings for RAG retrieval

---
