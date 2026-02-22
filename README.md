# 🎓 TDTU AI Assistant

<div align="center">


**Trợ lý ảo thông minh hỗ trợ sinh viên Đại học Tôn Đức Thắng**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-green)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## 📖 Giới thiệu

**TDTU AI Assistant** là hệ thống chatbot AI được xây dựng như khóa luận tốt nghiệp tại Đại học Tôn Đức Thắng (TDTU). Hệ thống ứng dụng kiến trúc **Dual-Layer Multi-Agent RAG** để trả lời các câu hỏi về thông tin học vụ, học phí, tuyển sinh và đời sống sinh viên bằng tiếng Việt.

## ✨ Tính năng chính

- 🤖 **Chatbot AI thông minh** — Hỏi đáp tự nhiên bằng tiếng Việt về mọi vấn đề liên quan đến nhà trường
- 🗂️ **Phân loại ý định** — Nhận diện câu hỏi ngoài phạm vi và lời chào hỏi bằng mô hình PhoBERT
- 🔀 **Định tuyến thông minh** — LLM tự động chọn agent phù hợp với từng loại câu hỏi
- 📚 **Cơ sở dữ liệu tài liệu** — Xem và tải 30+ văn bản quy chế, quy định PDF của trường
- 📞 **Thông tin liên hệ** — Tra cứu nhanh số điện thoại, email các phòng ban

## 🏗️ Kiến trúc hệ thống

```
Câu hỏi của người dùng
        │
        ▼
┌──────────────────────┐
│  Layer 1: PhoBERT    │  ← Phân loại ý định (GREETING / OUT_OF_SCOPE / IN_SCOPE)
│  Intent Classifier   │
└──────────┬───────────┘
           │ IN_SCOPE
           ▼
┌──────────────────────┐
│  Layer 2: Groq LLM   │  ← Router: Phân tích & lập kế hoạch
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
│  Groq LLM Synthesizer│  ← Tổng hợp câu trả lời cuối cùng
└──────────────────────┘
```

### Các Agent chuyên biệt

| Agent | Phụ trách | Nguồn dữ liệu |
|---|---|---|
| **ACADEMIC** | Điểm số, rèn luyện, quy chế đào tạo | SQLite + ChromaDB |
| **FINANCIAL** | Học phí, học bổng, công nợ | SQLite + ChromaDB |
| **ADMISSION** | Tuyển sinh, điểm chuẩn, thủ tục nhập học | ChromaDB |
| **STUDENT_LIFE** | Ký túc xá, bảo hiểm, câu lạc bộ | ChromaDB |
| **GENERAL** | Liên hệ, thông tin chung | ChromaDB |

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Web UI | Streamlit |
| LLM | Groq (Llama / Mixtral) |
| Embedding | HuggingFace Sentence Transformers |
| Vector Store | ChromaDB |
| Intent Classifier | PhoBERT (fine-tuned) |
| Vietnamese NLP | Underthesea |
| SQL Database | SQLite + SQLAlchemy |
| RAG Framework | LangChain |

## 📁 Cấu trúc thư mục

```
TDTU_Assistant/
├── .streamlit/                 # Cấu hình Streamlit & logo
│   ├── config.toml
│   └── Logo ĐH Tôn Đức Thắng-TDT.png
├── data/
│   ├── raw/                    # Dữ liệu JSON thu thập từ web
│   ├── processed/              # Dữ liệu đã xử lý
│   └── stdportal/
│       └── downloads_pdf/      # 30+ văn bản PDF quy chế/quy định
├── models/
│   └── intent_classifier/      # Mô hình PhoBERT đã fine-tune
├── src/
│   ├── app/
│   │   ├── app.py              # Giao diện Streamlit chính
│   │   ├── main.py             # Pipeline AI (Router + Synthesizer)
│   │   ├── agents.py           # 5 Agent chuyên biệt
│   │   ├── rag_engine.py       # RAG pipeline
│   │   └── intent_classifier.py
│   ├── data_collection/        # Scripts thu thập dữ liệu
│   ├── data_processing/        # Scripts xử lý dữ liệu
│   └── model_training/         # Scripts huấn luyện PhoBERT
├── requirements.txt
├── run_web.bat                  # Chạy app (Windows)
└── run_web.sh                   # Chạy app (Linux/macOS)
```

## 🚀 Cài đặt & Chạy

### Yêu cầu
- Python 3.10+
- API Key từ [Groq](https://console.groq.com)

### Bước 1: Clone & cài đặt
```bash
git clone <repo-url>
cd TDTU_Assistant

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

### Bước 2: Cấu hình API Key
Tạo file `.env` (copy từ `.env.example`):
```env
API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.1-8b-instant
```

### Bước 3: Chạy ứng dụng
```bash
# Windows (double-click hoặc chạy trong terminal)
run_web.bat

# Hoặc chạy trực tiếp
streamlit run src/app/app.py
```

Mở trình duyệt tại: **http://localhost:8501**

## 🖥️ Giao diện

| Trang | Mô tả |
|---|---|
| **🤖 Chatbot** | Chat với AI về thông tin TDTU |
| **📚 Cơ sở dữ liệu** | Xem và tải tài liệu PDF quy chế, quy định |
| **📞 Liên hệ** | Thông tin liên hệ các phòng ban |

## 📊 Dữ liệu

- **26+ file JSON** thu thập từ website TDTU (tuyển sinh, học phí, đào tạo...)
- **30+ file PDF** quy chế, quy định của nhà trường
- **SQLite database** chứa dữ liệu sinh viên (điểm, học phí, rèn luyện)
- **ChromaDB** vector store lưu embeddings cho RAG

---


