
import io
import os
import sys
from collections import defaultdict

import chromadb
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from embeddings import get_shared_embedding_model

PROCESSED_DIR = os.path.join(_ROOT_DIR, 'data', 'processed')

STORES = {
    "ACADEMIC_DB":     os.path.join(PROCESSED_DIR, 'academic_db'),
    "FINANCIAL_DB":    os.path.join(PROCESSED_DIR, 'financial_db'),
    "ADMISSION_DB":    os.path.join(PROCESSED_DIR, 'admission_db'),
    "STUDENT_LIFE_DB": os.path.join(PROCESSED_DIR, 'student_life_db'),
    "GENERAL_DB":      os.path.join(PROCESSED_DIR, 'general_db'),
}

DB_LABELS = {
    "ACADEMIC_DB":     " Học thuật / Đào tạo",
    "FINANCIAL_DB":    " Tài chính / Học bổng",
    "ADMISSION_DB":    " Tuyển sinh",
    "STUDENT_LIFE_DB": " Đời sống sinh viên",
    "GENERAL_DB":      " Thông tin chung",
}

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)


def _get_raw_collection(db_key: str):
    """Trả về raw chromadb Collection (tạo mới nếu chưa có)."""
    db_path = STORES[db_key]
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name="langchain",
        metadata={"hnsw:space": "cosine"},
    )

def list_sources(db_key: str) -> list:

    col = _get_raw_collection(db_key)
    results = col.get(include=["metadatas"])
    if not results["metadatas"]:
        return []

    groups = defaultdict(lambda: {"count": 0, "page_title": ""})
    for meta in results["metadatas"]:
        src = meta.get("source") or meta.get("page_title") or ""
        pt  = meta.get("page_title") or src
        groups[src]["count"] += 1
        groups[src]["page_title"] = pt

    return [
        {"source": k, "page_title": v["page_title"], "chunks": v["count"]}
        for k, v in sorted(groups.items(), key=lambda x: x[1]["page_title"].lower())
    ]

def add_texts(texts: list, metadatas: list, db_key: str) -> int:
    db_path = STORES[db_key]
    os.makedirs(db_path, exist_ok=True)
    emb = get_shared_embedding_model()
    client = chromadb.PersistentClient(path=db_path)
    db = Chroma(
        client=client,
        embedding_function=emb,
        collection_name="langchain",
        collection_metadata={"hnsw:space": "cosine"},
    )
    db.add_texts(texts=texts, metadatas=metadatas)
    return len(texts)


def add_raw_text(text: str, source_name: str, db_key: str) -> int:
    chunks = _SPLITTER.split_text(text)
    if not chunks:
        return 0
    metadatas = [{"source": source_name, "page_title": source_name}] * len(chunks)
    return add_texts(chunks, metadatas, db_key)


def add_pdf_bytes(pdf_bytes: bytes, source_name: str, db_key: str) -> int:
    pages = _extract_pdf_pages(pdf_bytes)
    if not pages:
        raise ValueError("Không trích xuất được text từ PDF (file có thể là scan ảnh).")

    all_texts = []
    all_metas = []
    for page_num, page_text in pages:
        chunks = _SPLITTER.split_text(page_text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_metas.append({
                "source":     source_name,
                "page_title": source_name,
                "page":       page_num,
            })

    return add_texts(all_texts, all_metas, db_key)


def _extract_pdf_pages(pdf_bytes: bytes) -> list:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "Cần thư viện pypdf. Chạy: pip install pypdf"
            )
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def delete_source(source_value: str, db_key: str) -> int:
    col = _get_raw_collection(db_key)
    results = col.get(where={"source": source_value}, include=[])
    ids = results["ids"]
    if ids:
        col.delete(ids=ids)
    return len(ids)


def get_db_stats() -> dict:
    stats = {}
    for key, label in DB_LABELS.items():
        try:
            col = _get_raw_collection(key)
            total = col.count()
            srcs  = list_sources(key)
            stats[key] = {
                "label":         label,
                "total_chunks":  total,
                "total_sources": len(srcs),
            }
        except Exception:
            stats[key] = {"label": label, "total_chunks": 0, "total_sources": 0}
    return stats
