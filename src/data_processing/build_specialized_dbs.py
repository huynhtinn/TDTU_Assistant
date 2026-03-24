import os
import shutil
import json
import re
import unicodedata

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"

import sys
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

_APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app')
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
from embeddings import E5Embeddings, get_shared_embedding_model

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

STORES = {
    "ADMISSION_DB": os.path.join(PROCESSED_DIR, 'admission_db'),
    "ACADEMIC_DB": os.path.join(PROCESSED_DIR, 'academic_db'),
    "FINANCIAL_DB": os.path.join(PROCESSED_DIR, 'financial_db'),
    "STUDENT_LIFE_DB": os.path.join(PROCESSED_DIR, 'student_life_db'),
    "GENERAL_DB": os.path.join(PROCESSED_DIR, 'general_db'),
}


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    no_diacritics = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = no_diacritics.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


_URL_ADMISSION_KEYWORDS = [
    "admission", "tuyen sinh", "xet tuyen", "nhap hoc", "diem trung tuyen",
    "guide for applicants", "how to apply",
]
_URL_FINANCIAL_KEYWORDS = [
    "hoc phi", "hoc bong", "tai chinh", "dong tien", "khen thuong", "ho tro",
]
_URL_STUDENT_LIFE_KEYWORDS = [
    "ctsv", "cong tac sinh vien", "bao hiem", "ky tuc xa", "ho tro sinh vien",
    "noi quy", "ky luat", "ung xu", "khong duoc lam", "ren luyen",
]
_URL_ACADEMIC_KEYWORDS = [
    "dao tao", "grad", "undergrad", "hoc vu", "chuan dau ra", "quy che",
    "tot nghiep", "tin chi", "quy dinh dao tao", "quan ly nguoi hoc", "tieng anh", "tap su",
]

def classify_file(filename, data):
    url = data.get('source_url', '')
    fname = filename
    text = _normalize_text(f"{url} {fname}")

    # --- Tuyển sinh ---
    if any(k in text for k in _URL_ADMISSION_KEYWORDS):
        return "ADMISSION_DB"

    # --- Tài chính ---
    if any(k in text for k in _URL_FINANCIAL_KEYWORDS):
        return "FINANCIAL_DB"

    # --- Đời sống sinh viên ---
    if any(k in text for k in _URL_STUDENT_LIFE_KEYWORDS):
        return "STUDENT_LIFE_DB"

    # --- Học thuật / Đào tạo ---
    if any(k in text for k in _URL_ACADEMIC_KEYWORDS):
        return "ACADEMIC_DB"

    return "GENERAL_DB"


_JSONL_FINANCIAL_KEYWORDS = [
    "hoc bong", "khen thuong", "ho tro", "hoc phi", "tai chinh", "dong tien",
]
_JSONL_STUDENT_LIFE_KEYWORDS = [
    "cong tac sinh vien", "khong duoc lam", "ky luat", "dao duc", "noi quy",
    "ung xu", "ren luyen",
]
_JSONL_ACADEMIC_KEYWORDS = [
    "dao tao", "quy che", "quy dinh", "tot nghiep", "tin chi", "tieng anh",
    "hoc tap", "chuan dau ra", "quan ly nguoi hoc", "phong thi", "chuyen doi tin chi", "tap su", "thu khen"
]

def _classify_jsonl_doc(source: str, content: str) -> str:
    text = _normalize_text(source + " " + content[:400])

    if any(k in text for k in _JSONL_FINANCIAL_KEYWORDS):
        return "FINANCIAL_DB"
    if any(k in text for k in _JSONL_STUDENT_LIFE_KEYWORDS):
        return "STUDENT_LIFE_DB"
    if any(k in text for k in _JSONL_ACADEMIC_KEYWORDS):
        return "ACADEMIC_DB"
    return "GENERAL_DB"


def load_stdportal_jsonl(docs_map):
    jsonl_path = os.path.join(BASE_DIR, 'data', 'stdportal', 'data.jsonl')
    if not os.path.exists(jsonl_path):
        print(f"   Không tìm thấy {jsonl_path}, bỏ qua.")
        return

    count = 0
    dist  = {k: 0 for k in ["FINANCIAL_DB", "STUDENT_LIFE_DB", "ACADEMIC_DB", "GENERAL_DB"]}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = doc.get('page_content', '').strip()
            if not content:
                continue

            meta     = doc.get('metadata', {})
            source   = meta.get('source', '')    # Tên tài liệu (tiêu đề)
            page_num = meta.get('page', None)    # Số trang
            db_key   = _classify_jsonl_doc(source, content)

            docs_map[db_key].append({
                "text":     content,
                "metadata": {"source": source, "page_title": source, "page": page_num}
            })
            dist[db_key] += 1
            count += 1

    print(f"   Đã nạp {count} docs từ data.jsonl:")
    for db, n in dist.items():
        if n: print(f"      {db}: {n} docs")

def main():
    print("--- REBUILDING DATA FOR GROQ STACK ---")
    
    if os.path.exists(PROCESSED_DIR):
        try:
            shutil.rmtree(PROCESSED_DIR)
        except: pass
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("   Loading Embedding Model (intfloat/multilingual-e5-base)...")
    embedding_model = get_shared_embedding_model()

    docs_map = {key: [] for key in STORES.keys()}

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]
    print(f"   [Nguồn 1] Found {len(files)} raw JSON files.")
    
    for filename in files:
        file_path = os.path.join(RAW_DATA_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        target_db = classify_file(filename, data)
        source_url = data.get('source_url', '')
        page_title = data.get('title', '')
        
        for section in data.get('segmented_content', []):
            content = section.get('content', '')
            title = section.get('title', '')

            if len(content.strip()) < 50: continue
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(content)
            
            for chunk in chunks:
                full_text = f"{title}: {chunk}"
                docs_map[target_db].append({
                    "text": full_text,
                    "metadata": {
                        "source": source_url,
                        "page_title": page_title,
                        "title": title,
                    }
                })

    print("\n   [Nguồn 2] Nạp data.jsonl từ stdportal...")
    load_stdportal_jsonl(docs_map)

    total = sum(len(v) for v in docs_map.values())
    print(f"\n   Tổng chunks toàn bộ nguồn: {total}")
    for db_name, docs in docs_map.items():
        if not docs: continue
        print(f"\n--- Building {db_name} ({len(docs)} chunks)... ---")
        save_path = STORES[db_name]
        
        Chroma.from_texts(
            texts=[d['text'] for d in docs],
            embedding=embedding_model,
            metadatas=[d['metadata'] for d in docs],
            persist_directory=save_path,
            collection_metadata={"hnsw:space": "cosine"}, 
        )
        print(f"   Saved {db_name}")

    print("\n=== DATA READY ===")

if __name__ == "__main__":
    main()