# file: src/data_processing/build_specialized_dbs.py
import os
import shutil
import json

# Tắt Telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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

def classify_file(filename, data):
    url = data.get('source_url', '').lower()
    title = data.get('title', '').lower()
    content_sample = str(data.get('segmented_content', ''))[:500].lower()
    
    if "admission" in url or "tuyen-sinh" in url or "xet-tuyen" in url: return "ADMISSION_DB"
    if "hoc-phi" in url or "hoc-bong" in url or "tai-chinh" in url: return "FINANCIAL_DB"
    if "ren-luyen" in url or "ctsv" in url or "bao-hiem" in url or "ky-tuc-xa" in url: return "STUDENT_LIFE_DB"
    if "dao-tao" in url or "grad" in url or "undergrad" in url or "hoc-vu" in url: return "ACADEMIC_DB"
    return "GENERAL_DB"

def main():
    print("--- REBUILDING DATA FOR GROQ STACK ---")
    
    # Xóa thư mục cũ nếu còn sót
    if os.path.exists(PROCESSED_DIR):
        try:
            shutil.rmtree(PROCESSED_DIR)
        except: pass
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("   Loading Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    docs_map = {key: [] for key in STORES.keys()}
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]
    print(f"   Found {len(files)} files.")
    
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
            if not content.strip(): continue
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(content)
            
            for chunk in chunks:
                full_text = f"{page_title} - {title}: {chunk}"
                docs_map[target_db].append({
                    "text": full_text,
                    "metadata": {"source": source_url, "title": title}
                })

    for db_name, docs in docs_map.items():
        if not docs: continue
        print(f"\n--- Building {db_name} ({len(docs)} chunks)... ---")
        save_path = STORES[db_name]
        
        Chroma.from_texts(
            texts=[d['text'] for d in docs],
            embedding=embedding_model,
            metadatas=[d['metadata'] for d in docs],
            persist_directory=save_path
        )
        print(f"   ✅ Saved {db_name}")

    print("\n=== DATA READY ===")

if __name__ == "__main__":
    main()