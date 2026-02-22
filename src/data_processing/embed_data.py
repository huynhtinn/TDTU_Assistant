# src/data_processing/embed_data.py

import json
import os
import shutil
import time 
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma

print("--- HUNKING & EMBEDDING (LOCAL) ---")

load_dotenv()
MY_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
VECTOR_DB_DIR = os.path.join(project_root, 'data', 'processed', 'chroma_db')

print(f"Đang tải dữ liệu từ thư mục: {RAW_DATA_DIR}")
all_documents = [] 
try:
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(RAW_DATA_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            source_url = data.get('source_url', 'Không rõ nguồn')
            page_title = data.get('title', 'Không rõ tiêu đề')
            print(f"   Đang xử lý file: {filename} ({len(data.get('segmented_content', []))} mục)")
            for section in data.get('segmented_content', []):
                section_title = section.get('title', 'Không có tiêu đề mục')
                section_content = section.get('content', '')
                if not section_content.strip(): continue
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                chunks = text_splitter.split_text(section_content)

                for chunk_text in chunks:
                    contextual_content = f"{page_title} - {section_title}: {chunk_text}"
                    doc = {
                        "page_content": contextual_content, # Lưu nội dung ĐÃ CÓ ngữ cảnh
                        "metadata": {
                            "source": source_url,
                            "page_title": page_title,
                            "section_title": section_title
                        }
                    }
                    all_documents.append(doc)
    print(f"\nTổng cộng đã đọc và chia nhỏ thành {len(all_documents)} mẩu (chunks).")
except Exception as e:
    print(f"[LỖI] Đã xảy ra lỗi khi đọc hoặc chia nhỏ file: {e}")
    exit()

if not all_documents:
    print("Lỗi: Không tìm thấy tài liệu nào để xử lý. Kiểm tra lại thư mục data/raw/")
    exit()

try:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print(f"   Đã cấu hình model: {model_name}")
    print("   (Lần chạy đầu tiên sẽ mất 1-2 phút để TẢI model về máy...)")

except Exception as e:
    print(f"[LỖI] Không thể khởi tạo mô hình Embedding HuggingFace: {e}")
    exit()


if os.path.exists(VECTOR_DB_DIR):
    print(f"Đang xóa Vector DB cũ tại: {VECTOR_DB_DIR}")
    shutil.rmtree(VECTOR_DB_DIR)

print(f"Đang chuẩn bị vector hóa {len(all_documents)} mẩu (bằng CPU/Local)...")

try:
    list_of_texts = [doc["page_content"] for doc in all_documents]
    list_of_metadata = [doc["metadata"] for doc in all_documents]

    print("   Khởi tạo Vector DB rỗng...")
    db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_model
    )
    
    batch_size = 32 
    print(f"   Sẽ xử lý theo các lô (batch) {batch_size} mẩu.")

    for i in range(0, len(list_of_texts), batch_size):
        batch_texts = list_of_texts[i : i + batch_size]
        batch_metadata = list_of_metadata[i : i + batch_size]
        
        current_batch_num = (i // batch_size) + 1
        print(f"   Đang xử lý lô {current_batch_num}/{ (len(list_of_texts) // batch_size) + 1 }...")
        
        db.add_texts(
            texts=batch_texts,
            metadatas=batch_metadata
        )
        
    print("   Đang lưu (persist) tất cả thay đổi xuống ổ đĩa...")
    db.persist()
    
    print(f"Đã tạo Vector DB (Local) thành công tại: {VECTOR_DB_DIR}")
    print(f"Tổng số vector đã được lưu: {db._collection.count()}")

except Exception as e:
    print(f"\n[LỖI] Đã xảy ra lỗi khi tạo Vector DB: {e}")