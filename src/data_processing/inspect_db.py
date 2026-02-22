# src/data_processing/inspect_db.py

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("--- BẮT ĐẦU KIỂM TRA VECTOR DB (ChromaDB) ---")

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_DB_DIR = os.path.join(project_root, 'data', 'processed', 'chroma_db')

if not os.path.exists(VECTOR_DB_DIR):
    print(f"LỖI: Không tìm thấy thư mục Vector DB tại: {VECTOR_DB_DIR}")
    print("Em hãy chạy file 'embed_data.py' trước nhé.")
    exit()

print("Đang khởi tạo mô hình Embedding (Local) để kết nối...")
try:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("   Mô hình Embedding sẵn sàng.")
except Exception as e:
    print(f"LỖI: Không thể tải mô hình embedding. {e}")
    exit()

print(f"Đang kết nối tới Vector DB tại: {VECTOR_DB_DIR}")

try:
    db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embedding_model
    )
    
    total_vectors = db._collection.count()
    print(f"   Kết nối thành công. Tổng số vector trong DB: {total_vectors}")
    
    if total_vectors == 0:
        print("LỖI: Vector DB bị rỗng. Em hãy chạy lại 'embed_data.py'.")
        exit()
        
    print("\n--- Lấy 5 mẩu (chunks) đầu tiên trong DB: ---")
    
    retrieved_data = db.get(
        limit=5,
        include=["metadatas", "documents"]
    )
    
    for i in range(len(retrieved_data["ids"])):
        doc_text = retrieved_data["documents"][i]
        doc_metadata = retrieved_data["metadatas"][i]
        
        print(f"\n--- MẨU (CHUNK) {i+1} ---")
        print(f"   Text (bị cắt ngắn): {doc_text[:150]}...")
        print(f"   Metadata:")
        print(f"     -> Nguồn (source): {doc_metadata.get('source')}")
        print(f"     -> Tiêu đề trang: {doc_metadata.get('page_title')}")
        print(f"     -> Tiêu đề mục: {doc_metadata.get('section_title')}")

except Exception as e:
    print(f"\n[LỖI] Đã xảy ra lỗi khi truy vấn ChromaDB: {e}")