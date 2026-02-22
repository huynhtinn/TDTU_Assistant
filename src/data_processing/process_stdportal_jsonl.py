# src/data_processing/process_stdportal_jsonl.py

"""
Script xử lý data.jsonl (243 documents) và đưa vào ChromaDB
"""

import json
import os
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

def load_jsonl(file_path):
    """Đọc file JSONL (mỗi dòng 1 JSON object)"""
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Bỏ qua dòng trống
                try:
                    data = json.loads(line)
                    documents.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Lỗi parse JSON: {e}")
                    continue
    
    return documents

def categorize_document(doc):
    """Phân loại document theo source"""
    source = doc['metadata'].get('source', '').lower()
    
    # Phân loại dựa vào keywords
    if any(kw in source for kw in ['học bổng', 'khen thưởng', 'hỗ trợ']):
        return 'financial'
    elif any(kw in source for kw in ['đạo đức', 'nội quy', 'quy tắc', 'vi phạm', 'kỷ luật']):
        return 'student_life'
    elif any(kw in source for kw in ['tốt nghiệp', 'rèn luyện', 'điểm']):
        return 'academic'
    else:
        return 'general'

def create_langchain_documents(jsonl_docs):
    """Chuyển JSONL docs thành LangChain Documents"""
    
    langchain_docs = []
    category_count = {'financial': 0, 'academic': 0, 'student_life': 0, 'general': 0}
    
    for doc in jsonl_docs:
        # Lấy nội dung
        content = doc.get('page_content', '')
        metadata = doc.get('metadata', {})
        
        # Phân loại
        category = categorize_document(doc)
        category_count[category] += 1
        
        # Tạo Document
        langchain_doc = Document(
            page_content=content,
            metadata={
                'source': metadata.get('source', 'Unknown'),
                'category': category,
                'page': metadata.get('page', 1),
                'file_name': metadata.get('file_name', ''),
                'processed_date': metadata.get('processed_date', '')
            }
        )
        
        langchain_docs.append(langchain_doc)
    
    print(f"\n📊 Thống kê phân loại:")
    for cat, count in category_count.items():
        print(f"   {cat}: {count} docs")
    
    return langchain_docs

def create_chromadb_by_category(documents, base_dir='data/processed'):
    """Tạo ChromaDB riêng cho từng category"""
    
    # Group documents by category
    categorized_docs = {
        'financial': [],
        'academic': [],
        'student_life': [],
        'general': []
    }
    
    for doc in documents:
        category = doc.metadata['category']
        categorized_docs[category].append(doc)
    
    # Khởi tạo embedding model (dimension 384 - matching ChromaDB)
    print("\n🔧 Khởi tạo embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    # Tạo ChromaDB cho từng category
    for category, docs in categorized_docs.items():
        if not docs:
            print(f"⚠️  Bỏ qua {category}: không có docs")
            continue
        
        db_path = os.path.join(base_dir, f'{category}_db')
        
        print(f"\n📂 Tạo ChromaDB cho {category}...")
        print(f"   Số documents: {len(docs)}")
        print(f"   Đường dẫn: {db_path}")
        
        # Tạo/cập nhật ChromaDB
        try:
            vectordb = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=db_path,
                collection_name="langchain"
            )
            
            print(f"   ✅ Đã lưu {vectordb._collection.count()} vectors")
            
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")

def main():
    """Main function"""
    
    print("=" * 60)
    print("PROCESS STDPORTAL JSONL → ChromaDB")
    print("=" * 60)
    
    # 1. Đọc JSONL
    jsonl_path = 'data/stdportal/data.jsonl'
    print(f"\n📂 Đọc file: {jsonl_path}")
    
    if not os.path.exists(jsonl_path):
        print(f"❌ File không tồn tại: {jsonl_path}")
        return
    
    jsonl_docs = load_jsonl(jsonl_path)
    print(f"✅ Đọc được {len(jsonl_docs)} documents")
    
    # 2. Chuyển sang LangChain Documents
    print("\n🔄 Chuyển đổi sang LangChain Documents...")
    langchain_docs = create_langchain_documents(jsonl_docs)
    
    # 3. Tạo ChromaDB
    print("\n💾 Tạo ChromaDB theo category...")
    create_chromadb_by_category(langchain_docs)
    
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH!")
    print("=" * 60)
    print("\n📌 Kết quả:")
    print("   - data/processed/financial_db/     (học bổng, khen thưởng)")
    print("   - data/processed/academic_db/      (tốt nghiệp, rèn luyện)")
    print("   - data/processed/student_life_db/  (nội quy, đạo đức)")
    print("   - data/processed/general_db/       (thông tin chung)")
    print("\n🎯 Agents giờ có thể truy vấn 243 documents quy chế!")

if __name__ == "__main__":
    main()
