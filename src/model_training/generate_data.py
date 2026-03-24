# src/model_training/generate_data.py

import json
import os
import csv
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
MY_API_KEY = os.getenv("GOOGLE_API_KEY")
GEN_MODEL = "gemini-2.5-flash" 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'training')
os.makedirs(TRAIN_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(TRAIN_DIR, 'train_data_auto.csv')

def get_label_from_url(url):
    url = url.lower()
    if "admission" in url or "tuyen-sinh" in url or "thinangkhieu" in url:
        return "ADMISSION"
    elif "undergrad" in url or "grad" in url or "dao-tao" in url or "hoc-vu" in url:
        return "ACADEMIC"
    elif "ctsv" in url or "ren-luyen" in url or "ho-tro" in url:
        return "STUDENT_AFFAIRS"
    else:
        return "GENERAL"

def generate_questions_with_ai(content_text, num_questions=5):
    try:
        model = ChatGoogleGenerativeAI(
            model=GEN_MODEL,
            google_api_key=MY_API_KEY,
            temperature=0.7
        )

        prompt_template = """
        Bạn là một sinh viên đại học Tôn Đức Thắng.
        Dưới đây là một đoạn thông tin trích từ quy chế/thông báo của trường:
        ---
        "{text}"
        ---
        Nhiệm vụ: Hãy đặt ra {num} câu hỏi ngắn gọn, tự nhiên (văn phong sinh viên, có thể viết tắt thông dụng như 'sv', 'đh', 'tdtu') mà sinh viên sẽ dùng để hỏi về thông tin trên.
        
        Yêu cầu định dạng:
        - Chỉ trả về danh sách câu hỏi.
        - Mỗi câu hỏi trên một dòng.
        - Không đánh số thứ tự (1. 2.), không gạch đầu dòng.
        - Không thêm lời dẫn.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | model | StrOutputParser()

        # Giới hạn text đầu vào để tiết kiệm token (chỉ lấy 2000 ký tự đầu của mục)
        input_text = content_text[:2000]
        
        response = chain.invoke({"text": input_text, "num": num_questions})
        
        questions = [line.strip() for line in response.split('\n') if line.strip()]
        return questions

    except Exception as e:
        print(f"   [Lỗi AI]: {e}")
        return []

def main():
    print("--- BẮT ĐẦU SINH DỮ LIỆU TRAINING TỰ ĐỘNG ---")
    
    total_generated = 0
    
    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['text', 'label'])
        
        files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
        
        for i, filename in enumerate(files):
            file_path = os.path.join(RAW_DIR, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            url = data.get('source_url', '')
            label = get_label_from_url(url)
            
            print(f"\n[{i+1}/{len(files)}] Xử lý file: {filename}")
            print(f"   -> Nhãn dự đoán: {label}")
            
            sections = data.get('segmented_content', [])
            
            for sec_idx, section in enumerate(sections[:10]): 
                content = section.get('content', '')
                
                if len(content) < 50: 
                    continue
                
                print(f"   Generating cho mục {sec_idx+1}...", end="", flush=True)
                
                questions = generate_questions_with_ai(content, num_questions=5)
                
                for q in questions:
                    writer.writerow([q, label])
                    total_generated += 1
                
                print(f" Xong ({len(questions)} câu).")
                
                time.sleep(1) 

    print(f"Đã sinh được tổng cộng: {total_generated} câu hỏi mẫu.")
    print(f"File lưu tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()