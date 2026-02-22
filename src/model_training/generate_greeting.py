# src/model_training/generate_greeting.py

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
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'training')
OUTPUT_CSV = os.path.join(TRAIN_DIR, 'train_data_greeting.csv')

LABEL_NAME = "GREETING"

GREETING_CONTEXTS = [
    "Các câu chào hỏi mở đầu thông thường (Xin chào, Hi, Hello, Alo)",
    "Các câu chào hỏi lễ phép với admin/thầy cô (Chào ad, Em chào thầy/cô)",
    "Các câu chào hỏi kèm ý muốn hỏi (Cho mình hỏi chút, Ad ơi giúp em với)",
    "Các câu chào theo buổi (Chào buổi sáng, Good morning)",
    "Các câu cảm ơn sau khi được giúp đỡ (Cảm ơn nhé, Thanks, Đã hiểu)",
    "Các câu tạm biệt (Bye bye, Tạm biệt, Hẹn gặp lại)"
]

def generate_greetings(context, num_questions=15):
    try:
        model = ChatGoogleGenerativeAI(
            model=GEN_MODEL, 
            google_api_key=MY_API_KEY,
            temperature=0.9
        )

        prompt_template = """
        Bạn là sinh viên đại học. Hãy liệt kê {num} câu nói ngắn gọn dùng trong tình huống: "{context}" khi chat với chatbot của trường.
        
        Yêu cầu:
        - Đa dạng phong cách: từ lịch sự, trang trọng đến thân mật, teencode nhẹ (vd: 'tks', 'hi ad').
        - Tuyệt đối KHÔNG đánh số thứ tự.
        - Mỗi câu một dòng.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | model | StrOutputParser()
        
        print(f"   -> Đang sinh mẫu câu: {context}...", end="", flush=True)
        response = chain.invoke({"context": context, "num": num_questions})
        
        questions = [line.strip() for line in response.split('\n') if line.strip()]
        print(f" Xong ({len(questions)} câu).")
        return questions

    except Exception as e:
        print(f"   [Lỗi]: {e}")
        return []

def main():
    print(f"--- BẮT ĐẦU SINH DỮ LIỆU {LABEL_NAME} ---")
    total = 0
    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['text', 'label'])
        
        for context in GREETING_CONTEXTS:
            # Sinh 15 câu cho mỗi ngữ cảnh -> Tổng cộng khoảng 90-100 câu
            questions = generate_greetings(context, num_questions=15)
            for q in questions:
                writer.writerow([q, LABEL_NAME])
                total += 1
            time.sleep(1)

    print(f"Đã sinh thêm: {total} câu GREETING.")
    print(f"Lưu tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()