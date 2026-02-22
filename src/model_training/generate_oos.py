# src/model_training/generate_oos.py

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
os.makedirs(TRAIN_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(TRAIN_DIR, 'train_data_oos.csv')

LABEL_NAME = "OUT_OF_SCOPE"

OOS_TOPICS = [
    "Chào hỏi xã giao (Hello, Hi, Chào bạn, Bot tên gì)",
    "Hỏi về thời tiết, khí hậu (Hôm nay mưa không, Nhiệt độ bao nhiêu)",
    "Hỏi về chính trị, lãnh đạo thế giới (Tổng thống Mỹ, Chủ tịch nước, Chiến tranh)",
    "Hỏi về giải trí, showbiz, ca sĩ, diễn viên (Sơn Tùng, Blackpink)",
    "Hỏi về kiến thức phổ thông, toán học, lịch sử (1+1 bằng mấy, Trái đất hình gì)",
    "Hỏi về ăn uống, món ăn ngon (Hôm nay ăn gì, Phở ở đâu ngon)",
    "Hỏi về tình cảm, tư vấn tâm lý cá nhân (Tôi buồn quá, Làm sao có người yêu)",
    "Các câu lệnh vô nghĩa hoặc spam (asdfgh, test, bla bla)"
]

def generate_oos_questions(topic, num_questions=15):
    try:
        model = ChatGoogleGenerativeAI(
            model=GEN_MODEL,
            google_api_key=MY_API_KEY,
            temperature=0.8
        )

        prompt_template = """
        Bạn là một người dùng đang chat với một chatbot của trường đại học.
        Tuy nhiên, bạn KHÔNG quan tâm đến chuyện học hành. Bạn đang muốn nói chuyện phím hoặc hỏi những thứ không liên quan.
        
        Nhiệm vụ: Hãy đặt ra {num} câu hỏi/câu nói ngắn gọn liên quan đến chủ đề: "{topic}".
        Ngôn ngữ: Tiếng Việt (có thể chèn vài từ tiếng Anh thông dụng hoặc teencode).
        
        Yêu cầu định dạng:
        - Chỉ trả về danh sách câu hỏi.
        - Mỗi câu trên một dòng.
        - Không đánh số thứ tự.
        - Không thêm lời dẫn.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | model | StrOutputParser()
        
        print(f"   -> Đang sinh dữ liệu cho chủ đề: {topic}...", end="", flush=True)
        response = chain.invoke({"topic": topic, "num": num_questions})
        
        questions = [line.strip() for line in response.split('\n') if line.strip()]
        print(f" Xong ({len(questions)} câu).")
        return questions

    except Exception as e:
        print(f"   [Lỗi AI]: {e}")
        return []

def main():
    print(f"--- BẮT ĐẦU SINH DỮ LIỆU OUT_OF_SCOPE ({LABEL_NAME}) ---")
    
    total_generated = 0
    
    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['text', 'label']) # Header
        
        for topic in OOS_TOPICS:
            # Sinh 20 câu cho mỗi chủ đề -> Tổng cộng khoảng 160 câu OOS
            questions = generate_oos_questions(topic, num_questions=20)
            
            for q in questions:
                writer.writerow([q, LABEL_NAME])
                total_generated += 1
            
            time.sleep(1)

    print(f"Đã sinh được: {total_generated} câu hỏi OOS.")
    print(f"File lưu tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()