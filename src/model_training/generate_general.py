# src/model_training/generate_general.py

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
OUTPUT_CSV = os.path.join(TRAIN_DIR, 'train_data_general_fix.csv')

LABEL_NAME = "GENERAL"

GENERAL_TOPICS = [
    "Địa chỉ các cơ sở (Quận 7, Bảo Lộc, Khánh Hòa)",
    "Hỏi đường đi, xe bus đến trường",
    "Thông tin về Thư viện (Tòa nhà, giờ mở cửa)",
    "Ký túc xá (Vị trí, cơ sở vật chất chung)",
    "Nhà thi đấu, Sân vận động, Hồ bơi",
    "Căn tin, quán cafe trong trường",
    "Bãi giữ xe (Hầm xe, giờ gửi xe)",
    "Đồng phục, tác phong khi đến trường (chung chung)",
    "Lịch sử hình thành trường, Ban giám hiệu",
    "Wifi, mạng internet trong trường",
    "Các tòa nhà (Tòa A, B, C, D, E...)",
    "Phòng y tế nằm ở đâu"
]

def generate_questions(topic, num_questions=10):
    try:
        model = ChatGoogleGenerativeAI(
            model=GEN_MODEL, 
            google_api_key=MY_API_KEY,
            temperature=0.7
        )

        prompt_template = """
        Bạn là sinh viên Đại học Tôn Đức Thắng (TDTU).
        Hãy đặt {num} câu hỏi ngắn gọn hỏi về chủ đề: "{topic}".
        
        Lưu ý: Chỉ hỏi thông tin chung (địa điểm, giờ giấc, cái gì, ở đâu), KHÔNG hỏi về quy chế học vụ hay điểm số.
        
        Yêu cầu:
        - Chỉ trả về danh sách câu hỏi.
        - Mỗi câu trên một dòng.
        - Không đánh số.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | model | StrOutputParser()
        
        print(f"   -> Đang sinh cho chủ đề: {topic}...", end="", flush=True)
        response = chain.invoke({"topic": topic, "num": num_questions})
        
        questions = [line.strip() for line in response.split('\n') if line.strip()]
        print(f" Xong ({len(questions)} câu).")
        return questions

    except Exception as e:
        print(f"   [Lỗi]: {e}")
        return []

def main():
    print(f"--- BẮT ĐẦU BỔ SUNG DỮ LIỆU {LABEL_NAME} ---")
    total = 0
    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['text', 'label'])
        
        for topic in GENERAL_TOPICS:
            # Sinh 12 câu cho mỗi chủ đề -> Tổng cộng khoảng 140 câu
            questions = generate_questions(topic, num_questions=12)
            for q in questions:
                writer.writerow([q, LABEL_NAME])
                total += 1
            time.sleep(1)

    print(f"Đã sinh thêm: {total} câu GENERAL.")

if __name__ == "__main__":
    main()