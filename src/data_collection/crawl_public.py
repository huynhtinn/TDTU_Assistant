# src/data_collection/crawl_public.py

import json
import os 
import time
import datetime
from dotenv import load_dotenv 
import requests 
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() 
MY_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MY_LLM_MODEL = os.getenv("LLM_MODEL")

if not (MY_GOOGLE_API_KEY and MY_LLM_MODEL):
    print("LỖI: Thiếu GOOGLE_API_KEY hoặc LLM_MODEL trong file .env")
    exit()

def get_raw_text_from_url(url):
    print(f"   Đang cào (requests): {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.body
        body_text = main_content.get_text(separator=' ', strip=True)
        if not body_text:
            print("   Lỗi: Trang không có nội dung text.")
            return None, "Trang trống"
        return body_text, soup.title.string 
    except requests.RequestException as e:
        print(f"   Lỗi khi cào (requests): {e}")
        return None, "Lỗi cào"

def structure_text_with_langchain(raw_text, model_name):
    print("   Đang gửi text thô đến Gemini để phân tích...")
    try:
        prompt_template = """
        Bạn là một bot trích xuất dữ liệu web.
        Nhiệm vụ của bạn là phân tích đoạn text thô sau đây, được lấy từ một trang web của trường đại học.
        Hãy bỏ qua tất cả các thành phần nhiễu (như menu, link điều hướng, footer, 'tin tức liên quan', 'Nhảy đến nội dung', 'Tuyển sinh English Main navigation').
        Chỉ tập trung vào nội dung chính (ví dụ: các quy định, thông báo, các mục chính).
        
        Hãy trả về kết quả dưới dạng một DANH SÁCH JSON (JSON list) [{{...}}, {{...}}].
        Mỗi đối tượng trong danh sách phải có 2 key:
        1. "title": Tiêu đề của phần nội dung đó (ví dụ: "1. Các phương thức xét tuyển").
        2. "content": Nội dung text của phần đó.
        
        QUAN TRỌNG: Chỉ được trả lời bằng JSON. Không thêm "```json" hay bất kỳ lời giải thích nào.

        --- BẮT ĐẦU TEXT THÔ ---
        {input_text}
        --- KẾT THÚC TEXT THÔ ---
        """
        
        model = ChatGoogleGenerativeAI(
            model=model_name, 
            google_api_key=MY_GOOGLE_API_KEY,
            temperature=0.0
        )
        
        # Tạo Chain
        prompt = ChatPromptTemplate.from_template(prompt_template)
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        
        response_text = chain.invoke({"input_text": raw_text})
        
        # Xử lý kết quả
        cleaned_json_text = response_text.strip().replace("```json", "").replace("```", "")
        structured_data = json.loads(cleaned_json_text)
        return structured_data

    except Exception as e:
        print(f"   Đã xảy ra lỗi khi phân tích: {e}")
        if 'response_text' in locals() and response_text:
             print(f"   (Debug: Phản hồi thô từ AI là: {response_text[:500]}...)")
        else:
             print("   (Debug: Không có phản hồi từ AI)")
        return None

def generate_safe_filename(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname.replace('.', '_') if parsed_url.hostname else 'no_host'
    path = parsed_url.path.strip('/').replace('/', '_').replace('-', '_')
    if not path: return f"{hostname}.json"
    safe_path = "".join(c for c in path if c.isalnum() or c == '_')
    return f"{hostname}_{safe_path}.json"

def main():
    urls_to_crawl = [
        # TUYỂN SINH
        "https://admission.tdtu.edu.vn/dai-hoc/tuyen-sinh/phuong-thuc-2025",
        "https://undergrad.tdtu.edu.vn/hoc-vu/ho-tro-sinh-vien",
        "https://thinangkhieu.tdtu.edu.vn/",
        "https://admission.tdtu.edu.vn/tuyen-sinh/2025/thu-tuc-nhap-hoc-2025",
        "https://admission.tdtu.edu.vn/dai-hoc/thong-bao-diem-trung-tuyen-va-thu-tuc-nhap-hoc-dai-hoc-nam-2025",
        "https://admission.tdtu.edu.vn/dai-hoc/thong-bao-huong-dan-dang-ky-xet-tuyen-dai-hoc-nam-2025",
        "https://admission.tdtu.edu.vn/en/undergraduate/guide-for-applicants",
        "https://admission.tdtu.edu.vn/en/undergraduate/How-to-apply",

        # SAU ĐẠI HỌC
        # 1. Chuẩn đầu ra
        "https://grad.tdtu.edu.vn/chuan-dau-ra-trinh-do-tien-si",
        "https://grad.tdtu.edu.vn/chuan-dau-ra-trinh-do-thac-si",
        "https://grad.tdtu.edu.vn/dao-tao/chuan-dau-ra-ngoai-ngu-bac-thac-si",
        # 2. Học phí, học bổng
        "https://grad.tdtu.edu.vn/hoc-phi-hoc-bong",
        "https://grad.tdtu.edu.vn/hoc-phi-hoc-bong/hoc-bong-chuong-trinh-4-1",
        "https://grad.tdtu.edu.vn/index.php/hoc-phi-hoc-bong/hoc-bong-thac-si",
        "https://grad.tdtu.edu.vn/hoc-phi-hoc-bong/hoc-bong-tien-si",
        "https://grad.tdtu.edu.vn/hoc-phi-hoc-bong/hoc-bong-sau-tien-si",
        "https://grad.tdtu.edu.vn/hoc-phi-hoc-bong/hoc-bong-chuong-trinh-lien-ket-quoc-te",
        "https://grad.tdtu.edu.vn/index.php/hoc-phi-hoc-bong/hoc-bong-luu-hoc-sinh",
        "https://grad.tdtu.edu.vn/hoc-phi-hoc-bong/hoc-bong-danh-cho-hoc-vien-hoc-tai-phan-hieu-khanh-hoa",
        "https://grad.tdtu.edu.vn/hoc-phi-hoc-bong/hoc-bong-danh-cho-can-bo-cong-doan",
        "https://grad.tdtu.edu.vn/hoc-phi-hoc-bong/khen-thuong-danh-cho-hoc-vien-dat-thanh-tich-nghien-cuu-khoa-hoc",
        "https://grad.tdtu.edu.vn/index.php/hoc-phi-hoc-bong/chinh-sach-hoc-bong",
        "https://grad.tdtu.edu.vn/hoc-phi-hoc-bong/hoc-phi",
        # 3. Hướng dẫn
        "https://grad.tdtu.edu.vn/dao-tao/thong-tin-huong-dan-dau-khoa-hoc",

        # TRUNG TÂM TIN HỌC
        "https://cait.tdtu.edu.vn/doi-ngu-giang-vien/",
        "https://cait.tdtu.edu.vn/phat-trien-phan-mem/",

        # KHOA HỌC + CÔNG NGHỆ
        "https://science.tdtu.edu.vn/en/research/institutes-and-research-groups",
        
        # HỢP TÁC QUỐC TẾ
        "https://international.tdtu.edu.vn/en/students/international-recruitment",
        "https://international.tdtu.edu.vn/en/students/language-program",
        "https://international.tdtu.edu.vn/en/students/international-internship-program",
        "https://international.tdtu.edu.vn/en/support/contact",
        "https://admission.tdtu.edu.vn/en/undergraduate/Scholarships",
        "https://admission.tdtu.edu.vn/du-hoc/chuong-trinh-lien-ket-dao-tao-quoc-te-trinh-do-dai-hoc",

        # HỢP TÁC DOANH NGHIỆP VÀ CỰU SINH VIÊN
        "https://ceca.tdtu.edu.vn/gioi-thieu",
        "https://www.tdtu.edu.vn/en/contact",

        # TƯ VẤN HỌC ĐƯỜNG
        "https://undergrad.tdtu.edu.vn/don-vi-quan-ly-hoc-vu/phong-dai-hoc",
        "https://undergrad.tdtu.edu.vn/hoc-vu/ho-tro-sinh-vien",
        "https://admission.tdtu.edu.vn/hoc-tai-tdtu/ho-tro-sinh-vien",
        "https://admission.tdtu.edu.vn/en/undergraduate/Student-support",

        # SINH VIÊN + HỌC VIÊN
        "https://pharmacy.tdtu.edu.vn/tin-tuc/quy-tac-ung-xu-sinh-vien-5-dac-trung-cua-sinh-vien-tdtu",

    ]
    
    print(f"--- BẮT ĐẦU QUÁ TRÌNH CRAWL HÀNG LOẠT ({len(urls_to_crawl)} URL) ---")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True) 
    
    success_count = 0
    for i, url in enumerate(urls_to_crawl):
        print(f"\n--- Đang xử lý URL {i+1}/{len(urls_to_crawl)}: {url} ---")
        
        raw_text, page_title = get_raw_text_from_url(url)
        if not raw_text:
            print("   Không thể lấy text thô. Bỏ qua URL này.")
            continue 

        print(f"   Cào thành công. Lấy được {len(raw_text)} ký tự.")
        
        structured_data = structure_text_with_langchain(raw_text, MY_LLM_MODEL) 
        if not structured_data:
            print("   Không thể bóc tách cấu trúc. Bỏ qua URL này.")
            continue
            
        print(f"   Gemini phân tích thành công! Tìm thấy {len(structured_data)} mục.")

        try:
            output_filename = generate_safe_filename(url)
            full_output_path = os.path.join(output_dir, output_filename)
            final_output = {
                "source_url": url,
                "crawl_date": datetime.datetime.now().isoformat(),
                "title": page_title if page_title else "Không có tiêu đề",
                "segmented_content": structured_data
            }
            print(f"   Đang lưu dữ liệu vào file: {output_filename}")
            with open(full_output_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
            success_count += 1
        except Exception as e:
            print(f"   [LỖI LƯU FILE] Không thể lưu file JSON: {e}")
        
        print("   Nghỉ 1 giây...")
        time.sleep(1)

    print(f"\n--- HOÀN TẤT ---")
    print(f"Đã cào và lưu thành công {success_count} file vào thư mục data/raw/")

if __name__ == "__main__":
    main()