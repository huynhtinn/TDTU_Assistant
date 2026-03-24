# file: src/data_processing/setup_sql.py

import sqlite3
import os
import random

# Đường dẫn lưu DB
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'student_data.db')

# --- DỮ LIỆU GIẢ LẬP ---
HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]
DEM = ["Văn", "Thị", "Minh", "Ngọc", "Thanh", "Đức", "Gia", "Bảo", "Hữu", "Tuấn", "Hoài", "Phương", "Quốc", "Thảo"]
TEN = ["Anh", "Bình", "Châu", "Dung", "Em", "Giang", "Hà", "Hùng", "Khanh", "Lan", "Minh", "Nam", "Oanh", "Phúc", "Quân", "Sơn", "Tâm", "Uyên", "Vinh", "Yến"]

NGANH = [
    "Kỹ thuật phần mềm", "Khoa học máy tính", "Mạng máy tính", "Hệ thống thông tin", 
    "Quản trị kinh doanh", "Kế toán", "Luật", "Dược học", 
    "Ngôn ngữ Anh", "Thiết kế đồ họa", "Kỹ thuật điện", "Tài chính ngân hàng"
]

def generate_name():
    """Tạo tên tiếng Việt ngẫu nhiên"""
    return f"{random.choice(HO)} {random.choice(DEM)} {random.choice(TEN)}"

def generate_student_data(num_students=2000):
    students = []
    print(f"--- Đang sinh dữ liệu cho {num_students} sinh viên... ---")
    
    for i in range(num_students):
        # 1. MSSV giả lập (VD: 5200001)
        mssv = f"52{random.randint(0, 3)}0{i:04d}" 
        
        # 2. Họ tên & Ngành
        ho_ten = generate_name()
        nganh = random.choice(NGANH)
        
        # 3. Điểm số (GPA) - Phân phối chuẩn để thực tế hơn
        # Phần lớn sinh viên sẽ nằm ở mức 6.0 - 8.0
        gpa = random.gauss(7.0, 1.2) 
        gpa = round(min(max(gpa, 0.0), 10.0), 2) # Kẹp giữa 0 và 10
        
        # 4. Điểm rèn luyện (ĐRL)
        # Điểm cao thường có ĐRL cao
        base_drl = int(gpa * 8) + random.randint(0, 30)
        drl = min(max(base_drl, 40), 100)
        
        # 5. Tín chỉ tích lũy (Năm 3-4 thường > 100)
        tin_chi = random.randint(50, 150)
        
        # 6. Nợ môn (Logic: GPA thấp dễ nợ môn)
        if gpa < 5.0:
            no_mon = 1 # Chắc chắn nợ
        elif gpa < 7.0:
            no_mon = random.choices([0, 1], weights=[0.6, 0.4])[0] # 40% khả năng nợ
        else:
            no_mon = 0 # Giỏi thì ít nợ
            
        students.append((mssv, ho_ten, nganh, gpa, drl, tin_chi, no_mon))
        
    return students

def create_dummy_db():
    # Xóa DB cũ nếu có để tạo mới
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Tạo bảng SINH_VIEN
    cursor.execute('''
    CREATE TABLE sinh_vien (
        mssv TEXT PRIMARY KEY,
        ho_ten TEXT,
        nganh_hoc TEXT,
        diem_tb_tich_luy REAL,
        diem_ren_luyen INTEGER,
        so_tin_chi_tich_luy INTEGER,
        no_mon INTEGER -- 0: Không nợ, 1: Có nợ
    )
    ''')

    # 2. Tạo bảng QUY_DINH_XEP_LOAI (Reference Data)
    cursor.execute('''
    CREATE TABLE quy_dinh_xep_loai (
        loai_tot_nghiep TEXT,
        min_gpa REAL,
        max_gpa REAL
    )
    ''')

    # 3. Insert dữ liệu Sinh viên (2000 dòng)
    students = generate_student_data(2000)
    cursor.executemany('INSERT INTO sinh_vien VALUES (?,?,?,?,?,?,?)', students)

    # 4. Insert dữ liệu Quy định
    rules = [
        ('Xuất sắc', 9.0, 10.0),
        ('Giỏi', 8.0, 8.99),
        ('Khá', 7.0, 7.99),
        ('Trung bình', 5.0, 6.99),
        ('Yếu/Kém', 0.0, 4.99)
    ]
    cursor.executemany('INSERT INTO quy_dinh_xep_loai VALUES (?,?,?)', rules)

    conn.commit()
    
    # --- TEST QUERY ---
    print("\n--- TEST TRUY VẤN (Kiểm tra dữ liệu) ---")
    cursor.execute("SELECT COUNT(*) FROM sinh_vien")
    print(f"Tổng số sinh viên: {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT * FROM sinh_vien WHERE diem_tb_tich_luy > 9.0 LIMIT 3")
    print("Top 3 sinh viên Xuất sắc:")
    for sv in cursor.fetchall():
        print(f" - {sv[1]} ({sv[2]}): GPA {sv[3]}")

    conn.close()
    print(f"\nĐã tạo Database giả lập thành công tại: {DB_PATH}")

if __name__ == "__main__":
    create_dummy_db()