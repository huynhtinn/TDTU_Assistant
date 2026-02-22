# src/model_training/merge_data.py
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING_DIR = os.path.join(BASE_DIR, 'data', 'training')

INPUT_FILES = [
    os.path.join(TRAINING_DIR, 'train_data_auto.csv'),
    os.path.join(TRAINING_DIR, 'train_data_oos.csv'),
    os.path.join(TRAINING_DIR, 'train_data_general_fix.csv'),
    os.path.join(TRAINING_DIR, 'train_data_greeting.csv'),
]

OUTPUT_FILE = os.path.join(TRAINING_DIR, 'final_dataset.csv')

def main():
    print("--- BẮT ĐẦU GỘP DỮ LIỆU ---")
    
    data_frames = []
    
    for file_path in INPUT_FILES:
        if os.path.exists(file_path):
            print(f"   -> Đang đọc file: {os.path.basename(file_path)}")
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                data_frames.append(df)
                print(f"      (Tìm thấy {len(df)} dòng)")
            except Exception as e:
                print(f"      [LỖI] Không đọc được file: {e}")
        else:
            print(f"   [CẢNH BÁO] Không tìm thấy file: {file_path}")

    if not data_frames:
        print("Lỗi: Không có dữ liệu nào để gộp!")
        return

    full_df = pd.concat(data_frames, ignore_index=True)
    
    # Xóa các dòng bị trùng lặp (nếu có)
    print("   -> Đang xóa dữ liệu trùng lặp...")
    before_dedup = len(full_df)
    full_df.drop_duplicates(subset=['text'], inplace=True)
    after_dedup = len(full_df)
    print(f"      (Đã xóa {before_dedup - after_dedup} dòng trùng)")

    # Xóa các dòng mà text hoặc label bị rỗng
    full_df.dropna(subset=['text', 'label'], inplace=True)

    # Xáo trộn dữ liệu
    # frac=1 nghĩa là lấy 100% dữ liệu nhưng xáo trộn vị trí
    print("   -> Đang xáo trộn ngẫu nhiên dữ liệu...")
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Thống kê
    print("\n--- THỐNG KÊ DỮ LIỆU SAU KHI GỘP ---")
    print(full_df['label'].value_counts())
    print("-" * 30)
    print(f"TỔNG CỘNG: {len(full_df)} dòng dữ liệu.")

    full_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\nĐã lưu file kết quả tại: {OUTPUT_FILE}")
    print("Sẵn sàng cho bước Train Model!")

if __name__ == "__main__":
    main()