# src/model_training/visualize_metrics.py

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
import torch.nn.functional as F
from tqdm import tqdm

os.environ["PYTORCH_ALLOW_INSECURE_LOAD"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'training', 'final_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'intent_classifier')
REPORT_DIR = os.path.join(BASE_DIR, 'data', 'reports')

os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    print("--- BẮT ĐẦU ĐÁNH GIÁ TOÀN DIỆN ---")
    
    # Load Dữ liệu & Chia tập Test
    print(f"1. Đang đọc dữ liệu từ: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"   Số lượng mẫu kiểm tra: {len(test_df)}")

    # Load Model
    print(f"2. Đang load model từ: {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    id2label = model.config.id2label
    label_list = [id2label[i] for i in range(len(id2label))]

    # Chạy dự đoán
    print("3. Đang chạy dự đoán (Inference)...")
    
    results = [] # Lưu kết quả chi tiết để phân tích

    for text, true_label in tqdm(zip(test_df['text'], test_df['label']), total=len(test_df)):
        # Pre-process
        text_segmented = word_tokenize(text, format="text")
        inputs = tokenizer(text_segmented, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            confidence, predicted_id = torch.max(probs, dim=-1)
            
        pred_label = id2label[predicted_id.item()]
        score = confidence.item()
        
        results.append({
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": score,
            "is_correct": true_label == pred_label
        })

    # Chuyển kết quả sang DataFrame để dễ xử lý
    res_df = pd.DataFrame(results)

    # Vẽ Confusion Matrix
    print("4. Vẽ Confusion Matrix...")
    cm = confusion_matrix(res_df['true_label'], res_df['pred_label'], labels=label_list)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_list, yticklabels=label_list)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrix.png'))
    plt.close()

    # Vẽ Biểu đồ Phân phối Độ tin cậy (Confidence Histogram) - MỚI
    print("5. Vẽ biểu đồ độ tin cậy...")
    plt.figure(figsize=(10, 6))
    
    # Vẽ histogram cho các câu đúng (Màu xanh)
    sns.histplot(data=res_df[res_df['is_correct']==True], x='confidence', color='green', label='Đúng', kde=True, alpha=0.5)
    # Vẽ histogram cho các câu sai (Màu đỏ)
    sns.histplot(data=res_df[res_df['is_correct']==False], x='confidence', color='red', label='Sai', kde=True, alpha=0.5)
    
    plt.title('Phân phối độ tin cậy của Model (Đúng vs Sai)')
    plt.xlabel('Độ tin cậy (Confidence Score)')
    plt.ylabel('Số lượng câu hỏi')
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, 'confidence_distribution.png'))
    plt.close()

    # Xuất báo cáo text
    print("6. Xuất báo cáo text...")
    report = classification_report(res_df['true_label'], res_df['pred_label'], target_names=label_list)
    with open(os.path.join(REPORT_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    # Xuất danh sách các câu SAI (Error Analysis)
    print("7. Xuất danh sách câu sai...")
    error_df = res_df[res_df['is_correct'] == False].sort_values(by='confidence', ascending=False)
    error_df.to_csv(os.path.join(REPORT_DIR, 'error_analysis.csv'), index=False, encoding='utf-8-sig')

    print(f"   -> Ảnh biểu đồ: {REPORT_DIR}")
    print(f"   -> File phân tích lỗi: {os.path.join(REPORT_DIR, 'error_analysis.csv')}")

if __name__ == "__main__":
    main()