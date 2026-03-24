# src/model_training/test_model.py

import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from underthesea import word_tokenize

# Để tránh lỗi bảo mật nếu dùng PyTorch cũ/mới lẫn lộn
os.environ["PYTORCH_ALLOW_INSECURE_LOAD"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'intent_classifier')

def load_model():
    print(f"--- ĐANG LOAD MODEL TỪ: {MODEL_PATH} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Thiết bị: {device.upper()}")

    try:
        # Load Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval() # Chuyển sang chế độ đánh giá

        # Load Label Mapping (để biết 0 là gì, 1 là gì)
        with open(os.path.join(MODEL_PATH, 'label_map.json'), 'r', encoding='utf-8') as f:
            id2label = json.load(f)
            # Chuyển key từ string sang int (do json lưu key là string)
            id2label = {int(k): v for k, v in id2label.items()}
            
        print("✅ Load model thành công!")
        return tokenizer, model, id2label, device

    except Exception as e:
        print(f"LỖI: Không tìm thấy model tại {MODEL_PATH}")
        print(f"   Chi tiết: {e}")
        print("   Em đã chạy 'train_classifier.py' chưa?")
        exit()

def predict(text, tokenizer, model, id2label, device):
    # Chuẩn hóa text trước khi đưa vào model
    text_segmented = word_tokenize(text, format="text")
    
    # Chuẩn bị dữ liệu đầu vào (dùng text đã segmented)
    inputs = tokenizer(text_segmented, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Tính xác suất (Softmax)
        probs = F.softmax(logits, dim=-1)
        confidence, predicted_class_id = torch.max(probs, dim=-1)

    # Lấy kết quả
    predicted_label = id2label[predicted_class_id.item()]
    score = confidence.item() * 100

    return predicted_label, score

def main():
    tokenizer, model, id2label, device = load_model()
    
    print("\n" + "="*50)
    print("CHATBOT INTENT CLASSIFIER (TEST MODE)")
    print("   Gõ 'exit' hoặc 'quit' để thoát.")
    print("="*50 + "\n")

    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Tạm biệt!")
            break
            
        if not user_input.strip():
            continue

        # Gọi hàm dự đoán
        label, score = predict(user_input, tokenizer, model, id2label, device)
        
        # In kết quả đẹp mắt
        print(f"Bot : [Nhãn: {label}] - (Độ tin cậy: {score:.2f}%)")
        print("-" * 30)

if __name__ == "__main__":
    main()