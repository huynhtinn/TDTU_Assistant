import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from underthesea import word_tokenize

os.environ["PYTORCH_ALLOW_INSECURE_LOAD"] = "1"

class IntentClassifier:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Đang tải Model Phân loại lên {self.device.upper()}... ---")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load nhãn
            with open(os.path.join(model_path, 'label_map.json'), 'r', encoding='utf-8') as f:
                self.id2label = json.load(f)
                # Chuyển key thành int
                self.id2label = {int(k): v for k, v in self.id2label.items()}
                
            print("✅ Model Phân loại đã sẵn sàng.")
        except Exception as e:
            print(f"❌ Lỗi tải model: {e}")
            raise e

    def predict(self, text):
        # 1. Tách từ (Word Segmentation)
        text_segmented = word_tokenize(text, format="text")
        
        # 2. Tokenize
        inputs = self.tokenizer(text_segmented, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 3. Dự đoán
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            confidence, predicted_id = torch.max(probs, dim=-1)

        label = self.id2label[predicted_id.item()]
        score = confidence.item() * 100
        return label, score