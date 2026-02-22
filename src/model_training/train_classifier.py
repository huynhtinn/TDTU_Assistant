# src/model_training/train_classifier.py

import os
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import torch
from underthesea import word_tokenize

os.environ["PYTORCH_ALLOW_INSECURE_LOAD"] = "1"

MODEL_NAME = "vinai/phobert-base"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'training', 'final_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'models', 'intent_classifier')

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

def compute_metrics(pred):
    """Hàm để đánh giá độ chính xác sau mỗi vòng train"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print(f"--- BẮT ĐẦU TRAINING MODEL PHÂN LOẠI ({MODEL_NAME}) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Thiết bị training: {device.upper()}")
    if device == "cpu":
        print("   [CẢNH BÁO] Train trên CPU sẽ rất chậm. Kiên nhẫn nhé!")

    print(f"   Đang đọc dữ liệu từ: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    
    # Tạo mapping nhãn (Label Mapping)
    # Biến đổi: "ADMISSION" -> 0, "ACADEMIC" -> 1...
    unique_labels = df['label'].unique()
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    print(f"   Tìm thấy {len(unique_labels)} nhãn: {unique_labels}")
    
    # Áp dụng mapping vào dataframe
    df['label_id'] = df['label'].map(label2id)
    
    # Chia tập Train (80%) và Test (20%)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Chuyển sang định dạng HuggingFace Dataset
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

    # Tokenizer (Bộ cắt từ)
    print("   Đang tải Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        # Bước 1: Tách từ tiếng Việt (Word Segmentation)
        # Ví dụ: "Học phí bao nhiêu" -> "Học_phí bao_nhiêu"
        segmented_texts = [word_tokenize(text, format="text") for text in examples["text"]]
        
        # Bước 2: Tokenize bằng PhoBERT tokenizer
        return tokenizer(segmented_texts, padding="max_length", truncation=True, max_length=128)
    
    print("   Đang mã hóa dữ liệu (Tokenizing)...")

    print("   Đang mã hóa dữ liệu (Tokenizing)...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Xóa các cột không cần thiết, chỉ giữ lại input_ids, attention_mask, labels
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "label", "__index_level_0__"])
    tokenized_datasets = tokenized_datasets.rename_column("label_id", "labels")
    tokenized_datasets.set_format("torch")

    # Khởi tạo Model
    print("   Đang tải Model PhoBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    # Cấu hình Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='./logs',
        logging_steps=10,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # TRAIN
    print("\n--- START TRAINING ---")
    trainer.train()

    # Đánh giá cuối cùng
    print("\n--- ĐÁNH GIÁ TRÊN TẬP TEST ---")
    eval_result = trainer.evaluate()
    print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")

    # Lưu Model hoàn chỉnh
    print(f"\n--- LƯU MODEL TẠI: {OUTPUT_DIR} ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Lưu thêm file mapping label
    with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()