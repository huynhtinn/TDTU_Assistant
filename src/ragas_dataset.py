"""
ragas_dataset.py
================
Bước 1: Thu thập dữ liệu đánh giá từ RAG system.

Script này đọc eval_layer2.csv (question + ground_truth),
gọi hệ thống RAG để lấy answer và contexts,
và lưu dataset hoàn chỉnh ra file JSON.

Chạy:
    python eval/ragas_dataset.py              # Toàn bộ dataset (99 câu)
    python eval/ragas_dataset.py --limit 10   # Chỉ 10 câu đầu (demo nhanh)
    python eval/ragas_dataset.py --start 5 --limit 10  # Câu 5→14
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# ── Setup paths ─────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
app_path = os.path.join(project_root, "src", "app")

for path in [app_path, project_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ── Import RAG system ────────────────────────────────────────────────────────
print("⏳ Đang khởi động hệ thống RAG...")
try:
    from main import process_query_with_context
    print("✅ Hệ thống RAG đã sẵn sàng.\n")
except Exception as e:
    print(f"❌ Không thể khởi động RAG system: {e}")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_FILE    = os.path.join(project_root, "data", "eval", "eval_dataset.csv")
OUTPUT_FILE = os.path.join(project_root, "data", "eval", "ragas_dataset.json")


def load_csv(filepath: str) -> list[dict]:
    """
    Đọc eval_layer2.csv → list of {question_id, question, ground_truth}.
    
    Format mới (comma-separated, 3 cột):
        question_id, question_text, ground_truth
    """
    import csv
    samples = []
    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question     = row.get("question_text", "").strip()
            ground_truth = row.get("ground_truth", "").strip()
            question_id  = row.get("question_id", "").strip()
            if question and ground_truth:
                samples.append({
                    "question_id":  question_id,
                    "question":     question,
                    "ground_truth": ground_truth,
                })
    return samples


def collect_rag_outputs(samples: list[dict], delay: float = 1.0) -> list[dict]:
    """
    Gọi RAG system cho từng câu hỏi,
    thu thập answer và contexts.
    
    Args:
        samples : list[{question, ground_truth}]
        delay   : số giây nghỉ giữa các lần gọi (tránh rate limit)
    Returns:
        list[{question, answer, contexts, ground_truth}]
    """
    dataset = []
    total   = len(samples)

    print(f"📋 Bắt đầu thu thập RAG outputs ({total} câu hỏi)...")
    print("─" * 60)

    for i, sample in enumerate(samples, 1):
        question     = sample["question"]
        ground_truth = sample["ground_truth"]

        print(f"[{i:3}/{total}] ❓ {question[:70]}{'...' if len(question)>70 else ''}")

        try:
            answer, ctx_docs = process_query_with_context(question)

            # Chuẩn hóa contexts thành list[str]
            contexts = []
            for doc in ctx_docs:
                if hasattr(doc, "page_content"):      # LangChain Document
                    contexts.append(doc.page_content.strip())
                elif isinstance(doc, str) and doc.strip():
                    contexts.append(doc.strip())

            entry = {
                "question_id":  sample.get("question_id", ""),
                "question":     question,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": ground_truth,
            }
            dataset.append(entry)

            ctx_count = len(contexts)
            print(f"       ✅ answer ({len(answer)} chars) | contexts: {ctx_count} chunks")

        except Exception as e:
            print(f"       ❌ Lỗi: {e}")
            dataset.append({
                "question_id":  sample.get("question_id", ""),
                "question":     question,
                "answer":       f"ERROR: {e}",
                "contexts":     [],
                "ground_truth": ground_truth,
            })

        # Delay giữa các lần gọi API
        if i < total:
            time.sleep(delay)

    return dataset


def save_dataset(dataset: list[dict], output_path: str):
    """Lưu dataset ra JSON với metadata."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "metadata": {
            "created_at":   datetime.now().isoformat(),
            "total_samples": len(dataset),
            "description":  "RAGAS evaluation dataset for TDTU RAG system",
            "fields":       ["question", "answer", "contexts", "ground_truth"],
        },
        "samples": dataset,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu {len(dataset)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Thu thập RAG outputs để tạo RAGAS evaluation dataset."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Giới hạn số câu (mặc định: tất cả)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Bắt đầu từ index thứ bao nhiêu (mặc định: 0)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Thời gian nghỉ giữa các API call (giây, mặc định: 1.0)"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE,
        help="Đường dẫn file JSON output"
    )
    args = parser.parse_args()

    # Đọc CSV
    print(f"📂 Đọc dataset: {CSV_FILE}")
    all_samples = load_csv(CSV_FILE)
    print(f"   → Tìm thấy {len(all_samples)} câu hỏi.\n")

    # Slice theo start/limit
    samples = all_samples[args.start:]
    if args.limit is not None:
        samples = samples[:args.limit]

    print(f"📊 Sẽ xử lý {len(samples)} câu (start={args.start}, limit={args.limit})\n")

    # Thu thập
    t0      = time.time()
    dataset = collect_rag_outputs(samples, delay=args.delay)
    elapsed = time.time() - t0

    # Lưu
    save_dataset(dataset, args.output)

    # Tóm tắt
    successful   = sum(1 for d in dataset if not d["answer"].startswith("ERROR"))
    has_contexts = sum(1 for d in dataset if d["contexts"])
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT THU THẬP DỮ LIỆU")
    print("=" * 60)
    print(f"  Tổng câu hỏi       : {len(dataset)}")
    print(f"  Thành công         : {successful}/{len(dataset)}")
    print(f"  Có contexts        : {has_contexts}/{len(dataset)}")
    print(f"  Thời gian chạy     : {elapsed:.1f}s")
    print(f"  Output             : {args.output}")
    print("=" * 60)
    print("\n💡 Bước tiếp theo: python eval/ragas_evaluate.py")


if __name__ == "__main__":
    main()
