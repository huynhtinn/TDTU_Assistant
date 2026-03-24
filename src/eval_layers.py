
import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_TEST_FILE = ROOT / "data" / "eval" / "data_labels.csv"
OUT_DIR = ROOT / "evaluate" / "layer_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompt gửi cho LLaMA ────────────────────────────────────────────
ROUTER_PROMPT = """You are a Vietnamese university question router. Classify the question into one or more agents.

## AGENTS & KEYWORDS (Vietnamese → Agent)

**ADMISSION** — tuyển sinh, xét tuyển, điểm chuẩn, trúng tuyển, nhập học (tân sinh viên), hồ sơ xét tuyển, phương thức tuyển sinh, đề án tuyển sinh, đăng ký xét tuyển, điểm sàn, chỉ tiêu tuyển sinh, liên kết quốc tế (điều kiện đầu vào)
**FINANCIAL** — học phí, đóng tiền, công nợ, miễn giảm, hoàn trả học phí, vay vốn, bảo hiểm y tế (mức đóng/phí), lệ phí, chi phí, mức phí, tài chính, kế hoạch tài chính
**ACADEMIC** — điểm số, GPA, tín chỉ, học phần, đăng ký môn, quy chế đào tạo, phòng đào tạo, tốt nghiệp (điều kiện), bảo lưu, phúc khảo, cấm thi, thời gian đào tạo, học bổng (điều kiện xét), chuyển ngành, bảng điểm
**STUDENT_LIFE** — ký túc xá (KTX), câu lạc bộ, tình nguyện, sự kiện ngoại khóa, thẻ sinh viên, phòng y tế, tư vấn tâm lý, sân bóng, phòng gym, giờ giới nghiêm, phòng CTSV, cổng sinh viên (mật khẩu), trang phục, thư viện, điểm rèn luyện
**GENERAL** — địa chỉ, website, năm thành lập, khoa viện (số lượng), khuôn viên, quận, thông tin chung về trường

## RULES
1. Match keywords FIRST — use the keyword list above to decide
2. If the question contains keywords from TWO agents, return BOTH agents
3. "điểm chuẩn/xét tuyển/tuyển sinh/nhập học" → ADMISSION (NOT ACADEMIC)
4. "học phí/đóng tiền/công nợ/miễn giảm/hoàn phí/lệ phí" → FINANCIAL (NOT ACADEMIC)
5. "địa chỉ/website/thành lập/khuôn viên" → GENERAL (NOT ACADEMIC)

## EXAMPLES
Q: "Điểm chuẩn ngành CNTT?" → {{"plan":[{{"agent":"ADMISSION","query":"Điểm chuẩn ngành CNTT"}}]}}
Q: "Học phí ngành Luật bao nhiêu?" → {{"plan":[{{"agent":"FINANCIAL","query":"Học phí ngành Luật"}}]}}
Q: "Điều kiện tốt nghiệp và lệ phí?" → {{"plan":[{{"agent":"ACADEMIC","query":"Điều kiện tốt nghiệp"}},{{"agent":"FINANCIAL","query":"Lệ phí tốt nghiệp"}}]}}
Q: "Website trường?" → {{"plan":[{{"agent":"GENERAL","query":"Website trường"}}]}}
Q: "Điểm chuẩn và học phí ngành QTKD?" → {{"plan":[{{"agent":"ADMISSION","query":"Điểm chuẩn QTKD"}},{{"agent":"FINANCIAL","query":"Học phí QTKD"}}]}}

Return ONLY valid JSON. No explanation.

Question: {question}
JSON:"""


# ── Helpers ──────────────────────────────────────────────────────────
def parse_agents(raw: str) -> set:
    if not raw or not raw.strip():
        return set()
    return {a.strip().upper() for a in raw.split(",") if a.strip()}


def extract_agents_from_json(output: str) -> set:
    start = output.find('{')
    if start == -1:
        return set()
    depth, end = 0, start
    for i, ch in enumerate(output[start:], start):
        depth += (ch == '{') - (ch == '}')
        if depth == 0:
            end = i + 1
            break
    plan = json.loads(output[start:end])
    return {s["agent"].upper() for s in plan.get("plan", [])}


# ── Đánh giá ────────────────────────────────────────────────────────
def evaluate(test_file: Path = DEFAULT_TEST_FILE):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_groq import ChatGroq

    # Đọc dữ liệu
    with open(test_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Khởi tạo LLaMA chain
    llm = ChatGroq(
        model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        api_key=os.getenv("API_KEY"),
        temperature=0,
    )
    chain = ChatPromptTemplate.from_template(ROUTER_PROMPT) | llm | StrOutputParser()

    print(f"\n{'='*60}")
    print(f"  ĐÁNH GIÁ NHẬN DIỆN NHÃN — LLaMA Router")
    print(f"  File: {test_file.name} | Số câu: {len(rows)}")
    print(f"  Tính điểm: đúng hoàn toàn = 1.0 | đúng 1/2 nhãn = 0.5")
    print(f"{'='*60}\n")

    total_score = 0.0
    details = []

    for row in rows:
        qid = row.get("ID", "?")
        question = row.get("Question", "").strip()
        expected = parse_agents(row.get("Expected Agents", ""))

        try:
            # Retry with backoff for rate limits
            for attempt in range(3):
                try:
                    output = chain.invoke({"question": question})
                    predicted = extract_agents_from_json(output)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        wait = 30 * (attempt + 1)
                        print(f"  [{qid:>2}] ⏳ Rate limit, chờ {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
        except Exception as e:
            print(f"  [{qid:>2}] ⚠️ Lỗi: {e}")
            predicted = set()

        # Tính điểm: đúng hoàn toàn = 1.0, đúng 1 phần = tp / len(expected)
        if expected == predicted:
            score = 1.0
            icon = "✅"
        else:
            tp = len(expected & predicted)
            score = tp / len(expected) if expected else 0.0
            icon = "⚠️" if score > 0 else "❌"

        total_score += score

        print(f"  [{qid:>2}] {icon} ({score:.1f})  Đúng: {sorted(expected)}  |  Dự đoán: {sorted(predicted)}")

        details.append({
            "id": qid,
            "question": question,
            "expected": sorted(expected),
            "predicted": sorted(predicted),
            "score": score,
        })

    # Kết quả
    n = len(rows)
    acc = round(total_score / n * 100, 2) if n else 0

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ: {total_score:.1f}/{n} điểm — Accuracy: {acc}%")
    print(f"{'='*60}")

    # Lưu JSON
    result = {
        "evaluated_at": datetime.now().isoformat(),
        "model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        "total": n,
        "total_score": total_score,
        "accuracy_pct": acc,
        "details": details,
    }
    out_file = OUT_DIR / "label_accuracy.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Đã lưu: {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Đánh giá nhận diện nhãn LLaMA")
    parser.add_argument("--file", type=str, default=None,
                        help="Đường dẫn file CSV (mặc định: data/eval/data_labels.csv)")
    args = parser.parse_args()
    test_file = Path(args.file) if args.file else DEFAULT_TEST_FILE
    evaluate(test_file)
