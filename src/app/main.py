import os
import json
import sys
import time
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from agents import get_agents, _SHARED_EMBEDDING_MODEL
from intent_classifier import IntentClassifier

load_dotenv()

print("--- Đang khởi tạo hệ thống ---")

# Lớp 1: PhoBERT
MODEL_PATH = os.path.join(project_root, 'models', 'intent_classifier')
try:
    classifier = IntentClassifier(MODEL_PATH)
    print("Layer 1 sẵn sàng.")
except Exception as e:
    print(f"Lỗi tải PhoBERT: {e}. Hệ thống sẽ bỏ qua Layer 1 (mọi câu hỏi → IN_SCOPE).")
    classifier = None  

# Lớp 2: Groq Router
llm_router = ChatGroq(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("API_KEY"),
    temperature=0
)

specialist_agents = get_agents()
print("Layer 2 sẵn sàng.")


_CACHE_MAX_SIZE = 128
_CACHE_SIMILARITY_THRESHOLD = 0.95  # cosine similarity >= 0.95 
_cache_entries: list = []
_cache_embedding_model = _SHARED_EMBEDDING_MODEL  


def _cache_lookup(question: str):
    if not _cache_entries:
        return None

    q_emb = np.array(_cache_embedding_model.embed_query(question), dtype=np.float32)

    best_score = -1.0
    best_idx = -1
    for i, entry in enumerate(_cache_entries):
        score = float(np.dot(q_emb, entry["embedding"]))
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score >= _CACHE_SIMILARITY_THRESHOLD:
        entry = _cache_entries[best_idx]
        _cache_entries.append(_cache_entries.pop(best_idx))
        print(f"   Cache HIT (similarity: {best_score:.3f})")
        return entry["agent_responses"], entry["contexts"]

    print(f"   Cache MISS (best similarity: {best_score:.3f})")
    return None


def _cache_store(question: str, agent_responses: str, contexts: list):
    q_emb = np.array(_cache_embedding_model.embed_query(question), dtype=np.float32)
    _cache_entries.append({
        "embedding":       q_emb,
        "agent_responses": agent_responses,
        "contexts":        contexts,
    })
    while len(_cache_entries) > _CACHE_MAX_SIZE:
        _cache_entries.pop(0)


def clear_cache():
    _cache_entries.clear()
    print("   Cache đã xoá.")

router_prompt = ChatPromptTemplate.from_template("""
You are an intelligent routing system for TDTU AI Assistant (Ton Duc Thang University - Đại học Tôn Đức Thắng).
Your task: Analyze the user's question and route it to the most appropriate specialized agent(s).

=== AVAILABLE AGENTS ===

**ACADEMIC**
- Student personal data (GPA, training points, credits, majors)
- Academic regulations and policies
- Graduation requirements, curriculum, credit transfer, internship/apprenticeship regulations
USE FOR: "Điểm của sinh viên X", "Quy chế đào tạo", "Thông tin học tập", "Chuẩn đầu ra", "Tập sự", "Thư khen"

**FINANCIAL**  
- Tuition fees, payment deadlines, payment methods
- Scholarships, rewards, financial aid, exemptions/reductions
- Student debts and payments
USE FOR: "Học phí", "Học bổng", "Khen thưởng", "Miễn giảm", "Công nợ", "Đóng tiền"

**ADMISSION**
- Entrance exams and admission criteria
- Application procedures
- Admission benchmarks
USE FOR: "Điểm chuẩn", "Tuyển sinh", "Đăng ký nhập học"

**STUDENT_LIFE**
- Student affairs regulations (CTSV), discipline, conduct, internal rules
- Dormitory (KTX), insurance
- Student activities and clubs
USE FOR: "Ký túc xá", "Bảo hiểm", "Nội quy", "Kỷ luật", "Công tác sinh viên", "Ứng xử"

**GENERAL**
- Contact information (emails, phones, addresses)
- General university information
- Out-of-scope or unclear questions
USE FOR: "Liên hệ", "Địa chỉ", "Email phòng ban"

=== ROUTING RULES ===

**Performance Optimization:**
1. **Prefer Single Agent**: If possible, route to ONE agent to minimize latency
2. **Group Related Queries**: "Thông tin sinh viên X" → Only ACADEMIC (don't split)
3. **Avoid Redundant Calls**: Don't call FINANCIAL if question has no finance/scholarship/payment intent

**Keyword Priority (must follow):**
4. If query mentions scholarship/finance/payment terms, ALWAYS include FINANCIAL:
    - "học bổng", "học phí", "khen thưởng", "hỗ trợ", "miễn giảm", "đóng tiền", "công nợ", "tài chính"
5. If query mentions admission terms, route to ADMISSION:
    - "tuyển sinh", "xét tuyển", "điểm chuẩn", "nhập học", "hồ sơ"
6. If query mentions student-life conduct/discipline terms, route to STUDENT_LIFE:
    - "công tác sinh viên", "nội quy", "kỷ luật", "ứng xử", "ký túc xá", "bảo hiểm"
7. If query asks about GPA/credits/curriculum/graduation regulations or "tập sự", route to ACADEMIC.
8. For multi-topic queries, split into multiple steps by topic.

**Edge Cases:**
9. **Out-of-Scope Questions**: Route to GENERAL with original query
10. **Ambiguous Questions**: Choose the most likely agent, but respect keyword priority above

=== EXAMPLES ===

**Example 1** - Simple query
Input: "Thông tin sinh viên Nguyễn Văn A"
Output: 
```json
{{
  "plan": [
    {{"agent": "ACADEMIC", "query": "Toàn bộ thông tin sinh viên Nguyễn Văn A"}}
  ]
}}
```

**Example 2** - Multi-topic query
Input: "Sinh viên B có nợ môn không và học phí bao nhiêu?"
Output:
```json
{{
  "plan": [
    {{"agent": "ACADEMIC", "query": "Sinh viên B có nợ môn không?"}},
    {{"agent": "FINANCIAL", "query": "Học phí của sinh viên B"}}
  ]
}}
```

**Example 3** - Regulations query
Input: "Quy định về điểm rèn luyện"
Output:
```json
{{
  "plan": [
    {{"agent": "ACADEMIC", "query": "Quy định về điểm rèn luyện"}}
  ]
}}
```

**Example 3b** - Scholarship query
Input: "Điều kiện cơ bản để sinh viên được xét cấp học bổng khuyến khích học tập là gì?"
Output:
```json
{{
    "plan": [
        {{"agent": "FINANCIAL", "query": "Điều kiện cơ bản để sinh viên được xét cấp học bổng khuyến khích học tập là gì?"}}
    ]
}}
```

**Example 3c** - Mixed academic + scholarship
Input: "Điểm rèn luyện bao nhiêu thì được học bổng?"
Output:
```json
{{
    "plan": [
        {{"agent": "ACADEMIC", "query": "Quy định về điểm rèn luyện"}},
        {{"agent": "FINANCIAL", "query": "Điều kiện học bổng liên quan đến điểm rèn luyện"}}
    ]
}}
```

**Example 4** - Contact info
Input: "Email phòng đại học"
Output:
```json
{{
  "plan": [
    {{"agent": "GENERAL", "query": "Email phòng đại học"}}
  ]
}}
```

**Example 5** - Out-of-scope
Input: "Thời tiết hôm nay thế nào?"
Output:
```json
{{
  "plan": [
    {{"agent": "GENERAL", "query": "Thời tiết hôm nay thế nào?"}}
  ]
}}
```

=== OUTPUT FORMAT ===

**CRITICAL**: Return ONLY valid JSON. No markdown, no explanation.

Format:
```json
{{
  "plan": [
    {{"agent": "AGENT_NAME", "query": "specific question for this agent"}}
  ]
}}
```

User Question: {question}

JSON Response:
""")
router_chain = router_prompt | llm_router | StrOutputParser()

question_rewriter_prompt = ChatPromptTemplate.from_template("""Dưới đây là lịch sử hội thoại và một câu hỏi follow-up.
Nếu câu hỏi có dùng đại từ hoặc tham chiếu không rõ ràng (tôi, nó, đó, ngành học của tôi, điểm của tôi...),
hãy viết lại thành câu hỏi độc lập bằng cách thay thế tham chiếu bằng thông tin cụ thể từ lịch sử.
Nếu câu hỏi đã rõ ràng, trả về nguyên văn.
QUAN TRỌNG: Chỉ trả về câu hỏi đã viết lại, KHÔNG giải thích gì thêm.

Lịch sử hội thoại:
{chat_history}

Câu hỏi follow-up: {question}
Câu hỏi độc lập:""")

question_rewriter_chain = question_rewriter_prompt | llm_router | StrOutputParser()


def _format_chat_history(chat_history: list) -> str:
    lines = []
    for msg in chat_history:
        role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _rewrite_question(question: str, chat_history: list) -> str:
    """Viết lại câu hỏi follow-up thành standalone nếu có lịch sử."""
    if not chat_history:
        return question
    history_str = _format_chat_history(chat_history)
    try:
        rewritten = question_rewriter_chain.invoke({
            "chat_history": history_str,
            "question": question,
        }).strip()
        if rewritten and rewritten != question:
            print(f"   [Rewrite] '{question}' → '{rewritten}'")
        return rewritten or question
    except Exception as e:
        print(f"   [Rewrite] Lỗi: {e}. Dùng câu hỏi gốc.")
        return question


synthesizer_prompt = ChatPromptTemplate.from_template("""
Bạn là Trợ lý ảo của Trường Đại học Tôn Đức Thắng (TDTU).
NHIỆM VỤ: Đọc kết quả từ các agent và trả lời chính xác, thân thiện cho sinh viên.
QUY TẮC BẮT BUỘC:
- Chỉ đề cập đến Trường ĐH Tôn Đức Thắng (TDTU). TUYỆT ĐỐI KHÔNG nhắc đến trường đại học khác.
- Nếu không có thông tin, nói: "Tôi chưa tìm thấy thông tin này trong dữ liệu của TDTU."
- KHÔNG tự điền thông tin ngoài dữ liệu được cung cấp.
- TUYỆT ĐỐI KHÔNG bịa số liệu, ngưỡng điểm, tỉ lệ phần trăm hay bảng xếp loại nếu agent KHÔNG cung cấp cụ thể.
- Nếu agent trả về nội dung chung chung (không có con số cụ thể), CHỈ tóm tắt nội dung đó, KHÔNG thêm chi tiết từ kiến thức riêng.
- Khi có dữ liệu cụ thể từ agent → trích dẫn chính xác. Khi KHÔNG có → nói rõ "thông tin chi tiết chưa có trong dữ liệu".
Your task: Read agent responses and create a clear, helpful answer for the user.

User Question: "{question}"

Agent Responses:
{agent_responses}

=== DATA PROCESSING RULES ===

**1. Raw Data Recognition**
- If agent returns lists/tuples like `[('522001', 'Nguyễn Văn A', ...)]`, this IS the answer
- DO NOT say "không tìm thấy" when you see data in `[]` or `()`
- Transform raw data into natural Vietnamese sentences
- If result has MULTIPLE rows → list ALL of them, do NOT pick just one
- Single-element tuples like `[('52100064',), ('52200747',)]` = 2 results for the queried field

**2. Empty Results**
- Empty list `[]` or `None` = No data found
- Response: "Xin lỗi, tôi không tìm thấy thông tin về [topic] trong hệ thống."

**3. Error Handling**
- If agent returns error message → Apologize politely
- Example: "Xin lỗi, hệ thống gặp vấn đề khi tra cứu. Vui lòng thử lại sau."

**4. Conflicting Data**
- If multiple agents return different info → Prioritize most relevant
- Note any inconsistencies if critical

=== OUTPUT FORMAT RULES ===

**Structure:**
- Use **bold** for important info (names, numbers, grades)
- Use bullet points (•) for lists
- Break into paragraphs if lengthy
- Add line breaks for readability

**Tone & Style:**
- Friendly but professional
- Use "bạn" for casual tone
- Concise but complete
- End with helpful suggestion if appropriate

**Examples:**

**Input:** [('522001', 'Lê Văn A', 'CNTT', 7.2, 80, 101, 0)]
**Output:**
Tìm thấy thông tin sinh viên:
• **Họ tên**: Lê Văn A
• **Mã số SV**: 522001
• **Ngành học**: Công nghệ thông tin
• **Điểm TB tích lũy**: 7.2/10
• **Điểm rèn luyện**: 80
• **Số tín chỉ tích lũy**: 101
• **Nợ môn**: 0

**Input:** []
**Output:** 
Xin lỗi, tôi không tìm thấy thông tin sinh viên trong hệ thống. Bạn có thể kiểm tra lại tên hoặc mã số sinh viên không?

**Input:** Error: Database connection failed
**Output:**
Xin lỗi, hệ thống đang gặp sự cố kỹ thuật. Vui lòng thử lại sau.

=== QUALITY CHECKLIST ===

Before responding, verify:
-  All data from agents is included
-  Format is clean and readable
-  Tone is friendly and helpful
-  Vietnamese grammar is correct
-  No hallucination (stick to provided data)

Response (Vietnamese):
""")

synthesizer_chain = synthesizer_prompt | llm_router | StrOutputParser()

def deduplicate_contexts(contexts):
    if not contexts:
        return []

    def get_key(ctx):
        if isinstance(ctx, dict):
            return ctx.get("content", "").strip()
        return str(ctx).strip()

    seen_hashes = set()
    unique_contexts = []
    for ctx in contexts:
        text = get_key(ctx)
        if not text:
            continue
        ctx_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if ctx_hash not in seen_hashes:
            seen_hashes.add(ctx_hash)
            unique_contexts.append(ctx)
    return unique_contexts


def _parse_plan(router_output: str) -> list:
    json_start = router_output.find('{')
    if json_start == -1:
        raise ValueError("Không tìm thấy JSON trong output của router")
    depth = 0
    json_end = json_start
    for i, char in enumerate(router_output[json_start:], json_start):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break
    plan = json.loads(router_output[json_start:json_end])
    return plan.get("plan", [])


def _run_agents(steps: list) -> tuple:

    def run_step(step):
        agent_name = step.get("agent")
        sub_query  = step.get("query")
        agent = specialist_agents.get(agent_name)
        if not agent:
            print(f"   -> Không tìm thấy agent: {agent_name}")
            return agent_name, "", []
        print(f"   -> Gọi {agent_name}: '{sub_query}'")
        resp, ctx = agent.answer_with_context(sub_query)
        return agent_name, resp, ctx

    agent_responses = ""
    retrieved_contexts = []

    if len(steps) == 1:
        # Single agent: không cần prefix, gửi response trực tiếp
        agent_name, resp, ctx = run_step(steps[0])
        if resp:
            agent_responses = resp
        retrieved_contexts.extend(ctx)
    else:
        # Multiple agents: thêm prefix để phân biệt nguồn
        with ThreadPoolExecutor(max_workers=len(steps)) as executor:
            futures = {executor.submit(run_step, step): step for step in steps}
            for future in as_completed(futures):
                agent_name, resp, ctx = future.result()
                if resp:
                    agent_responses += f"- Thông tin từ {agent_name}: {resp}\n"
                retrieved_contexts.extend(ctx)

    return agent_responses, retrieved_contexts


def _prepare_agent_responses(question: str) -> tuple:

    #  Layer 1: PhoBERT 
    if classifier is None:
        bert_label, bert_score = "IN_SCOPE", 0.0
        print("   [Layer 1 - PhoBERT] Skipped (model not loaded)")
    else:
        bert_label, bert_score = classifier.predict(question)
        print(f"   [Layer 1 - PhoBERT] Nhãn: {bert_label} (Tin cậy: {bert_score:.1f}%)")

    if bert_label == "OUT_OF_SCOPE" and bert_score > 50.0:
        return "Xin lỗi, mình chỉ chuyên về thông tin của Đại học Tôn Đức Thắng thôi ạ.", "", []

    if bert_label == "GREETING" and bert_score > 50.0:
        return "Chào bạn! Mình là Trợ lý ảo TDTU. Mình có thể giúp gì?", "", []

    # Layer 2: Router
    print("   [Layer 2 - Groq] Đang phân tích chuyên sâu...")
    try:
        router_output = router_chain.invoke({"question": question})
        steps = _parse_plan(router_output)
    except Exception as e:
        print(f" Lỗi Router: {e}. -> Fallback to GENERAL agent")
        resp, ctx = specialist_agents["GENERAL"].answer_with_context(question)
        return resp, "", ctx

    if not steps:
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp.", "", []

    print(f"   Detected Plan: {len(steps)} bước")
    agent_responses, contexts = _run_agents(steps)
    return None, agent_responses, contexts


def process_query(question: str) -> str:
    """Process query và trả về response string."""
    response, _ = process_query_with_context(question)
    return response


def process_query_with_context(question: str, provider: str = "groq_llama", chat_history: list = None) -> tuple:

    print(f"\n User [{provider}]: {question}")

    standalone = _rewrite_question(question, chat_history or [])
    was_rewritten = (standalone.strip() != question.strip())

    cached = _cache_lookup(standalone)
    if cached is not None:
        agent_responses, contexts = cached
        unique_contexts = contexts
    else:
        early, agent_responses, contexts = _prepare_agent_responses(standalone)
        if early is not None:
            return early, []
        unique_contexts = deduplicate_contexts(contexts)
        if len(contexts) != len(unique_contexts):
            print(f"   [Dedup] {len(contexts)} → {len(unique_contexts)} contexts")
        if not was_rewritten:
            _cache_store(standalone, agent_responses, unique_contexts)
        else:
            print("   [Cache] Bỏ qua lưu cache (câu hỏi chứa ngữ cảnh cá nhân)")

    if provider == "groq_llama":
        synth_chain = synthesizer_chain  # chain mặc định
    else:
        try:
            llm = get_llm(provider)
            synth_chain = synthesizer_prompt | llm | StrOutputParser()
        except Exception as e:
            print(f"   Không thể dùng provider '{provider}': {e}. Fallback LLaMA.")
            synth_chain = synthesizer_chain

    print(f"   Synthesizing ({provider})...")
    final_answer = synth_chain.invoke({
        "question": question,
        "agent_responses": agent_responses,
    })

    return final_answer, unique_contexts


def process_query_streaming(question: str, provider: str = "groq_llama", chat_history: list = None) -> tuple:

    print(f"\nUser (stream) [{provider}]: {question}")

    standalone = _rewrite_question(question, chat_history or [])
    was_rewritten = (standalone.strip() != question.strip())

    cached = _cache_lookup(standalone)
    if cached is not None:
        agent_responses, unique_contexts = cached
    else:
        early, agent_responses, contexts = _prepare_agent_responses(standalone)
        if early is not None:
            return early, [], None
        unique_contexts = deduplicate_contexts(contexts)
        if not was_rewritten:
            _cache_store(standalone, agent_responses, unique_contexts)
        else:
            print("   [Cache] Bỏ qua lưu cache (câu hỏi chứa ngữ cảnh cá nhân)")

    if provider == "groq_llama":
        synth_chain = synthesizer_chain  
    else:
        try:
            llm = get_llm(provider)
            synth_chain = synthesizer_prompt | llm | StrOutputParser()
        except Exception as e:
            print(f"   Không thể dùng provider '{provider}': {e}. Fallback LLaMA.")
            synth_chain = synthesizer_chain

    print(f"   Synthesizing (stream) với {provider}...")

    def _stream_gen():
        for token in synth_chain.stream({
            "question": question,
            "agent_responses": agent_responses,
        }):
            yield token

    return None, unique_contexts, _stream_gen()


PROVIDER_LABELS = {
    "groq_llama":  f"LLaMA ({os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')})",
    "gemini":      f"Gemini ({os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')})",
}

PROVIDER_META = {
    "groq_llama": {
        "icon":        "⚡",
        "name":        os.getenv('LLM_MODEL', 'llama-3.1-8b-instant'),
        "full_name":   f"LLaMA — {os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')}",
        "badge":       "Groq",
    },
    "gemini": {
        "icon":        "✨",
        "name":        os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'),
        "full_name":   f"Gemini — {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}",
        "badge":       "Google",
    },
}


def get_available_providers() -> list[str]:
    """Chỉ trả về các provider còn lại: groq_llama, gemini."""
    providers = ["groq_llama"]
    if _GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        providers.append("gemini")
    return providers


def get_llm(provider: str):
    """
    Factory: trả về LLM object theo provider.
    Hỗ trợ: 'groq_llama', 'gemini'
    """
    if provider == "groq_llama":
        return ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("API_KEY"),
            temperature=0,
        )
    elif provider == "gemini":
        if not _GEMINI_AVAILABLE:
            raise ImportError("langchain-google-genai chưa được cài. Chạy: pip install langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
    else:
        raise ValueError(f"Provider không hợp lệ: '{provider}'. Chọn: groq_llama, gemini")


def process_query_compare(question: str, providers: list[str] | None = None) -> dict:
    if providers is None:
        providers = get_available_providers()

    print(f"\n[Compare] Đang xử lý retrieval cho: '{question}'")
    early, agent_responses, contexts = _prepare_agent_responses(question)
    unique_contexts = deduplicate_contexts(contexts)

    if early is not None:
        return {
            p: {
                "response": early,
                "contexts": [],
                "elapsed":  0.0,
                "label":    PROVIDER_LABELS.get(p, p),
                "error":    None,
            }
            for p in providers
        }

    def _synthesize(provider: str) -> tuple:
        label = PROVIDER_LABELS.get(provider, provider)
        try:
            llm = get_llm(provider)
            chain = synthesizer_prompt | llm | StrOutputParser()
            t0 = time.time()
            resp = chain.invoke({"question": question, "agent_responses": agent_responses})
            elapsed = time.time() - t0
            print(f"   [{label}] xong trong {elapsed:.1f}s")
            return provider, resp, elapsed, None
        except Exception as e:
            print(f"   [{label}] lỗi: {e}")
            return provider, f"Lỗi từ {label}: {str(e)}", 0.0, str(e)

    print(f"   Synthesis song song với {len(providers)} mô hình: {providers}")
    results = {}
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {executor.submit(_synthesize, p): p for p in providers}
        for future in as_completed(futures):
            provider, resp, elapsed, err = future.result()
            results[provider] = {
                "response": resp,
                "contexts": unique_contexts,
                "elapsed":  elapsed,
                "label":    PROVIDER_LABELS.get(provider, provider),
                "error":    err,
            }

    return results

if __name__ == "__main__":
    print("=== HỆ THỐNG DUAL-LAYER MULTI-AGENT ===")
    while True:
        q = input("\nBạn hỏi gì? (exit để thoát): ")
        if q.lower() == "exit":
            break
        ans = process_query(q)
        print(f"\nBot: {ans}")
