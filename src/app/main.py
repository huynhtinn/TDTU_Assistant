import os
import json
import sys
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from agents import get_agents
from intent_classifier import IntentClassifier

load_dotenv()

print("--- Đang khởi tạo hệ thống ---")

# Lớp 1: PhoBERT
MODEL_PATH = os.path.join(project_root, 'models', 'intent_classifier')
try:
    classifier = IntentClassifier(MODEL_PATH)
    print("✅ Layer 1: PhoBERT Classifier sẵn sàng.")
except Exception as e:
    print(f"❌ Lỗi tải PhoBERT: {e}")
    exit()

# Lớp 2: Groq Router
# --- Khởi tạo ChatGroq ---
llm_router = ChatGroq(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("API_KEY"),
    temperature=0
)

specialist_agents = get_agents()
print("✅ Layer 2: Groq Router & Agents sẵn sàng.")

router_prompt = ChatPromptTemplate.from_template("""
You are an intelligent routing system for TDTU AI Assistant.
Your task: Analyze the user's question and route it to the most appropriate specialized agent(s).

=== AVAILABLE AGENTS ===

**ACADEMIC**
- Student personal data (GPA, training points, credits, majors)
- Academic regulations and policies
- Graduation requirements
USE FOR: "Điểm của sinh viên X", "Quy chế đào tạo", "Thông tin học tập"

**FINANCIAL**  
- Tuition fees, payment deadlines
- Scholarships and financial aid
- Student debts and payments
USE FOR: "Học phí", "Học bổng", "Công nợ"

**ADMISSION**
- Entrance exams and admission criteria
- Application procedures
- Admission benchmarks
USE FOR: "Điểm chuẩn", "Tuyển sinh", "Đăng ký nhập học"

**STUDENT_LIFE**
- Dormitory (KTX), insurance
- Student activities and clubs
- Campus facilities
USE FOR: "Ký túc xá", "Bảo hiểm", "Câu lạc bộ"

**GENERAL**
- Contact information (emails, phones, addresses)
- General university information
- Out-of-scope or unclear questions
USE FOR: "Liên hệ", "Địa chỉ", "Email phòng ban"

=== ROUTING RULES ===

**Performance Optimization:**
1. **Prefer Single Agent**: If possible, route to ONE agent to minimize latency
2. **Group Related Queries**: "Thông tin sinh viên X" → Only ACADEMIC (don't split)
3. **Avoid Redundant Calls**: Don't call FINANCIAL if question doesn't mention money

**Edge Cases:**
4. **Out-of-Scope Questions**: Route to GENERAL with original query
5. **Ambiguous Questions**: Choose the most likely agent
6. **Multi-Topic Questions**: Split ONLY if topics are clearly separate

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

synthesizer_prompt = ChatPromptTemplate.from_template("""
You are TDTU AI Assistant - a friendly and professional AI helper for students and staff.
Your task: Read agent responses and create a clear, helpful answer for the user.

User Question: "{question}"

Agent Responses:
{agent_responses}

=== DATA PROCESSING RULES ===

**1. Raw Data Recognition**
- If agent returns lists/tuples like `[('522001', 'Nguyễn Văn A', ...)]`, this IS the answer
- DO NOT say "không tìm thấy" when you see data in `[]` or `()`
- Transform raw data into natural Vietnamese sentences

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

**Input:** [('522001', 'Lê Văn A', 'CNTT', 8.5)]
**Output:**
Tìm thấy thông tin sinh viên:
• **Họ tên**: Lê Văn A
• **Mã số SV**: 522001
• **Ngành học**: Công nghệ thông tin
• **Điểm TB tích lũy**: 8.5/4.0

**Input:** []
**Output:** 
Xin lỗi, tôi không tìm thấy thông tin sinh viên trong hệ thống. Bạn có thể kiểm tra lại tên hoặc mã số sinh viên không?

**Input:** Error: Database connection failed
**Output:**
Xin lỗi, hệ thống đang gặp sự cố kỹ thuật. Vui lòng thử lại sau.

=== QUALITY CHECKLIST ===

Before responding, verify:
- ✅ All data from agents is included
- ✅ Format is clean and readable
- ✅ Tone is friendly and helpful
- ✅ Vietnamese grammar is correct
- ✅ No hallucination (stick to provided data)

Response (Vietnamese):
""")

synthesizer_chain = synthesizer_prompt | llm_router | StrOutputParser()

def process_query(question):
    """Process query và trả về response"""
    response, _ = process_query_with_context(question)
    return response

def process_query_with_context(question):
    """
    Process query và trả về cả response + retrieved contexts
    Returns: (response: str, contexts: list[str])
    """
    print(f"\n📢 User: {question}")
    retrieved_contexts = []  # Lưu context để trả về
    
    # Layer 1: PhoBERT
    bert_label, bert_score = classifier.predict(question)
    print(f"   [Layer 1 - PhoBERT] Nhãn: {bert_label} (Tin cậy: {bert_score:.1f}%)")
    
    if bert_label == "OUT_OF_SCOPE" and bert_score > 60.0:
         return "Xin lỗi, mình chỉ chuyên về thông tin của Đại học Tôn Đức Thắng thôi ạ.", []

    if bert_label == "GREETING" and bert_score > 60.0:
        return "Chào bạn! Mình là Trợ lý ảo TDTU. Mình có thể giúp gì?", []

    # Layer 2: Groq Gemini
    print("   [Layer 2 - Groq] Đang phân tích chuyên sâu...")
    
    try:
        router_output = router_chain.invoke({"question": question})
        # Parse JSON an toàn - tìm JSON object đầu tiên hoàn chỉnh
        json_start = router_output.find('{')
        if json_start == -1:
            raise ValueError("Không tìm thấy JSON")
        
        # Đếm {} để tìm đúng vị trí kết thúc
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
        
        router_output = router_output[json_start:json_end]
        plan = json.loads(router_output)
    except Exception as e:
        print(f"❌ Lỗi Router: {e}")
        # Fallback: gọi GENERAL agent
        print("   -> Fallback to GENERAL agent")
        response, contexts = specialist_agents["GENERAL"].answer_with_context(question)
        return response, contexts

    steps = plan.get("plan", [])
    if not steps:
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp.", []

    agent_responses = ""
    print(f"   Detected Plan: {len(steps)} bước")
    
    for step in steps:
        agent_name = step.get("agent")
        sub_query = step.get("query")
        
        agent = specialist_agents.get(agent_name)
        if agent:
            print(f"   -> Gọi {agent_name}: '{sub_query}'")
            response, context = agent.answer_with_context(sub_query)
            agent_responses += f"- Thông tin từ {agent_name}: {response}\n"
            if context:
                retrieved_contexts.extend(context)
        else:
            print(f"   -> Không tìm thấy agent: {agent_name}")

    print("   Synthesizing...")
    final_answer = synthesizer_chain.invoke({
        "question": question,
        "agent_responses": agent_responses
    })
    
    return final_answer, retrieved_contexts

if __name__ == "__main__":
    print("=== HỆ THỐNG DUAL-LAYER MULTI-AGENT ===")
    while True:
        q = input("\nBạn hỏi gì? (exit để thoát): ")
        if q.lower() == "exit": break
        ans = process_query(q)
        print(f"\n🤖 Bot: {ans}")