import os
import re
import shutil

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from embeddings import E5Embeddings, get_shared_embedding_model

import chromadb
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SQL_DB_PATH = os.path.join(PROCESSED_DIR, 'student_data.db')


print("Đang tải Embedding model (intfloat/multilingual-e5-base)...")
_SHARED_EMBEDDING_MODEL = get_shared_embedding_model()

def _handle_error(error) -> str:
    error_str = str(error)
    
    if "Final Answer:" in error_str:
        last_final = error_str.rfind("Final Answer:")
        if last_final != -1:
            answer = error_str[last_final + len("Final Answer:"):].strip()
            if "Error encountered:" in answer:
                answer = answer.split("Error encountered:")[0].strip()
            return answer
    
    if "[(" in error_str and ")]" in error_str:
        return f"Found data: {error_str}"
    
    if "Could not parse LLM output" in error_str:
        content = error_str.replace("Could not parse LLM output: `", "").replace("`", "")
        if "For troubleshooting, visit:" in content:
            content = content.split("For troubleshooting, visit:")[0].strip()
            
        lines = content.split('\n')
        useful_lines = [l for l in lines if not l.strip().startswith(('Action:', 'Thought:', 'Action Input:', 'Observation:'))]
        return '\n'.join(useful_lines).strip()
    
    return f"Lỗi: {error_str[:200]}"

def create_rag_tool(vector_db_folder, tool_name, captured_docs=None):

    db_path = os.path.join(PROCESSED_DIR, vector_db_folder)
    embedding_model = _SHARED_EMBEDDING_MODEL
    _SCORE_THRESHOLD = 0.78

    def retrieve_func(query):
        if not os.path.exists(db_path):
            if captured_docs is not None:
                captured_docs.clear()
            return "Chưa có dữ liệu trong nhóm này."

        try:
            client = chromadb.PersistentClient(path=db_path)
            db = Chroma(
                client=client,
                embedding_function=embedding_model,
                collection_name="langchain",
            )
        except Exception as e:
            print(f"Lỗi Chroma ({vector_db_folder}): {e}")
            return "Lỗi DB."

        results_with_scores = db.similarity_search_with_relevance_scores(query, k=5)

        print(f"\n{'='*60}")
        print(f"[RAG] Query: \"{query[:80]}...\"" if len(query) > 80 else f"[RAG] Query: \"{query}\"")
        print(f"   Threshold: {_SCORE_THRESHOLD}")
        if results_with_scores:
            for i, (doc, score) in enumerate(results_with_scores):
                status = "✅" if score >= _SCORE_THRESHOLD else "❌"
                snippet = doc.page_content[:100].replace('\n', ' ')
                print(f"   {status} Doc {i+1}: score={score:.4f} | \"{snippet}...\"")
        else:
            print("   Không có kết quả nào từ vector DB.")
        print(f"{'='*60}\n")

        filtered = [(doc, score) for doc, score in results_with_scores if score >= _SCORE_THRESHOLD]

        if filtered:
            docs = [doc for doc, _ in filtered[:3]] 
            if captured_docs is not None:
                captured_docs.clear()
                captured_docs.extend(docs)
            return "\n\n".join([d.page_content for d in docs])
        else:
            if results_with_scores:
                top_doc, top_score = results_with_scores[0]
                print(f"   Fallback: lấy top-1 (score={top_score:.4f}) — không hiển thị nguồn")
                if captured_docs is not None:
                    captured_docs.clear()
                return top_doc.page_content
            else:
                if captured_docs is not None:
                    captured_docs.clear()
                return "Không tìm thấy thông tin liên quan trong dữ liệu TDTU."

    return Tool(
        name=tool_name,
        func=retrieve_func,
        description="Tra cứu thông tin về: quy chế, quy định, học bổng, kỷ luật, điểm rèn luyện, nội quy, THÔNG TIN LIÊN HỆ các khoa/phòng ban, email, số điện thoại. LUÔN dùng tool này trước khi query SQL."
    )

class HybridAgent:
    def __init__(self, name, vector_db_folder, role_instruction):
        self.name = name
        self.role_instruction = role_instruction
        print(f"--- [INIT] Agent: {self.name} ---")
        
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL"),
            api_key=os.getenv("API_KEY"),
            temperature=0
        )

        self._rag_docs = []

        self.tools = []
        if os.path.exists(SQL_DB_PATH):
            try:
                db = SQLDatabase.from_uri(f"sqlite:///{SQL_DB_PATH}")
                all_sql_tools = SQLDatabaseToolkit(db=db, llm=self.llm).get_tools()
                for tool in all_sql_tools:
                    if tool.name == "sql_db_query":
                        tool.description = "Use this to execute SQL queries. Output is raw data."
                        tool.return_direct = True
                    
                    if tool.name in ["sql_db_query", "sql_db_schema"]:
                        self.tools.append(tool)
            except Exception as e:
                print(f"[{self.name}] Không thể khởi tạo SQL tools: {e}")
        
        self.tools.append(create_rag_tool(vector_db_folder, "search_regulations", self._rag_docs))

        template = """
        You are a specialized AI agent for Ton Duc Thang University (TDTU) data system.
        Your role: {role_instruction}
        Use the provided tools to answer questions accurately.

        === TOOLS ===
        search_regulations: University regulations, policies, scholarships, contact info, emails, phone numbers.
        sql_db_query: Personal student data (GPA, credits, training points, major, failed subjects).
        sql_db_schema: Check DB schema if needed (use sparingly).

        === DATABASE SCHEMA ===
        Table sinh_vien: mssv(TEXT), ho_ten(TEXT), nganh_hoc(TEXT), diem_tb_tich_luy(REAL 0-10), diem_ren_luyen(INT 0-100), so_tin_chi_tich_luy(INT), no_mon(INT 0=không nợ 1=có nợ)
        Table quy_dinh_xep_loai: loai_tot_nghiep(TEXT), min_gpa(REAL), max_gpa(REAL)

        === SQL RULES ===
        - Use EXACT column names, LIKE for text: WHERE ho_ten LIKE '%Name%'
        - Plain SQL, no markdown. Always WHERE clause. LIMIT 10.
        - NEVER guess column names or omit WHERE.

        === CRITICAL: NO HALLUCINATION ===
        - ONLY return information that ACTUALLY appears in tool Observation.
        - DO NOT invent numbers, thresholds, percentages, or classification tables.
        - If the Observation does not contain specific data (e.g. GPA thresholds), say "Thông tin chi tiết không có trong dữ liệu" instead of guessing.
        - Return raw data from tools as-is. Let the synthesizer format it.

        === FORMAT ===
        Thought: [reasoning]
        Action: [tool_name only]
        Action Input: [query]
        Observation: [result]
        Final Answer: [answer or raw data]

        Rules:
        - Return raw tool data as-is, let synthesizer format it.
        - NEVER add details not found in Observation.
        - If empty []: "Không tìm thấy thông tin."
        - If error: "Lỗi truy vấn."
        - Max 1-2 tool calls. Stop after getting data.

        Tools: {tools}
        Tool Names: {tool_names}

        Question: {input}
        Thought: {agent_scratchpad}
        """
        
        react_prompt = PromptTemplate.from_template(template).partial(
            role_instruction=role_instruction
        )
        agent = create_react_agent(self.llm, self.tools, react_prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=False,
            handle_parsing_errors=_handle_error,
            max_iterations=2,         
            return_intermediate_steps=True
        )

    def answer(self, query):
        """Trả về response string"""
        response, _ = self.answer_with_context(query)
        return response

    def _build_structured_contexts(self) -> list:
        structured_contexts = []
        for doc in self._rag_docs:
            meta = doc.metadata if hasattr(doc, 'metadata') else {}
            _raw = meta.get("page_title") or meta.get("title") or ""
            if re.match(r'^page_\d+\.(png|jpg|jpeg|pdf)$', _raw, re.IGNORECASE):
                _raw = ""
            page_title = _raw or meta.get("source", "")
            source_url = meta.get("source", "")
            if not source_url.startswith("http"):
                source_url = ""
            page_num = meta.get("page", None)
            if page_num is None:
                _fname = meta.get("title", "") or meta.get("file_name", "")
                _pm = re.match(r'^page_(\d+)\.(png|jpe?g|pdf)$', str(_fname), re.IGNORECASE)
                if _pm:
                    page_num = int(_pm.group(1))
            structured_contexts.append({
                "content":    doc.page_content if hasattr(doc, 'page_content') else str(doc),
                "source":     source_url,
                "page_title": page_title,
                "page":       page_num,
            })
        return structured_contexts

    def answer_with_context(self, query):
        structured_contexts = []
        try:
            self._rag_docs.clear()  
            clean_q = query.strip().replace("`", "")

            result = self.agent_executor.invoke({"input": clean_q})
            
            output = result.get("output", "")
            if "For troubleshooting, visit:" in output:
                output = output.split("For troubleshooting, visit:")[0].strip()
            
            steps = result.get("intermediate_steps", [])
            
            if self._rag_docs:
                structured_contexts = self._build_structured_contexts()
            else:
                for action, observation in steps:
                    obs_str = str(observation)
                    if observation and len(obs_str) > 20:
                        if obs_str.startswith("Lỗi:") or "Could not parse LLM output" in obs_str or "For troubleshooting, visit:" in obs_str:
                            continue
                        structured_contexts.append({
                            "content":    obs_str,
                            "source":     "",
                            "page_title": "",
                        })
            
            if not output or "Agent stopped" in output or "iteration limit" in output:
                for action, observation in reversed(steps):
                    obs_str = str(observation)
                    if observation and len(obs_str) > 50:
                        if "For troubleshooting, visit:" in obs_str or "Could not parse LLM output" in obs_str:
                            continue
                        if obs_str.startswith("Lỗi:"):
                            continue
                        return obs_str, structured_contexts
            
            return (output if output else "Không tìm thấy thông tin."), structured_contexts
        except Exception as e:
            return f"Agent Error: {str(e)}", structured_contexts
        
def get_agents():
    return {
        "ACADEMIC":     HybridAgent("Phòng Đại Học",   "academic_db",      "Chuyên về quy chế đào tạo, điểm số, GPA, rèn luyện, tín chỉ."),
        "FINANCIAL":    HybridAgent("Phòng Tài Chính",  "financial_db",     "Chuyên về học phí, học bổng, khen thưởng, công nợ."),
        "ADMISSION":    HybridAgent("Ban Tuyển Sinh",   "admission_db",     "Chuyên về tuyển sinh, điểm chuẩn, thủ tục nhập học."),
        "STUDENT_LIFE": HybridAgent("Phòng CTSV",       "student_life_db",  "Chuyên về ký túc xá, bảo hiểm, rèn luyện, câu lạc bộ."),
        "GENERAL":      HybridAgent("Trợ Lý",           "general_db",       "Thông tin chung về TDTU, liên hệ, địa chỉ các phòng ban.")
    }