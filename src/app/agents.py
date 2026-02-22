import os
import shutil

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
# --- DÙNG GROQ ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
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

def _handle_error(error) -> str:
    error_str = str(error)
    
    # Trường hợp 1: LLM trả về cả Action và Final Answer cùng lúc
    # => Lấy phần Final Answer
    if "Final Answer:" in error_str:
        # Tìm vị trí Final Answer cuối cùng
        last_final = error_str.rfind("Final Answer:")
        if last_final != -1:
            answer = error_str[last_final + len("Final Answer:"):].strip()
            # Cắt bỏ phần lỗi phía sau nếu có
            if "Error encountered:" in answer:
                answer = answer.split("Error encountered:")[0].strip()
            return answer
    
    # Trường hợp 2: Có dữ liệu SQL trong lỗi
    if "[(" in error_str and ")]" in error_str:
        return f"Found data: {error_str}"
    
    # Trường hợp 3: Lỗi parse output nhưng có nội dung hữu ích
    if "Could not parse LLM output" in error_str:
        content = error_str.replace("Could not parse LLM output: `", "").replace("`", "")
        # Loại bỏ các dòng Action/Thought nếu có
        lines = content.split('\n')
        useful_lines = [l for l in lines if not l.strip().startswith(('Action:', 'Thought:', 'Action Input:', 'Observation:'))]
        return '\n'.join(useful_lines).strip()
    
    return f"Lỗi: {error_str[:200]}"

def create_rag_tool(vector_db_folder, tool_name):
    db_path = os.path.join(PROCESSED_DIR, vector_db_folder)
    
    # Sử dụng cùng embedding model với regulations DB
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    if not os.path.exists(db_path):
        return Tool(name=tool_name, func=lambda x: "Chưa có dữ liệu.", description="Tra cứu quy chế.")

    try:
        # Tạo Client thủ công
        client = chromadb.PersistentClient(path=db_path)
        
        db = Chroma(
            client=client,
            embedding_function=embedding_model,
            collection_name="langchain"
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})

        def retrieve_func(query):
            docs = retriever.invoke(query)
            if not docs: return "Không tìm thấy thông tin."
            return "\n\n".join([d.page_content for d in docs])

        return Tool(
            name=tool_name, 
            func=retrieve_func, 
            description="Tra cứu thông tin về: quy chế, quy định, học bổng, kỷ luật, điểm rèn luyện, nội quy, THÔNG TIN LIÊN HỆ các khoa/phòng ban, email, số điện thoại. LUÔN dùng tool này trước khi query SQL."
        )
    except Exception as e:
        print(f"Lỗi Chroma: {e}")
        return Tool(name=tool_name, func=lambda x: "Lỗi DB.", description="Lỗi.")

class HybridAgent:
    def __init__(self, name, vector_db_folder, role_instruction):
        self.name = name
        print(f"--- [INIT] Agent: {self.name} ---")
        
        # --- GROQ CONFIG ---
        self.llm = ChatGroq(
            model=os.getenv("LLM_MODEL"),
            api_key=os.getenv("API_KEY"),
            temperature=0
        )

        self.tools = []
        if os.path.exists(SQL_DB_PATH):
            try:
                db = SQLDatabase.from_uri(f"sqlite:///{SQL_DB_PATH}")
                all_sql_tools = SQLDatabaseToolkit(db=db, llm=self.llm).get_tools()
                for tool in all_sql_tools:
                    if tool.name == "sql_db_query":
                        tool.description = "Use this to execute SQL queries. Output is raw data."
                        tool.return_direct = True
                    
                    # Chỉ lấy query và schema
                    if tool.name in ["sql_db_query", "sql_db_schema"]:
                        self.tools.append(tool)
            except: pass
        
        self.tools.append(create_rag_tool(vector_db_folder, "search_regulations"))

        template = """
        You are a specialized AI agent for TDTU university data system.
        Use the provided tools to answer questions accurately and efficiently.

        === TOOL SELECTION GUIDE ===

        **search_regulations**
        USE FOR:
        • University regulations, policies, guidelines
        • Academic rules, training points, discipline
        • Scholarships, financial aid information
        • Contact information (emails, phones, addresses)
        • Department and faculty information

        Examples:
        - "Quy định điểm rèn luyện" → search_regulations
        - "Email phòng đại học" → search_regulations
        - "Học bổng khuyến khích" → search_regulations

        **sql_db_query**
        USE FOR:
        • Personal student data (name, ID, major)
        • Academic records (GPA, credits, grades)
        • Student-specific information

        Examples:
        - "Điểm của Nguyễn Văn A" → sql_db_query
        - "GPA sinh viên 522001" → sql_db_query
        - "Ngành học của Lê B" → sql_db_query

        ⚠️ NEVER mix tools for one query. Choose the most appropriate tool.

        === DATABASE SCHEMA ===

        **Table: sinh_vien**
        ```
        mssv                    TEXT      -- Student ID (6-7 digits, e.g., "522001")
        ho_ten                  TEXT      -- Full name (Vietnamese, e.g., "Nguyễn Văn A")
        nganh_hoc              TEXT      -- Major (e.g., "Công nghệ thông tin")
        diem_tb_tich_luy       REAL      -- Cumulative GPA (0.0-4.0)
        diem_ren_luyen         INTEGER   -- Training points (0-100)
        so_tin_chi_tich_luy    INTEGER   -- Total credits earned
        no_mon                 TEXT      -- Failed subjects (comma-separated)
        ```

        **Table: quy_dinh_xep_loai**
        ```
        loai_tot_nghiep        TEXT      -- Graduation classification
        min_gpa                REAL      -- Minimum GPA
        max_gpa                REAL      -- Maximum GPA
        ```

        === SQL QUERY RULES ===

        **✅ MUST DO:**
        1. Use EXACT column names: `ho_ten`, `nganh_hoc`, `diem_tb_tich_luy`
        2. Use LIKE for text matching: `WHERE ho_ten LIKE '%Nguyễn%'`
        3. Write plain SQL (NO markdown backticks)
        4. Always include WHERE clause (never query entire table)
        5. Use LIMIT for safety (max 10 rows)

        **❌ NEVER DO:**
        1. Guess column names (e.g., "HoTen", "Họ_tên")
        2. Use `=` for name matching (always use LIKE)
        3. Wrap SQL in markdown: ~~```sql~~
        4. Query without WHERE clause
        5. Analyze results yourself (return raw data)

        **SQL Examples:**

        Good ✅:
        ```
        SELECT mssv, ho_ten, nganh_hoc, diem_tb_tich_luy 
        FROM sinh_vien 
        WHERE ho_ten LIKE '%Nguyễn Văn%' 
        LIMIT 5
        ```

        Bad ❌:
        ```sql  ← Don't use markdown
        SELECT * FROM sinh_vien  ← No WHERE clause
        WHERE HoTen = 'Nguyễn Văn A'  ← Wrong column name + wrong operator
        ```

        === DATA VALIDATION ===

        **Student ID (mssv)**
        - Format: 6-7 digits
        - Valid: "522001", "19520123"
        - Invalid: "abc123", "52"

        **GPA (diem_tb_tich_luy)**
        - Range: 0.0 - 4.0
        - If query asks for GPA > 4.0: Impossible, return empty

        **Training Points (diem_ren_luyen)**
        - Range: 0 - 100
        - Common: 70-90 (good students)

        **Name (ho_ten)**
        - Vietnamese names (2-4 words)
        - Always use LIKE '%partial_name%'

        === ERROR HANDLING ===

        **Empty Results ([], None, no data)**
        ```
        Final Answer: Không tìm thấy thông tin về [query subject] trong hệ thống.
        ```

        **SQL Error**
        ```
        Final Answer: Lỗi truy vấn dữ liệu. Vui lòng kiểm tra lại thông tin.
        ```

        **Multiple Results (>5 matches)**
        ```
        Final Answer: Tìm thấy nhiều kết quả (X sinh viên). Vui lòng cung cấp thêm thông tin (MSSV hoặc tên đầy đủ).
        ```

        **Ambiguous Query**
        ```
        Thought: Query is unclear. I need to make a reasonable assumption.
        Action: [choose most likely tool]
        ```

        === OUTPUT FORMAT (ReAct) ===

        ```
        Question: [user's question]
        Thought: [your reasoning about which tool to use and why]
        Action: [tool_name ONLY - no brackets, no extra text]
        Action Input: [exact query for the tool]
        Observation: [tool's output]
        ... (repeat if needed)
        Final Answer: [concise answer with data, or error message]
        ```

        **Critical Rules:**
        - Action MUST be tool name only: `sql_db_query` NOT `sql_db_query()`
        - Stop immediately after getting data from tools
        - Return raw data in Final Answer, don't format it
        - If data is `[('522...', 'Name', ...)]`, that's perfect - return as-is

        === COMPLETE EXAMPLES ===

        **Example 1: Student Info**
        ```
        Question: Thông tin sinh viên Lê Văn A
        Thought: This asks for specific student data. Use sql_db_query.
        Action: sql_db_query
        Action Input: SELECT mssv, ho_ten, nganh_hoc, diem_tb_tich_luy FROM sinh_vien WHERE ho_ten LIKE '%Lê Văn%' LIMIT 5
        Observation: [('522015', 'Lê Văn A', 'Kinh tế', 3.2)]
        Final Answer: [('522015', 'Lê Văn A', 'Kinh tế', 3.2)]
        ```

        **Example 2: Regulation**
        ```
        Question: Quy định về điểm rèn luyện
        Thought: This asks about regulations. Use search_regulations.
        Action: search_regulations
        Action Input: Quy định về điểm rèn luyện
        Observation: Điểm rèn luyện được đánh giá theo thang điểm 100...
        Final Answer: Điểm rèn luyện được đánh giá theo thang điểm 100...
        ```

        **Example 3: Contact Info**
        ```
        Question: Email phòng đại học
        Thought: Contact information should be in regulations/documents.
        Action: search_regulations
        Action Input: Email phòng đại học
        Observation: phongdaihoc@tdtu.edu.vn
        Final Answer: Email phòng đại học: phongdaihoc@tdtu.edu.vn
        ```

        **Example 4: No Results**
        ```
        Question: Thông tin sinh viên XYZ không tồn tại
        Thought: Search for student XYZ.
        Action: sql_db_query
        Action Input: SELECT * FROM sinh_vien WHERE ho_ten LIKE '%XYZ%' LIMIT 5
        Observation: []
        Final Answer: Không tìm thấy thông tin sinh viên XYZ trong hệ thống.
        ```

        **Example 5: Multiple Results**
        ```
        Question: Sinh viên Nguyễn
        Thought: "Nguyễn" is too generic, will return many results.
        Action: sql_db_query
        Action Input: SELECT mssv, ho_ten FROM sinh_vien WHERE ho_ten LIKE '%Nguyễn%' LIMIT 10
        Observation: [('522001', 'Nguyễn A'), ('522002', 'Nguyễn B'), ... (8 more)]
        Final Answer: Tìm thấy 10+ sinh viên có họ Nguyễn. Vui lòng cung cấp tên đầy đủ hoặc MSSV.
        ```

        Tools Available: {tools}
        Tool Names: {tool_names}

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        """
        
        agent = create_react_agent(self.llm, self.tools, PromptTemplate.from_template(template))
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=_handle_error,
            max_iterations=2,  # Tăng lên 2 để Agent có thể hoàn thành
            return_intermediate_steps=True  # Lấy cả intermediate steps
        )

    def answer(self, query):
        """Trả về response string"""
        response, _ = self.answer_with_context(query)
        return response

    def answer_with_context(self, query):
        """
        Trả về (response, contexts) để hỗ trợ RAGAS evaluation
        Returns: (str, list[str])
        """
        contexts = []
        try:
            clean_q = query.strip().replace("`", "")
            result = self.agent_executor.invoke({"input": clean_q})
            
            output = result.get("output", "")
            
            # Extract contexts từ intermediate_steps
            steps = result.get("intermediate_steps", [])
            for action, observation in steps:
                if observation and len(str(observation)) > 20:
                    # Nếu là RAG tool output, lưu làm context
                    contexts.append(str(observation))
            
            # Nếu output rỗng hoặc báo "Agent stopped", lấy từ intermediate_steps
            if not output or "Agent stopped" in output or "iteration limit" in output:
                if steps:
                    # Lấy observation từ step cuối cùng
                    for action, observation in reversed(steps):
                        if observation and len(str(observation)) > 50:
                            return str(observation), contexts
            
            return (output if output else "Không tìm thấy thông tin."), contexts
        except Exception as e:
            return f"Agent Error: {str(e)}", contexts
        
def get_agents():
    # Tất cả agents đều dùng general_db (chứa 243 docs quy chế)
    return {
        "ACADEMIC": HybridAgent("Phòng Đại Học", "general_db", "Chuyên về quy chế, điểm số, GPA, rèn luyện."),
        "FINANCIAL": HybridAgent("Phòng Tài Chính", "general_db", "Chuyên về học phí, học bổng, khen thưởng."),
        "ADMISSION": HybridAgent("Ban Tuyển Sinh", "general_db", "Chuyên về tuyển sinh đầu vào."),
        "STUDENT_LIFE": HybridAgent("Phòng CTSV", "general_db", "Chuyên về rèn luyện, nội quy, kỷ luật."),
        "GENERAL": HybridAgent("Trợ Lý", "general_db", "Thông tin chung về TDTU.")
    }