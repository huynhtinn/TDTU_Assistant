import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# CLASS CHA
class BaseAgent:
    def __init__(self, name, vector_db_path, role_description):
        self.name = name
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("LLM_MODEL")
        
        print(f"--- Đang khởi tạo Agent: {self.name} ---")
        
        # 1. Embedding
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # 2. Vector DB (Mỗi Agent có thể dùng chung DB nhưng lọc Metadata khác nhau)
        if not os.path.exists(vector_db_path):
            raise FileNotFoundError(f"Không tìm thấy DB tại {vector_db_path}")
            
        self.db = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        
        self.retriever = self.db.as_retriever(search_kwargs={"k": 6})
        
        # 3. LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.3
        )
        
        self.role_description = role_description
        self.prompt = ChatPromptTemplate.from_template(f"""
        Bạn là {{agent_name}}, một trợ lý ảo chuyên trách của Đại học Tôn Đức Thắng.
        {{role_description}}
        
        Nhiệm vụ: Trả lời câu hỏi dựa trên thông tin được cung cấp (Context).
        
        Yêu cầu:
        - Trả lời chính xác, ngắn gọn, đi thẳng vào vấn đề.
        - Nếu thông tin không có trong Context, hãy nói: "Xin lỗi, vấn đề này nằm ngoài dữ liệu của tôi."
        - Luôn trích dẫn nguồn (nếu có).

        --- THÔNG TIN TRA CỨU ĐƯỢC ---
        {{context}}
        -------------------------------

        Câu hỏi: {{question}}
        Câu trả lời:
        """)

    def format_docs(self, docs):
        return "\n\n".join(f"[Nguồn: {doc.metadata.get('source')}] \nNội dung: {doc.page_content}" for doc in docs)

    def process(self, question):
        # 1. Tìm kiếm
        docs = self.retriever.invoke(question)
        
        # 2. Chạy Chain
        chain = (
            {"context": lambda x: self.format_docs(docs), "question": RunnablePassthrough(), 
             "agent_name": lambda x: self.name, "role_description": lambda x: self.role_description}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(question), docs

# CÁC AGENT CON

class AdmissionAgent(BaseAgent):
    def __init__(self, vector_db_path):
        super().__init__(
            name="Ban Tư Vấn Tuyển Sinh",
            vector_db_path=vector_db_path,
            role_description="Bạn chuyên giải đáp về tuyển sinh, điểm chuẩn, phương thức xét tuyển và ngành nghề đào tạo."
        )

class AcademicAgent(BaseAgent):
    def __init__(self, vector_db_path):
        super().__init__(
            name="Phòng Đại Học (Giáo Vụ)",
            vector_db_path=vector_db_path,
            role_description="Bạn chuyên giải đáp về đăng ký môn học, lịch thi, quy chế đào tạo, tốt nghiệp và chứng chỉ."
        )

class StudentLifeAgent(BaseAgent):
    def __init__(self, vector_db_path):
        super().__init__(
            name="Phòng Công Tác Sinh Viên",
            vector_db_path=vector_db_path,
            role_description="Bạn chuyên giải đáp về học bổng, rèn luyện, bảo hiểm y tế, ký túc xá và các hoạt động phong trào."
        )

class GeneralAgent(BaseAgent):
    def __init__(self, vector_db_path):
        super().__init__(
            name="Trợ Lý Tổng Hợp",
            vector_db_path=vector_db_path,
            role_description="Bạn giải đáp các thông tin chung về trường như địa chỉ, sơ đồ, các phòng ban chức năng."
        )