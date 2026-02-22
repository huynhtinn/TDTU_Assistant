import streamlit as st
import sys
import os
import time
import json
import sqlite3
import base64
from datetime import datetime

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, current_dir)

# Import core system
try:
    from main import process_query_with_context
except ImportError as e:
    st.error(f"❌ Không thể import hệ thống AI: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="TDTU AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium aesthetics with sidebar navigation
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar Navigation */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E8ECF2 0%, #F5F7FA 100%);
        border-right: 2px solid #D0D7E3;
        padding: 0;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 0;
    }
    
    /* Navigation buttons */
    .nav-button {
        width: 100%;
        padding: 1.5rem 0;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        background: transparent;
        color: #4A5568;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .nav-button:hover {
        background: rgba(0, 102, 204, 0.08);
        color: #0066CC;
    }
    
    .nav-button.active {
        background: rgba(0, 102, 204, 0.12);
        color: #0066CC;
        border-left: 4px solid #0066CC;
    }
    
    /* Override Streamlit default sidebar button styles for light sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: #4A5568 !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0, 102, 204, 0.08) !important;
        color: #0066CC !important;
        transform: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(0, 102, 204, 0.15) !important;
        color: #0066CC !important;
        font-weight: 700 !important;
        border-left: 3px solid #0066CC !important;
        border-radius: 0 8px 8px 0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #D0D7E3;
    }
    
    .nav-icon {
        font-size: 2rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0066CC 0%, #004C99 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 102, 204, 0.2);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-weight: 700;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Chat message containers */
    .user-message {
        background: linear-gradient(135deg, #0066CC 0%, #0052A3 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0;
        margin-left: auto;
        margin-right: 0;
        max-width: 70%;
        width: fit-content;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
        animation: slideInRight 0.3s ease-out;
        word-wrap: break-word;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #212529;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 1rem 0;
        margin-left: 0;
        margin-right: auto;
        max-width: 70%;
        width: fit-content;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #DC143C;
        animation: slideInLeft 0.3s ease-out;
        word-wrap: break-word;
    }
    
    /* Document card */
    .doc-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .doc-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        border-color: #0066CC;
    }
    
    .doc-title {
        font-weight: 600;
        color: #0066CC;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .doc-meta {
        color: #6c757d;
        font-size: 0.85rem;
    }
    
    /* Contact card */
    .contact-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #DC143C;
    }
    
    .contact-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0066CC;
        margin-bottom: 1rem;
    }
    
    .contact-item {
        margin: 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .contact-icon {
        font-size: 1.2rem;
        color: #DC143C;
    }
    
    .contact-text {
        color: #495057;
        font-size: 0.95rem;
    }
    
    .contact-link {
        color: #0066CC;
        text-decoration: none;
        font-weight: 500;
    }
    
    .contact-link:hover {
        text-decoration: underline;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066CC 0%, #004C99 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 102, 204, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #dee2e6;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: border-color 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0066CC;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'chatbot'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'total_response_time' not in st.session_state:
    st.session_state.total_response_time = 0.0
if 'viewing_pdf' not in st.session_state:
    st.session_state.viewing_pdf = None


# ======================
# SIDEBAR NAVIGATION
# ======================
with st.sidebar:
    # Logo chính thức của trường
    logo_path = os.path.join(project_root, '.streamlit', 'Logo ĐH Tôn Đức Thắng-TDT.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width='stretch')
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h2 style="color: #0066CC; margin: 0; font-size: 1.5rem;">🎓 TDTU</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 0.25rem 0 0.75rem 0;">
        <p style="color: #6c757d; font-size: 0.85rem; margin: 0;">AI Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation buttons with active state highlighting
    # Chatbot
    chatbot_type = "primary" if st.session_state.current_page == 'chatbot' else "secondary"
    if st.button("🤖\n\nChatbot", key="nav_chatbot", width='stretch', type=chatbot_type):
        st.session_state.current_page = 'chatbot'
        st.rerun()
    
    # Database
    database_type = "primary" if st.session_state.current_page == 'database' else "secondary"
    if st.button("📚\n\nCơ sở dữ liệu", key="nav_database", width='stretch', type=database_type):
        st.session_state.current_page = 'database'
        st.rerun()
    
    # Contact
    contact_type = "primary" if st.session_state.current_page == 'contact' else "secondary"
    if st.button("📞\n\nLiên hệ", key="nav_contact", width='stretch', type=contact_type):
        st.session_state.current_page = 'contact'
        st.rerun()
    
    st.markdown("---")
    

# ======================
# PAGE ROUTING
# ======================

def render_chatbot_page():
    """Chatbot page - Main chat interface"""
    st.markdown("""
    <div class="main-header">
        <h1>Chatbot AI</h1>
        <p>Trợ lý ảo thông minh - Hỏi đáp về TDTU</p>
    </div>
    """, unsafe_allow_html=True)
    

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 0.5rem; opacity: 0.8;">👤 Bạn</div>
                    <div>{message['content']}</div>
                    <div style="font-size: 0.75rem; opacity: 0.6; margin-top: 0.5rem; text-align: right;">{message.get('time', '')}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 0.5rem; opacity: 0.8;">🤖 TDTU Assistant</div>
                    <div>{message['content']}</div>
                    <div style="font-size: 0.75rem; opacity: 0.6; margin-top: 0.5rem; text-align: right;">{message.get('time', '')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show contexts if available
                if 'contexts' in message and message['contexts']:
                    with st.expander(f"📚 Nguồn tham khảo ({len(message['contexts'])} tài liệu)"):
                        for i, ctx in enumerate(message['contexts'], 1):
                            if hasattr(ctx, 'metadata'):
                                source = ctx.metadata.get('source', 'Unknown')
                                content = ctx.page_content[:200] + "..." if len(ctx.page_content) > 200 else ctx.page_content
                            else:
                                source = f"Context {i}"
                                content = str(ctx)[:200] + "..." if len(str(ctx)) > 200 else str(ctx)
                            
                            st.markdown(f"**📄 {source}**")
                            st.text(content)
                            st.markdown("---")
    
    # Welcome message if no messages
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6c757d;">
            <h3>👋 Xin chào! Tôi là TDTU AI Assistant</h3>
            <p>Tôi có thể giúp bạn với:</p>
            <ul style="list-style: none; padding: 0;">
                <li>📝 Thông tin sinh viên (điểm, rèn luyện, ngành học)</li>
                <li>💰 Học phí và công nợ</li>
                <li>🎓 Tuyển sinh và điểm chuẩn</li>
                <li>🏠 Ký túc xá và hoạt động sinh viên</li>
                <li>📚 Quy chế đào tạo</li>
            </ul>
            <p style="margin-top: 2rem;"><strong>Hãy đặt câu hỏi của bạn bên dưới!</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Input area - Using chat_input for better UX (supports Enter key)
    user_input = st.chat_input(
        "Nhập câu hỏi... (Nhấn Enter để gửi)",
        key="user_input"
    )
    
    # Process input
    if user_input:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'time': current_time
        })
        
        with st.spinner("🤔 Đang suy nghĩ..."):
            try:
                start_time = time.time()
                response, contexts = process_query_with_context(user_input)
                end_time = time.time()
                
                response_time = end_time - start_time
                st.session_state.query_count += 1
                st.session_state.total_response_time += response_time
                
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': response,
                    'contexts': contexts,
                    'time': datetime.now().strftime("%H:%M:%S")
                })
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': f"Xin lỗi, đã xảy ra lỗi: {str(e)}. Vui lòng thử lại.",
                    'time': datetime.now().strftime("%H:%M:%S")
                })
        
        st.rerun()

def render_database_page():
    """Database page - Browse regulations and documents"""
    st.markdown("""
    <div class="main-header">
        <h1>📚 Cơ sở dữ liệu</h1>
        <p>Tra cứu quy định, quy chế và tài liệu</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for viewing and uploading
    tab1, tab2 = st.tabs(["📖 Xem tài liệu", "📤 Tải lên PDF"])
    
    with tab1:
        # Search box
        search_term = st.text_input(
            "🔍 Tìm kiếm tài liệu", 
            placeholder="Nhập từ khóa: tuyển sinh, học phí, quy chế, đào tạo...",
            help="Tìm kiếm trong tên tài liệu và nội dung"
        )
        
        # Show PDF files section
        st.markdown("---")
        st.markdown("### 📚 Quy chế - Quy định (PDF)")
        
        pdf_dir = os.path.join(project_root, 'data', 'stdportal', 'downloads_pdf')
        if os.path.exists(pdf_dir):
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            
            # Filter PDF files by search term
            if search_term:
                search_lower = search_term.lower()
                filtered_pdf_files = [f for f in pdf_files if search_lower in f.lower()]
            else:
                filtered_pdf_files = pdf_files
            
            if filtered_pdf_files:
                st.markdown(f"**Tìm thấy: {len(filtered_pdf_files)} file PDF** (Tổng: {len(pdf_files)})")
                
                for pdf_file in filtered_pdf_files:
                    file_path = os.path.join(pdf_dir, pdf_file)
                    file_size = os.path.getsize(file_path)
                    file_size_mb = file_size / (1024 * 1024)
                    is_viewing = st.session_state.viewing_pdf == pdf_file
                    
                    col1, col2, col3 = st.columns([5, 1, 1])
                    with col1:
                        st.markdown(f"📄 **{pdf_file}** ({file_size_mb:.2f} MB)")
                    with col2:
                        view_label = "✖️ Đóng" if is_viewing else "👁️ Xem"
                        if st.button(
                            view_label,
                            key=f"view_{pdf_file}",
                            width='stretch'
                        ):
                            if is_viewing:
                                st.session_state.viewing_pdf = None
                            else:
                                st.session_state.viewing_pdf = pdf_file
                            st.rerun()
                    with col3:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Tải",
                                data=f,
                                file_name=pdf_file,
                                mime="application/pdf",
                                key=f"download_{pdf_file}",
                                width='stretch'
                            )
                    
                    # Inline PDF viewer
                    if is_viewing:
                        with open(file_path, 'rb') as f:
                            pdf_bytes = f.read()
                        b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f"""
                        <div style="
                            border: 2px solid #D0D7E3;
                            border-radius: 12px;
                            overflow: hidden;
                            margin: 0.5rem 0 1.5rem 0;
                            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
                        ">
                            <div style="
                                background: #E8ECF2;
                                padding: 0.5rem 1rem;
                                font-weight: 600;
                                color: #0066CC;
                                font-size: 0.85rem;
                                border-bottom: 1px solid #D0D7E3;
                            ">📄 {pdf_file}</div>
                            <iframe
                                src="data:application/pdf;base64,{b64_pdf}"
                                width="100%"
                                height="750px"
                                style="display: block; border: none;"
                                type="application/pdf"
                            ></iframe>
                        </div>
                        """
                        st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                if search_term:
                    st.info(f"Không tìm thấy file PDF nào phù hợp với '{search_term}'.")
                else:
                    st.info("Chưa có tài liệu PDF nào.")
        else:
            st.info("Thư mục PDF chưa tồn tại.")
    
    with tab2:
        st.markdown("### 📤 Tải lên tài liệu PDF")
        st.info("💡 Tải lên các tài liệu PDF của bạn để thêm vào cơ sở dữ liệu")
        
        # Upload widget
        uploaded_files = st.file_uploader(
            "Chọn file PDF",
            type=['pdf'],
            accept_multiple_files=True,
            help="Bạn có thể chọn nhiều file PDF cùng lúc"
        )
        
        if uploaded_files:
            st.markdown(f"**Đã chọn {len(uploaded_files)} file:**")
            
            # Show selected files
            for uploaded_file in uploaded_files:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.markdown(f"- 📄 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
            
            # Save button
            if st.button("💾 Lưu tài liệu", type="primary", width='stretch'):
                pdf_dir = os.path.join(project_root, 'data', 'stdportal', 'downloads_pdf')
                
                # Create directory if it doesn't exist
                if not os.path.exists(pdf_dir):
                    os.makedirs(pdf_dir)
                
                success_count = 0
                error_count = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Update progress
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Đang lưu {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}...")
                        
                        # Save file
                        file_path = os.path.join(pdf_dir, uploaded_file.name)
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        success_count += 1
                        
                    except Exception as e:
                        error_count += 1
                        st.error(f"❌ Lỗi khi lưu {uploaded_file.name}: {str(e)}")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show results
                if success_count > 0:
                    st.success(f"✅ Đã lưu thành công {success_count} tài liệu vào thư mục `data/stdportal/downloads_pdf`")
                
                if error_count > 0:
                    st.warning(f"⚠️ Có {error_count} file gặp lỗi khi lưu")
                
                st.balloons()
    
    st.markdown("---")


def render_contact_page():
    """Contact page - Department contact information"""
    st.markdown("""
    <div class="main-header">
        <h1>📞 Thông tin liên hệ</h1>
        <p>Liên hệ các phòng ban - Đại học Tôn Đức Thắng</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact information for major departments
    contacts = [
        {
            "name": "Ban Tuyển sinh",
            "icon": "🎓",
            "phone": ["19002024", "(028) 37 755 051", "(028) 37 755 052"],
            "email": "tuvantuyensinh@tdtu.edu.vn",
            "website": "https://admission.tdtu.edu.vn"
        },
        {
            "name": "Phòng Đại học",
            "icon": "📚",
            "phone": ["(028) 37755052"],
            "email": "phongdaihoc@tdtu.edu.vn",
            "website": "https://undergrad.tdtu.edu.vn"
        },
        {
            "name": "Phòng Công tác Sinh viên",
            "icon": "👥",
            "phone": ["(028) 37755054"],
            "email": "phongctct-hssv@tdtu.edu.vn",
            "website": "https://student.tdtu.edu.vn"
        },
        {
            "name": "Phòng Kế hoạch Tài chính",
            "icon": "💰",
            "phone": ["(028) 37755070"],
            "email": "phongkhtc@tdtu.edu.vn",
            "website": None
        },
        {
            "name": "Phòng Sau đại học",
            "icon": "🎯",
            "phone": ["(028) 3775-5059"],
            "email": "sdh@tdtu.edu.vn",
            "website": "http://grad.tdtu.edu.vn"
        },
        {
            "name": "Ký túc xá (KTX)",
            "icon": "🏠",
            "phone": ["(028) 37 760 652"],
            "email": "ktx@tdtu.edu.vn",
            "website": "https://dormitory.tdtu.edu.vn"
        },
        {
            "name": "Trung tâm Tin học",
            "icon": "💻",
            "phone": ["(028) 37761046", "0908 110 456"],
            "email": "trungtamtinhoc@tdtu.edu.vn",
            "website": "http://cait.tdtu.edu.vn"
        },
        {
            "name": "Trung tâm Tư vấn học đường",
            "icon": "🧑‍🏫",
            "phone": ["(028) 22477215"],
            "email": "tuvanhocduong@tdtu.edu.vn",
            "website": "https://tuvanhocduong.tdtu.edu.vn"
        }
    ]
    
    # Display in 2 columns
    col1, col2 = st.columns(2)
    
    for i, contact in enumerate(contacts):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="contact-card">
                <div class="contact-title">{contact['icon']} {contact['name']}</div>
            """, unsafe_allow_html=True)
            
            # Phone numbers
            for phone in contact['phone']:
                st.markdown(f"""
                <div class="contact-item">
                    <span class="contact-icon">📞</span>
                    <span class="contact-text">{phone}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Email
            st.markdown(f"""
            <div class="contact-item">
                <span class="contact-icon">📧</span>
                <a href="mailto:{contact['email']}" class="contact-link">{contact['email']}</a>
            </div>
            """, unsafe_allow_html=True)
            
            # Website
            if contact['website']:
                st.markdown(f"""
                <div class="contact-item">
                    <span class="contact-icon">🌐</span>
                    <a href="{contact['website']}" target="_blank" class="contact-link">Website</a>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # General information
    st.markdown("---")
    st.markdown("""
    ### 📍 Địa chỉ
    **Đại học Tôn Đức Thắng**  
    Số 19 Nguyễn Hữu Thọ, Phường Tân Phong, Quận 7, TP. Hồ Chí Minh
    
    **Giờ làm việc:**  
    Thứ Hai - Thứ Bảy: 7h30 - 11h30, 13h00 - 17h00
    """)

# Route to appropriate page
if st.session_state.current_page == 'chatbot':
    render_chatbot_page()
elif st.session_state.current_page == 'database':
    render_database_page()

elif st.session_state.current_page == 'contact':
    render_contact_page()

# Footer
st.markdown("---")
st.markdown("""

""", unsafe_allow_html=True)
