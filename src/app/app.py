import streamlit as st
import sys
import os
import time
import json
import sqlite3
import base64
from datetime import datetime
from auth import (
    init_db, login_user, register_user,
    create_conversation, get_conversations, update_conversation_title,
    rename_conversation, pin_conversation,
    save_message, load_messages, delete_conversation,
    save_feedback, get_feedbacks, update_feedback_reply, get_feedback_stats,
    get_my_feedbacks, mark_feedbacks_seen,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, current_dir)

try:
    from main import (
        process_query_with_context,
        process_query_streaming,
        process_query_compare,
        get_available_providers,
        clear_cache,
        PROVIDER_LABELS,
        PROVIDER_META,
    )
except ImportError as e:
    st.error(f"Không thể import hệ thống AI: {str(e)}")
    st.stop()

try:
    from doc_manager import (
        DB_LABELS, STORES,
        list_sources, add_raw_text, add_pdf_bytes, delete_source, get_db_stats,
    )
    _DOC_MANAGER_OK = True
except ImportError:
    _DOC_MANAGER_OK = False

st.set_page_config(
    page_title="TDTU AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    /* ── Typing indicator (Messenger-style) ─────────────────────── */
    .typing-bubble {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 14px 20px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 18px 18px 18px 4px;
        border-left: 4px solid #DC143C;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-top: 4px;
    }

    .typing-bubble span {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: #adb5bd;
        display: inline-block;
        animation: tdtu-bounce 1.3s infinite ease-in-out;
    }

    .typing-bubble span:nth-child(1) { animation-delay: 0.0s; }
    .typing-bubble span:nth-child(2) { animation-delay: 0.18s; }
    .typing-bubble span:nth-child(3) { animation-delay: 0.36s; }

    @keyframes tdtu-bounce {
        0%, 55%, 100% { transform: translateY(0);  background-color: #adb5bd; }
        27.5%          { transform: translateY(-9px); background-color: #DC143C; }
    }

    /* ── Streaming cursor ───────────────────────────────────── */
    .stream-cursor {
        display: inline-block;
        width: 2px;
        height: 1em;
        background: #DC143C;
        margin-left: 2px;
        vertical-align: text-bottom;
        animation: cursor-blink 0.6s step-end infinite;
    }

    @keyframes cursor-blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0; }
    }

    /* ── Model picker — inline pill above chat input ─────────── */
    .model-picker-bar {
        display: inline-flex;
        align-items: center;
        margin-bottom: 4px;
    }
    /* Popover trigger button */
    .model-picker-bar > div > div > button,
    .model-picker-bar > div > button {
        background: #f0f2f6 !important;
        border: 1px solid #d0d7e3 !important;
        border-radius: 20px !important;
        padding: 4px 14px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        box-shadow: none !important;
        transition: all 0.15s !important;
        white-space: nowrap !important;
    }
    .model-picker-bar > div > div > button:hover,
    .model-picker-bar > div > button:hover {
        background: #e2eaf5 !important;
        border-color: #0066CC !important;
        color: #0066CC !important;
        transform: none !important;
        box-shadow: none !important;
    }
    /* Popover panel */
    div[data-testid="stPopoverBody"] {
        background: #ffffff !important;
        border: 1px solid #d0d7e3 !important;
        border-radius: 14px !important;
        padding: 8px 6px !important;
        min-width: 240px !important;
        box-shadow: 0 6px 24px rgba(0,0,0,0.12) !important;
    }
    /* Each option row inside popover */
    div[data-testid="stPopoverBody"] .stButton > button {
        text-align: left !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        color: #212529 !important;
        font-size: 0.88rem !important;
        width: 100% !important;
    }
    div[data-testid="stPopoverBody"] .stButton > button:hover {
        background: rgba(0,102,204,0.06) !important;
        box-shadow: none !important;
        transform: none !important;
    }
    /* Old pill wrap kept for backwards compat */
    .model-pill-wrap > div > button {
        display: none !important;
    }

    /* ── Conversation history: title buttons fixed 1 line ───────────── */
    [data-testid="stSidebar"] .stButton > button {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        height: 2.4rem !important;
        min-height: 2.4rem !important;
        max-height: 2.4rem !important;
        display: flex !important;
        align-items: center !important;
    }
</style>
""", unsafe_allow_html=True)

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
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'selected_provider' not in st.session_state:
    st.session_state.selected_provider = 'groq_llama'
if 'pending_provider' not in st.session_state:
    st.session_state.pending_provider = 'groq_llama'
if 'compare_history' not in st.session_state:
    st.session_state.compare_history = []
if 'compare_pending' not in st.session_state:
    st.session_state.compare_pending = None
if 'compare_providers' not in st.session_state:
    st.session_state.compare_providers = []

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None
if 'conversation_list' not in st.session_state:
    st.session_state.conversation_list = []

if 'last_feedback_idx' not in st.session_state:
    st.session_state.last_feedback_idx = -1
if 'last_feedback_state' not in st.session_state:
    st.session_state.last_feedback_state = None  

init_db()

from dotenv import load_dotenv
load_dotenv()
_LECTURER_CODE = os.getenv("LECTURER_CODE", "TDTU@LECTURER2026")


with st.sidebar:
    logo_path = os.path.join(project_root, '.streamlit', 'Logo ĐH Tôn Đức Thắng-TDT.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=160)
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

    if st.session_state.logged_in:
        user = st.session_state.user_info
        st.markdown(f"""
        <div style="padding:0.5rem 0.75rem;background:rgba(0,102,204,0.08);
                    border-radius:8px;margin-bottom:0.75rem;">
            <div style="font-weight:700;color:#0066CC;font-size:0.88rem;">
                👤 {user['display_name']}
            </div>
            <div style="color:#6c757d;font-size:0.75rem;">@{user['username']}</div>
            <div style="margin-top:4px;">
                {"<span style='background:#1a7f37;color:#fff;font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:10px;'>Giảng viên</span>" if user.get('role') == 'lecturer' else "<span style='background:#0066CC;color:#fff;font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:10px;'>Sinh viên</span>"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Đoạn chat mới", key="new_conv_btn",
                     use_container_width=True, type="primary"):
            clear_cache()
            st.session_state.current_conversation_id = None
            st.session_state.messages = []
            st.session_state.current_page = 'chatbot'
            st.rerun()

        convs = st.session_state.conversation_list
        if convs:
            st.markdown(
                "<div style='font-size:0.72rem;color:#999;font-weight:700;"
                "letter-spacing:1px;text-transform:uppercase;margin:8px 4px 4px;'>"
                "Lịch sử</div>",
                unsafe_allow_html=True,
            )
            for conv in convs[:25]:
                is_active = (conv['id'] == st.session_state.current_conversation_id)
                is_pinned = conv.get('pinned', 0) == 1
                t = conv['title']
                pin_prefix = "📌 " if is_pinned else ""
                title_display = pin_prefix + ((t[:17] + '…') if len(t) > 17 else t)
                col_title, col_menu = st.columns([5, 1])
                with col_title:
                    if st.button(
                        title_display,
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        clear_cache()
                        st.session_state.current_conversation_id = conv['id']
                        st.session_state.messages = load_messages(conv['id'])
                        st.session_state.current_page = 'chatbot'
                        st.rerun()
                with col_menu:
                    with st.popover(" ", use_container_width=True):
                        new_title = st.text_input(
                            "Tên mới",
                            value=t,
                            key=f"rename_input_{conv['id']}",
                            placeholder="Nhập tên mới...",
                        )
                        if st.button(" Đổi tên", key=f"ren_{conv['id']}", use_container_width=True):
                            if new_title.strip():
                                rename_conversation(conv['id'], new_title.strip())
                                st.session_state.conversation_list = get_conversations(
                                    st.session_state.user_info['id']
                                )
                                st.rerun()
                        pin_label = " Bỏ ghim" if is_pinned else "📌 Ghim"
                        if st.button(pin_label, key=f"pin_{conv['id']}", use_container_width=True):
                            pin_conversation(conv['id'], 0 if is_pinned else 1)
                            st.session_state.conversation_list = get_conversations(
                                st.session_state.user_info['id']
                            )
                            st.rerun()
                        if st.button(" Xóa", key=f"del_conv_{conv['id']}", use_container_width=True):
                            delete_conversation(conv['id'])
                            if is_active:
                                clear_cache()
                                st.session_state.current_conversation_id = None
                                st.session_state.messages = []
                            st.session_state.conversation_list = get_conversations(
                                st.session_state.user_info['id']
                            )
                            st.rerun()

        st.markdown("---")

    chatbot_type = "primary" if st.session_state.current_page == 'chatbot' else "secondary"
    if st.button("Chatbot", key="nav_chatbot", use_container_width=True, type=chatbot_type):
        st.session_state.current_page = 'chatbot'
        st.rerun()

    database_type = "primary" if st.session_state.current_page == 'database' else "secondary"
    if st.button("Cơ sở dữ liệu", key="nav_database", use_container_width=True, type=database_type):
        st.session_state.current_page = 'database'
        st.rerun()

    compare_type = "primary" if st.session_state.current_page == 'compare' else "secondary"
    if st.button("So sánh LLM", key="nav_compare", use_container_width=True, type=compare_type):
        st.session_state.current_page = 'compare'
        st.rerun()

    contact_type = "primary" if st.session_state.current_page == 'contact' else "secondary"
    if st.button("Liên hệ", key="nav_contact", use_container_width=True, type=contact_type):
        st.session_state.current_page = 'contact'
        st.rerun()

    if st.session_state.get('logged_in') and (st.session_state.user_info or {}).get('role') == 'student':
        _my_fbs = get_my_feedbacks(st.session_state.user_info['id'])
        _unread_n = sum(1 for f in _my_fbs if f['status'] == 'resolved' and not f.get('student_seen'))
        _notif_label = f"Thông báo ({_unread_n})" if _unread_n else "Thông báo"
        notif_type = "primary" if st.session_state.current_page == 'notifications' else "secondary"
        if st.button(_notif_label, key="nav_notifications", use_container_width=True, type=notif_type):
            st.session_state.current_page = 'notifications'
            st.rerun()

    if st.session_state.get('logged_in') and (st.session_state.user_info or {}).get('role') == 'lecturer':
        _stats = get_feedback_stats()
        _pending_n = _stats.get('pending', 0)
        _inbox_label = f"Hộp thư ({_pending_n})" if _pending_n else "Hộp thư"
        inbox_type = "primary" if st.session_state.current_page == 'inbox' else "secondary"
        if st.button(_inbox_label, key="nav_inbox", use_container_width=True, type=inbox_type):
            st.session_state.current_page = 'inbox'
            st.rerun()

    st.markdown("---")

    if st.session_state.logged_in:
        if st.button("Đăng xuất", key="logout_btn", use_container_width=True):
            clear_cache()
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.session_state.current_conversation_id = None
            st.session_state.conversation_list = []
            st.session_state.messages = []
            st.rerun()


def _serialize_contexts(contexts):
    if not contexts:
        return None
    result = []
    for ctx in contexts:
        if isinstance(ctx, dict):
            result.append(ctx)
        elif hasattr(ctx, 'metadata'):
            result.append({
                "source":     ctx.metadata.get('source', ''),
                "page_title": ctx.metadata.get('page_title') or ctx.metadata.get('title', ''),
                "page":       ctx.metadata.get('page'),
                "content":    ctx.page_content if hasattr(ctx, 'page_content') else str(ctx),
            })
        else:
            result.append({"content": str(ctx)})
    return result


def render_login_page():
    """Login / Register full-page form."""
    st.markdown("""
    <div class="main-header">
        <h1>TDTU AI Assistant</h1>
        <p>Đăng nhập để sử dụng trợ lý ảo thông minh</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])

        with tab_login:
            with st.form("login_form"):
                username  = st.text_input("Tên đăng nhập", placeholder="Nhập tên đăng nhập")
                password  = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu")
                submitted = st.form_submit_button("Đăng nhập", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Vui lòng nhập đầy đủ thông tin.")
                else:
                    ok, user = login_user(username, password)
                    if ok:
                        st.session_state.logged_in        = True
                        st.session_state.user_info        = user
                        st.session_state.conversation_list = get_conversations(user['id'])
                        st.rerun()
                    else:
                        st.error("Tên đăng nhập hoặc mật khẩu không đúng.")

        with tab_register:
            with st.form("register_form"):
                reg_display   = st.text_input("Họ và tên", placeholder="Nguyễn Văn A")
                reg_username  = st.text_input("Tên đăng nhập", placeholder="nguyenvana")
                reg_password  = st.text_input("Mật khẩu", type="password", placeholder="Ít nhất 6 ký tự")
                reg_password2 = st.text_input("Xác nhận mật khẩu", type="password", placeholder="Nhập lại mật khẩu")
                reg_role      = st.selectbox(
                    "Vai trò",
                    options=["student", "lecturer"],
                    format_func=lambda r: "Sinh viên" if r == "student" else "Giảng viên / Viên chức",
                )
                reg_code = ""
                if reg_role == "lecturer":
                    reg_code = st.text_input(
                        "Mã xác nhận giảng viên",
                        type="password",
                        placeholder="Nhập mã xác nhận do nhà trường cung cấp",
                    )
                reg_submitted = st.form_submit_button("Đăng ký", use_container_width=True, type="primary")

            if reg_submitted:
                if not reg_display or not reg_username or not reg_password:
                    st.error("Vui lòng nhập đầy đủ thông tin.")
                elif reg_password != reg_password2:
                    st.error("Mật khẩu xác nhận không khớp.")
                elif len(reg_username.strip()) < 3:
                    st.error("Tên đăng nhập phải có ít nhất 3 ký tự.")
                elif len(reg_password) < 6:
                    st.error("Mật khẩu phải có ít nhất 6 ký tự.")
                elif reg_role == "lecturer" and reg_code != _LECTURER_CODE:
                    st.error("Mã xác nhận giảng viên không đúng.")
                else:
                    ok, msg = register_user(reg_username, reg_password, reg_display, role=reg_role)
                    if ok:
                        st.success(msg + " Vui lòng chuyển sang tab Đăng nhập.")
                    else:
                        st.error(msg)


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
    _is_student = (
        st.session_state.get('logged_in') and
        st.session_state.user_info.get('role') == 'student'
    )
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
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
                    <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 0.5rem; opacity: 0.8;">
                        TDTU Assistant
                        <span style="font-size:0.72rem;opacity:0.55;margin-left:6px;">
                            {PROVIDER_META.get(message.get('provider','groq_llama'), PROVIDER_META['groq_llama'])['icon']}
                            {PROVIDER_META.get(message.get('provider','groq_llama'), PROVIDER_META['groq_llama'])['name']}
                        </span>
                    </div>
                    <div>{message['content']}</div>
                    <div style="font-size: 0.75rem; opacity: 0.6; margin-top: 0.5rem; text-align: right;">{message.get('time', '')}</div>
                </div>
                """, unsafe_allow_html=True)

                if 'contexts' in message and message['contexts']:
                    with st.expander(f"Nguồn tham khảo ({len(message['contexts'])} tài liệu)"):
                        for i, ctx in enumerate(message['contexts'], 1):
                            if isinstance(ctx, dict):
                                source     = ctx.get("source", "")
                                page_title = ctx.get("page_title", "")
                                page_num   = ctx.get("page", None)
                                content    = ctx.get("content", "")
                            elif hasattr(ctx, 'metadata'):
                                meta       = ctx.metadata
                                source     = meta.get('source', '')
                                if not source.startswith('http'):
                                    source = ''
                                page_title = meta.get('page_title') or meta.get('title') or meta.get('source', '')
                                page_num   = meta.get('page', None)
                                content    = ctx.page_content if hasattr(ctx, 'page_content') else str(ctx)
                            else:
                                source     = ""
                                page_title = ""
                                page_num   = None
                                content    = str(ctx)

                            display_title = page_title if page_title else f"Tài liệu {i}"
                            page_badge = f" — trang {page_num}" if page_num is not None else ""
                            st.markdown(f"**{i}. {display_title}{page_badge}**")

                            if source:
                                st.markdown(f"[Xem nguồn ]({source})")

                            preview = content[:300] + "..." if len(content) > 300 else content
                            st.caption(preview)
                            st.markdown("---")

                if _is_student and idx == st.session_state.get('last_feedback_idx', -1):
                    fb_state = st.session_state.get('last_feedback_state')
                    _prev_msgs = st.session_state.messages[:idx]
                    _user_q = next(
                        (m['content'] for m in reversed(_prev_msgs) if m['role'] == 'user'),
                        message.get('content', '')
                    )
                    _bot_ans = message.get('content', '')
                    if fb_state == 'liked':
                        st.markdown(
                            "<div style='font-size:0.8rem;color:#1a7f37;margin:4px 0 12px 0;'>"
                            "Cảm ơn phản hồi tích cực của bạn!</div>",
                            unsafe_allow_html=True,
                        )
                    elif fb_state == 'forwarded':
                        st.markdown(
                            "<div style='font-size:0.8rem;color:#0066CC;margin:4px 0 12px 0;'>"
                            "Câu hỏi đã được gửi đến giảng viên. Bạn sẽ sớm nhận được giải đáp!</div>",
                            unsafe_allow_html=True,
                        )
                    elif fb_state == 'disliked':
                        st.markdown(
                            "<div style='font-size:0.82rem;color:#856404;background:#fff3cd;"
                            "border-radius:8px;padding:10px 14px;margin:6px 0 8px 0;'>"
                            "Bạn chưa hài lòng với câu trả lời này. "
                            "Bạn có muốn gửi câu hỏi đến giảng viên để được giải đáp trực tiếp không?</div>",
                            unsafe_allow_html=True,
                        )
                        note_val = st.text_area(
                            "Ghi chú thêm cho giảng viên (tuỳ chọn):",
                            key="forward_note_input",
                            height=72,
                            placeholder="Ví dụ: Tôi muốn hỏi thêm về điều kiện xét học bổng loại A...",
                        )
                        col_fwd, col_skip = st.columns([2, 1])
                        with col_fwd:
                            if st.button("Gửi đến giảng viên", key="btn_forward_q", type="primary"):
                                user = st.session_state.user_info
                                save_feedback(
                                    user_id=user['id'],
                                    username=user['username'],
                                    display_name=user['display_name'],
                                    question=_user_q,
                                    bot_answer=_bot_ans,
                                    satisfied=0,
                                    student_note=note_val.strip() or None,
                                )
                                st.session_state.last_feedback_state = 'forwarded'
                                st.rerun()
                        with col_skip:
                            if st.button("Bỏ qua", key="btn_skip_fwd"):
                                st.session_state.last_feedback_state = None
                                st.rerun()
                    else:
                        st.markdown(
                            "<div style='font-size:0.8rem;color:#555;margin:6px 0 4px 0;'>"
                            "Câu trả lời này có hữu ích với bạn không?</div>",
                            unsafe_allow_html=True,
                        )
                        col_like, col_dislike, _ = st.columns([1, 1, 4])
                        with col_like:
                            if st.button("Hài lòng", key=f"fb_like_{idx}"):
                                user = st.session_state.user_info
                                save_feedback(
                                    user_id=user['id'],
                                    username=user['username'],
                                    display_name=user['display_name'],
                                    question=_user_q,
                                    bot_answer=_bot_ans,
                                    satisfied=1,
                                )
                                st.session_state.last_feedback_state = 'liked'
                                st.rerun()
                        with col_dislike:
                            if st.button("Chưa hài lòng", key=f"fb_dislike_{idx}"):
                                st.session_state.last_feedback_state = 'disliked'
                                st.rerun()
    
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #6c757d;">
            <h3>Xin chào! Tôi là trợ lý công tác sinh viên của TDTU</h3>
            <p style="margin-top: 2rem;"><strong>Hãy đặt câu hỏi của bạn bên dưới!</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    available  = get_available_providers()
    cur_prov   = st.session_state.selected_provider
    if cur_prov not in available:
        cur_prov = available[0]
        st.session_state.selected_provider = cur_prov
    cur_meta   = PROVIDER_META.get(cur_prov, PROVIDER_META["groq_llama"])

    st.markdown('<div class="model-picker-bar">', unsafe_allow_html=True)
    with st.popover(f"{cur_meta['icon']} {cur_meta['name']} ▾"):
        st.markdown(
            """<p style='font-size:0.68rem;color:#999;font-weight:700;
                         letter-spacing:1.1px;text-transform:uppercase;
                         margin:2px 8px 10px;'>Mô hình</p>""",
            unsafe_allow_html=True,
        )
        for p in available:
            m      = PROVIDER_META.get(p, {})
            is_sel = (p == cur_prov)
            sel_bg = "background:rgba(0,102,204,0.08);border-left:3px solid #0066CC;" if is_sel else "border-left:3px solid transparent;"
            check  = "<span style='color:#0066CC;font-size:1rem;'>&#10003;</span>" if is_sel else ""
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                         {sel_bg}padding:9px 12px 5px;border-radius:10px;margin:2px 0;'>
                <div>
                    <div style='font-weight:700;font-size:0.88rem;color:#1a1a2e;'>
                        {m.get('icon','')} {m.get('name','')}
                    </div>
                    <div style='font-size:0.73rem;color:#6c757d;margin-top:2px;'>{m.get('description','')}</div>
                </div>
                {check}
            </div>""", unsafe_allow_html=True)
            if st.button(
                m.get("full_name", p),
                key=f"mpick_{p}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                st.session_state.selected_provider = p
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.chat_input(
        "Nhập câu hỏi... (Nhấn Enter để gửi)",
        key="user_input"
    )
    
    if user_input:
        now_time = datetime.now().strftime("%H:%M:%S")

        if st.session_state.get('logged_in') and st.session_state.current_conversation_id is None:
            title = user_input.strip()[:50]
            conv_id = create_conversation(st.session_state.user_info['id'], title)
            st.session_state.current_conversation_id = conv_id
            st.session_state.conversation_list = get_conversations(st.session_state.user_info['id'])

        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'time': now_time
        })

        if st.session_state.get('logged_in') and st.session_state.current_conversation_id:
            save_message(st.session_state.current_conversation_id, 'user', user_input, timestamp=now_time)
            st.session_state.conversation_list = get_conversations(st.session_state.user_info['id'])

        st.session_state.pending_question = user_input
        st.session_state.pending_provider = st.session_state.selected_provider
        st.rerun()  
    if st.session_state.pending_question:
        pending  = st.session_state.pending_question
        provider = st.session_state.get("pending_provider", st.session_state.selected_provider)
        st.session_state.pending_question = None  
        prov_meta = PROVIDER_META.get(provider, PROVIDER_META["groq_llama"])

        try:
            start_time = time.time()

            history_msgs = st.session_state.messages[:-1]
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in history_msgs
            ]

            typing_slot = st.empty()
            typing_slot.markdown(
                f"""<div style='margin:6px 0 12px 0;'>
                    <div style='font-size:0.78rem;color:#999;font-weight:500;
                                margin-bottom:6px;'>
                        TDTU Assistant
                        <span style='opacity:0.5;margin-left:5px;font-size:0.7rem;'>
                            {prov_meta['icon']} {prov_meta['name']}
                        </span>
                    </div>
                    <div class='typing-bubble'>
                        <span></span><span></span><span></span>
                    </div>
                </div>""",
                unsafe_allow_html=True
            )

            early_response, contexts, stream = process_query_streaming(pending, provider, chat_history=chat_history)

            if early_response is not None:
                typing_slot.empty()
                response = early_response
            else:
                response = ""
                _last_render = 0.0
                _last_len = 0
                for chunk in stream:
                    response += chunk
                    _now = time.time()
                    # Re-render tối đa 12 lần/giây hoặc mỗi 30 ký tự mới — giảm thời gian chờ hiển thị
                    if (_now - _last_render >= 0.08) or (len(response) - _last_len >= 30):
                        typing_slot.markdown(
                            f"""<div class='bot-message'>
                                <div style='font-weight:600;font-size:0.78rem;
                                            margin-bottom:6px;opacity:0.7;'>
                                    TDTU Assistant
                                    <span style='opacity:0.6;margin-left:5px;font-size:0.68rem;'>
                                        {prov_meta['icon']} {prov_meta['name']}
                                    </span>
                                </div>
                                {response}<span class='stream-cursor'></span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        _last_render = _now
                        _last_len = len(response)
                typing_slot.empty()

            end_time = time.time()
            st.session_state.query_count += 1
            st.session_state.total_response_time += end_time - start_time

            ai_time = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response,
                'contexts': contexts,
                'provider': provider,
                'time': ai_time
            })

            if st.session_state.get('logged_in') and st.session_state.user_info.get('role') == 'student':
                st.session_state.last_feedback_idx   = len(st.session_state.messages) - 1
                st.session_state.last_feedback_state = 'pending'

            if st.session_state.get('logged_in') and st.session_state.current_conversation_id:
                save_message(
                    st.session_state.current_conversation_id,
                    'assistant', response,
                    provider=provider,
                    contexts=_serialize_contexts(contexts),
                    timestamp=ai_time,
                )

        except Exception as e:
            st.error(f"Lỗi: {str(e)}")
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f"Xin lỗi, đã xảy ra lỗi: {str(e)}. Vui lòng thử lại.",
                'time': datetime.now().strftime("%H:%M:%S")
            })

        st.rerun()

def render_compare_page():
    """So sánh câu trả lời của nhiều LLM trên cùng câu hỏi."""
    st.markdown("""
    <div class="main-header">
        <h1>So sánh LLM</h1>
        <p>Chạy song song nhiều mô hình, so sánh câu trả lời</p>
    </div>
    """, unsafe_allow_html=True)

    available = get_available_providers()
    provider_options = {PROVIDER_LABELS.get(p, p): p for p in available}

    st.markdown("### Cấu hình")
    col_cfg1, col_cfg2 = st.columns([2, 1])
    with col_cfg1:
        selected_labels = st.multiselect(
            "Chọn mô hình muốn so sánh:",
            options=list(provider_options.keys()),
            default=list(provider_options.keys()),
            help="Chỉ các mô hình đã có API key mới xuất hiện ở đây.",
        )
    with col_cfg2:
        if st.button("Xóa lịch sử", key="clear_compare"):
            st.session_state.compare_history = []
            st.rerun()

    selected_providers = [provider_options[lbl] for lbl in selected_labels]

    st.markdown("---")

    for entry in st.session_state.compare_history:
        st.markdown(f"""
        <div class="user-message">
            <div style="font-weight:600;font-size:0.85rem;margin-bottom:0.5rem;">👤 Bạn</div>
            <div>{entry['question']}</div>
        </div>
        """, unsafe_allow_html=True)

        n = len(entry['results'])
        cols = st.columns(n)
        for col, (provider, data) in zip(cols, entry['results'].items()):
            with col:
                elapsed_str = f" ({data['elapsed']:.1f}s)" if data['elapsed'] > 0 else ""
                err_badge = " ❌" if data.get('error') else ""
                st.markdown(f"""
                <div class="bot-message" style="border-left:4px solid #0066CC;">
                    <div style="font-weight:700;font-size:0.85rem;margin-bottom:8px;
                                color:#0066CC;">
                        🤖 {data['label']}{elapsed_str}{err_badge}
                    </div>
                    <div>{data['response']}</div>
                </div>
                """, unsafe_allow_html=True)

                if data.get('contexts'):
                    with st.expander(f"Nguồn ({len(data['contexts'])} tài liệu)"):
                        for i, ctx in enumerate(data['contexts'][:3], 1):
                            if isinstance(ctx, dict):
                                title   = ctx.get('page_title', '') or f"Tài liệu {i}"
                                source  = ctx.get('source', '')
                                content = ctx.get('content', '')[:250]
                            else:
                                title, source, content = f"Tài liệu {i}", "", str(ctx)[:250]
                            st.markdown(f"**{i}. {title}**")
                            if source:
                                st.markdown(f"[Xem nguồn]({source})")
                            st.caption(content)
        st.markdown("---")

    user_input = st.chat_input(
        "Nhập câu hỏi để so sánh... (Nhấn Enter)",
        key="compare_input",
    )

    if user_input:
        if not selected_providers:
            st.warning("Vui lòng chọn ít nhất một mô hình.")
        else:
            st.session_state.compare_pending = user_input
            st.session_state.compare_providers = selected_providers
            st.rerun()

    if st.session_state.compare_pending:
        pending   = st.session_state.compare_pending
        providers = st.session_state.compare_providers
        st.session_state.compare_pending   = None
        st.session_state.compare_providers = []

        with st.spinner(f"Đang hỏi {len(providers)} mô hình cùng lúc..."):
            try:
                results = process_query_compare(pending, providers)
                st.session_state.compare_history.append({
                    "question": pending,
                    "results":  results,
                })
            except Exception as e:
                st.error(f"Lỗi: {e}")
        st.rerun()


def render_database_page():
    """Database page - Browse regulations and documents"""
    st.markdown("""
    <div class="main-header">
        <h1>Cơ sở dữ liệu</h1>
        <p>Tra cứu quy định, quy chế và tài liệu</p>
    </div>
    """, unsafe_allow_html=True)
    
    _is_lecturer = st.session_state.get('user_info', {}).get('role') == 'lecturer'
    _tabs = ["Xem tài liệu", "Quản lý kiến thức"] if _is_lecturer else ["Xem tài liệu"]
    _tab_results = st.tabs(_tabs)
    tab1 = _tab_results[0]
    tab2 = _tab_results[1] if _is_lecturer else None

    with tab1:
        search_term = st.text_input(
            "Tìm kiếm tài liệu", 
            placeholder="Nhập từ khóa: tuyển sinh, học phí, quy chế, đào tạo...",
            help="Tìm kiếm trong tên tài liệu và nội dung"
        )
        
        st.markdown("---")
        st.markdown("### Quy chế - Quy định (PDF)")
        
        pdf_dir = os.path.join(project_root, 'data', 'stdportal', 'downloads_pdf')
        if os.path.exists(pdf_dir):
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            
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
                        st.markdown(f"**{pdf_file}** ({file_size_mb:.2f} MB)")
                    with col2:
                        view_label = "Đóng" if is_viewing else "Xem"
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
                                label="⬇Tải",
                                data=f,
                                file_name=pdf_file,
                                mime="application/pdf",
                                key=f"download_{pdf_file}",
                                width='stretch'
                            )
                    
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
                            ">{pdf_file}</div>
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


    if tab2 is not None:
        with tab2:
            if not _DOC_MANAGER_OK:
                st.error("Module doc_manager không tải được.")
            else:
                sub_add, sub_del = st.tabs(["Thêm tài liệu", "Xóa tài liệu"])

                # Thêm tài liệu 
                with sub_add:
                    st.markdown("### Thêm tài liệu vào hệ thống")

                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        source_name = st.text_input(
                            "Tên tài liệu",
                            placeholder="Ví dụ: Quy định học bổng 2026",
                        )
                    with col_b:
                        target_db = st.selectbox(
                            "Nhóm kiến thức",
                            options=list(DB_LABELS.keys()),
                            format_func=lambda k: DB_LABELS[k],
                        )

                    input_type = st.radio(
                        "Loại nội dung",
                        ["Tải lên PDF", "Dán văn bản"],
                        horizontal=True,
                    )

                    if input_type == "Tải lên PDF":
                        uploaded = st.file_uploader("Chọn file PDF", type=["pdf"])
                        if uploaded and source_name and st.button("Thêm vào hệ thống", type="primary", key="add_pdf_btn"):
                            with st.spinner("Đang đọc PDF và nhúng embed..."):
                                try:
                                    n = add_pdf_bytes(uploaded.read(), source_name, target_db)
                                    st.success(f"Đã thêm {n} chunks từ **{uploaded.name}** vào {DB_LABELS[target_db]}")
                                except Exception as e:
                                    st.error(f"❌ Lỗi: {e}")
                    else:
                        pasted_text = st.text_area(
                            "Nội dung văn bản",
                            height=250,
                            placeholder="Dán nội dung quy định, thông báo... vào đây",
                        )
                        if pasted_text and source_name and st.button("Thêm vào hệ thống", type="primary", key="add_text_btn"):
                            with st.spinner("Đang nhúng embed..."):
                                try:
                                    n = add_raw_text(pasted_text, source_name, target_db)
                                    st.success(f"Đã thêm {n} chunks vào {DB_LABELS[target_db]}")
                                except Exception as e:
                                    st.error(f"Lỗi: {e}")

                with sub_del:
                    st.markdown("### Xóa tài liệu khỏi hệ thống")

                    stats = get_db_stats()
                    cols = st.columns(len(DB_LABELS))
                    for col, (k, s) in zip(cols, stats.items()):
                        with col:
                            st.metric(s["label"].split(" ", 1)[1], s["total_sources"], f"{s['total_chunks']} chunks")

                    st.markdown("---")

                    sel_db = st.selectbox(
                        "Chọn nhóm kiến thức",
                        options=list(DB_LABELS.keys()),
                        format_func=lambda k: DB_LABELS[k],
                        key="del_db_select",
                    )

                    sources = list_sources(sel_db)
                    if not sources:
                        st.info("Nhóm này chưa có tài liệu nào.")
                    else:
                        st.markdown(f"**{len(sources)} tài liệu** trong {DB_LABELS[sel_db]}:")
                        for i, doc in enumerate(sources):
                            col_title, col_info, col_del = st.columns([4, 1, 1])
                            with col_title:
                                display = doc['page_title'] or doc['source']
                                st.markdown(f"• {display[:80]}")
                            with col_info:
                                st.caption(f"{doc['chunks']} chunks")
                            with col_del:
                                if st.button(
                                    " Xóa",
                                    key=f"deldoc_{sel_db}_{i}",
                                    type="secondary",
                                ):
                                    n = delete_source(doc['source'], sel_db)
                                    st.success(f"Đã xóa {n} chunks của '{display[:50]}'")
                                    st.rerun()

    st.markdown("---")

def render_contact_page():
    """Contact page - Department contact information"""
    st.markdown("""
    <div class="main-header">
        <h1>Thông tin liên hệ</h1>
        <p>Liên hệ các phòng ban - Đại học Tôn Đức Thắng</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    st.markdown("---")
    st.markdown("""
    ### Địa chỉ
    **Đại học Tôn Đức Thắng**  
    Số 19 Nguyễn Hữu Thọ, Phường Tân Phong, Quận 7, TP. Hồ Chí Minh
    
    **Giờ làm việc:**  
    Thứ Hai - Thứ Bảy: 7h30 - 11h30, 13h00 - 17h00
    """)

def render_notifications_page():
    """Trang thông báo phản hồi từ giảng viên (dành cho sinh viên)"""
    user_info = st.session_state.get('user_info') or {}
    if user_info.get('role') != 'student':
        st.error("Trang này chỉ dành cho sinh viên.")
        return

    mark_feedbacks_seen(user_info['id'])

    st.markdown("""
    <div class="main-header">
        <h1>Thông báo</h1>
        <p>Phản hồi từ giảng viên cho các câu hỏi bạn đã gửi</p>
    </div>
    """, unsafe_allow_html=True)

    fbs = get_my_feedbacks(user_info['id'])
    if not fbs:
        st.info("Bạn chưa gửi câu hỏi nào đến giảng viên.")
        return

    for fb in fbs:
        is_resolved = fb['status'] == 'resolved'
        border_color = "#28a745" if is_resolved else "#ffc107"
        status_text = "Đã giải đáp" if is_resolved else "Đang chờ giải đáp"

        st.markdown(
            f"<div style='border:1.5px solid {border_color};border-radius:10px;"
            f"padding:16px 20px;margin-bottom:16px;background:#fff;'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:10px;'>"
            f"<span style='font-size:0.78rem;color:#888;'>"
            f"{fb['created_at'][:16] if fb['created_at'] else ''}</span>"
            f"<span style='font-size:0.82rem;font-weight:600;'>{status_text}</span>"
            f"</div>"
            f"<div style='margin-bottom:8px;'>"
            f"<b>Câu hỏi của bạn:</b><br>"
            f"<span style='color:#333;'>{fb['question']}</span>"
            f"</div>"
            + (
                f"<div style='color:#856404;background:#fff3cd;padding:8px 12px;"
                f"border-radius:6px;font-size:0.85rem;margin-bottom:8px;'>"
                f"<b>Ghi chú bạn đã gửi:</b> {fb['student_note']}</div>"
                if fb.get('student_note') else ""
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        if is_resolved and fb.get('lecturer_reply'):
            st.markdown(
                f"<div style='background:#d4edda;border:1px solid #c3e6cb;"
                f"border-radius:8px;padding:14px 18px;margin:-8px 0 16px 0;'>"
                f"<b>Phản hồi từ giảng viên:</b><br>"
                f"<span style='color:#155724;'>{fb['lecturer_reply']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif not is_resolved:
            st.markdown(
                "<div style='color:#856404;font-size:0.82rem;margin:-4px 0 16px 0;'>"
                "Giảng viên chưa phản hồi. Bạn sẽ thấy câu trả lời tại đây khi có.</div>",
                unsafe_allow_html=True,
            )

def render_inbox_page():
    """Hộp thư giảng viên — danh sách câu hỏi sinh viên gửi đến"""
    user_info = st.session_state.get('user_info') or {}
    if user_info.get('role') != 'lecturer':
        st.error("Bạn không có quyền truy cập trang này.")
        return

    st.markdown("""
    <div class="main-header">
        <h1>Hộp thư</h1>
        <p>Câu hỏi sinh viên chuyển đến để giải đáp</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_feedback_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng câu hỏi", stats.get('total', 0))
    c2.metric("Chờ giải đáp", stats.get('pending', 0))
    c3.metric("Đã giải đáp", stats.get('resolved', 0))

    st.markdown("---")

    tab_all, tab_pending, tab_resolved = st.tabs(["Tất cả", "Chờ giải đáp", "Đã giải đáp"])

    def _render_feedback_cards(feedbacks, key_prefix=''):
        if not feedbacks:
            st.info("Không có câu hỏi nào.")
            return
        for fb in feedbacks:
            with st.container():
                status_badge = (
                    "🟢 Đã giải đáp" if fb['status'] == 'resolved' else "🟡 Chờ giải đáp"
                )
                st.markdown(
                    f"<div style='border:1px solid #ddd;border-radius:10px;"
                    f"padding:16px 20px;margin-bottom:16px;background:#fff;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;margin-bottom:8px;'>"
                    f"<b>🎓 {fb['display_name']}</b> "
                    f"<span style='color:#888;font-size:0.8rem;'>@{fb['username']} — "
                    f"{fb['created_at'][:16] if fb['created_at'] else ''}</span>"
                    f"<span style='font-size:0.78rem;color:#555;'>{status_badge}</span>"
                    f"</div>"
                    f"<div style='margin-bottom:6px;'>"
                    f"<b>Câu hỏi:</b><br>"
                    f"<span style='color:#333;'>{fb['question']}</span>"
                    f"</div>"
                    f"<details style='margin:6px 0;'>"
                    f"<summary style='cursor:pointer;color:#666;font-size:0.85rem;'>"
                    f"Xem câu trả lời của bot</summary>"
                    f"<div style='background:#f8f8f8;padding:10px;border-radius:6px;"
                    f"font-size:0.85rem;margin-top:6px;'>{fb['bot_answer']}</div>"
                    f"</details>"
                    + (
                        f"<div style='color:#856404;background:#fff3cd;padding:8px 12px;"
                        f"border-radius:6px;font-size:0.85rem;margin-bottom:8px;'>"
                        f"<b>Ghi chú sinh viên:</b> {fb['student_note']}</div>"
                        if fb.get('student_note') else ""
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )

                if fb['status'] == 'resolved' and fb.get('lecturer_reply'):
                    st.markdown(
                        f"<div style='background:#d4edda;border-radius:8px;"
                        f"padding:10px 14px;font-size:0.88rem;margin-bottom:8px;'>"
                        f"<b>Phản hồi của bạn:</b><br>{fb['lecturer_reply']}</div>",
                        unsafe_allow_html=True,
                    )
                elif fb['status'] == 'pending':
                    reply_key = f"reply_text_{key_prefix}_{fb['id']}"
                    reply = st.text_area(
                        "Nhập phản hồi cho sinh viên:",
                        key=reply_key,
                        height=90,
                        placeholder="Giải đáp câu hỏi...",
                    )
                    if st.button("Gửi phản hồi", key=f"btn_reply_{key_prefix}_{fb['id']}", type="primary"):
                        if reply.strip():
                            update_feedback_reply(fb['id'], reply.strip())
                            st.success("Đã gửi phản hồi!")
                            st.rerun()
                        else:
                            st.warning("Vui lòng nhập nội dung phản hồi.")

    with tab_all:
        _render_feedback_cards(get_feedbacks(), key_prefix='all')
    with tab_pending:
        _render_feedback_cards(get_feedbacks(status='pending'), key_prefix='pend')
    with tab_resolved:
        _render_feedback_cards(get_feedbacks(status='resolved'), key_prefix='res')


if not st.session_state.get('logged_in'):
    render_login_page()
elif st.session_state.current_page == 'chatbot':
    render_chatbot_page()
elif st.session_state.current_page == 'database':
    render_database_page()
elif st.session_state.current_page == 'compare':
    render_compare_page()
elif st.session_state.current_page == 'contact':
    render_contact_page()
elif st.session_state.current_page == 'inbox':
    render_inbox_page()
elif st.session_state.current_page == 'notifications':
    render_notifications_page()

# Footer
st.markdown("---")
st.markdown("""

""", unsafe_allow_html=True)
