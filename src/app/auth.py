import sqlite3
import hashlib
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'users.db')

_SALT = "tdtu_assistant_salt_2025"


def init_db():
    """Create tables if they don't exist yet."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT,
            role         TEXT NOT NULL DEFAULT 'student',
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            title      TEXT DEFAULT 'Hội thoại mới',
            pinned     INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            provider        TEXT,
            contexts_json   TEXT,
            timestamp       TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            username        TEXT NOT NULL,
            display_name    TEXT NOT NULL,
            question        TEXT NOT NULL,
            bot_answer      TEXT NOT NULL,
            satisfied       INTEGER DEFAULT 0,
            student_note    TEXT,
            lecturer_reply  TEXT,
            status          TEXT DEFAULT 'pending',
            student_seen    INTEGER DEFAULT 0,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            resolved_at     DATETIME,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    ''')
    conn.commit()
    for migration_sql in [
        "ALTER TABLE conversations ADD COLUMN pinned INTEGER DEFAULT 0",
        "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'student'",
        "ALTER TABLE feedback ADD COLUMN student_seen INTEGER DEFAULT 0",
        "ALTER TABLE feedback ADD COLUMN username TEXT",
        "ALTER TABLE feedback ADD COLUMN display_name TEXT",
        "ALTER TABLE feedback ADD COLUMN bot_answer TEXT",
        "ALTER TABLE feedback ADD COLUMN satisfied INTEGER DEFAULT 0",
        "ALTER TABLE feedback ADD COLUMN student_note TEXT",
        "ALTER TABLE feedback ADD COLUMN lecturer_reply TEXT",
    ]:
        try:
            conn.execute(migration_sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass 
    conn.close()



def save_feedback(
    user_id: int,
    username: str,
    display_name: str,
    question: str,
    bot_answer: str,
    satisfied: int,
    student_note: str = None,
) -> int:
    """Lưu đánh giá câu trả lời. satisfied: 1=hài lòng, 0=chưa hài lòng."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO feedback (user_id, username, display_name, question, bot_answer, satisfied, student_note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, username, display_name, question, bot_answer, satisfied, student_note),
    )
    fb_id = c.lastrowid
    conn.commit()
    conn.close()
    return fb_id


def get_feedbacks(status: str = None) -> list:
    """Lấy danh sách feedback dành cho hộp thư giảng viên.
    Chỉ hiển thị các câu sinh viên chưa hài lòng và đã/chưa được giảng viên xử lý.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if status:
        c.execute(
            "SELECT id, user_id, username, display_name, question, bot_answer, "
            "satisfied, student_note, lecturer_reply, status, created_at, resolved_at "
            "FROM feedback WHERE satisfied=0 AND status=? ORDER BY created_at DESC",
            (status,),
        )
    else:
        c.execute(
            "SELECT id, user_id, username, display_name, question, bot_answer, "
            "satisfied, student_note, lecturer_reply, status, created_at, resolved_at "
            "FROM feedback WHERE satisfied=0 ORDER BY created_at DESC"
        )
    cols = ["id","user_id","username","display_name","question","bot_answer",
            "satisfied","student_note","lecturer_reply","status","created_at","resolved_at"]
    rows = c.fetchall()
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def update_feedback_reply(feedback_id: int, reply: str):
    """Lưu phản hồi của giảng viên và đánh dấu resolved."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE feedback SET lecturer_reply=?, status='resolved', resolved_at=? WHERE id=?",
        (reply, datetime.now().isoformat(), feedback_id),
    )
    conn.commit()
    conn.close()


def get_feedback_stats() -> dict:
    """Thống kê số lượng feedback hiển thị trong hộp thư giảng viên."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT status, COUNT(*) FROM feedback WHERE satisfied=0 GROUP BY status")
    rows = c.fetchall()
    c.execute("SELECT COUNT(*) FROM feedback WHERE satisfied=0")
    total = c.fetchone()[0]
    conn.close()
    stats = {'pending': 0, 'resolved': 0, 'total': total}
    for status, count in rows:
        stats[status] = count
    return stats


def get_my_feedbacks(user_id: int) -> list:
    """Lấy danh sách feedback của một sinh viên cụ thể."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, user_id, username, display_name, question, bot_answer, "
        "satisfied, student_note, lecturer_reply, status, student_seen, created_at, resolved_at "
        "FROM feedback WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    )
    cols = ["id","user_id","username","display_name","question","bot_answer",
            "satisfied","student_note","lecturer_reply","status","student_seen","created_at","resolved_at"]
    rows = c.fetchall()
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def mark_feedbacks_seen(user_id: int):
    """Đánh dấu tất cả phản hồi của giảng viên cho user_id này là đã đọc."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE feedback SET student_seen=1 WHERE user_id=? AND status='resolved' AND student_seen=0",
        (user_id,),
    )
    conn.commit()
    conn.close()


def _hash_password(password: str) -> str:
    return hashlib.sha256(f"{_SALT}{password}".encode()).hexdigest()


def register_user(username: str, password: str, display_name: str = None, role: str = 'student'):
    """Return (success: bool, message: str). role: 'student' | 'lecturer'"""
    if role not in ('student', 'lecturer'):
        role = 'student'
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
            (username.strip(), _hash_password(password), display_name or username.strip(), role),
        )
        conn.commit()
        return True, "Đăng ký thành công!"
    except sqlite3.IntegrityError:
        return False, "Tên đăng nhập đã tồn tại."
    finally:
        conn.close()


def login_user(username: str, password: str):
    """Return (success: bool, user_info: dict | None)."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, username, display_name, role FROM users WHERE username=? AND password_hash=?",
        (username.strip(), _hash_password(password)),
    )
    row = c.fetchone()
    conn.close()
    if row:
        return True, {"id": row[0], "username": row[1], "display_name": row[2], "role": row[3] or 'student'}
    return False, None


def create_conversation(user_id: int, title: str = "Hội thoại mới") -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
        (user_id, title),
    )
    conv_id = c.lastrowid
    conn.commit()
    conn.close()
    return conv_id


def get_conversations(user_id: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, title, pinned, created_at, updated_at FROM conversations "
        "WHERE user_id=? ORDER BY pinned DESC, updated_at DESC",
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [
        {"id": r[0], "title": r[1], "pinned": r[2], "created_at": r[3], "updated_at": r[4]}
        for r in rows
    ]


def update_conversation_title(conv_id: int, title: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE conversations SET title=? WHERE id=?", (title, conv_id))
    conn.commit()
    conn.close()


def rename_conversation(conv_id: int, new_title: str):
    """Đổi tên conversation."""
    update_conversation_title(conv_id, new_title.strip())


def pin_conversation(conv_id: int, pinned: int):
    """Ghim (pinned=1) hoặc bỏ ghim (pinned=0) conversation."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE conversations SET pinned=? WHERE id=?", (pinned, conv_id))
    conn.commit()
    conn.close()


def touch_conversation(conv_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE conversations SET updated_at=? WHERE id=?",
        (datetime.now().isoformat(), conv_id),
    )
    conn.commit()
    conn.close()


def delete_conversation(conv_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))
    c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    conn.commit()
    conn.close()


def save_message(
    conv_id: int,
    role: str,
    content: str,
    provider: str = None,
    contexts=None,
    timestamp: str = None,
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    contexts_json = json.dumps(contexts, ensure_ascii=False) if contexts else None
    c.execute(
        "INSERT INTO messages "
        "(conversation_id, role, content, provider, contexts_json, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (conv_id, role, content, provider, contexts_json,
         timestamp or datetime.now().strftime("%H:%M:%S")),
    )
    conn.commit()
    conn.close()
    touch_conversation(conv_id)


def load_messages(conv_id: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, provider, contexts_json, timestamp "
        "FROM messages WHERE conversation_id=? ORDER BY id ASC",
        (conv_id,),
    )
    rows = c.fetchall()
    conn.close()
    result = []
    for row in rows:
        msg = {
            "role": row[0],
            "content": row[1],
            "provider": row[2],
            "time": row[4] or "",
        }
        if row[3]:
            try:
                msg["contexts"] = json.loads(row[3])
            except Exception:
                msg["contexts"] = []
        result.append(msg)
    return result
