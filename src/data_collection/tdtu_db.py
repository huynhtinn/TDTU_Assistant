from datetime import datetime
from typing import Optional
from loguru import logger
from sqlalchemy import create_engine, String, Text, DateTime
from sqlalchemy import select
from sqlalchemy.orm import Session, DeclarativeBase, Mapped, mapped_column

# Use separate database for TDTU data
tdtu_engine = None


def open_tdtu_db():
    """Open TDTU database"""
    global tdtu_engine
    tdtu_engine = create_engine("sqlite+pysqlite:///tdtu.sqlite", echo=False)
    Base.metadata.create_all(tdtu_engine)
    return tdtu_engine

class Base(DeclarativeBase):
    pass


class QuyCheDocument(Base):
    """Quy Che document model"""
    __tablename__ = "quy_che_document"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[Optional[str]] = mapped_column(String(1000))
    doc_type: Mapped[Optional[str]] = mapped_column(String(100))  # quy_che, quy_dinh, huong_dan
    department: Mapped[Optional[str]] = mapped_column(String(200))  # Phòng ban phụ trách
    issue_date: Mapped[Optional[str]] = mapped_column(String(50))  # Ngày ban hành
    effective_date: Mapped[Optional[str]] = mapped_column(String(50))  # Ngày hiệu lực
    status: Mapped[Optional[str]] = mapped_column(String(50))  # Con hieu luc, Het hieu luc
    file_path: Mapped[Optional[str]] = mapped_column(String(500))  # Local file path if downloaded
    content: Mapped[Optional[str]] = mapped_column(Text)  # Text content if extracted
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    @staticmethod
    def from_json(data: dict) -> 'QuyCheDocument':
        """Create QuyCheDocument from JSON data"""
        doc = QuyCheDocument()
        doc.title = data.get('title', '')
        doc.url = data.get('url', '')
        doc.doc_type = data.get('type', 'quy_che')
        doc.department = data.get('department', '')
        doc.issue_date = data.get('issue_date', '')
        doc.effective_date = data.get('effective_date', '')
        doc.status = data.get('status', '')
        doc.file_path = data.get('file_path', '')
        doc.content = data.get('content', '')
        return doc
    
    @staticmethod
    def get_by_title(sess: Session, title: str) -> Optional['QuyCheDocument']:
        """Get document by title"""
        return sess.execute(
            select(QuyCheDocument).where(QuyCheDocument.title == title)
        ).scalar_one_or_none()
    
    @staticmethod
    def get_by_url(sess: Session, url: str) -> Optional['QuyCheDocument']:
        """Get document by URL"""
        return sess.execute(
            select(QuyCheDocument).where(QuyCheDocument.url == url)
        ).scalar_one_or_none()
    
    @staticmethod
    def save(sess: Session, doc: 'QuyCheDocument'):
        """Save or update document"""
        existing = QuyCheDocument.get_by_url(sess, doc.url) if doc.url else None
        
        if existing:
            # Update existing
            existing.title = doc.title
            existing.doc_type = doc.doc_type
            existing.department = doc.department
            existing.issue_date = doc.issue_date
            existing.effective_date = doc.effective_date
            existing.status = doc.status
            existing.file_path = doc.file_path
            existing.content = doc.content
            existing.updated_at = datetime.now()
            logger.debug(f"Updated document: {doc.title}")
        else:
            # Insert new
            sess.add(doc)
            logger.debug(f"Inserted new document: {doc.title}")
    
    def __repr__(self) -> str:
        return f"<QuyCheDocument(id={self.id}, title='{self.title}', type='{self.doc_type}')>"
    

if __name__ == "__main__":
    # Test database creation
    logger.info("Creating TDTU database...")
    engine = open_tdtu_db()
    
    with Session(engine) as sess:

        
        logger.info("✓ Database created successfully")
        
        # Test query
        docs = sess.query(QuyCheDocument).all()
        logger.info(f"Found {len(docs)} documents")
        for d in docs:
            print(d)
