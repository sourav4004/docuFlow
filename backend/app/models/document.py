from sqlalchemy import Column, String, DateTime, Integer, BigInteger, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class Document(Base):
    """Document model for storing user-uploaded files.

    Status lifecycle:
        UPLOADED → QUEUED → PROCESSING → READY
                                      → FAILED
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    storage_key = Column(String(255), unique=True, nullable=False, index=True)
    mime_type = Column(String(100), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    status = Column(String(50), nullable=False, default="UPLOADED", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship to User
    user = relationship("User", backref="documents")

    # Relationship to extracted content
    content = relationship("DocumentContent", back_populates="document", uselist=False, cascade="all, delete-orphan")

    # Composite index for common queries
    __table_args__ = (
        Index("ix_documents_user_created", "user_id", "created_at"),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.original_filename}, user_id={self.user_id})>"
