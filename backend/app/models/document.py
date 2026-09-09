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
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True, index=True)
    original_filename = Column(String(255), nullable=False)
    storage_key = Column(String(255), unique=True, nullable=False, index=True)
    mime_type = Column(String(100), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    status = Column(String(50), nullable=False, default="UPLOADED", index=True)
    # Data sensitivity classification (Phase 15) — drives AI provider routing.
    # PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED
    sensitivity = Column(String(20), nullable=False, default="INTERNAL")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", backref="documents")
    content = relationship("DocumentContent", back_populates="document", uselist=False, cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan", order_by="DocumentChunk.chunk_index")
    versions = relationship("DocumentVersion", back_populates="document", cascade="all, delete-orphan")
    classifications = relationship("DocumentClassification", back_populates="document", cascade="all, delete-orphan")
    summaries = relationship("DocumentSummary", back_populates="document", cascade="all, delete-orphan")
    # passive_deletes: DB CASCADE removes jobs; avoids nulling the NOT NULL FK.
    jobs = relationship("ProcessingJob", back_populates="document", passive_deletes=True)

    # Composite index for common queries
    __table_args__ = (
        Index("ix_documents_user_created", "user_id", "created_at"),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.original_filename}, user_id={self.user_id})>"
