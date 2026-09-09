"""Document tags and classification models."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Table
from sqlalchemy.orm import relationship
from ..core.database import Base

# Many-to-many relationship between documents and tags
document_tags = Table(
    "document_tags",
    Base.metadata,
    Column("document_id", Integer, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
    Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)),
)


class Tag(Base):
    """Document tag for categorization."""
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    category = Column(String(50), nullable=True)  # e.g., "topic", "type", "custom"
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Tag {self.name}>"


class DocumentClassification(Base):
    """Document classification results."""
    __tablename__ = "document_classifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    document_type = Column(String(100), nullable=False)  # report, invoice, contract, etc.
    confidence = Column(Float, nullable=False, default=0.0)
    classifier_version = Column(String(50), nullable=True)
    is_user_override = Column(Integer, default=0)  # 0=auto, 1=user override
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    document = relationship("Document", back_populates="classifications")

    def __repr__(self):
        return f"<DocumentClassification doc={self.document_id} type={self.document_type}>"


class DocumentSummary(Base):
    """Document summaries for quick understanding."""
    __tablename__ = "document_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    version_id = Column(Integer, ForeignKey("document_versions.id", ondelete="SET NULL"), nullable=True)
    summary_type = Column(String(50), nullable=False)  # short, detailed, key_points
    content = Column(String(5000), nullable=False)
    generated_by = Column(String(50), default="llm")  # llm, extractive, user
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    document = relationship("Document", back_populates="summaries")

    def __repr__(self):
        return f"<DocumentSummary doc={self.document_id} type={self.summary_type}>"
