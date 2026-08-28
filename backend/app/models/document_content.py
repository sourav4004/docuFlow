from sqlalchemy import Column, String, DateTime, Integer, BigInteger, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class DocumentContent(Base):
    """Stores extracted text content for a processed document.

    One document has at most one DocumentContent row.
    """

    __tablename__ = "document_content"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    extracted_text = Column(Text, nullable=False, default="")
    page_count = Column(Integer, nullable=True)
    char_count = Column(BigInteger, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship back to Document
    document = relationship("Document", back_populates="content")

    def __repr__(self):
        return f"<DocumentContent(id={self.id}, document_id={self.document_id}, chars={self.char_count})>"
