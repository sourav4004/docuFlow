from sqlalchemy import Column, DateTime, Integer, ForeignKey, Float, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class MessageSource(Base):
    """A source reference associated with an assistant message.

    When RAG generates an answer, the sources (document chunks) that
    supported the answer are persisted here. This allows conversation
    history replay without re-calling RAG.

    One message → many sources.
    Cascade: deleting a message deletes its sources.
    """

    __tablename__ = "message_sources"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(
        Integer,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    similarity_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    message = relationship("Message", back_populates="sources")
    document = relationship("Document")
    chunk = relationship("DocumentChunk")

    __table_args__ = (
        Index("ix_message_sources_message_id", "message_id"),
    )

    def __repr__(self):
        return (
            f"<MessageSource(id={self.id}, message={self.message_id}, "
            f"doc={self.document_id}, chunk={self.chunk_id}, "
            f"sim={self.similarity_score})>"
        )
