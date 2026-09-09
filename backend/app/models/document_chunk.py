from sqlalchemy import Column, DateTime, Integer, BigInteger, ForeignKey, Text, UniqueConstraint, Index
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class DocumentChunk(Base):
    """Stores text chunks for a processed document.

    One document → many chunks.
    Each chunk holds a slice of the normalized extracted text
    with positional metadata for retrieval.
    """

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    char_start = Column(BigInteger, nullable=False)
    char_end = Column(BigInteger, nullable=False)

    # Page metadata — nullable because current pipeline does not
    # reliably preserve per-page boundaries.
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)

    # Embedding vector — pgvector vector type for similarity search.
    # Nullable because embeddings are generated in a separate step.
    # Dimension matches EMBEDDING_DIMENSION setting (default 384).
    embedding = Column(Vector(384), nullable=True)

    # NOTE: search_vector (TSVECTOR) column exists in PostgreSQL but is NOT
    # defined here because TSVECTOR is not compatible with SQLite (used in tests).
    # The column is managed by migration 009 and a PostgreSQL trigger that
    # auto-populates it from the 'text' column on INSERT/UPDATE.
    # See keyword_search.py for full-text search queries.

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationship back to Document
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
        Index("ix_document_chunks_doc_index", "document_id", "chunk_index"),
    )

    def __repr__(self):
        preview = self.text[:40].replace("\n", "\\n") if self.text else ""
        return (
            f"<DocumentChunk(doc={self.document_id}, idx={self.chunk_index}, "
            f"chars=[{self.char_start}:{self.char_end}], text={preview!r})>"
        )
