"""Document schemas for request/response validation."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class DocumentBase(BaseModel):
    """Base document schema with shared fields."""
    original_filename: str
    mime_type: str
    file_size: int


class DocumentCreate(DocumentBase):
    """Schema for document creation (internal use)."""
    storage_key: str
    user_id: int


class DocumentResponse(BaseModel):
    """Schema for document response to client."""
    id: int
    original_filename: str
    mime_type: str
    file_size: int
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Schema for paginated document list response."""
    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class DocumentStatusResponse(BaseModel):
    """Schema for document processing status response."""
    document_id: int
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentContentResponse(BaseModel):
    """Schema for document extracted content response."""
    document_id: int
    status: str
    extracted_text: Optional[str] = None
    error_message: Optional[str] = None

    model_config = {"from_attributes": True}


class ChunkResponse(BaseModel):
    """Schema for a single document chunk."""
    id: int
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    model_config = {"from_attributes": True}


class DocumentChunksResponse(BaseModel):
    """Schema for document chunks list response."""
    document_id: int
    total_chunks: int
    chunks: list[ChunkResponse]


# ---------------------------------------------------------------------------
# RAG schemas
# ---------------------------------------------------------------------------

class RAGSourceResponse(BaseModel):
    """Schema for a single RAG source reference."""
    document_id: int
    filename: Optional[str] = None
    chunk_id: int
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    similarity_score: Optional[float] = None


class RAGRequest(BaseModel):
    """Schema for RAG question request."""
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask about the documents")
    document_id: Optional[int] = Field(None, description="Optional: restrict to a specific document")
    top_k: int = Field(5, ge=1, le=20, description="Number of retrieval candidates")


class RAGResponse(BaseModel):
    """Schema for RAG answer response."""
    answer: str
    sources: list[RAGSourceResponse]
    grounded: bool
    retrieval_count: int
    model: Optional[str] = None
    provider: Optional[str] = None
