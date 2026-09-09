"""Conversation and Message schemas for request/response validation."""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List


class ConversationCreate(BaseModel):
    """Schema for creating a new conversation."""
    title: str = Field(
        default="New Conversation",
        min_length=1,
        max_length=500,
        description="Conversation title",
    )

    @field_validator("title")
    @classmethod
    def title_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Title must not be empty or whitespace-only")
        return v


class ConversationUpdate(BaseModel):
    """Schema for updating a conversation."""
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="New conversation title",
    )

    @field_validator("title")
    @classmethod
    def title_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Title must not be empty or whitespace-only")
        return v.strip()


class MessageResponse(BaseModel):
    """Schema for a single message in a conversation."""
    id: int
    role: str
    content: str
    created_at: datetime
    sources: List["SourceInfoResponse"] = []

    model_config = {"from_attributes": True}


class SourceInfoResponse(BaseModel):
    """Schema for a persisted source reference."""
    id: int
    document_id: int
    chunk_id: int
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    similarity_score: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    """Schema for conversation response to client."""
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailResponse(BaseModel):
    """Schema for conversation with messages."""
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse]

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    """Schema for paginated conversation list response."""
    items: list[ConversationResponse]
    total: int
    limit: int
    offset: int
    has_next: bool = False
    has_previous: bool = False


# ---------------------------------------------------------------------------
# Message creation schemas
# ---------------------------------------------------------------------------

class MessageCreate(BaseModel):
    """Schema for creating a new message in a conversation.

    Triggers RAG pipeline to generate an assistant response.
    Supports optional RAG controls for advanced users.
    """
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message content",
    )
    document_id: Optional[int] = Field(
        None,
        description="Optional: restrict RAG to a specific document",
    )
    collection_id: Optional[int] = Field(
        None,
        description="Optional: restrict RAG to documents in this collection",
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Optional: number of retrieval results (1-50)",
    )
    min_similarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional: minimum similarity threshold (0-1)",
    )

    @field_validator("content")
    @classmethod
    def content_must_not_be_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content must not be empty or whitespace-only")
        return v


class SourceInfo(BaseModel):
    """Schema for a source reference in the RAG response."""
    document_id: int
    filename: Optional[str] = None
    chunk_id: int
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    similarity_score: Optional[float] = None


class PaginatedMessageResponse(BaseModel):
    """Schema for paginated message list response."""
    messages: List[MessageResponse]
    page: int
    page_size: int
    total: int
    has_next: bool
    has_previous: bool


class SendMessageResponse(BaseModel):
    """Schema for the response after sending a message."""
    user_message: MessageResponse
    assistant_message: MessageResponse
    sources: List[SourceInfo]
    grounded: bool
