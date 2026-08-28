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
