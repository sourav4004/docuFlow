"""Database-backed job model for document processing persistence.

Provides durable job tracking that survives process restarts.
Replaces the in-memory job_service for production use.
"""

import enum
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class JobStatus(str, enum.Enum):
    """Document processing job status."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProcessingJob(Base):
    """Database-backed document processing job.
    
    Tracks the full lifecycle of document processing with:
    - Retry support (attempts, max_attempts)
    - Error recording
    - Timestamps for auditing
    - Ownership enforcement via user_id
    """
    
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_type = Column(String(50), nullable=False, default="document_processing")
    status = Column(String(50), nullable=False, default=JobStatus.QUEUED.value, index=True)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    # passive_deletes: let the DB CASCADE remove jobs when a document is
    # deleted instead of nulling the NOT NULL document_id FK (which would
    # raise IntegrityError and break document deletion).
    document = relationship("Document", back_populates="jobs", passive_deletes=True)
    user = relationship("User", backref="processing_jobs")
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_processing_jobs_status_created", "status", "created_at"),
        Index("ix_processing_jobs_user_status", "user_id", "status"),
    )
    
    def __repr__(self):
        return f"<ProcessingJob(id={self.id}, document_id={self.document_id}, status={self.status}, attempts={self.attempts})>"
