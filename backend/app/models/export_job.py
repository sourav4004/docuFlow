"""Asynchronous export job model."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class ExportJob(Base):
    """Asynchronous data export job with status tracking and expiry."""

    __tablename__ = "export_jobs"
    __table_args__ = (
        Index("ix_export_jobs_workspace_id", "workspace_id"),
        Index("ix_export_jobs_user_id", "user_id"),
        Index("ix_export_jobs_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    export_type = Column(String(50), nullable=False)  # documents, metadata, conversations, audit, all
    format = Column(String(10), nullable=False, default="json")  # json, zip
    status = Column(String(20), nullable=False, default="QUEUED")  # QUEUED, PROCESSING, COMPLETED, FAILED, EXPIRED
    storage_path = Column(String(1000), nullable=True)  # relative safe path, never absolute
    download_token_hash = Column(String(128), nullable=True)  # short-lived download token
    download_expires_at = Column(DateTime(timezone=True), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    workspace = relationship("Workspace")
    organization = relationship("Organization")
    user = relationship("User")

    def __repr__(self):
        return f"<ExportJob id={self.id} type={self.export_type} status={self.status}>"