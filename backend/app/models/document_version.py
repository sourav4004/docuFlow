"""Document versioning model for version control."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from ..core.database import Base


class DocumentVersion(Base):
    """Document version tracking."""
    __tablename__ = "document_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False, default=1)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    storage_key = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=True)  # SHA-256 hash for change detection
    mime_type = Column(String(100), nullable=False, default="application/pdf")
    is_current = Column(Boolean, default=True)
    processing_status = Column(String(50), default="PENDING")  # PENDING, PROCESSING, READY, FAILED
    metadata_json = Column(Text, nullable=True)  # JSON metadata for version-specific info
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    document = relationship("Document", back_populates="versions")
    creator = relationship("User")

    def __repr__(self):
        return f"<DocumentVersion doc={self.document_id} v{self.version_number}>"

    @property
    def is_ready(self) -> bool:
        return self.processing_status == "READY"

    @property
    def is_failed(self) -> bool:
        return self.processing_status == "FAILED"
