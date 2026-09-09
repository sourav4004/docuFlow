"""Document activity timeline model."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class DocumentActivity(Base):
    """Track document activity history."""
    __tablename__ = "document_activities"
    __table_args__ = (
        Index("ix_document_activities_document_id", "document_id"),
        Index("ix_document_activities_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(50), nullable=False)  # uploaded, processed, archived, restored, downloaded, viewed, etc.
    details = Column(Text, nullable=True)  # JSON details
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    document = relationship("Document")
    user = relationship("User")

    def __repr__(self):
        return f"<DocumentActivity document={self.document_id} action={self.action}>"
