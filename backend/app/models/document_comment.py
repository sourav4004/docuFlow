"""Document comment model for collaboration."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class DocumentComment(Base):
    """Document comment model for collaboration."""
    __tablename__ = "document_comments"
    __table_args__ = (
        Index("ix_document_comments_document_id", "document_id"),
        Index("ix_document_comments_user_id", "user_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    parent_id = Column(Integer, ForeignKey("document_comments.id", ondelete="SET NULL"), nullable=True)
    content = Column(Text, nullable=False)
    is_resolved = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    document = relationship("Document")
    user = relationship("User")
    replies = relationship("DocumentComment", backref="parent", remote_side=[id])

    def resolve(self):
        """Mark comment as resolved."""
        self.is_resolved = True

    def reopen(self):
        """Reopen resolved comment."""
        self.is_resolved = False

    def __repr__(self):
        return f"<DocumentComment document={self.document_id} user={self.user_id}>"
