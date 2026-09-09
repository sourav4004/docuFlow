"""Notification model for in-app notifications."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class Notification(Base):
    """In-app notification model."""
    __tablename__ = "notifications"
    __table_args__ = (
        Index("ix_notifications_user_id", "user_id"),
        Index("ix_notifications_read", "user_id", "is_read"),
        Index("ix_notifications_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False)  # processing_completed, processing_failed, workspace_invite, etc.
    resource_type = Column(String(50), nullable=True)  # document, workspace, etc.
    resource_id = Column(Integer, nullable=True)
    is_read = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User")

    def mark_read(self):
        """Mark notification as read."""
        self.is_read = True

    def __repr__(self):
        return f"<Notification user={self.user_id} type={self.notification_type} read={self.is_read}>"
