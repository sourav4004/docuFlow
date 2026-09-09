"""Idempotency key model for safe retries of repeatable operations."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Index, UniqueConstraint
from sqlalchemy.orm import relationship
from ..core.database import Base


class IdempotencyKey(Base):
    """Persisted idempotency record linking a key to a stored response."""

    __tablename__ = "idempotency_keys"
    __table_args__ = (
        UniqueConstraint("workspace_id", "key", name="uq_idempotency_workspace_key"),
        Index("ix_idempotency_keys_key", "key"),
        Index("ix_idempotency_keys_expires_at", "expires_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    key = Column(String(255), nullable=False)
    request_hash = Column(String(128), nullable=False)  # hash of request body to detect conflicts
    response_status = Column(Integer, nullable=True)
    response_json = Column(Text, nullable=True)  # stored response payload
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)

    # Relationships
    workspace = relationship("Workspace")
    user = relationship("User")

    def __repr__(self):
        return f"<IdempotencyKey workspace={self.workspace_id} key={self.key}>"