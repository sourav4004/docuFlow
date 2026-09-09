"""Usage tracking model for billing readiness."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, BigInteger, Float, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class UsageRecord(Base):
    """Track resource usage for billing/quota purposes."""
    __tablename__ = "usage_records"
    __table_args__ = (
        Index("ix_usage_records_workspace_id", "workspace_id"),
        Index("ix_usage_records_created_at", "created_at"),
        Index("ix_usage_records_type", "usage_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    usage_type = Column(String(50), nullable=False)  # document_upload, ai_request, embedding, storage, etc.
    quantity = Column(BigInteger, default=1, nullable=False)
    units = Column(String(20), default="count", nullable=False)  # count, bytes, tokens
    metadata_json = Column(String(1000), nullable=True)  # Additional details
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    user = relationship("User")

    def __repr__(self):
        return f"<UsageRecord workspace={self.workspace_id} type={self.usage_type} qty={self.quantity}>"
