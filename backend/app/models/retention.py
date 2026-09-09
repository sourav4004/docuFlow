"""Retention policy and idempotent usage event models."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, BigInteger, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class RetentionPolicy(Base):
    """Configurable data retention policy for a workspace or organization."""

    __tablename__ = "retention_policies"
    __table_args__ = (
        Index("ix_retention_policies_scope", "scope_type", "scope_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope_type = Column(String(20), nullable=False, default="WORKSPACE")  # GLOBAL, ORGANIZATION, WORKSPACE
    scope_id = Column(Integer, nullable=True)
    data_type = Column(String(50), nullable=False)  # documents, trash, audit_logs, ai_executions, artifacts, exports, sessions
    retention_days = Column(Integer, nullable=False, default=30)
    is_enabled = Column(Boolean, nullable=False, default=True)
    last_cleanup_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<RetentionPolicy {self.data_type} days={self.retention_days} enabled={self.is_enabled}>"


class UsageEvent(Base):
    """Idempotent metering event for quota/billing calculations."""

    __tablename__ = "usage_events"
    __table_args__ = (
        Index("ix_usage_events_workspace_id", "workspace_id"),
        Index("ix_usage_events_metric", "metric"),
        Index("ix_usage_events_period", "period"),
        Index("ix_usage_events_event_key", "event_key"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    metric = Column(String(50), nullable=False)  # storage_bytes, documents, ai_requests, api_requests, etc.
    quantity = Column(BigInteger, nullable=False, default=1)
    period = Column(String(10), nullable=False)  # daily, monthly, lifetime
    event_key = Column(String(128), nullable=True)  # idempotency — same key is counted once
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    organization = relationship("Organization")
    workspace = relationship("Workspace")
    user = relationship("User")

    def __repr__(self):
        return f"<UsageEvent workspace={self.workspace_id} metric={self.metric} qty={self.quantity}>"