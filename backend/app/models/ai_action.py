"""AI Action Center models — actions, suggestions, and their lifecycles."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Index
from sqlalchemy.orm import relationship
from ..core.database import Base

# Unified action lifecycle (valid transitions enforced in ai_action_service)
ACTION_STATUSES = (
    "SUGGESTED",
    "APPROVAL_REQUIRED",
    "APPROVED",
    "QUEUED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "REJECTED",
)

ACTION_TYPES = (
    "summarize",
    "extract",
    "classify",
    "compare",
    "tag",
    "generate_report",
    "create_workflow",
    "send_notification",
    "request_review",
    "detect_conflict",
    "detect_missing_info",
    "deadline_alert",
    "duplicate_review",
    "health_review",
)

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


class AIAction(Base):
    """A centralized AI action with an explicit lifecycle."""

    __tablename__ = "ai_actions"
    __table_args__ = (
        Index("ix_ai_actions_workspace_id", "workspace_id"),
        Index("ix_ai_actions_owner_id", "owner_id"),
        Index("ix_ai_actions_status", "status"),
        Index("ix_ai_actions_created_at", "created_at"),
        Index("ix_ai_actions_source", "source_type", "source_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    action_type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="SUGGESTED")
    risk_level = Column(String(20), nullable=False, default="LOW")
    priority = Column(String(20), nullable=False, default="NORMAL")  # CRITICAL, HIGH, NORMAL, LOW

    # Source of the action
    source_type = Column(String(50), nullable=False, default="manual")  # suggestion, workflow, document, manual
    source_id = Column(Integer, nullable=True)
    execution_id = Column(String(36), nullable=True)  # linked AI execution

    # Dependency graph (Phase 15) — a child action waits for parent actions
    parent_id = Column(Integer, ForeignKey("ai_actions.id", ondelete="SET NULL"), nullable=True)
    dependency_status = Column(String(20), nullable=False, default="NONE")  # NONE/PENDING/SATISFIED/BLOCKED
    blocked_reason = Column(Text, nullable=True)

    # Metadata
    payload_json = Column(Text, nullable=True)  # parameters/inputs
    result_json = Column(Text, nullable=True)  # outputs
    estimated_cost = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Approval metadata
    approved_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    organization = relationship("Organization")
    owner = relationship("User", foreign_keys=[owner_id])
    approver = relationship("User", foreign_keys=[approved_by])

    def __repr__(self):
        return f"<AIAction id={self.id} type={self.action_type} status={self.status}>"


class AISuggestion(Base):
    """Deterministic, explainable AI suggestion (never autonomous)."""

    __tablename__ = "ai_suggestions"
    __table_args__ = (
        Index("ix_ai_suggestions_workspace_id", "workspace_id"),
        Index("ix_ai_suggestions_owner_id", "owner_id"),
        Index("ix_ai_suggestions_status", "status"),
        Index("ix_ai_suggestions_source", "source_type", "source_id"),
        Index("ix_ai_suggestions_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    reason = Column(Text, nullable=True)  # explainable rationale
    suggestion_type = Column(String(50), nullable=False)  # conflict, deadline, duplicate, collection, health, metadata
    source_type = Column(String(50), nullable=False, default="system")  # document, workflow, search, system
    source_id = Column(Integer, nullable=True)
    priority = Column(String(20), nullable=False, default="NORMAL")
    status = Column(String(20), nullable=False, default="OPEN")  # OPEN, DISMISSED, ACTIONED, EXPIRED
    evidence_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    organization = relationship("Organization")
    owner = relationship("User")

    def __repr__(self):
        return f"<AISuggestion id={self.id} type={self.suggestion_type} status={self.status}>"