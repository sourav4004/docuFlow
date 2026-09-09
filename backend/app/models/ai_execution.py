"""AI Execution persistence models."""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, JSON, Boolean, Index
from sqlalchemy.orm import relationship
from ..core.database import Base


class AIExecution(Base):
    """Persistent AI execution record."""
    __tablename__ = "ai_executions"
    __table_args__ = (
        Index("ix_ai_executions_workspace_id", "workspace_id"),
        Index("ix_ai_executions_user_id", "user_id"),
        Index("ix_ai_executions_status", "status"),
        Index("ix_ai_executions_created_at", "created_at"),
    )

    id = Column(String(36), primary_key=True)  # UUID
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    action_id = Column(Integer, ForeignKey("ai_actions.id", ondelete="SET NULL"), nullable=True)
    execution_type = Column(String(50), nullable=False, default="rag")  # rag/agent/workflow/extraction/report
    parent_execution_id = Column(String(36), nullable=True)
    task_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default="QUEUED")
    priority = Column(String(20), nullable=False, default="NORMAL")
    query = Column(Text, nullable=True)
    model = Column(String(100), nullable=True)
    provider = Column(String(50), nullable=True)

    # Idempotency / references
    idempotency_key = Column(String(128), nullable=True)
    request_hash = Column(String(64), nullable=True)
    input_reference = Column(String(255), nullable=True)
    output_reference = Column(String(255), nullable=True)
    trace_id = Column(String(64), nullable=True)

    # Token tracking
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    estimated_cost = Column(Float, default=0.0)
    actual_cost = Column(Float, default=0.0)

    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)

    # Failure / retry
    failure_reason = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Results
    result_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Metadata
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    user = relationship("User")
    steps = relationship("AIExecutionStep", back_populates="execution", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<AIExecution id={self.id} status={self.status}>"


class AIExecutionStep(Base):
    """Individual step in an AI execution."""
    __tablename__ = "ai_execution_steps"
    __table_args__ = (
        Index("ix_ai_execution_steps_execution_id", "execution_id"),
    )

    id = Column(String(36), primary_key=True)  # UUID
    execution_id = Column(String(36), ForeignKey("ai_executions.id", ondelete="CASCADE"), nullable=False)
    step_index = Column(Integer, nullable=False)
    step_type = Column(String(50), nullable=False)  # retrieval, tool_call, reasoning, validation
    description = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    
    # Tool call details
    tool_name = Column(String(100), nullable=True)
    tool_input_json = Column(JSON, nullable=True)
    tool_output_json = Column(JSON, nullable=True)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)
    
    # Error
    error_message = Column(Text, nullable=True)
    
    # Token tracking
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    execution = relationship("AIExecution", back_populates="steps")

    def __repr__(self):
        return f"<AIExecutionStep execution={self.execution_id} type={self.step_type}>"


class AIApproval(Base):
    """Human-in-the-loop approval record."""
    __tablename__ = "ai_approvals"
    __table_args__ = (
        Index("ix_ai_approvals_execution_id", "execution_id"),
        Index("ix_ai_approvals_status", "status"),
    )

    id = Column(String(36), primary_key=True)  # UUID
    execution_id = Column(String(36), ForeignKey("ai_executions.id", ondelete="CASCADE"), nullable=False)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    action = Column(String(255), nullable=False)
    reason = Column(Text, nullable=True)
    risk_level = Column(String(20), nullable=False, default="read_only")
    affected_resources_json = Column(JSON, nullable=True)
    proposed_parameters_json = Column(JSON, nullable=True)
    
    status = Column(String(20), nullable=False, default="pending")
    approved_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    execution = relationship("AIExecution")
    workspace = relationship("Workspace")
    user = relationship("User", foreign_keys=[user_id])
    approver = relationship("User", foreign_keys=[approved_by])

    def __repr__(self):
        return f"<AIApproval id={self.id} status={self.status}>"


class AIArtifact(Base):
    """AI-generated artifact with versioning."""
    __tablename__ = "ai_artifacts"
    __table_args__ = (
        Index("ix_ai_artifacts_workspace_id", "workspace_id"),
        Index("ix_ai_artifacts_execution_id", "execution_id"),
        Index("ix_ai_artifacts_family_id", "family_id"),
    )

    id = Column(String(36), primary_key=True)  # UUID — physical row id per version
    # Logical artifact family: every version row shares the original artifact id.
    family_id = Column(String(36), nullable=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    execution_id = Column(String(36), ForeignKey("ai_executions.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    artifact_type = Column(String(50), nullable=False)  # report, extraction, comparison, summary
    name = Column(String(255), nullable=False)
    content_json = Column(JSON, nullable=True)
    version = Column(Integer, default=1, nullable=False)
    
    # Source tracking
    source_document_ids_json = Column(JSON, nullable=True)
    model_used = Column(String(100), nullable=True)
    provider_used = Column(String(50), nullable=True)
    
    # Status
    status = Column(String(20), default="draft")  # draft, ai_generated, under_review, approved, rejected
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    workspace = relationship("Workspace")
    execution = relationship("AIExecution")
    user = relationship("User")

    def __repr__(self):
        return f"<AIArtifact id={self.id} type={self.artifact_type} v{self.version}>"
