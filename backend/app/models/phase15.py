"""Phase 15 models — autonomous knowledge operating system infrastructure.

All timestamps are UTC. Every table is tenant/workspace scoped where the
feature touches tenant data. No secrets are ever stored in these tables.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean, Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from ..core.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# AI execution extensions (orchestrator 2.0)
# ---------------------------------------------------------------------------

EXECUTION_STATUSES = (
    "QUEUED", "PLANNING", "RUNNING", "WAITING_APPROVAL", "WAITING_TOOL",
    "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "RETRYING",
)

EXECUTION_PRIORITIES = ("CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND")


class AIExecutionIdempotency(Base):
    """Idempotency record for AI executions (same key + tenant = one execution)."""

    __tablename__ = "ai_execution_idempotency"
    __table_args__ = (
        UniqueConstraint("workspace_id", "idempotency_key", name="uq_execution_idempotency_key"),
        Index("ix_execution_idem_expires", "expires_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    idempotency_key = Column(String(128), nullable=False)
    request_hash = Column(String(64), nullable=False)
    execution_id = Column(String(36), ForeignKey("ai_executions.id", ondelete="CASCADE"), nullable=False)
    result_status = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)


class AIExecutionCheckpoint(Base):
    """Resumable execution checkpoint (never stores secrets or hidden reasoning)."""

    __tablename__ = "ai_execution_checkpoints"
    __table_args__ = (
        Index("ix_exec_checkpoints_execution", "execution_id"),
        Index("ix_exec_checkpoints_step", "execution_id", "step_number"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(String(36), ForeignKey("ai_executions.id", ondelete="CASCADE"), nullable=False)
    step_number = Column(Integer, nullable=False)
    state_json = Column(Text, nullable=True)  # resume state (inputs, cursor), no secrets
    tool_output_reference = Column(String(255), nullable=True)
    artifact_reference = Column(String(255), nullable=True)
    checksum = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    execution = relationship("AIExecution")


# ---------------------------------------------------------------------------
# Knowledge event outbox
# ---------------------------------------------------------------------------

EVENT_STATUSES = ("PENDING", "PROCESSED", "FAILED", "DEAD")

EVENT_TYPES = (
    "DOCUMENT_UPLOADED", "DOCUMENT_READY", "DOCUMENT_UPDATED",
    "DOCUMENT_VERSION_CREATED", "DOCUMENT_DELETED", "DOCUMENT_RESTORED",
    "DOCUMENT_CLASSIFIED", "DOCUMENT_SUMMARIZED", "DOCUMENT_DUPLICATE_DETECTED",
    "DOCUMENT_CONFLICT_DETECTED", "DEADLINE_APPROACHING", "DEADLINE_OVERDUE",
    "SEARCH_ALERT_MATCHED", "KNOWLEDGE_GAP_DETECTED", "WORKFLOW_COMPLETED",
    "AI_ACTION_COMPLETED", "AI_ACTION_FAILED", "POLICY_CONFLICT_DETECTED",
)


class KnowledgeEvent(Base):
    """Transactional outbox event — the source of truth for async processing."""

    __tablename__ = "knowledge_events"
    __table_args__ = (
        Index("ix_knowledge_events_status_next_retry", "status", "next_retry_at"),
        Index("ix_knowledge_events_workspace", "workspace_id"),
        Index("ix_knowledge_events_type", "event_type"),
        UniqueConstraint("workspace_id", "event_type", "aggregate_type", "aggregate_id", "dedupe_key",
                         name="uq_knowledge_event_dedupe"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    event_type = Column(String(50), nullable=False)
    aggregate_type = Column(String(50), nullable=False, default="unknown")
    aggregate_id = Column(Integer, nullable=False, default=0)
    dedupe_key = Column(String(64), nullable=False, default="")
    payload_json = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="PENDING")
    retry_count = Column(Integer, nullable=False, default=0)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    workspace = relationship("Workspace")


# ---------------------------------------------------------------------------
# Knowledge engine
# ---------------------------------------------------------------------------

class KnowledgeChange(Base):
    """A detected knowledge change (never fabricated; evidence-backed)."""

    __tablename__ = "knowledge_changes"
    __table_args__ = (
        Index("ix_knowledge_changes_workspace", "workspace_id"),
        Index("ix_knowledge_changes_document", "document_id"),
        Index("ix_knowledge_changes_type", "change_type"),
        Index("ix_knowledge_changes_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    version_from = Column(Integer, nullable=True)
    version_to = Column(Integer, nullable=True)
    change_type = Column(String(50), nullable=False)  # CONTENT/DATE/POLICY/NUMERIC/ENTITY/STRUCTURAL/OWNERSHIP/REQUIREMENT/TERMINOLOGY/RISK
    severity = Column(String(20), nullable=False, default="MEDIUM")  # LOW/MEDIUM/HIGH/CRITICAL
    confidence = Column(String(20), nullable=False, default="MEDIUM")
    summary = Column(Text, nullable=True)
    old_evidence_json = Column(Text, nullable=True)
    new_evidence_json = Column(Text, nullable=True)
    affected_sections_json = Column(Text, nullable=True)
    affected_entities_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    document = relationship("Document")


class ImpactLink(Base):
    """Lightweight dependency graph between knowledge objects."""

    __tablename__ = "impact_links"
    __table_args__ = (
        Index("ix_impact_links_source", "source_type", "source_id"),
        Index("ix_impact_links_target", "target_type", "target_id"),
        Index("ix_impact_links_workspace", "workspace_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    source_type = Column(String(30), nullable=False)  # document/entity/deadline/workflow/collection/policy
    source_id = Column(Integer, nullable=False)
    target_type = Column(String(30), nullable=False)
    target_id = Column(Integer, nullable=False)
    relation = Column(String(20), nullable=False, default="EXPLICIT")  # EXPLICIT/INFERRED
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class PolicyStatement(Base):
    """A structured policy statement extracted from a document."""

    __tablename__ = "policy_statements"
    __table_args__ = (
        Index("ix_policy_statements_workspace", "workspace_id"),
        Index("ix_policy_statements_document", "document_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    source_version = Column(Integer, nullable=True)
    statement = Column(Text, nullable=False)
    requirement_type = Column(String(50), nullable=True)  # approval/limit/obligation/exception
    applicability = Column(Text, nullable=True)
    effective_date = Column(DateTime(timezone=True), nullable=True)
    expiration_date = Column(DateTime(timezone=True), nullable=True)
    responsible_party = Column(String(255), nullable=True)
    evidence_reference = Column(String(255), nullable=True)  # chunk/page reference
    source_chunk = Column(Text, nullable=True)
    # Phase 16 semantic normalization (subject/action/threshold/…)
    subject = Column(String(255), nullable=True)
    action = Column(String(100), nullable=True)
    threshold_value = Column(Float, nullable=True)
    threshold_unit = Column(String(50), nullable=True)
    condition_text = Column(Text, nullable=True)
    timeframe_text = Column(String(255), nullable=True)
    scope_text = Column(String(255), nullable=True)
    exception_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class PolicyConflict(Base):
    """A detected conflict between policy requirements (with conditions)."""

    __tablename__ = "policy_conflicts"
    __table_args__ = (
        Index("ix_policy_conflicts_workspace", "workspace_id"),
        Index("ix_policy_conflicts_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    policy_a_id = Column(Integer, ForeignKey("policy_statements.id", ondelete="CASCADE"), nullable=False)
    policy_b_id = Column(Integer, ForeignKey("policy_statements.id", ondelete="CASCADE"), nullable=False)
    conflict_type = Column(String(50), nullable=False)  # NUMERIC_THRESHOLD/DATE/REQUIREMENT/TERMINOLOGY
    severity = Column(String(20), nullable=False, default="MEDIUM")
    description = Column(Text, nullable=True)
    conditions_json = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="OPEN")  # OPEN/RESOLVED/DISMISSED
    resolution_note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    policy_a = relationship("PolicyStatement", foreign_keys=[policy_a_id])
    policy_b = relationship("PolicyStatement", foreign_keys=[policy_b_id])


class TemporalFact(Base):
    """A fact with validity windows — stale knowledge is never current."""

    __tablename__ = "temporal_facts"
    __table_args__ = (
        Index("ix_temporal_facts_workspace", "workspace_id"),
        Index("ix_temporal_facts_valid_from", "valid_from"),
        Index("ix_temporal_facts_superseded", "superseded_by"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    entity_id = Column(Integer, nullable=True)
    fact_type = Column(String(50), nullable=False)  # policy_effective/policy_expiry/amount/deadline/ownership
    fact_value = Column(Text, nullable=False)
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_until = Column(DateTime(timezone=True), nullable=True)  # None = still current
    superseded_by = Column(Integer, nullable=True)
    observed_at = Column(DateTime(timezone=True), default=_utcnow)
    source = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class KnowledgeSnapshot(Base):
    """Versioned reproducible workspace knowledge snapshot."""

    __tablename__ = "knowledge_snapshots"
    __table_args__ = (
        Index("ix_knowledge_snapshots_workspace", "workspace_id"),
        Index("ix_knowledge_snapshots_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    name = Column(String(255), nullable=False)
    snapshot_type = Column(String(30), nullable=False, default="workspace")
    data_json = Column(Text, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

MEMORY_TYPES = (
    "CONVERSATION", "USER_PREFERENCE", "WORKSPACE_FACT", "DOCUMENT_FACT",
    "POLICY_FACT", "TASK_CONTEXT", "DECISION", "APPROVAL",
)

MEMORY_SCOPES = ("WORKSPACE", "USER", "DOCUMENT", "CONVERSATION")


class AIMemory(Base):
    """Persistent AI memory with scope, source, confidence, and expiration."""

    __tablename__ = "ai_memories"
    __table_args__ = (
        Index("ix_ai_memories_workspace", "workspace_id"),
        Index("ix_ai_memories_user", "user_id"),
        Index("ix_ai_memories_scope", "scope"),
        Index("ix_ai_memories_expires", "expires_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    memory_type = Column(String(30), nullable=False)
    scope = Column(String(30), nullable=False, default="WORKSPACE")
    content = Column(Text, nullable=False)
    source = Column(String(255), nullable=True)
    confidence = Column(String(20), nullable=False, default="MEDIUM")
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    # Memory 2.0 lifecycle: ACTIVE / EXPIRED / SUPERSEDED / DELETED (soft)
    lifecycle_status = Column(String(20), nullable=False, default="ACTIVE")
    supersedes_id = Column(Integer, ForeignKey("ai_memories.id", ondelete="SET NULL"), nullable=True)


# ---------------------------------------------------------------------------
# Human review queue
# ---------------------------------------------------------------------------

REVIEW_ITEM_TYPES = (
    "AI_ACTION", "DOCUMENT_CONFLICT", "POLICY_CONFLICT", "EXTRACTION_UNCERTAINTY",
    "RECOMMENDATION", "WORKFLOW_APPROVAL", "AI_ANSWER",
)

REVIEW_STATUSES = ("PENDING", "APPROVED", "REJECTED", "EDITED", "NEEDS_EVIDENCE", "DELEGATED", "DEFERRED")


class ReviewItem(Base):
    """Centralized human review queue item with SLA."""

    __tablename__ = "review_items"
    __table_args__ = (
        Index("ix_review_items_workspace", "workspace_id"),
        Index("ix_review_items_status", "status"),
        Index("ix_review_items_due", "due_at"),
        Index("ix_review_items_assignee", "assignee_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    item_type = Column(String(40), nullable=False)
    status = Column(String(20), nullable=False, default="PENDING")
    priority = Column(String(20), nullable=False, default="NORMAL")  # CRITICAL/HIGH/NORMAL/LOW
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=True)
    source_type = Column(String(30), nullable=True)
    source_id = Column(Integer, nullable=True)
    assignee_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    due_at = Column(DateTime(timezone=True), nullable=True)
    escalated = Column(Boolean, nullable=False, default=False)
    decision_note = Column(Text, nullable=True)
    decided_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    decided_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    workspace = relationship("Workspace")
    assignee = relationship("User", foreign_keys=[assignee_id])
    decider = relationship("User", foreign_keys=[decided_by])


# ---------------------------------------------------------------------------
# Automation reliability
# ---------------------------------------------------------------------------

class WorkflowNodeExecution(Base):
    """Per-node execution record — same node never runs twice unintentionally."""

    __tablename__ = "workflow_node_executions"
    __table_args__ = (
        Index("ix_wf_node_exec_execution", "workflow_execution_id"),
        Index("ix_wf_node_exec_node", "workflow_id", "node_id"),
        UniqueConstraint("workflow_execution_id", "node_id", "attempt", name="uq_wf_node_attempt"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_execution_id = Column(String(64), nullable=False)
    workflow_id = Column(String(64), nullable=False)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    node_id = Column(String(64), nullable=False)
    attempt = Column(Integer, nullable=False, default=1)
    input_hash = Column(String(64), nullable=True)
    output_reference = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default="PENDING")  # PENDING/RUNNING/COMPLETED/FAILED/SKIPPED
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class WorkflowCompensation(Base):
    """Compensation metadata for workflow side effects (reversible vs not)."""

    __tablename__ = "workflow_compensations"
    __table_args__ = (
        Index("ix_wf_comp_execution", "workflow_execution_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_execution_id = Column(String(64), nullable=False)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    node_id = Column(String(64), nullable=False)
    side_effect_type = Column(String(50), nullable=False)  # notification/email/webhook/metadata_update/delete
    reversible = Column(Boolean, nullable=False, default=False)
    status = Column(String(20), nullable=False, default="RECORDED")  # RECORDED/COMPENSATED/FAILED
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Provider health + AI quality
# ---------------------------------------------------------------------------

PROVIDER_CIRCUIT_STATES = ("CLOSED", "OPEN", "HALF_OPEN")


class ProviderHealth(Base):
    """Per provider/model health + circuit breaker state."""

    __tablename__ = "provider_health"
    __table_args__ = (
        Index("ix_provider_health_provider", "provider"),
        UniqueConstraint("provider", "model", name="uq_provider_model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False, default="UNKNOWN")  # UNKNOWN/UP/DEGRADED/DOWN
    circuit_state = Column(String(20), nullable=False, default="CLOSED")
    consecutive_failures = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
    avg_latency_ms = Column(Float, nullable=True)
    last_error = Column(Text, nullable=True)
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class AIQualityMetric(Base):
    """Aggregated AI quality/cost/latency metric (tenant scoped)."""

    __tablename__ = "ai_quality_metrics"
    __table_args__ = (
        Index("ix_ai_quality_workspace", "workspace_id"),
        Index("ix_ai_quality_type", "metric_type"),
        Index("ix_ai_quality_model", "model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    metric_type = Column(String(50), nullable=False)  # grounding_rate/citation_correctness/failure_rate/latency_ms/cost_usd
    feature = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)
    provider = Column(String(100), nullable=True)
    value = Column(Float, nullable=False, default=0.0)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class AIStreamEvent(Base):
    """Persisted reconnectable execution stream events (sequence per execution)."""

    __tablename__ = "ai_stream_events"
    __table_args__ = (
        UniqueConstraint("execution_id", "seq", name="uq_stream_seq"),
        Index("ix_ai_stream_execution", "execution_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(String(36), ForeignKey("ai_executions.id", ondelete="CASCADE"), nullable=False)
    seq = Column(Integer, nullable=False)
    event_type = Column(String(30), nullable=False)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class KnowledgeGap(Base):
    """Detected knowledge gap with type/severity/evidence/remediation."""

    __tablename__ = "knowledge_gaps"
    __table_args__ = (
        Index("ix_knowledge_gaps_workspace", "workspace_id"),
        Index("ix_knowledge_gaps_type", "gap_type"),
        Index("ix_knowledge_gaps_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True)
    gap_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False, default="MEDIUM")
    title = Column(String(255), nullable=False)
    detail = Column(Text, nullable=True)
    evidence_json = Column(Text, nullable=True)
    remediation = Column(Text, nullable=True)
    document_id = Column(Integer, nullable=True)
    entity_id = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, default="OPEN")  # OPEN/ACKNOWLEDGED/RESOLVED/DISMISSED
    created_at = Column(DateTime(timezone=True), default=_utcnow)


class RequestDedup(Base):
    """Deterministic request fingerprint dedup for expensive operations."""

    __tablename__ = "request_dedup"
    __table_args__ = (
        UniqueConstraint("workspace_id", "kind", "fingerprint", name="uq_request_dedup"),
        Index("ix_request_dedup_expires", "expires_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(50), nullable=False)  # summarize/embed/classify/research
    fingerprint = Column(String(64), nullable=False)
    result_reference = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default="PENDING")
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)


def _json_dumps(value) -> str:
    return json.dumps(value, default=str) if value is not None else None