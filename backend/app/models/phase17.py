"""Phase 17 models — Distributed AI + Enterprise Knowledge Cloud.

Tables cover: explicit job leasing, durable multi-stage ingestion, document
fingerprints, knowledge connectors/sync state, entity resolution candidates,
relationship suggestions, memory supersession, agent plan graphs + human
handoffs, workflow-run orchestration state, alert rules/events, AI policy
rules (governance), legal holds, vector backfill runs, cost anomalies, and
API-key lifecycle events.

All tenant scoping mirrors the Phase 0-16 conventions (workspace_id always
required where data is workspace-scoped; organization_id optional but never
used to bypass a workspace boundary).
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Float,
    ForeignKey, Index, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from ..core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Distributed worker platform — explicit job leases
# ---------------------------------------------------------------------------

class JobLease(Base):
    """Explicit lease row for a claimed job (distributed lease ownership).

    The claim UPDATE on worker_jobs is authoritative; the lease row gives a
    second owner/heartbeat layer so a crashed worker's lease can be recovered
    without executing the same non-idempotent side effect twice.
    """

    __tablename__ = "job_leases"
    __table_args__ = (
        Index("ix_job_leases_expiry", "expires_at"),
        Index("ix_job_leases_worker", "worker_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("worker_jobs.id", ondelete="CASCADE"),
                    nullable=False, unique=True)
    worker_id = Column(String(64), nullable=False)
    lease_token = Column(String(64), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    heartbeat_at = Column(DateTime(timezone=True), nullable=False,
                          default=_utcnow)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Durable ingestion pipeline 2.0
# ---------------------------------------------------------------------------

class IngestionRun(Base):
    """A durable, resumable ingestion run over one or more documents."""

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        Index("ix_ingestion_runs_workspace_status", "workspace_id", "status"),
        Index("ix_ingestion_runs_document", "document_id"),
        Index("ix_ingestion_runs_idem", "workspace_id", "idempotency_key"),
        UniqueConstraint("workspace_id", "idempotency_key",
                         name="uq_ingestion_run_idem"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"),
                     nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id",
                                             ondelete="SET NULL"),
                         nullable=True)
    batch_json = Column(Text, nullable=True)  # extra document ids for batches
    current_stage = Column(String(20), nullable=False, default="UPLOAD")
    status = Column(String(20), nullable=False, default="RUNNING")
    progress_pct = Column(Integer, nullable=False, default=0)
    current_page = Column(Integer, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    error = Column(String(1000), nullable=True)
    idempotency_key = Column(String(120), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class IngestionStage(Base):
    """Per-stage durable state with its own idempotency key."""

    __tablename__ = "ingestion_stages"
    __table_args__ = (
        Index("ix_ingestion_stages_run", "run_id", "stage"),
        UniqueConstraint("run_id", "stage", name="uq_ingestion_stage"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("ingestion_runs.id",
                                        ondelete="CASCADE"), nullable=False)
    stage = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default="PENDING")
    attempts = Column(Integer, nullable=False, default=0)
    idempotency_key = Column(String(120), nullable=True)
    output_reference = Column(String(255), nullable=True)
    error = Column(String(1000), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Document fingerprints + duplicate classification
# ---------------------------------------------------------------------------

class DocumentFingerprint(Base):
    __tablename__ = "document_fingerprints"
    __table_args__ = (
        Index("ix_fingerprints_content", "content_hash"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id",
                                             ondelete="CASCADE"),
                         nullable=False, unique=True)
    version_id = Column(Integer, nullable=True)
    content_hash = Column(String(64), nullable=False)
    structural_hash = Column(String(64), nullable=False)
    metadata_hash = Column(String(64), nullable=False)
    shingles_json = Column(Text, nullable=True)  # sampled shingle hashes
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class DuplicateCandidate(Base):
    """Persisted duplicate classification (never auto-deletes anything)."""

    __tablename__ = "duplicate_candidates"
    __table_args__ = (
        Index("ix_duplicate_candidates_doc", "document_id"),
        UniqueConstraint("document_id", "other_document_id",
                         name="uq_duplicate_pair"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id",
                                             ondelete="CASCADE"), nullable=False)
    other_document_id = Column(Integer, ForeignKey("documents.id",
                                                   ondelete="CASCADE"),
                               nullable=False)
    classification = Column(String(30), nullable=False)  # EXACT/NEAR/VERSION/RELATED/UNRELATED
    similarity = Column(Float, nullable=False, default=0.0)
    method = Column(String(40), nullable=False, default="fingerprint")
    reviewed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Knowledge federation — connectors
# ---------------------------------------------------------------------------

class ConnectorSource(Base):
    """External knowledge source. Credentials are never stored here — only a
    reference (``credential_ref``) resolved by a configured secret backend."""

    __tablename__ = "connector_sources"
    __table_args__ = (
        Index("ix_connector_sources_workspace", "workspace_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    name = Column(String(120), nullable=False)
    kind = Column(String(40), nullable=False)  # uploaded, connector stub
    config_ref = Column(String(255), nullable=True)
    scopes_json = Column(Text, nullable=True)
    permissions_json = Column(Text, nullable=True)
    allowed_domains_json = Column(Text, nullable=True)
    credential_ref = Column(String(255), nullable=True)
    retention_days = Column(Integer, nullable=True)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    syncs = relationship("ConnectorSync", back_populates="source",
                         cascade="all, delete-orphan")


class ConnectorSync(Base):
    """Incremental sync state — cursor, counts, errors. Idempotent by design:
    items are matched on external_id + content hash, never blindly inserted."""

    __tablename__ = "connector_syncs"
    __table_args__ = (
        Index("ix_connector_syncs_source", "source_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey("connector_sources.id",
                                           ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False, default="RUNNING")
    cursor_json = Column(Text, nullable=True)
    items_added = Column(Integer, nullable=False, default=0)
    items_changed = Column(Integer, nullable=False, default=0)
    items_deleted = Column(Integer, nullable=False, default=0)
    error = Column(String(1000), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    source = relationship("ConnectorSource", back_populates="syncs")


class ConnectorItem(Base):
    """Federated item ingested from a connector (deduplicated by external_id)."""

    __tablename__ = "connector_items"
    __table_args__ = (
        Index("ix_connector_items_source", "source_id", "external_id"),
        UniqueConstraint("source_id", "external_id",
                         name="uq_connector_item_ext"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey("connector_sources.id",
                                           ondelete="CASCADE"), nullable=False)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    external_id = Column(String(255), nullable=False)
    external_parent_id = Column(String(255), nullable=True)
    item_type = Column(String(40), nullable=False, default="document")
    title = Column(String(500), nullable=True)
    content_hash = Column(String(64), nullable=True)
    payload_ref = Column(String(255), nullable=True)
    deleted = Column(Boolean, nullable=False, default=False)
    first_seen_at = Column(DateTime(timezone=True), nullable=False,
                           default=_utcnow)
    last_seen_at = Column(DateTime(timezone=True), nullable=False,
                          default=_utcnow)


# ---------------------------------------------------------------------------
# Knowledge graph 4.0 — resolution + suggestions
# ---------------------------------------------------------------------------

class EntityCandidate(Base):
    """Deterministic entity-resolution candidate. Never auto-merged."""

    __tablename__ = "entity_candidates"
    __table_args__ = (
        Index("ix_entity_candidates_entity", "entity_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"),
                       nullable=False)
    candidate_id = Column(Integer,
                          ForeignKey("entities.id", ondelete="CASCADE"),
                          nullable=False)
    score = Column(Float, nullable=False, default=0.0)
    method = Column(String(40), nullable=False)  # normalized_name, alias, shared_evidence
    reason = Column(String(500), nullable=True)
    status = Column(String(20), nullable=False, default="PENDING")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RelationshipSuggestion(Base):
    """AI/deterministic relationship suggestion requiring validation."""

    __tablename__ = "relationship_suggestions"
    __table_args__ = (
        Index("ix_rel_suggestions_entity_a", "entity_a_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    entity_a_id = Column(Integer, ForeignKey("entities.id",
                                             ondelete="CASCADE"), nullable=False)
    entity_b_id = Column(Integer, ForeignKey("entities.id",
                                             ondelete="CASCADE"), nullable=False)
    relation_type = Column(String(60), nullable=False)
    confidence = Column(Float, nullable=False, default=0.0)
    evidence = Column(String(1000), nullable=True)
    method = Column(String(40), nullable=False, default="shared_evidence")
    status = Column(String(20), nullable=False, default="PENDING")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Organizational memory 3.0
# ---------------------------------------------------------------------------

class MemorySupersession(Base):
    """Explicit 'old fact replaced by new fact' record (both preserved)."""

    __tablename__ = "memory_supersessions"
    __table_args__ = (
        Index("ix_memory_supersessions_old", "old_memory_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    old_memory_id = Column(Integer, ForeignKey("ai_memories.id",
                                               ondelete="CASCADE"),
                           nullable=False)
    new_memory_id = Column(Integer, ForeignKey("ai_memories.id",
                                               ondelete="CASCADE"),
                           nullable=False)
    reason = Column(String(500), nullable=True)
    created_by_user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Agent platform 3.0
# ---------------------------------------------------------------------------

class AgentPlan(Base):
    """Validated agent plan graph (DAG of steps)."""

    __tablename__ = "agent_plans"
    __table_args__ = (
        Index("ix_agent_plans_execution", "execution_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    execution_id = Column(String(36), ForeignKey("ai_executions.id",
                                                 ondelete="CASCADE"),
                          nullable=False)
    objective = Column(String(1000), nullable=False)
    plan_json = Column(Text, nullable=False)  # {"steps":[{id,deps,tool,...}]}
    risk = Column(String(20), nullable=False, default="LOW")
    budgets_json = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="VALIDATED")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class HumanHandoff(Base):
    """Agent pause + human input request."""

    __tablename__ = "human_handoffs"
    __table_args__ = (
        Index("ix_human_handoffs_execution", "execution_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    execution_id = Column(String(36), ForeignKey("ai_executions.id",
                                                 ondelete="CASCADE"),
                          nullable=False)
    question = Column(String(1000), nullable=False)
    context_ref = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default="PENDING")
    answer = Column(Text, nullable=True)
    answered_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    answered_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Workflow orchestration 3.0
# ---------------------------------------------------------------------------

class WorkflowRun(Base):
    """Run-level durable workflow orchestration state (pause/resume/timeout)."""

    __tablename__ = "workflow_runs"
    __table_args__ = (
        Index("ix_workflow_runs_workspace_status", "workspace_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    workflow_version_id = Column(Integer, nullable=True)
    definition_hash = Column(String(64), nullable=False)
    definition_json = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="RUNNING")
    control_state = Column(String(20), nullable=False, default="ACTIVE")
    # node scheduling state for resume
    node_index = Column(Integer, nullable=False, default=0)
    node_state_json = Column(Text, nullable=True)
    timeout_at = Column(DateTime(timezone=True), nullable=True)
    idle_timeout_at = Column(DateTime(timezone=True), nullable=True)
    pause_reason = Column(String(500), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(String(1000), nullable=True)


# ---------------------------------------------------------------------------
# Enterprise governance — AI policy rules, retention, legal holds
# ---------------------------------------------------------------------------

class AIPolicyRule(Base):
    """Hierarchical policy rule (org > workspace > feature > execution scope).
    Evaluation picks the most restrictive applicable rule."""

    __tablename__ = "ai_policy_rules"
    __table_args__ = (
        Index("ix_ai_policy_rules_org", "organization_id"),
        Index("ix_ai_policy_rules_workspace", "workspace_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=True)
    feature = Column(String(60), nullable=True)  # None => all features
    rule_type = Column(String(30), nullable=False)  # MODEL / TOOL / SENSITIVITY / BUDGET
    allowlist_json = Column(Text, nullable=True)
    deny_json = Column(Text, nullable=True)
    sensitivity_max = Column(String(20), nullable=True)
    budget_max_usd = Column(Float, nullable=True)
    enabled = Column(Boolean, nullable=False, default=True)
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RetentionAssignment(Base):
    """Retention policy application to an entity type + scope."""

    __tablename__ = "retention_assignments"
    __table_args__ = (
        Index("ix_retention_assignments_scope", "workspace_id", "entity_type"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=True)
    entity_type = Column(String(40), nullable=False)
    retention_days = Column(Integer, nullable=False, default=90)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class LegalHold(Base):
    """Non-destructive legal hold. Data under hold is never auto-deleted."""

    __tablename__ = "legal_holds"
    __table_args__ = (
        Index("ix_legal_holds_workspace", "workspace_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=False)
    name = Column(String(255), nullable=False)
    reason = Column(String(1000), nullable=True)
    status = Column(String(20), nullable=False, default="ACTIVE")
    started_by = Column(Integer, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    released_at = Column(DateTime(timezone=True), nullable=True)
    released_by = Column(Integer, nullable=True)


class HoldEntity(Base):
    """Entity (type+id) covered by a legal hold."""

    __tablename__ = "hold_entities"
    __table_args__ = (
        Index("ix_hold_entities_hold", "hold_id"),
        Index("ix_hold_entities_target", "entity_type", "entity_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    hold_id = Column(Integer, ForeignKey("legal_holds.id",
                                         ondelete="CASCADE"), nullable=False)
    entity_type = Column(String(40), nullable=False)
    entity_id = Column(Integer, nullable=False)


# ---------------------------------------------------------------------------
# Vector backfill + cost anomalies + alerts + API key events
# ---------------------------------------------------------------------------

class VectorBackfillRun(Base):
    __tablename__ = "vector_backfill_runs"
    __table_args__ = (
        Index("ix_vector_backfill_runs_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, nullable=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=True)
    model = Column(String(120), nullable=False)
    dimensions = Column(Integer, nullable=False)
    dry_run = Column(Boolean, nullable=False, default=True)
    status = Column(String(20), nullable=False, default="PREVIEW")
    processed = Column(Integer, nullable=False, default=0)
    total = Column(Integer, nullable=False, default=0)
    failed = Column(Integer, nullable=False, default=0)
    errors_json = Column(Text, nullable=True)
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class CostAnomaly(Base):
    __tablename__ = "cost_anomalies"
    __table_args__ = (
        Index("ix_cost_anomalies_scope", "organization_id", "workspace_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, nullable=True)
    workspace_id = Column(Integer, nullable=True)
    period = Column(String(20), nullable=False)  # e.g. 2026-09-01
    metric = Column(String(40), nullable=False)  # cost_usd, tokens, executions
    expected = Column(Float, nullable=False)
    actual = Column(Float, nullable=False)
    deviation = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False, default="MEDIUM")
    status = Column(String(20), nullable=False, default="OPEN")
    detected_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class AlertRule(Base):
    __tablename__ = "alert_rules"
    __table_args__ = (
        Index("ix_alert_rules_workspace", "workspace_id", "enabled"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, nullable=True)
    workspace_id = Column(Integer,
                          ForeignKey("workspaces.id", ondelete="CASCADE"),
                          nullable=True)
    name = Column(String(255), nullable=False)
    metric = Column(String(60), nullable=False)  # queue_depth, error_rate, ...
    operator = Column(String(10), nullable=False, default=">")  # >, >=, <
    threshold = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False, default="WARNING")
    cooldown_minutes = Column(Integer, nullable=False, default=30)
    enabled = Column(Boolean, nullable=False, default=True)
    last_fired_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class AlertEvent(Base):
    __tablename__ = "alert_events"
    __table_args__ = (
        Index("ix_alert_events_rule", "rule_id", "fired_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_id = Column(Integer, ForeignKey("alert_rules.id",
                                         ondelete="CASCADE"), nullable=True)
    workspace_id = Column(Integer, nullable=True)
    severity = Column(String(20), nullable=False, default="WARNING")
    message = Column(String(1000), nullable=False)
    metric_value = Column(Float, nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    fired_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ApiKeyEvent(Base):
    """API-key lifecycle + abuse events (rotation, revocation, suspicious use)."""

    __tablename__ = "api_key_events"
    __table_args__ = (
        Index("ix_api_key_events_key", "api_key_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id",
                                            ondelete="CASCADE"), nullable=False)
    workspace_id = Column(Integer, nullable=True)
    event_type = Column(String(30), nullable=False)  # CREATED/ROTATED/REVOKED/ABUSE
    detail = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
