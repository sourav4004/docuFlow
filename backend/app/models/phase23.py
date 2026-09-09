"""Phase 23 models — Global AI Cloud Platform.

Phase 23 closes the final deployment boundary: REAL infrastructure
activation, global control plane, multi-region data placement, production
AI provider operations, and zero-downtime platform engineering.

These models ADD only what Phases 0-22 do not already cover:

- storage migration plans/batches/objects (local -> object storage, resumable)
- provider readiness scores + routing decisions (routing 3.0 admission log)
- residency decision log (per-operation evidence, extends ResidencyRule)
- embedding model versions (immutable) + per-document migration status
- search shadow comparisons (dual-index/dual-retrieval coexistence)
- region drain operations (bounded, audited drain progress)
- dependency graph edges + component dependency impact snapshots
- knowledge freshness states (per-document freshness classification)
- data consistency check runs (bounded, repair-plan only)
- request deduplication records (idempotent request identity)
- scheduler task runs (deduplicated scheduled execution)
- webhook delivery attempts (signed, replay-protected, backoff)
- review decisions (Phase 15 ReviewItem unified human review, audited)
- retention legal holds already exist (phase17) — reused, not duplicated

Conventions from Phases 0-22: tenant scoping, no secrets, bounded text,
indexes on ops-console query paths, migrations via Alembic only.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Float,
    Index, UniqueConstraint,
)

from ..core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Storage migration (Steps 4-5): local -> object storage, resumable
# ---------------------------------------------------------------------------

class StorageMigrationPlan(Base):
    """A governed plan to migrate documents from local to object storage."""

    __tablename__ = "storage_migration_plans"
    __table_args__ = (
        Index("ix_smp_workspace_status", "workspace_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    organization_id = Column(Integer, nullable=True)
    source_backend = Column(String(30), nullable=False, default="local")
    target_backend = Column(String(30), nullable=False, default="s3")
    status = Column(String(20), nullable=False, default="DRAFT")
    # DRAFT / DRY_RUN / RUNNING / PAUSED / COMPLETED / FAILED / ROLLED_BACK
    batch_size = Column(Integer, nullable=False, default=25)
    dry_run = Column(Boolean, nullable=False, default=False)
    total_objects = Column(Integer, nullable=False, default=0)
    migrated_objects = Column(Integer, nullable=False, default=0)
    failed_objects = Column(Integer, nullable=False, default=0)
    verified_objects = Column(Integer, nullable=False, default=0)
    progress_json = Column(Text, nullable=True)   # resumability cursor
    audit_json = Column(Text, nullable=True)
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class StorageMigrationObject(Base):
    """Per-document migration state — idempotency + verification."""

    __tablename__ = "storage_migration_objects"
    __table_args__ = (
        Index("ix_smo_plan_state", "plan_id", "state"),
        UniqueConstraint("plan_id", "document_id", name="uq_smo_plan_doc"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, nullable=False, index=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    document_id = Column(Integer, nullable=False)
    state = Column(String(20), nullable=False, default="PENDING")
    # PENDING / MIGRATED / VERIFIED / FAILED / SKIPPED
    attempts = Column(Integer, nullable=False, default=0)
    last_error = Column(String(1000), nullable=True)
    source_checksum = Column(String(128), nullable=True)
    target_checksum = Column(String(128), nullable=True)
    migrated_at = Column(DateTime(timezone=True), nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Provider readiness + routing 3.0 (Steps 6-8)
# ---------------------------------------------------------------------------

class ProviderReadinessScore(Base):
    """Rolling readiness score per provider kind (validation-driven)."""

    __tablename__ = "provider_readiness_scores"
    __table_args__ = (
        Index("ix_prs_kind", "provider_kind"),
        UniqueConstraint("provider_kind", "environment", name="uq_prs_kind_env"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_kind = Column(String(40), nullable=False)
    environment = Column(String(40), nullable=False, default="default")
    score = Column(Float, nullable=False, default=0.0)      # 0..100
    validated_capabilities_json = Column(Text, nullable=True)
    failure_summary_json = Column(Text, nullable=True)
    last_validation_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ProviderRoutingDecision(Base):
    """Audit log of deterministic routing decisions (Routing 3.0)."""

    __tablename__ = "provider_routing_decisions"
    __table_args__ = (
        Index("ix_prd_workspace", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    organization_id = Column(Integer, nullable=True)
    operation = Column(String(60), nullable=False)          # completion/embedding/...
    decision = Column(String(20), nullable=False)           # ROUTED / REJECTED
    selected_provider = Column(String(60), nullable=True)
    selected_model = Column(String(120), nullable=True)
    fallback_provider = Column(String(60), nullable=True)
    reasons_json = Column(Text, nullable=True)              # concise policy reasons
    candidate_ranking_json = Column(Text, nullable=True)
    estimated_cost = Column(Float, nullable=True)
    sensitivity = Column(String(20), nullable=True)
    region = Column(String(60), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Residency decision log (Step 14) — evidence for every routing decision
# ---------------------------------------------------------------------------

class ResidencyDecisionLog(Base):
    """Per-operation residency evidence: request -> decision -> reason."""

    __tablename__ = "residency_decision_logs"
    __table_args__ = (
        Index("ix_rdl_workspace", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    organization_id = Column(Integer, nullable=True)
    operation = Column(String(60), nullable=False)
    source_region = Column(String(60), nullable=True)
    destination_region = Column(String(60), nullable=True)
    classification = Column(String(20), nullable=True)
    policy_id = Column(Integer, nullable=True)
    allowed = Column(Boolean, nullable=False, default=False)
    reason = Column(String(600), nullable=True)
    actor = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Vector model migration + search coexistence (Steps 10-12)
# ---------------------------------------------------------------------------

class EmbeddingModelVersion(Base):
    """Immutable embedding model version registry."""

    __tablename__ = "embedding_model_versions"
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_emv_name_ver"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(120), nullable=False)
    version = Column(String(60), nullable=False)
    dimension = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="ACTIVE")
    # ACTIVE / DUAL / RETIRING / RETIRED
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class DocumentEmbeddingStatus(Base):
    """Per-document embedding-model migration state (resumable)."""

    __tablename__ = "document_embedding_status"
    __table_args__ = (
        Index("des_workspace_state", "workspace_id", "state"),
        UniqueConstraint("workspace_id", "document_id", "model_version_id",
                         name="uq_des_doc_model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    document_id = Column(Integer, nullable=False)
    model_version_id = Column(Integer, nullable=False)
    state = Column(String(20), nullable=False, default="PENDING")
    # PENDING / DUAL_GENERATED / VERIFIED / PROMOTED / FAILED / SKIPPED
    chunks_total = Column(Integer, nullable=False, default=0)
    chunks_done = Column(Integer, nullable=False, default=0)
    quality_delta = Column(Float, nullable=True)
    last_error = Column(String(1000), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class SearchShadowComparison(Base):
    """Shadow comparison between old and new vector versions (Step 12)."""

    __tablename__ = "search_shadow_comparisons"
    __table_args__ = (
        Index("ssc_workspace", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    query = Column(String(500), nullable=False)
    baseline_ranking_json = Column(Text, nullable=True)
    candidate_ranking_json = Column(Text, nullable=True)
    overlap_at_5 = Column(Float, nullable=True)
    mrr_delta = Column(Float, nullable=True)
    verdict = Column(String(20), nullable=True)   # BETTER / WORSE / TIE / INSUFFICIENT
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Region drain + dependency graph (Steps 16, 19)
# ---------------------------------------------------------------------------

class RegionDrainOperation(Base):
    """Bounded, audited region drain (stop-new / finish / migrate / verify)."""

    __tablename__ = "region_drain_operations"
    __table_args__ = (
        Index("rdo_region_state", "region", "state"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    region = Column(String(60), nullable=False, index=True)
    organization_id = Column(Integer, nullable=True)
    state = Column(String(20), nullable=False, default="DRAINING")
    # DRAINING / DRAINED / RECOVERED / FAILED / ROLLED_BACK
    jobs_retried = Column(Integer, nullable=False, default=0)
    jobs_marked = Column(Integer, nullable=False, default=0)
    jobs_remaining = Column(Integer, nullable=False, default=0)
    progress_json = Column(Text, nullable=True)
    actor = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class DependencyEdge(Base):
    """Static dependency graph edge between platform components."""

    __tablename__ = "dependency_edges"
    __table_args__ = (
        UniqueConstraint("component", "depends_on", name="uq_dep_edge"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    component = Column(String(60), nullable=False, index=True)
    depends_on = Column(String(60), nullable=False, index=True)
    criticality = Column(String(20), nullable=False, default="HARD")
    # HARD = component cannot serve without it; SOFT = degraded only
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Knowledge freshness engine (Step 24)
# ---------------------------------------------------------------------------

class KnowledgeFreshnessState(Base):
    """Per-document freshness classification with explainable reasons."""

    __tablename__ = "knowledge_freshness_states"
    __table_args__ = (
        Index("kfs_workspace_state", "workspace_id", "state"),
        UniqueConstraint("workspace_id", "document_id", name="uq_kfs_doc"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    document_id = Column(Integer, nullable=False)
    state = Column(String(20), nullable=False, default="UNKNOWN")
    # FRESH / AGING / STALE / EXPIRED / UNKNOWN
    age_days = Column(Float, nullable=True)
    source_reliability = Column(Float, nullable=True)  # 0..1
    expiration_days = Column(Integer, nullable=True)
    reasons_json = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Data consistency engine (Step 43) + request dedup (Steps 44-45)
# ---------------------------------------------------------------------------

class ConsistencyCheckRun(Base):
    """Bounded consistency check run over a domain (report + repair plan)."""

    __tablename__ = "consistency_check_runs"
    __table_args__ = (
        Index("ccr_domain", "domain", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=True, index=True)
    domain = Column(String(40), nullable=False)
    # documents/chunks/embeddings/jobs/executions/workflows/artifacts/graph/
    # memory/notifications/audit/usage/cost
    status = Column(String(20), nullable=False, default="COMPLETED")
    items_checked = Column(Integer, nullable=False, default=0)
    issues_found = Column(Integer, nullable=False, default=0)
    repaired = Column(Integer, nullable=False, default=0)
    findings_json = Column(Text, nullable=True)
    repair_plan_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RequestDedupRecord(Base):
    """Idempotent request identity — result reuse within TTL."""

    __tablename__ = "request_dedup_records"
    __table_args__ = (
        UniqueConstraint("workspace_id", "idempotency_key", name="uq_rdr_key"),
        Index("ix_rdr_expires", "expires_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    idempotency_key = Column(String(200), nullable=False)
    request_hash = Column(String(128), nullable=False)
    response_status = Column(Integer, nullable=True)
    response_json = Column(Text, nullable=True)
    state = Column(String(20), nullable=False, default="IN_FLIGHT")
    # IN_FLIGHT / COMPLETED / FAILED / CONFLICT
    conflict = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Scheduler task dedup (Step 48) + webhook deliveries (Steps 50-51)
# ---------------------------------------------------------------------------

class SchedulerTaskRun(Base):
    """Deduplicated scheduled task execution record."""

    __tablename__ = "scheduler_task_runs"
    __table_args__ = (
        UniqueConstraint("task_key", "due_bucket", name="uq_str_task_bucket"),
        Index("ix_str_started", "started_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_key = Column(String(120), nullable=False)
    due_bucket = Column(String(40), nullable=False)   # e.g. 2026-09-06T12:00Z
    leader_id = Column(String(120), nullable=True)
    status = Column(String(20), nullable=False, default="RUNNING")
    # RUNNING / COMPLETED / FAILED / SKIPPED_DUPLICATE
    started_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    detail_json = Column(Text, nullable=True)


class WebhookDeliveryAttempt(Base):
    """Signed webhook delivery with backoff + endpoint health tracking."""

    __tablename__ = "webhook_delivery_attempts"
    __table_args__ = (
        Index("wda_endpoint_state", "endpoint_id", "state"),
        Index("ix_wda_created", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    endpoint_id = Column(Integer, nullable=False)
    delivery_id = Column(String(80), nullable=False)
    event_type = Column(String(80), nullable=False)
    state = Column(String(20), nullable=False, default="PENDING")
    # PENDING / DELIVERED / FAILED / DEAD / DISABLED_ENDPOINT
    attempt = Column(Integer, nullable=False, default=0)
    response_status = Column(Integer, nullable=True)
    last_error = Column(String(600), nullable=True)
    signature_ts = Column(Integer, nullable=True)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class WebhookEndpointHealth(Base):
    """Endpoint health with automatic disable after repeated failures."""

    __tablename__ = "webhook_endpoint_health"
    __table_args__ = (
        UniqueConstraint("endpoint_id", name="uq_weh_endpoint"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    endpoint_id = Column(Integer, nullable=False)
    workspace_id = Column(Integer, nullable=False, index=True)
    consecutive_failures = Column(Integer, nullable=False, default=0)
    disabled = Column(Boolean, nullable=False, default=False)
    disabled_reason = Column(String(600), nullable=True)
    last_success_at = Column(DateTime(timezone=True), nullable=True)
    last_failure_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Review decisions (Step 35) — audits Phase 15 ReviewItem lifecycle
# ---------------------------------------------------------------------------

class ReviewDecision(Base):
    """Audited decision on a Phase 15 review item."""

    __tablename__ = "review_decisions"
    __table_args__ = (
        Index("ix_revdec_item", "item_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, nullable=False, index=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    decision = Column(String(20), nullable=False)
    # APPROVE / REJECT / REQUEST_CHANGES / DELEGATE / EXPIRE
    actor_user_id = Column(Integer, nullable=True)
    delegate_to = Column(Integer, nullable=True)
    reason = Column(String(1000), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
