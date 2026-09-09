"""Phase 19 models — Enterprise AI Cloud 2.0: control plane, region
abstraction, scheduler leadership, worker quarantine/migration, vector
lifecycle + coverage, ingestion quality, connector conflicts, memory
suppression, agent dead letters, cost reservations/recommendations,
SLO definitions/burn windows, API abuse events, quality gates,
evaluation runs, backup records, and consistency reports.

All tenant scoping mirrors Phases 0–18 (workspace_id required where
workspace-scoped; organization_id never bypasses a workspace boundary).
No secrets are stored in any of these tables.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Float,
    ForeignKey, Index, UniqueConstraint,
)

from ..core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enterprise control plane
# ---------------------------------------------------------------------------

class ControlPlaneSnapshot(Base):
    """Immutable, versioned configuration snapshot with diff + rollback.

    Never silently changes production policy: activating a snapshot records
    the actor, reason and a structured diff, and every rollback is itself an
    audited, idempotent snapshot activation.
    """

    __tablename__ = "control_plane_snapshots"
    __table_args__ = (
        Index("ix_cp_snapshots_scope", "scope_type", "scope_id"),
        UniqueConstraint("scope_type", "scope_id", "version",
                         name="uq_cp_snapshot_version"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope_type = Column(String(20), nullable=False)  # GLOBAL/ORGANIZATION/WORKSPACE
    scope_id = Column(Integer, nullable=True)
    version = Column(Integer, nullable=False)
    config_json = Column(Text, nullable=False)
    diff_json = Column(Text, nullable=True)
    actor_user_id = Column(Integer, nullable=True)
    reason = Column(String(1000), nullable=True)
    is_active = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RegionRecord(Base):
    """Region/deployment abstraction (no physical multi-region required).

    Carries health, capacity, provider/vector availability and a residency
    policy so routing decisions stay deterministic and auditable.
    """

    __tablename__ = "region_records"
    __table_args__ = (
        Index("ix_region_records_org", "organization_id", "region_id"),
        UniqueConstraint("organization_id", "region_id",
                         name="uq_region_record"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    region_id = Column(String(60), nullable=False)   # e.g. us-east-1
    name = Column(String(255), nullable=True)
    deployment = Column(String(120), nullable=True)  # env label / deployment id
    status = Column(String(20), nullable=False, default="HEALTHY")
    # HEALTHY / DEGRADED / OUTAGE / MAINTENANCE / FAILED_OVER
    health_score = Column(Float, nullable=False, default=1.0)
    capabilities_json = Column(Text, nullable=True)
    capacity_json = Column(Text, nullable=True)
    provider_availability_json = Column(Text, nullable=True)
    vector_availability_json = Column(Text, nullable=True)
    residency_policy_json = Column(Text, nullable=True)
    failover_to = Column(String(60), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ResidencyRule(Base):
    """Data-residency rule per sensitivity classification."""

    __tablename__ = "residency_rules"
    __table_args__ = (
        Index("ix_residency_rules_org_class", "organization_id",
              "classification"),
        UniqueConstraint("organization_id", "classification",
                         name="uq_residency_class"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    classification = Column(String(20), nullable=False)  # PUBLIC/INTERNAL/CONFIDENTIAL/RESTRICTED
    allowed_regions_json = Column(Text, nullable=True)
    prohibited_regions_json = Column(Text, nullable=True)
    default_region = Column(String(60), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RegionFailover(Base):
    """Explicit, audited failover between regions (never automatic data loss)."""

    __tablename__ = "region_failovers"
    __table_args__ = (
        Index("ix_region_failovers_org", "organization_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    region_from = Column(String(60), nullable=False)
    region_to = Column(String(60), nullable=False)
    status = Column(String(20), nullable=False, default="INITIATED")
    # INITIATED / COMPLETED / REVERTED / FAILED
    reason = Column(String(1000), nullable=True)
    actor_user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Worker platform 2.0 / scheduler 2.0
# ---------------------------------------------------------------------------

class SchedulerLeader(Base):
    """Database-backed scheduler leadership lease (single owner at a time)."""

    __tablename__ = "scheduler_leaders"
    __table_args__ = (
        Index("ix_scheduler_leaders_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    leader_id = Column(String(64), nullable=False, unique=True)
    hostname = Column(String(255), nullable=True)
    pid = Column(Integer, nullable=True)
    version = Column(String(50), nullable=True)
    acquired_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    lease_until = Column(DateTime(timezone=True), nullable=False)
    last_heartbeat = Column(DateTime(timezone=True), nullable=False,
                            default=_utcnow)
    status = Column(String(20), nullable=False, default="LEADER")
    # LEADER / STANDBY / RELEASED / STALE


class WorkerQuarantine(Base):
    """Repeatedly failing workers can be quarantined (operator-overridable)."""

    __tablename__ = "worker_quarantines"
    __table_args__ = (
        Index("ix_worker_quarantines_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String(64), nullable=False, unique=True)
    reason = Column(String(1000), nullable=True)
    failure_count = Column(Integer, nullable=False, default=0)
    status = Column(String(20), nullable=False, default="QUARANTINED")
    # QUARANTINED / OPERATOR_OVERRIDE / AUTO_RECOVERED / RELEASED
    auto_recover_after = Column(DateTime(timezone=True), nullable=True)
    recovery_criteria_json = Column(Text, nullable=True)
    operator_user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class JobMigrationRecord(Base):
    """Audit trail for safe job migration between workers."""

    __tablename__ = "job_migrations"
    __table_args__ = (
        Index("ix_job_migrations_job", "job_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("worker_jobs.id",
                                        ondelete="CASCADE"), nullable=False)
    from_worker = Column(String(64), nullable=True)
    to_worker = Column(String(64), nullable=True)
    reason = Column(String(500), nullable=True)
    status = Column(String(20), nullable=False, default="MIGRATED")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Ingestion 3.0 quality + failure explanations
# ---------------------------------------------------------------------------

class IngestionQualityReport(Base):
    """Document-level ingestion quality indicators (never stores secrets)."""

    __tablename__ = "ingestion_quality_reports"
    __table_args__ = (
        Index("ix_ingestion_quality_doc", "document_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id",
                                              ondelete="CASCADE"),
                          nullable=False)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id",
                                             ondelete="CASCADE"),
                         nullable=True)
    run_id = Column(Integer, nullable=True)
    extraction_completeness = Column(Float, nullable=False, default=0.0)
    ocr_quality = Column(Float, nullable=True)
    metadata_completeness = Column(Float, nullable=False, default=0.0)
    chunk_quality = Column(Float, nullable=False, default=0.0)
    embedding_coverage = Column(Float, nullable=False, default=0.0)
    quality_score = Column(Float, nullable=False, default=0.0)
    detail_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Federation 2.0 — connector conflicts
# ---------------------------------------------------------------------------

class ConnectorConflict(Base):
    """Externally-detected connector conflicts — never silently overwritten."""

    __tablename__ = "connector_conflicts"
    __table_args__ = (
        Index("ix_connector_conflicts_source", "source_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id",
                                              ondelete="CASCADE"),
                          nullable=False)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    source_id = Column(Integer, ForeignKey("connector_sources.id",
                                           ondelete="CASCADE"),
                       nullable=False)
    external_id = Column(String(255), nullable=False)
    conflict_type = Column(String(40), nullable=False)
    # MODIFIED_EXTERNALLY / DELETED_RECREATED / METADATA / CONTENT
    local_ref = Column(String(500), nullable=True)
    external_ref = Column(String(500), nullable=True)
    detail = Column(String(1000), nullable=True)
    status = Column(String(20), nullable=False, default="OPEN")
    # OPEN / RESOLVED / IGNORED
    resolved_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Memory 4.0 — suppression
# ---------------------------------------------------------------------------

class MemorySuppression(Base):
    """User/workspace/organization suppression of a memory (soft, auditable)."""

    __tablename__ = "memory_suppressions"
    __table_args__ = (
        Index("ix_memory_suppressions_memory", "memory_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(Integer, ForeignKey("ai_memories.id",
                                           ondelete="CASCADE"), nullable=False)
    scope_type = Column(String(20), nullable=False)  # USER/WORKSPACE/ORGANIZATION
    suppressed_by_user_id = Column(Integer, nullable=True)
    workspace_id = Column(Integer, nullable=True)
    organization_id = Column(Integer, nullable=True)
    reason = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Agent platform 4.0 — dead letters
# ---------------------------------------------------------------------------

class AgentDeadLetter(Base):
    """Persisted failed agent execution with safe-replay eligibility."""

    __tablename__ = "agent_dead_letters"
    __table_args__ = (
        Index("ix_agent_dead_letters_execution", "execution_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id",
                                              ondelete="CASCADE"),
                          nullable=False)
    execution_id = Column(String(36), ForeignKey("ai_executions.id",
                                                 ondelete="CASCADE"),
                          nullable=False)
    reason = Column(String(1000), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    last_checkpoint_json = Column(Text, nullable=True)
    safe_replay = Column(Boolean, nullable=False, default=False)
    status = Column(String(20), nullable=False, default="OPEN")
    # OPEN / REPLAYED / RESOLVED / ABANDONED
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    resolved_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# Cost platform 4.0
# ---------------------------------------------------------------------------

class CostReservation(Base):
    """Budget reservation before an expensive AI operation."""

    __tablename__ = "cost_reservations"
    __table_args__ = (
        Index("ix_cost_reservations_scope", "organization_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id",
                                              ondelete="CASCADE"),
                          nullable=False)
    execution_ref = Column(String(64), nullable=True)
    feature = Column(String(60), nullable=False, default="ai")
    reserved_amount = Column(Float, nullable=False, default=0.0)
    released_amount = Column(Float, nullable=False, default=0.0)
    actual_cost = Column(Float, nullable=True)
    currency = Column(String(8), nullable=False, default="usd")
    status = Column(String(20), nullable=False, default="RESERVED")
    # RESERVED / RELEASED / RECONCILED / EXPIRED
    reserved_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    settled_at = Column(DateTime(timezone=True), nullable=True)


class CostRecommendation(Base):
    """Cost-optimization suggestion — never auto-applied to production policy."""

    __tablename__ = "cost_recommendations"
    __table_args__ = (
        Index("ix_cost_recommendations_scope", "organization_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer,
                             ForeignKey("organizations.id",
                                        ondelete="CASCADE"), nullable=True)
    workspace_id = Column(Integer, nullable=True)
    kind = Column(String(40), nullable=False)  # MODEL_DOWNGRADE/CACHING/BATCHING/PROMPT_REDUCTION/EMBEDDING_REUSE/WORKFLOW
    reason = Column(String(1000), nullable=False)
    savings_estimate = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="SUGGESTED")
    # SUGGESTED / DISMISSED / APPLIED_MANUALLY
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Vector platform 2.0
# ---------------------------------------------------------------------------

class EmbeddingLifecycleEvent(Base):
    """Embedding-model lifecycle transition (active/deprecated/migration/retired)."""

    __tablename__ = "embedding_lifecycle_events"
    __table_args__ = (
        Index("ix_embedding_lifecycle_model", "embedding_model_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_model_id = Column(Integer,
                                ForeignKey("embedding_models.id",
                                           ondelete="CASCADE"), nullable=False)
    lifecycle = Column(String(30), nullable=False)
    # ACTIVE / DEPRECATED / MIGRATION_REQUIRED / RETIRED
    reason = Column(String(500), nullable=True)
    decided_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class VectorCoverageSnapshot(Base):
    """Periodic embedding-coverage snapshot (total/embedded/stale/failed)."""

    __tablename__ = "vector_coverage_snapshots"
    __table_args__ = (
        Index("ix_vector_coverage_computed", "computed_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=False, default=0)
    embedded_chunks = Column(Integer, nullable=False, default=0)
    stale_chunks = Column(Integer, nullable=False, default=0)
    failed_chunks = Column(Integer, nullable=False, default=0)
    model_distribution_json = Column(Text, nullable=True)
    computed_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# SLO engine
# ---------------------------------------------------------------------------

class SloDefinition(Base):
    """Configurable SLO: metric, target, operator, window, burn threshold."""

    __tablename__ = "slo_definitions"
    __table_args__ = (
        Index("ix_slo_definitions_enabled", "enabled"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, nullable=True)
    name = Column(String(120), nullable=False)
    metric = Column(String(60), nullable=False)  # api_latency_p95/api_availability/worker_completion/provider_success/ingestion/rag/workflow
    operator = Column(String(10), nullable=False, default="<=")
    target_value = Column(Float, nullable=False)
    window_minutes = Column(Integer, nullable=False, default=60)
    burn_rate_threshold = Column(Float, nullable=False, default=2.0)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class SloBudgetWindow(Base):
    """Rolling error-budget window used for burn-rate style indicators."""

    __tablename__ = "slo_budget_windows"
    __table_args__ = (
        Index("ix_slo_budget_windows_def", "definition_id", "window_start"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    definition_id = Column(Integer, ForeignKey("slo_definitions.id",
                                               ondelete="CASCADE"),
                           nullable=False)
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    budget_ratio = Column(Float, nullable=False, default=0.0)   # consumed budget
    burn_rate = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# API platform 3.0 — abuse detection
# ---------------------------------------------------------------------------

class ApiAbuseEvent(Base):
    """API abuse signal (repeated failures, enumeration, scope violations)."""

    __tablename__ = "api_abuse_events"
    __table_args__ = (
        Index("ix_api_abuse_events_window", "kind", "window_start"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key_id = Column(Integer, nullable=True)
    user_id = Column(Integer, nullable=True)
    workspace_id = Column(Integer, nullable=True)
    kind = Column(String(40), nullable=False)
    # REPEATED_FAILURES / SUSPICIOUS_KEY / EXCESSIVE_REQUESTS / ENUMERATION / SCOPE_VIOLATION
    detail = Column(String(500), nullable=True)
    window_start = Column(DateTime(timezone=True), nullable=False,
                          default=_utcnow)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Data platform — consistency reports
# ---------------------------------------------------------------------------

class ConsistencyReport(Base):
    """Periodic consistency-check report (documents/chunks/graph/memory/cross-tenant)."""

    __tablename__ = "consistency_reports"
    __table_args__ = (
        Index("ix_consistency_reports_scope", "workspace_id", "check_kind"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=True)
    organization_id = Column(Integer, nullable=True)
    check_kind = Column(String(40), nullable=False)
    status = Column(String(20), nullable=False, default="CLEAN")
    # CLEAN / ISSUES / ERROR
    issue_count = Column(Integer, nullable=False, default=0)
    issues_json = Column(Text, nullable=True)
    dry_run = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# DR 2.0 — backup inventory
# ---------------------------------------------------------------------------

class BackupRecord(Base):
    """Tracked backup with migration head + validation status."""

    __tablename__ = "backup_records"
    __table_args__ = (
        Index("ix_backup_records_status", "status", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope = Column(String(40), nullable=False)  # DATABASE/OBJECT_STORAGE/CONFIGURATION
    backup_ref = Column(String(255), nullable=False)
    database_version = Column(String(120), nullable=True)
    migration_head = Column(String(120), nullable=True)
    checksum = Column(String(128), nullable=True)
    size_bytes = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, default="PENDING")
    # PENDING / VALIDATED / FAILED
    note = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    validated_at = Column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# AI quality platform
# ---------------------------------------------------------------------------

class QualityGate(Base):
    """Configurable quality threshold (retrieval/citations/confidence/latency/cost)."""

    __tablename__ = "quality_gates"
    __table_args__ = (
        Index("ix_quality_gates_org", "organization_id", "name"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, nullable=True)
    name = Column(String(120), nullable=False)
    metric = Column(String(60), nullable=False)
    operator = Column(String(10), nullable=False, default=">=")
    threshold = Column(Float, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class EvaluationRun(Base):
    """Offline evaluation-run result over a golden dataset."""

    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index("ix_evaluation_runs_dataset", "dataset_name", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_name = Column(String(120), nullable=False)
    mode = Column(String(20), nullable=False, default="offline")
    sample_count = Column(Integer, nullable=False, default=0)
    metric_json = Column(Text, nullable=True)
    passed = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Search 5.0 — personalization
# ---------------------------------------------------------------------------

class SearchPersonalizationProfile(Base):
    """Per-user (workspace-scoped) safe search personalization state."""

    __tablename__ = "search_personalization"
    __table_args__ = (
        Index("ix_search_personalization_user", "user_id", "workspace_id"),
        UniqueConstraint("user_id", "workspace_id",
                         name="uq_search_personalization"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False)
    workspace_id = Column(Integer, ForeignKey("workspaces.id",
                                              ondelete="CASCADE"),
                          nullable=False)
    preferences_json = Column(Text, nullable=True)
    recent_intents_json = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
