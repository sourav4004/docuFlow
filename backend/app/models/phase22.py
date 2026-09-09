"""Phase 22 models — Real-World Production AI Cloud + Continuous Operations.

Phase 22 closes the gap between simulated/fallback architecture and real
production execution. These models ADD only what Phases 0-21 do not already
cover (extension over duplication):

- infrastructure capability registry (REAL/UNAVAILABLE/DEGRADED/...)
- real provider validation runs (contract + failure modes + cost reconciliation)
- vector production: coverage snapshots, drift, benchmarks (pgvector or fallback)
- durable evaluation executions (idempotent, checkpointed, cancellable)
- improvement gate runs (evaluation/simulation/safety/cost/latency/autonomy)
- knowledge maintenance runs + ingestion governor/quarantine events
- connector sync checkpoints/conflicts/backoff
- region capacity snapshots + backup/restore drills (RPO/RTO measurement)
- SLO burn-rate events, retention executions, cost reconciliation runs
- worker runtime events (heartbeat/lease/backpressure/shedding/drain)
- DB growth/partition audits, API platform audits, security scan runs
- self-healing loop runs + autonomous AI operating loop runs

Every table follows Phases 0-21 conventions: tenant scoping, no secrets,
bounded text, indexes on ops-console query paths.
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
# Infrastructure capability registry (Steps 1-4)
# ---------------------------------------------------------------------------

class InfraCapability(Base):
    """One detected infrastructure capability and its honest state."""

    __tablename__ = "infra_capabilities"
    __table_args__ = (
        UniqueConstraint("component", name="uq_infra_capability_component"),
        Index("ix_infra_cap_state", "state", "component"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    component = Column(String(48), nullable=False)
    # postgres|pgvector|redis|object_storage|provider|smtp|webhook|container
    state = Column(String(20), nullable=False, default="UNKNOWN")
    # AVAILABLE|UNAVAILABLE|DEGRADED|NOT_CONFIGURED|UNKNOWN
    realization = Column(String(20), nullable=False, default="SIMULATED")
    # REAL | SIMULATED | UNAVAILABLE — never claim REAL without a live check
    detail = Column(Text, nullable=True)          # bounded, never credentials
    version = Column(String(64), nullable=True)
    checked_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Real provider validation (Steps 23-37)
# ---------------------------------------------------------------------------

class ProviderValidationRun(Base):
    """One provider contract/failure-mode validation execution.

    ``simulated=True`` rows used deterministic fake providers; REAL rows only
    exist when credentials were configured and the check actually ran.
    """

    __tablename__ = "provider_validation_runs"
    __table_args__ = (
        Index("ix_prov_val_provider", "provider", "kind", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(48), nullable=False)
    kind = Column(String(40), nullable=False)
    # completion|streaming|embedding|structured|tool_calling|multimodal|
    # rate_limit|timeout|server_error|malformed|fallback|circuit|health|cost
    simulated = Column(Boolean, nullable=False, default=True)
    passed = Column(Boolean, nullable=False, default=False)
    latency_ms = Column(Float, nullable=True)
    detail = Column(Text, nullable=True)          # bounded; never secrets
    metrics = Column(Text, nullable=True)         # bounded JSON
    workspace_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


class CostReconciliationRun(Base):
    """Provider-reported usage vs local usage ledger reconciliation."""

    __tablename__ = "cost_reconciliation_runs"
    __table_args__ = (
        Index("ix_cost_recon_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    provider = Column(String(48), nullable=False, default="fake")
    local_usage = Column(Float, nullable=False, default=0.0)
    provider_usage = Column(Float, nullable=False, default=0.0)
    delta = Column(Float, nullable=False, default=0.0)
    delta_pct = Column(Float, nullable=False, default=0.0)
    reconciled = Column(Boolean, nullable=False, default=False)
    simulated = Column(Boolean, nullable=False, default=True)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Vector production platform (Steps 11-22)
# NOTE: coverage snapshots reuse the Phase 19 ``VectorCoverageSnapshot``
# (vector_coverage_snapshots) — no duplicate table is created.
# ---------------------------------------------------------------------------

class VectorDriftSnapshot(Base):
    """Embedding model/version drift across the corpus."""

    __tablename__ = "vector_drift_snapshots"
    __table_args__ = (
        Index("ix_vec_drift_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    active_model = Column(String(120), nullable=False, default="")
    active_dimensions = Column(Integer, nullable=False, default=0)
    versions = Column(Text, nullable=True)         # bounded JSON {model: count}
    drifted_chunks = Column(Integer, nullable=False, default=0)
    drift_pct = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


class VectorBenchmarkRun(Base):
    """Native (pgvector) or simulated vector retrieval benchmark."""

    __tablename__ = "vector_benchmark_runs"
    __table_args__ = (
        Index("ix_vec_bench_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    backend = Column(String(20), nullable=False, default="json")
    native = Column(Boolean, nullable=False, default=False)
    queries = Column(Integer, nullable=False, default=0)
    p50_ms = Column(Float, nullable=True)
    p95_ms = Column(Float, nullable=True)
    recall_at_10 = Column(Float, nullable=True)
    passed = Column(Boolean, nullable=False, default=False)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Durable continuous evaluation (Steps 38-47)
# ---------------------------------------------------------------------------

class EvalExecution(Base):
    """One worker-executed evaluation with idempotency + checkpoint."""

    __tablename__ = "eval_executions"
    __table_args__ = (
        UniqueConstraint("idempotency_key",
                         name="uq_eval_execution_idem"),
        Index("ix_eval_exec_sched", "schedule_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    schedule_id = Column(Integer, nullable=True, index=True)
    domain = Column(String(32), nullable=False, default="retrieval")
    dataset_id = Column(Integer, nullable=True)
    dataset_version = Column(String(32), nullable=False, default="v1")
    idempotency_key = Column(String(120), nullable=False)
    status = Column(String(20), nullable=False, default="QUEUED")
    # QUEUED|RUNNING|CHECKPOINTED|COMPLETED|FAILED|CANCELLED
    attempt = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    checkpoint = Column(Text, nullable=True)       # bounded JSON
    metrics = Column(Text, nullable=True)          # bounded JSON
    cancelled = Column(Boolean, nullable=False, default=False)
    regression_detected = Column(Boolean, nullable=False, default=False)
    job_id = Column(Integer, nullable=True, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Continuous improvement loop gates (Steps 48-58)
# ---------------------------------------------------------------------------

class ImprovementGateRun(Base):
    """Gates applied to one improvement proposal before promotion."""

    __tablename__ = "improvement_gate_runs"
    __table_args__ = (
        Index("ix_improv_gates_proposal", "proposal_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    proposal_id = Column(Integer, nullable=False, index=True)
    gate = Column(String(24), nullable=False)
    # evaluation|simulation|safety|cost|latency|autonomy
    passed = Column(Boolean, nullable=False, default=False)
    decision = Column(String(20), nullable=False, default="PENDING")
    # PENDING|PASS|FAIL|SKIPPED
    evidence = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Knowledge maintenance + ingestion hardening (Steps 59-74)
# ---------------------------------------------------------------------------

class MaintenanceRun(Base):
    """One scheduled knowledge-maintenance execution."""

    __tablename__ = "maintenance_runs_p22"
    __table_args__ = (
        Index("ix_maint_runs_p22", "workspace_id", "kind", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    kind = Column(String(32), nullable=False)
    # stale_documents|embeddings|graph|memory|summaries|connectors
    findings = Column(Text, nullable=True)         # bounded JSON
    auto_repaired = Column(Integer, nullable=False, default=0)
    proposals_created = Column(Integer, nullable=False, default=0)
    status = Column(String(20), nullable=False, default="COMPLETED")
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


class IngestionGovernorEvent(Base):
    """Resource-governor enforcement during ingestion."""

    __tablename__ = "ingestion_governor_events"
    __table_args__ = (
        Index("ix_ing_gov_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    document_id = Column(Integer, nullable=True, index=True)
    limit_kind = Column(String(32), nullable=False)
    # file_size|pages|ocr|memory|cpu|execution_time|batch
    limit_value = Column(Float, nullable=True)
    observed_value = Column(Float, nullable=True)
    action = Column(String(20), nullable=False, default="REJECTED")
    # REJECTED|DEFERRED|TRUNCATED|ALLOWED
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


class PoisonQuarantine(Base):
    """Documents repeatedly failing ingestion are quarantined, not retried."""

    __tablename__ = "poison_quarantines"
    __table_args__ = (
        UniqueConstraint("workspace_id", "document_id",
                         name="uq_poison_doc"),
        Index("ix_poison_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    document_id = Column(Integer, nullable=False)
    failure_count = Column(Integer, nullable=False, default=0)
    last_error_class = Column(String(48), nullable=True)
    status = Column(String(20), nullable=False, default="QUARANTINED")
    # QUARANTINED|RELEASED|DISCARDED
    released_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow)


# ---------------------------------------------------------------------------
# Connector production platform (Steps 75-80)
# ---------------------------------------------------------------------------

class ConnectorSyncState(Base):
    """Per-connector incremental sync state: checkpoint + idempotency."""

    __tablename__ = "connector_sync_states"
    __table_args__ = (
        UniqueConstraint("workspace_id", "connector_id",
                         name="uq_connector_sync_ws"),
        Index("ix_conn_sync_health", "health", "workspace_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    connector_id = Column(Integer, nullable=False, index=True)
    checkpoint = Column(Text, nullable=True)       # bounded JSON cursor
    last_sync_at = Column(DateTime(timezone=True), nullable=True)
    sync_age_s = Column(Integer, nullable=True)
    item_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Float, nullable=True)
    health = Column(String(20), nullable=False, default="UNKNOWN")
    # HEALTHY|DEGRADED|UNHEALTHY|UNKNOWN
    backoff_seconds = Column(Integer, nullable=False, default=0)
    conflicts = Column(Integer, nullable=False, default=0)
    last_error_class = Column(String(48), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow)


# ---------------------------------------------------------------------------
# Multi-region production + DR (Steps 81-95)
# ---------------------------------------------------------------------------

class RegionCapacitySnapshot(Base):
    """Observed capacity for one region."""

    __tablename__ = "region_capacity_snapshots"
    __table_args__ = (
        Index("ix_region_cap_region", "region", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    region = Column(String(32), nullable=False, index=True)
    workers = Column(Integer, nullable=False, default=0)
    queue_depth = Column(Integer, nullable=False, default=0)
    db_healthy = Column(Boolean, nullable=False, default=False)
    provider_healthy = Column(Boolean, nullable=False, default=False)
    simulated = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


class BackupRestoreDrill(Base):
    """Restore validation drill with measured RPO/RTO or honest simulation."""

    __tablename__ = "backup_restore_drills"
    __table_args__ = (
        Index("ix_drill_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    simulated = Column(Boolean, nullable=False, default=True)
    passed = Column(Boolean, nullable=False, default=False)
    tenant_isolation_ok = Column(Boolean, nullable=False, default=False)
    measured_rpo_s = Column(Integer, nullable=True)   # REAL only
    measured_rto_s = Column(Integer, nullable=True)   # REAL only
    target_rpo_s = Column(Integer, nullable=True)
    target_rto_s = Column(Integer, nullable=True)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# SLO platform 2.0 (Steps 103-107)
# ---------------------------------------------------------------------------

class SloBurnEvent(Base):
    """Burn-rate threshold crossing for one SLO."""

    __tablename__ = "slo_burn_events"
    __table_args__ = (
        Index("ix_slo_burn_slo", "slo_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    slo_id = Column(Integer, nullable=False, index=True)
    domain = Column(String(32), nullable=False, default="api")
    burn_rate = Column(Float, nullable=False, default=0.0)
    threshold = Column(Float, nullable=False, default=1.0)
    error_budget_remaining = Column(Float, nullable=False, default=1.0)
    incident_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Data lifecycle (Steps 123-129)
# ---------------------------------------------------------------------------

class RetentionExecution(Base):
    """One retention job execution (always dry-run capable)."""

    __tablename__ = "retention_executions"
    __table_args__ = (
        Index("ix_retention_exec", "kind", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=True, index=True)
    kind = Column(String(32), nullable=False)
    # artifacts|traces|evaluations|events|usage|temporary
    dry_run = Column(Boolean, nullable=False, default=True)
    candidates = Column(Integer, nullable=False, default=0)
    deleted = Column(Integer, nullable=False, default=0)
    held = Column(Integer, nullable=False, default=0)   # legal-hold protected
    status = Column(String(20), nullable=False, default="COMPLETED")
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Worker production runtime (Steps 136-144)
# ---------------------------------------------------------------------------

class WorkerRuntimeEvent(Base):
    """Backpressure / load-shedding / drain / heartbeat runtime events."""

    __tablename__ = "worker_runtime_events"
    __table_args__ = (
        Index("ix_worker_rt_kind", "kind", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=True, index=True)
    worker_id = Column(String(64), nullable=True, index=True)
    kind = Column(String(32), nullable=False)
    # heartbeat|lease_recovered|backpressure|load_shed|drain_start|
    # drain_complete|graceful_shutdown|fairness_defer
    detail = Column(Text, nullable=True)           # bounded JSON
    depth = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Database + API platform audits (Steps 145-160)
# ---------------------------------------------------------------------------

class DbGrowthEstimate(Base):
    """Estimated growth + partition readiness for large tables."""

    __tablename__ = "db_growth_estimates"
    __table_args__ = (
        UniqueConstraint("table_name", name="uq_db_growth_table"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(64), nullable=False)
    estimated_rows = Column(Integer, nullable=False, default=0)
    est_bytes = Column(Float, nullable=False, default=0.0)
    growth_per_day = Column(Float, nullable=False, default=0.0)
    partition_recommended = Column(Boolean, nullable=False, default=False)
    recommendation = Column(Text, nullable=True)
    simulated = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


class ApiPlatformAudit(Base):
    """One API platform audit result (versioning/idempotency/errors/...)."""

    __tablename__ = "api_platform_audits"
    __table_args__ = (
        Index("ix_api_audit_kind", "kind", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(32), nullable=False)
    # versioning|idempotency|rate_limits|abuse|error_contract|
    # pagination|request_ids
    passed = Column(Boolean, nullable=False, default=False)
    checked = Column(Integer, nullable=False, default=0)
    violations = Column(Integer, nullable=False, default=0)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Security operations (Steps 115-122)
# ---------------------------------------------------------------------------

class SecurityScanRun(Base):
    """One continuous security scan execution over a named corpus."""

    __tablename__ = "security_scan_runs"
    __table_args__ = (
        Index("ix_sec_scan_corpus", "corpus", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    corpus = Column(String(40), nullable=False)
    # prompt_injection|exfiltration|tool_abuse|ssrf|tenant_matrix|
    # api_abuse|autonomy_bypass
    cases = Column(Integer, nullable=False, default=0)
    blocked = Column(Integer, nullable=False, default=0)
    leaked = Column(Integer, nullable=False, default=0)
    passed = Column(Boolean, nullable=False, default=False)
    findings = Column(Text, nullable=True)         # bounded JSON
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Production operating loops (Steps 175-191)
# ---------------------------------------------------------------------------

class SelfHealLoopRun(Base):
    """detection -> diagnosis -> recovery decision -> recovery -> validation
    -> escalation -> learning, one auditable row per loop."""

    __tablename__ = "selfheal_loop_runs"
    __table_args__ = (
        Index("ix_selfheal_loop_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    trigger = Column(String(64), nullable=False)
    stage = Column(String(24), nullable=False, default="DETECTION")
    # DETECTION|DIAGNOSIS|DECISION|RECOVERY|VALIDATION|ESCALATION|LEARNING|DONE
    detected = Column(Boolean, nullable=False, default=False)
    diagnosis_id = Column(Integer, nullable=True)
    decision = Column(String(24), nullable=True)   # AUTO|PROPOSAL|MANUAL
    recovered = Column(Boolean, nullable=False, default=False)
    validated = Column(Boolean, nullable=False, default=False)
    escalated = Column(Boolean, nullable=False, default=False)
    learned = Column(Boolean, nullable=False, default=False)
    timeline = Column(Text, nullable=True)         # bounded JSON
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


class AutonomyLoopRun(Base):
    """observe -> detect -> propose -> evaluate -> simulate -> govern ->
    approve -> activate -> monitor -> rollback, one row per loop."""

    __tablename__ = "autonomy_loop_runs"
    __table_args__ = (
        Index("ix_autonomy_loop_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    domain = Column(String(32), nullable=False, default="retrieval")
    stage = Column(String(24), nullable=False, default="OBSERVE")
    deviation_detected = Column(Boolean, nullable=False, default=False)
    proposal_id = Column(Integer, nullable=True, index=True)
    evaluated = Column(Boolean, nullable=False, default=False)
    simulated = Column(Boolean, nullable=False, default=False)
    governance = Column(String(24), nullable=True)  # ALLOWED|APPROVAL|BLOCKED
    approved = Column(Boolean, nullable=False, default=False)
    activated = Column(Boolean, nullable=False, default=False)
    monitored = Column(Boolean, nullable=False, default=False)
    rolled_back = Column(Boolean, nullable=False, default=False)
    timeline = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Real-time operations streams (Steps 170-174)
# ---------------------------------------------------------------------------

class OpsStreamEvent(Base):
    """Durable stream event for execution/incident/worker/provider feeds.

    Consumers track a monotonic cursor for safe reconnection; ``stream`` +
    ``seq`` are the reconnect contract. Dedup keys prevent duplicate side
    effects on replay.
    """

    __tablename__ = "ops_stream_events"
    __table_args__ = (
        UniqueConstraint("stream", "seq", name="uq_ops_stream_seq"),
        UniqueConstraint("dedup_key", name="uq_ops_stream_dedup"),
        Index("ix_ops_stream_stream", "stream", "seq"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    stream = Column(String(32), nullable=False)
    # execution|incident|worker|provider
    seq = Column(Integer, nullable=False)
    workspace_id = Column(Integer, nullable=True, index=True)
    kind = Column(String(32), nullable=False)
    payload = Column(Text, nullable=True)          # bounded JSON, PII-redacted
    dedup_key = Column(String(120), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Chaos / load / soak (Steps 192-208) — persisted in chaos_test_runs (P21)
# via services; no new table needed.
# ---------------------------------------------------------------------------
