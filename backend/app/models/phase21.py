"""Phase 21 models — Autonomous Enterprise AI + Self-Healing Knowledge Cloud.

Adds the governed-autonomy substrate on top of Phase 20's self-improvement
platform: persisted autonomy policies and levels, autonomous operations with
audited decisions, recovery playbooks + attempts with idempotency/cooldowns,
system health snapshots, root-cause diagnoses, knowledge recovery, adaptive
ingestion/retrieval/RAG candidates, model/provider autopilot events, cost
forecasts/guards/optimizations, agent plan risk + dead letters, workflow risk
gates, durable platform events with dedup + replay, continuous evaluation
schedules, security center 3.0 health, data governance monitoring, region
operations (health/capacity/failover readiness), worker quarantine/recovery,
broker/scheduler/cache healing state, personal AI settings and activity feed,
artifact quality, API abuse monitoring, maintenance plans, and DR records.

Every table follows Phases 0-20 conventions: tenant scoping via workspace_id
or organization_id, no secrets stored, bounded text columns, indexes on the
query paths used by the ops console.
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
# Autonomous operations control plane
# ---------------------------------------------------------------------------

class AutonomyPolicy(Base):
    """Persisted autonomy policy for one scope (workspace or organization)."""

    __tablename__ = "autonomy_policies"
    __table_args__ = (
        Index("ix_autonomy_policies_scope", "workspace_id", "operation_type"),
        UniqueConstraint("workspace_id", "operation_type", "risk_level",
                         name="uq_autonomy_policy_scope_op_risk"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    organization_id = Column(Integer, nullable=True, index=True)
    operation_type = Column(String(64), nullable=False)   # e.g. knowledge.reprocess
    risk_level = Column(String(16), nullable=False, default="LOW")  # LOW|MEDIUM|HIGH|CRITICAL
    autonomy_level = Column(String(20), nullable=False, default="RECOMMEND")
    # OBSERVE | RECOMMEND | AUTO_LOW_RISK | AUTO_APPROVAL | MANUAL_ONLY
    requires_approval = Column(Boolean, nullable=False, default=True)
    budget_limit_usd = Column(Float, nullable=True)
    execution_limit_per_hour = Column(Integer, nullable=True)
    cooldown_seconds = Column(Integer, nullable=False, default=300)
    requires_audit = Column(Boolean, nullable=False, default=True)
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class AutonomyTransition(Base):
    """Audited change of an autonomy level/policy."""

    __tablename__ = "autonomy_transitions"
    __table_args__ = (
        Index("ix_autonomy_transitions_policy", "policy_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(Integer, nullable=False, index=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    previous_level = Column(String(20), nullable=True)
    new_level = Column(String(20), nullable=False)
    actor = Column(String(120), nullable=False, default="system")
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class AutonomousOperation(Base):
    """One autonomous (or simulated) operation with full decision/audit trail."""

    __tablename__ = "autonomous_operations"
    __table_args__ = (
        Index("ix_autonomous_operations_ws", "workspace_id", "created_at"),
        Index("ix_autonomous_operations_op", "workspace_id", "operation_type"),
        UniqueConstraint("workspace_id", "idempotency_key",
                         name="uq_autonomous_op_idem"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    operation_type = Column(String(64), nullable=False)
    risk_level = Column(String(16), nullable=False, default="LOW")
    actor = Column(String(120), nullable=False, default="system")
    source = Column(String(32), nullable=False, default="SYSTEM")  # AI|SYSTEM|OPERATOR
    policy_id = Column(Integer, nullable=True)
    policy_level = Column(String(20), nullable=True)
    input_payload = Column(Text, nullable=True)     # bounded JSON
    decision = Column(String(24), nullable=False)   # ALLOWED|REQUIRES_APPROVAL|BLOCKED
    decision_reason = Column(Text, nullable=True)
    status = Column(String(24), nullable=False, default="DECIDED")
    # DECIDED|EXECUTING|SUCCEEDED|FAILED|ROLLED_BACK|ESCALATED|CANCELLED
    result = Column(Text, nullable=True)            # bounded JSON result
    rollback_info = Column(Text, nullable=True)     # bounded JSON for reversibility
    simulated = Column(Boolean, nullable=False, default=False)
    idempotency_key = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


# ---------------------------------------------------------------------------
# Self-healing platform
# ---------------------------------------------------------------------------

class SystemHealthSnapshot(Base):
    """Aggregated system health across all platform components."""

    __tablename__ = "system_health_snapshots"
    __table_args__ = (
        Index("ix_system_health_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    overall_state = Column(String(16), nullable=False, default="UNKNOWN")
    # HEALTHY|DEGRADED|UNHEALTHY|RECOVERING|UNKNOWN
    components = Column(Text, nullable=False)  # bounded JSON {component: {state, detail}}
    unhealthy_count = Column(Integer, nullable=False, default=0)
    degraded_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class RecoveryPlaybook(Base):
    """Persisted safe recovery playbook."""

    __tablename__ = "recovery_playbooks"
    __table_args__ = (
        Index("ix_recovery_playbooks_trigger", "workspace_id", "trigger"),
        UniqueConstraint("workspace_id", "name", name="uq_recovery_playbook_name"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    name = Column(String(120), nullable=False)
    trigger = Column(String(64), nullable=False)     # failure kind, e.g. worker_stall
    detection_criteria = Column(Text, nullable=True)  # bounded JSON
    actions = Column(Text, nullable=False)           # bounded JSON list of action names
    risk_level = Column(String(16), nullable=False, default="LOW")
    cooldown_seconds = Column(Integer, nullable=False, default=300)
    max_attempts = Column(Integer, nullable=False, default=3)
    rollback_strategy = Column(Text, nullable=True)
    approved = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class RecoveryAttempt(Base):
    """One recovery attempt — idempotent, cooldown-aware, escalating."""

    __tablename__ = "recovery_attempts"
    __table_args__ = (
        Index("ix_recovery_attempts_ws", "workspace_id", "created_at"),
        Index("ix_recovery_attempts_playbook", "playbook_id", "created_at"),
        UniqueConstraint("workspace_id", "idempotency_key",
                         name="uq_recovery_attempt_idem"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    playbook_id = Column(Integer, nullable=True, index=True)
    trigger = Column(String(64), nullable=False)
    risk_level = Column(String(16), nullable=False, default="LOW")
    status = Column(String(24), nullable=False, default="RUNNING")
    # RUNNING|SUCCEEDED|FAILED|COOLDOWN_BLOCKED|LIMIT_BLOCKED|POLICY_BLOCKED|ESCALATED
    attempts_so_far = Column(Integer, nullable=False, default=1)
    action_results = Column(Text, nullable=True)   # bounded JSON
    escalated_incident_id = Column(Integer, nullable=True)
    auto_applied = Column(Boolean, nullable=False, default=False)
    idempotency_key = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Self-diagnosis engine
# ---------------------------------------------------------------------------

class DiagnosisReport(Base):
    """Operator-readable diagnosis with ranked hypotheses and evidence."""

    __tablename__ = "diagnosis_reports"
    __table_args__ = (
        Index("ix_diagnosis_reports_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    incident_id = Column(Integer, nullable=True, index=True)
    symptom = Column(String(200), nullable=False)
    hypotheses = Column(Text, nullable=False)     # bounded JSON [{cause, confidence, evidence}]
    top_cause = Column(String(200), nullable=True)
    top_confidence = Column(Float, nullable=True)
    correlated_failures = Column(Integer, nullable=False, default=0)
    report = Column(Text, nullable=True)          # operator-readable summary
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Self-healing knowledge
# ---------------------------------------------------------------------------

class KnowledgeRecoveryPlan(Base):
    """Bounded, authorized knowledge recovery plan."""

    __tablename__ = "knowledge_recovery_plans"
    __table_args__ = (
        Index("ix_knowledge_recovery_plans_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    issue_kind = Column(String(48), nullable=False)
    # ingestion_failure|stale_document|embedding_gap|graph_corruption|memory_conflict|connector_drift
    target_type = Column(String(32), nullable=False)   # document|embedding|graph|memory|connector
    target_id = Column(Integer, nullable=True)
    plan = Column(Text, nullable=False)                # bounded JSON actions
    risk_level = Column(String(16), nullable=False, default="LOW")
    status = Column(String(24), nullable=False, default="PROPOSED")
    # PROPOSED|APPROVED|EXECUTING|COMPLETED|FAILED|REJECTED
    decision = Column(String(24), nullable=True)       # autonomy decision
    executed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Adaptive ingestion / retrieval / RAG
# ---------------------------------------------------------------------------

class IngestionQualitySample(Base):
    """Quality measurement for one ingestion run."""

    __tablename__ = "ingestion_quality_samples"
    __table_args__ = (
        Index("ix_ingestion_quality_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    document_id = Column(Integer, nullable=True, index=True)
    extraction_quality = Column(Float, nullable=True)
    ocr_quality = Column(Float, nullable=True)
    chunk_quality = Column(Float, nullable=True)
    metadata_completeness = Column(Float, nullable=True)
    embedding_coverage = Column(Float, nullable=True)
    processing_latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class IngestionAnomaly(Base):
    """Detected sudden ingestion degradation."""

    __tablename__ = "ingestion_anomalies"
    __table_args__ = (
        Index("ix_ingestion_anomalies_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    metric = Column(String(48), nullable=False)
    severity = Column(String(16), nullable=False, default="MEDIUM")
    baseline_value = Column(Float, nullable=True)
    observed_value = Column(Float, nullable=True)
    drop_percent = Column(Float, nullable=True)
    recommendation = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default="OPEN")  # OPEN|RESOLVED
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class AdaptiveCandidate(Base):
    """Bounded, evaluated adaptation candidate (ingestion/retrieval/RAG/search)."""

    __tablename__ = "adaptive_candidates"
    __table_args__ = (
        Index("ix_adaptive_candidates_ws", "workspace_id", "domain"),
        UniqueConstraint("workspace_id", "idempotency_key",
                         name="uq_adaptive_candidate_idem"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    domain = Column(String(32), nullable=False)
    # ingestion|retrieval|rag|search|routing
    change_kind = Column(String(48), nullable=False)
    # vector_weight|keyword_weight|rerank|freshness|diversity|chunk_size|ocr|context|evidence_threshold...
    proposed_value = Column(Text, nullable=True)      # bounded JSON
    rationale = Column(Text, nullable=True)
    evaluation_score = Column(Float, nullable=True)
    baseline_score = Column(Float, nullable=True)
    gates = Column(Text, nullable=True)               # bounded JSON {gate: pass/fail}
    gates_passed = Column(Boolean, nullable=False, default=False)
    status = Column(String(24), nullable=False, default="CANDIDATE")
    # CANDIDATE|EVALUATED|APPROVED|PROMOTED|REJECTED|ROLLED_BACK
    promoted = Column(Boolean, nullable=False, default=False)
    idempotency_key = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class RetrievalDriftSnapshot(Base):
    """Retrieval ranking drift measurement over time."""

    __tablename__ = "retrieval_drift_snapshots"
    __table_args__ = (
        Index("ix_retrieval_drift_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    metric = Column(String(48), nullable=False, default="ndcg_proxy")
    baseline_value = Column(Float, nullable=True)
    current_value = Column(Float, nullable=True)
    drift_percent = Column(Float, nullable=True)
    degraded = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class RagDriftSnapshot(Base):
    """RAG quality drift measurement over time."""

    __tablename__ = "rag_drift_snapshots"
    __table_args__ = (
        Index("ix_rag_drift_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    metric = Column(String(48), nullable=False, default="groundedness")
    baseline_value = Column(Float, nullable=True)
    current_value = Column(Float, nullable=True)
    drift_percent = Column(Float, nullable=True)
    degraded = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class RagFailureCluster(Base):
    """Clustered RAG failures by cause."""

    __tablename__ = "rag_failure_clusters"
    __table_args__ = (
        Index("ix_rag_failure_clusters_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    cause = Column(String(64), nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    representative_question = Column(Text, nullable=True)
    recommendation = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Model / cost autopilot
# ---------------------------------------------------------------------------

class ModelPerformanceSample(Base):
    """Continuous model performance measurement."""

    __tablename__ = "model_performance_samples"
    __table_args__ = (
        Index("ix_model_perf_ws", "workspace_id", "model", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    model = Column(String(120), nullable=False)
    provider = Column(String(64), nullable=False)
    quality_score = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    cost_usd = Column(Float, nullable=True)
    availability = Column(Float, nullable=True)
    tool_reliability = Column(Float, nullable=True)
    structured_output_reliability = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class ModelDriftEvent(Base):
    """Detected model or provider degradation."""

    __tablename__ = "model_drift_events"
    __table_args__ = (
        Index("ix_model_drift_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    model = Column(String(120), nullable=True)
    provider = Column(String(64), nullable=True)
    kind = Column(String(32), nullable=False, default="MODEL")
    # MODEL|PROVIDER
    metric = Column(String(48), nullable=False)
    direction = Column(String(16), nullable=False)  # DEGRADED|IMPROVED
    baseline_value = Column(Float, nullable=True)
    current_value = Column(Float, nullable=True)
    drop_percent = Column(Float, nullable=True)
    recommendation = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class RoutingSimulation(Base):
    """Simulated routing change against a representative workload."""

    __tablename__ = "routing_simulations"
    __table_args__ = (
        Index("ix_routing_sims_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    description = Column(String(200), nullable=False)
    workload = Column(Text, nullable=True)          # bounded JSON
    current_metrics = Column(Text, nullable=True)   # bounded JSON
    simulated_metrics = Column(Text, nullable=True)  # bounded JSON
    safety_checks = Column(Text, nullable=True)     # bounded JSON {check: pass/fail}
    safe = Column(Boolean, nullable=False, default=False)
    candidate_id = Column(Integer, nullable=True)
    executed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class CostForecast(Base):
    """Live + forecasted spend per scope."""

    __tablename__ = "cost_forecasts"
    __table_args__ = (
        Index("ix_cost_forecasts_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    scope_kind = Column(String(24), nullable=False, default="WORKSPACE")
    # WORKSPACE|ORG|MODEL|PROVIDER|WORKFLOW
    scope_key = Column(String(120), nullable=True)
    period_days = Column(Integer, nullable=False, default=30)
    current_spend_usd = Column(Float, nullable=False, default=0.0)
    forecast_usd = Column(Float, nullable=False, default=0.0)
    budget_usd = Column(Float, nullable=True)
    over_budget = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class CostAnomaly(Base):
    """Detected abnormal usage/spend."""

    __tablename__ = "cost_anomalies_p21"
    __table_args__ = (
        Index("ix_cost_anomalies_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    metric = Column(String(48), nullable=False, default="spend")
    severity = Column(String(16), nullable=False, default="MEDIUM")
    baseline_value = Column(Float, nullable=True)
    observed_value = Column(Float, nullable=True)
    increase_percent = Column(Float, nullable=True)
    recommendation = Column(Text, nullable=True)
    status = Column(String(16), nullable=False, default="OPEN")
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class CostGuardDecision(Base):
    """Pre-execution cost guard: estimate vs budget vs policy."""

    __tablename__ = "cost_guard_decisions"
    __table_args__ = (
        Index("ix_cost_guard_ws", "workspace_id", "created_at"),
        UniqueConstraint("workspace_id", "idempotency_key",
                         name="uq_cost_guard_idem"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    operation_type = Column(String(64), nullable=False)
    estimated_cost_usd = Column(Float, nullable=False, default=0.0)
    remaining_budget_usd = Column(Float, nullable=True)
    policy_level = Column(String(20), nullable=True)
    decision = Column(String(24), nullable=False)
    # ALLOWED|BLOCKED|REQUIRES_APPROVAL
    reason = Column(Text, nullable=True)
    idempotency_key = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class CostOptimizationEvent(Base):
    """Audited application of a pre-approved low-risk cost optimization."""

    __tablename__ = "cost_optimization_events"
    __table_args__ = (
        Index("ix_cost_optim_ws", "workspace_id", "created_at"),
        UniqueConstraint("workspace_id", "idempotency_key",
                         name="uq_cost_optim_idem"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    optimization_kind = Column(String(48), nullable=False)
    # cache_reuse|context_reduction|batching|cheaper_model
    policy_level = Column(String(20), nullable=True)
    applied = Column(Boolean, nullable=False, default=False)
    estimated_savings_usd = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)
    idempotency_key = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Agent autonomy 2.0
# ---------------------------------------------------------------------------

class AgentPlanRisk(Base):
    """Risk classification + simulation for one agent plan."""

    __tablename__ = "agent_plan_risks"
    __table_args__ = (
        Index("ix_agent_plan_risk_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    agent_run_id = Column(Integer, nullable=True, index=True)
    plan_summary = Column(String(200), nullable=True)
    risk_level = Column(String(16), nullable=False, default="LOW")
    risk_factors = Column(Text, nullable=True)     # bounded JSON list
    simulation = Column(Text, nullable=True)       # bounded JSON result
    optimizations = Column(Text, nullable=True)    # bounded JSON list
    applied_optimizations = Column(Text, nullable=True)
    handed_off = Column(Boolean, nullable=False, default=False)
    decision = Column(String(24), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class AgentRecoveryEvent(Base):
    """Retry/checkpoint/resume/rollback/handoff for a failed agent run."""

    __tablename__ = "agent_recovery_events"
    __table_args__ = (
        Index("ix_agent_recovery_ws", "workspace_id", "created_at"),
        UniqueConstraint("workspace_id", "idempotency_key",
                         name="uq_agent_recovery_idem"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    agent_run_id = Column(Integer, nullable=True, index=True)
    recovery_kind = Column(String(24), nullable=False)
    # RETRY|CHECKPOINT|RESUME|ROLLBACK|HANDOFF
    status = Column(String(24), nullable=False, default="RUNNING")
    detail = Column(Text, nullable=True)
    idempotency_key = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Workflow autonomy
# ---------------------------------------------------------------------------

class WorkflowRiskAssessment(Base):
    """Workflow risk engine output + autonomy decision."""

    __tablename__ = "workflow_risk_assessments"
    __table_args__ = (
        Index("ix_workflow_risk_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    workflow_id = Column(Integer, nullable=True, index=True)
    workflow_version_id = Column(Integer, nullable=True)
    risk_level = Column(String(16), nullable=False, default="LOW")
    risk_factors = Column(Text, nullable=True)     # bounded JSON
    autonomy_decision = Column(String(24), nullable=True)
    requires_approval = Column(Boolean, nullable=False, default=False)
    simulation = Column(Text, nullable=True)
    optimizations = Column(Text, nullable=True)
    recurring_failures = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Event-driven intelligence
# ---------------------------------------------------------------------------

class PlatformEvent(Base):
    """Durable platform event with dedup and bounded replay support."""

    __tablename__ = "platform_events"
    __table_args__ = (
        Index("ix_platform_events_ws", "workspace_id", "created_at"),
        Index("ix_platform_events_kind", "workspace_id", "event_kind"),
        UniqueConstraint("workspace_id", "event_key", name="uq_platform_event_key"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    event_kind = Column(String(48), nullable=False)
    # knowledge.* | ai.* | ops.*
    event_key = Column(String(160), nullable=False)  # dedup key
    payload = Column(Text, nullable=True)            # bounded JSON
    deduplicated = Column(Boolean, nullable=False, default=False)
    replayed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Continuous evaluation
# ---------------------------------------------------------------------------

class EvaluationSchedule(Base):
    """Scheduled evaluation job against an immutable dataset version."""

    __tablename__ = "evaluation_schedules"
    __table_args__ = (
        Index("ix_eval_schedules_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    domain = Column(String(32), nullable=False, default="retrieval")
    dataset_id = Column(Integer, nullable=True, index=True)
    dataset_version = Column(String(32), nullable=False, default="v1")
    interval_minutes = Column(Integer, nullable=False, default=1440)
    config = Column(Text, nullable=True)            # bounded JSON (immutable per schedule)
    active = Column(Boolean, nullable=False, default=True)
    last_run_at = Column(DateTime, nullable=True)
    last_metrics = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class EvaluationRun(Base):
    """One reproducible evaluation execution."""

    __tablename__ = "evaluation_runs_p21"
    __table_args__ = (
        Index("ix_eval_runs_p21_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    schedule_id = Column(Integer, nullable=True, index=True)
    domain = Column(String(32), nullable=False)
    dataset_version = Column(String(32), nullable=False)
    model = Column(String(120), nullable=True)
    provider = Column(String(64), nullable=True)
    config = Column(Text, nullable=True)
    metrics = Column(Text, nullable=True)           # bounded JSON
    regression_detected = Column(Boolean, nullable=False, default=False)
    gate_passed = Column(Boolean, nullable=False, default=True)
    incident_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# AI safety 10.0 / security center 3.0
# ---------------------------------------------------------------------------

class EmergencyStop(Base):
    """Authenticated operator emergency stop for autonomous subsystems."""

    __tablename__ = "emergency_stops"
    __table_args__ = (
        Index("ix_emergency_stops_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    scope = Column(String(24), nullable=False, default="ALL")
    # ALL|AI_ACTIONS|AGENTS|WORKFLOWS|AUTONOMOUS_RECOVERY
    active = Column(Boolean, nullable=False, default=True)
    actor = Column(String(120), nullable=False, default="operator")
    reason = Column(Text, nullable=True)
    lifted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class AutonomyAbuseAttempt(Base):
    """Detected autonomy bypass / scope escalation attempt."""

    __tablename__ = "autonomy_abuse_attempts"
    __table_args__ = (
        Index("ix_autonomy_abuse_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    abuse_kind = Column(String(48), nullable=False)
    # approval_bypass|budget_bypass|policy_bypass|scope_escalation|recursive_execution
    blocked = Column(Boolean, nullable=False, default=True)
    detail = Column(Text, nullable=True)
    actor = Column(String(120), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class ToolSafetyViolation(Base):
    """Tool allowlist/argument/budget/timeout/scope violation."""

    __tablename__ = "tool_safety_violations"
    __table_args__ = (
        Index("ix_tool_safety_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    tool_name = Column(String(120), nullable=False)
    violation_kind = Column(String(48), nullable=False)
    # not_allowlisted|argument_invalid|budget_exceeded|timeout|scope_violation|unsafe_output
    blocked = Column(Boolean, nullable=False, default=True)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class SecurityHealthScore(Base):
    """Aggregated security health per workspace."""

    __tablename__ = "security_health_scores"
    __table_args__ = (
        Index("ix_security_health_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    score = Column(Float, nullable=False, default=100.0)
    auth_failures = Column(Integer, nullable=False, default=0)
    authz_failures = Column(Integer, nullable=False, default=0)
    injection_attempts = Column(Integer, nullable=False, default=0)
    ssrf_attempts = Column(Integer, nullable=False, default=0)
    tool_abuse = Column(Integer, nullable=False, default=0)
    exfiltration_attempts = Column(Integer, nullable=False, default=0)
    suspicious_api = Column(Integer, nullable=False, default=0)
    state = Column(String(16), nullable=False, default="HEALTHY")  # HEALTHY|ELEVATED|CRITICAL
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class SecurityIncidentP21(Base):
    """Security incident created from threshold breaches."""

    __tablename__ = "security_incidents_p21"
    __table_args__ = (
        Index("ix_sec_incidents_p21_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    severity = Column(String(8), nullable=False, default="SEV3")
    category = Column(String(48), nullable=False)
    # injection|exfiltration|tool_abuse|ssrf|api_abuse|auth|autonomy_abuse
    correlated_count = Column(Integer, nullable=False, default=1)
    summary = Column(String(300), nullable=True)
    recommended_playbook = Column(Text, nullable=True)
    auto_response = Column(Text, nullable=True)    # only pre-approved non-destructive
    status = Column(String(16), nullable=False, default="OPEN")
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Data governance 6.0
# ---------------------------------------------------------------------------

class DataClassificationSnapshot(Base):
    """Data classification distribution + drift tracking."""

    __tablename__ = "data_classification_snapshots"
    __table_args__ = (
        Index("ix_data_class_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    counts = Column(Text, nullable=False)          # bounded JSON {classification: n}
    drifted = Column(Boolean, nullable=False, default=False)
    drift_detail = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class DataPolicyImpact(Base):
    """Impact of a classification change on models/providers/workflows/regions."""

    __tablename__ = "data_policy_impacts"
    __table_args__ = (
        Index("ix_data_policy_impact_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    affected_models = Column(Text, nullable=True)
    affected_providers = Column(Text, nullable=True)
    affected_workflows = Column(Text, nullable=True)
    affected_connectors = Column(Text, nullable=True)
    affected_regions = Column(Text, nullable=True)
    minimization_required = Column(Boolean, nullable=False, default=False)
    residency_violations = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Multi-region operations
# ---------------------------------------------------------------------------

class RegionHealthP21(Base):
    """Per-region health + capacity snapshot."""

    __tablename__ = "region_health_p21"
    __table_args__ = (
        Index("ix_region_health_p21", "organization_id", "region_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, nullable=False, index=True)
    region_id = Column(String(64), nullable=False)
    state = Column(String(16), nullable=False, default="HEALTHY")
    workers = Column(Integer, nullable=False, default=0)
    queue_depth = Column(Integer, nullable=False, default=0)
    provider_healthy = Column(Boolean, nullable=False, default=True)
    db_healthy = Column(Boolean, nullable=False, default=True)
    failover_ready = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class FailoverSimulation(Base):
    """Dry-run failover simulation (zero side effects)."""

    __tablename__ = "failover_simulations"
    __table_args__ = (
        Index("ix_failover_sims", "organization_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, nullable=False, index=True)
    from_region = Column(String(64), nullable=False)
    to_region = Column(String(64), nullable=False)
    residency_ok = Column(Boolean, nullable=False, default=True)
    capacity_ok = Column(Boolean, nullable=False, default=True)
    ready = Column(Boolean, nullable=False, default=False)
    detail = Column(Text, nullable=True)
    simulated = Column(Boolean, nullable=False, default=True)
    executed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class ResidencyGuardEvent(Base):
    """Blocked illegal cross-region routing."""

    __tablename__ = "residency_guard_events"
    __table_args__ = (
        Index("ix_residency_guard", "organization_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, nullable=False, index=True)
    requested_region = Column(String(64), nullable=False)
    allowed_region = Column(String(64), nullable=True)
    blocked = Column(Boolean, nullable=False, default=True)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Worker / broker / scheduler / database / cache self-healing
# ---------------------------------------------------------------------------

class WorkerHealthScore(Base):
    """Per-worker health score from heartbeat/throughput/failures/queue."""

    __tablename__ = "worker_health_scores"
    __table_args__ = (
        Index("ix_worker_health_p21", "workspace_id", "worker_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    worker_id = Column(String(120), nullable=False)
    region = Column(String(64), nullable=True)
    heartbeat_age_seconds = Column(Integer, nullable=False, default=0)
    throughput = Column(Float, nullable=False, default=0.0)
    failure_count = Column(Integer, nullable=False, default=0)
    queue_latency_ms = Column(Float, nullable=False, default=0.0)
    score = Column(Float, nullable=False, default=100.0)
    state = Column(String(16), nullable=False, default="HEALTHY")
    # HEALTHY|DEGRADED|QUARANTINED|RECOVERED
    quarantined = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class CapacityRecommendation(Base):
    """Autoscaling recommendation with tenant fairness constraints."""

    __tablename__ = "capacity_recommendations"
    __table_args__ = (
        Index("ix_capacity_rec_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    region = Column(String(64), nullable=True)
    current_workers = Column(Integer, nullable=False, default=0)
    recommended_workers = Column(Integer, nullable=False, default=0)
    fairness_preserved = Column(Boolean, nullable=False, default=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class BrokerHealthSnapshot(Base):
    """Broker depth/latency/visibility/errors/reconnects snapshot."""

    __tablename__ = "broker_health_snapshots"
    __table_args__ = (
        Index("ix_broker_health_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    broker = Column(String(24), nullable=False, default="postgres")
    depth = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Float, nullable=False, default=0.0)
    visibility_timeouts = Column(Integer, nullable=False, default=0)
    error_count = Column(Integer, nullable=False, default=0)
    reconnects = Column(Integer, nullable=False, default=0)
    state = Column(String(16), nullable=False, default="HEALTHY")
    degraded = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class SchedulerHealthSnapshot(Base):
    """Leader health, dedup and missed-schedule recovery state."""

    __tablename__ = "scheduler_health_snapshots"
    __table_args__ = (
        Index("ix_scheduler_health_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    has_leader = Column(Boolean, nullable=False, default=False)
    leader_age_seconds = Column(Integer, nullable=False, default=0)
    missed_schedules = Column(Integer, nullable=False, default=0)
    recovered_schedules = Column(Integer, nullable=False, default=0)
    dedup_blocked = Column(Integer, nullable=False, default=0)
    state = Column(String(16), nullable=False, default="HEALTHY")
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class SlowQueryRecord(Base):
    """Detected slow database query (recommendations only, never auto-DDL)."""

    __tablename__ = "slow_query_records"
    __table_args__ = (
        Index("ix_slow_queries_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    statement = Column(String(300), nullable=False)
    duration_ms = Column(Float, nullable=False, default=0.0)
    recommendation = Column(Text, nullable=True)
    auto_applied = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class CacheHealthSnapshot(Base):
    """Cache hit/miss/stale/usage health."""

    __tablename__ = "cache_health_snapshots"
    __table_args__ = (
        Index("ix_cache_health_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    cache = Column(String(48), nullable=False, default="embedding_cache")
    hit_rate = Column(Float, nullable=False, default=0.0)
    miss_rate = Column(Float, nullable=False, default=1.0)
    stale_entries = Column(Integer, nullable=False, default=0)
    memory_usage_mb = Column(Float, nullable=False, default=0.0)
    anomaly = Column(Boolean, nullable=False, default=False)
    recommendation = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Graph autonomy / memory autonomy
# ---------------------------------------------------------------------------

class GraphRepairProposal(Base):
    """Graph repair proposal (approved low-risk repairs only)."""

    __tablename__ = "graph_repair_proposals"
    __table_args__ = (
        Index("ix_graph_repairs_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    repair_kind = Column(String(48), nullable=False)
    # orphan_cleanup|conflict_resolution|stale_edge_refresh|confidence_recompute
    target_entity_id = Column(Integer, nullable=True)
    target_relationship_id = Column(Integer, nullable=True)
    proposal = Column(Text, nullable=True)
    risk_level = Column(String(16), nullable=False, default="LOW")
    impact = Column(Text, nullable=True)            # memories/RAG/reports affected
    status = Column(String(24), nullable=False, default="PROPOSED")
    # PROPOSED|APPROVED|EXECUTING|COMPLETED|REJECTED
    decision = Column(String(24), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class MemoryAutonomyEvent(Base):
    """Deterministic suppression / conflict routing / audit for memories."""

    __tablename__ = "memory_autonomy_events"
    __table_args__ = (
        Index("ix_memory_autonomy_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    memory_id = Column(Integer, nullable=True, index=True)
    event_kind = Column(String(32), nullable=False)
    # suppressed|conflict_review|audit
    automatic = Column(Boolean, nullable=False, default=False)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Personal AI control
# ---------------------------------------------------------------------------

class PersonalAutonomySetting(Base):
    """Per-user autonomy preferences within policy bounds."""

    __tablename__ = "personal_autonomy_settings"
    __table_args__ = (
        UniqueConstraint("workspace_id", "user_id", name="uq_personal_autonomy"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    autonomy_level = Column(String(20), nullable=False, default="RECOMMEND")
    allow_memory_suppression = Column(Boolean, nullable=False, default=True)
    allow_memory_reset = Column(Boolean, nullable=False, default=False)
    allow_memory_delete = Column(Boolean, nullable=False, default=False)
    activity_feed_enabled = Column(Boolean, nullable=False, default=True)
    explanation_style = Column(String(24), nullable=False, default="CONCISE")
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class AIActivityItem(Base):
    """Per-user AI activity feed item (suggestions/actions/decisions/approvals)."""

    __tablename__ = "ai_activity_items"
    __table_args__ = (
        Index("ix_ai_activity_ws_user", "workspace_id", "user_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    item_kind = Column(String(32), nullable=False, default="SUGGESTION")
    # SUGGESTION|ACTION|DECISION|APPROVAL|RECOVERY
    title = Column(String(200), nullable=False)
    explanation = Column(Text, nullable=True)  # evidence/policy only, never CoT
    ref_kind = Column(String(32), nullable=True)
    ref_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Incident management 2.0
# ---------------------------------------------------------------------------

class IncidentP21(Base):
    """Phase 21 incidents with timeline + postmortem learning."""

    __tablename__ = "incidents_p21"
    __table_args__ = (
        Index("ix_incidents_p21_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    severity = Column(String(8), nullable=False, default="SEV3")
    # SEV0..SEV4
    title = Column(String(200), nullable=False)
    source = Column(String(32), nullable=False, default="THRESHOLD")
    # THRESHOLD|DIAGNOSIS|EVALUATION|OPERATOR
    timeline = Column(Text, nullable=True)          # bounded JSON [{at, kind, detail}]
    diagnosis_id = Column(Integer, nullable=True)
    recovery_attempt_id = Column(Integer, nullable=True)
    postmortem = Column(Text, nullable=True)        # draft, human review required
    postmortem_finalized = Column(Boolean, nullable=False, default=False)
    learnings = Column(Text, nullable=True)         # tests/monitoring/playbooks/eval cases
    status = Column(String(16), nullable=False, default="OPEN")
    # OPEN|MITIGATED|RESOLVED
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Continuous learning (no weight training)
# ---------------------------------------------------------------------------

class LearningDatasetCandidate(Base):
    """Approved feedback/examples converted into versioned dataset entries."""

    __tablename__ = "learning_dataset_candidates"
    __table_args__ = (
        Index("ix_learning_candidates_ws", "workspace_id", "created_at"),
        UniqueConstraint("workspace_id", "idempotency_key",
                         name="uq_learning_candidate_idem"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    source_kind = Column(String(32), nullable=False, default="FEEDBACK")
    # FEEDBACK|VERIFIED_FAILURE|KNOWLEDGE_GAP
    dataset_id = Column(Integer, nullable=True, index=True)
    dataset_version = Column(String(32), nullable=False, default="v1")
    input_text = Column(Text, nullable=False)
    expected_output = Column(Text, nullable=True)
    quality = Column(String(16), nullable=False, default="APPROVED")
    # APPROVED|NOISY|ABUSIVE
    regression_case_id = Column(Integer, nullable=True)
    idempotency_key = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# AI artifact intelligence
# ---------------------------------------------------------------------------

class ArtifactQualityScore(Base):
    """Quality scoring for AI artifacts (reports/summaries/extractions)."""

    __tablename__ = "artifact_quality_scores"
    __table_args__ = (
        Index("ix_artifact_quality_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    artifact_id = Column(Integer, nullable=True, index=True)
    artifact_kind = Column(String(32), nullable=False, default="REPORT")
    version = Column(Integer, nullable=False, default=1)
    quality_score = Column(Float, nullable=True)
    groundedness = Column(Float, nullable=True)
    citation_coverage = Column(Float, nullable=True)
    completeness = Column(Float, nullable=True)
    provenance_ok = Column(Boolean, nullable=False, default=True)
    regression = Column(Boolean, nullable=False, default=False)
    comparison = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# API autonomy safety
# ---------------------------------------------------------------------------

class ApiAbuseSignalP21(Base):
    """API abuse signal (enumeration/brute force/anomalous usage/key abuse)."""

    __tablename__ = "api_abuse_signals_p21"
    __table_args__ = (
        Index("ix_api_abuse_p21_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    abuse_kind = Column(String(48), nullable=False)
    # enumeration|brute_force|abnormal_usage|key_abuse
    subject = Column(String(160), nullable=True)   # ip/api key id (never the secret)
    severity = Column(String(16), nullable=False, default="MEDIUM")
    rate_limit_recommendation = Column(Text, nullable=True)
    incident_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Cost-aware worker scheduling
# ---------------------------------------------------------------------------

class CostAwareQueueDecision(Base):
    """Queue prioritization decision balancing urgency/budget/fairness/cost."""

    __tablename__ = "cost_aware_queue_decisions"
    __table_args__ = (
        Index("ix_cost_queue_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    job_id = Column(Integer, nullable=True)
    priority = Column(Integer, nullable=False, default=5)
    urgency = Column(String(16), nullable=False, default="NORMAL")
    estimated_cost_usd = Column(Float, nullable=False, default=0.0)
    budget_state = Column(String(16), nullable=False, default="OK")  # OK|TIGHT|EXCEEDED
    fairness_preserved = Column(Boolean, nullable=False, default=True)
    guard = Column(String(24), nullable=True)      # ALLOWED|REQUIRES_APPROVAL|BLOCKED
    decision = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Autonomous data maintenance
# ---------------------------------------------------------------------------

class MaintenancePlan(Base):
    """Bounded maintenance plan with dry-run + approval for destructive ops."""

    __tablename__ = "maintenance_plans"
    __table_args__ = (
        Index("ix_maintenance_plans_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    plan_kind = Column(String(48), nullable=False)
    # cleanup|stale_artifacts|expired_traces|old_evaluations|temporary_data
    targets = Column(Text, nullable=True)          # bounded JSON
    destructive = Column(Boolean, nullable=False, default=False)
    dry_run = Column(Boolean, nullable=False, default=True)
    dry_run_result = Column(Text, nullable=True)
    legal_hold_respected = Column(Boolean, nullable=False, default=True)
    retention_policy_id = Column(Integer, nullable=True)
    requires_approval = Column(Boolean, nullable=False, default=False)
    status = Column(String(24), nullable=False, default="PROPOSED")
    # PROPOSED|APPROVED|EXECUTING|COMPLETED|FAILED|REJECTED
    executed_result = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# Backup / DR
# ---------------------------------------------------------------------------

class BackupHealthRecord(Base):
    """Backup status + restore validation (infrastructure permitting)."""

    __tablename__ = "backup_health_records"
    __table_args__ = (
        Index("ix_backup_health_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    backup_kind = Column(String(32), nullable=False, default="SNAPSHOT")
    status = Column(String(24), nullable=False, default="UNKNOWN")
    # HEALTHY|STALE|MISSING|UNKNOWN
    last_backup_at = Column(DateTime, nullable=True)
    age_hours = Column(Float, nullable=True)
    restore_validated = Column(Boolean, nullable=False, default=False)
    tenant_isolation_ok = Column(Boolean, nullable=False, default=True)
    rto_target_minutes = Column(Integer, nullable=True)
    rpo_target_minutes = Column(Integer, nullable=True)
    rto_observed_minutes = Column(Float, nullable=True)
    rpo_observed_minutes = Column(Float, nullable=True)
    simulated = Column(Boolean, nullable=False, default=False)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)


class ChaosTestRun(Base):
    """Bounded chaos/soak execution record (simulated where infra absent)."""

    __tablename__ = "chaos_test_runs"
    __table_args__ = (
        Index("ix_chaos_runs_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    scenario = Column(String(48), nullable=False)
    # provider_timeout|provider_429|provider_5xx|provider_malformed|broker_failure|
    # worker_loss|scheduler_leader_loss|db_failure|api_load|worker_load|noisy_neighbor|soak
    bounded = Column(Boolean, nullable=False, default=True)
    simulated = Column(Boolean, nullable=False, default=False)
    passed = Column(Boolean, nullable=False, default=False)
    metrics = Column(Text, nullable=True)          # bounded JSON
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)
