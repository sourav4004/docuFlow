"""Phase 20 models — Self-Improving Enterprise AI Knowledge OS.

Adds the governed self-improvement substrate: improvement proposals and their
audited lifecycle, the experiment platform (datasets, runs, comparisons),
AI quality scorecards/trends/alerts, retrieval + RAG failure classification
and recommendations, knowledge freshness/health/gaps, document change
intelligence, policy versioning + drift, provider/model performance profiles,
routing recommendations, cost baselines + token efficiency, agent/workflow/
memory/graph intelligence, search quality events, unified feedback,
incident management, SLO history + error budgets, alert prioritization,
notification preferences, versioned reports, API/database health metrics,
and retention policy records.

Every table follows Phases 0-19 conventions: tenant scoping via workspace_id
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
# Improvement control plane
# ---------------------------------------------------------------------------

class ImprovementProposal(Base):
    """Persisted, human-governed improvement proposal."""

    __tablename__ = "improvement_proposals"
    __table_args__ = (
        Index("ix_improv_proposals_status", "status", "domain"),
        Index("ix_improv_proposals_domain", "domain"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(40), nullable=False)  # retrieval/rag/routing/prompt/workflow/...
    title = Column(String(300), nullable=False)
    problem = Column(Text, nullable=False)
    evidence = Column(Text, nullable=True)
    expected_benefit = Column(Text, nullable=True)
    risk = Column(String(20), nullable=False, default="LOW")  # LOW/MEDIUM/HIGH
    estimated_cost = Column(Float, nullable=True)
    proposed_change = Column(Text, nullable=False)
    evaluation_requirements = Column(Text, nullable=True)
    author_source = Column(String(100), nullable=True)  # 'ai' | 'human' | email
    author_user_id = Column(Integer, nullable=True)
    status = Column(String(24), nullable=False, default="PROPOSED")
    # PROPOSED -> EVALUATING -> APPROVAL_REQUIRED -> APPROVED -> STAGED
    #   -> ACTIVE -> SUPERSEDED | ROLLED_BACK | REJECTED
    workspace_id = Column(Integer, nullable=True)
    organization_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ImprovementAudit(Base):
    """Every lifecycle transition is audited (actor, prev/new, reason)."""

    __tablename__ = "improvement_audits"
    __table_args__ = (
        Index("ix_improv_audits_proposal", "proposal_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    proposal_id = Column(Integer, ForeignKey("improvement_proposals.id",
                                             ondelete="CASCADE"),
                         nullable=False, index=True)
    actor_user_id = Column(Integer, nullable=True)
    previous_state = Column(String(24), nullable=True)
    new_state = Column(String(24), nullable=False)
    reason = Column(String(1000), nullable=True)
    evidence = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ImprovementTransition(Base):
    """Allowed lifecycle transitions, validated server-side."""

    __tablename__ = "improvement_transitions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    from_state = Column(String(24), nullable=False)
    to_state = Column(String(24), nullable=False)
    requires_approval = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("from_state", "to_state",
                         name="uq_improv_transition"),
    )


# ---------------------------------------------------------------------------
# Experiment platform
# ---------------------------------------------------------------------------

class ExperimentDataset(Base):
    """Golden / synthetic / anonymized / curated evaluation datasets."""

    __tablename__ = "experiment_datasets"
    __table_args__ = (
        UniqueConstraint("name", name="uq_experiment_dataset_name"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    kind = Column(String(20), nullable=False)  # golden/synthetic/anonymized/curated
    domain = Column(String(40), nullable=False, default="retrieval")
    items_json = Column(Text, nullable=False)  # JSON list of examples
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class Experiment(Base):
    """A controlled experiment; config is immutable once created."""

    __tablename__ = "experiments"
    __table_args__ = (
        Index("ix_experiments_status", "status"),
        Index("ix_experiments_domain", "domain"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    proposal_id = Column(Integer, ForeignKey("improvement_proposals.id",
                                             ondelete="SET NULL"),
                         nullable=True)
    name = Column(String(200), nullable=False)
    domain = Column(String(40), nullable=False)
    config_json = Column(Text, nullable=False)  # immutable configuration
    config_fingerprint = Column(String(64), nullable=False)
    dataset_id = Column(Integer, ForeignKey("experiment_datasets.id",
                                            ondelete="SET NULL"), nullable=True)
    status = Column(String(20), nullable=False, default="DRAFT")  # DRAFT/RUNNING/DONE/FAILED
    created_by = Column(Integer, nullable=True)
    workspace_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ExperimentRun(Base):
    """A single persisted evaluation run with metrics/cost/latency."""

    __tablename__ = "experiment_runs"
    __table_args__ = (
        Index("ix_experiment_runs_experiment", "experiment_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id",
                                               ondelete="CASCADE"),
                           nullable=False, index=True)
    dataset_id = Column(Integer, nullable=True)
    dataset_name = Column(String(120), nullable=True)
    model = Column(String(120), nullable=True)
    provider = Column(String(120), nullable=True)
    config_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=False)  # quality/latency/cost metrics
    cost = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    environment = Column(String(40), nullable=True)
    status = Column(String(20), nullable=False, default="RUNNING")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ExperimentComparison(Base):
    """Deterministic comparison between baseline and candidate runs."""

    __tablename__ = "experiment_comparisons"
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id",
                                               ondelete="CASCADE"),
                           nullable=False, index=True)
    baseline_run_id = Column(Integer, nullable=False)
    candidate_run_id = Column(Integer, nullable=False)
    metrics_json = Column(Text, nullable=False)
    verdict = Column(String(20), nullable=False)  # CANDIDATE_BETTER/BASELINE_BETTER/INCONCLUSIVE
    sample_size = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# AI quality platform 3.0
# ---------------------------------------------------------------------------

class QualityScorecard(Base):
    """Unified quality scorecard per domain and period."""

    __tablename__ = "quality_scorecards"
    __table_args__ = (
        Index("ix_quality_scorecards_scope", "domain", "workspace_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(40), nullable=False)
    workspace_id = Column(Integer, nullable=True)
    organization_id = Column(Integer, nullable=True)
    period = Column(String(10), nullable=False, default="daily")  # daily/weekly/monthly
    scorecard_json = Column(Text, nullable=False)
    overall_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class QualityTrend(Base):
    """Quality over time per domain (for trending + regression)."""

    __tablename__ = "quality_trends"
    __table_args__ = (
        Index("ix_quality_trends_domain", "domain", "period"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(40), nullable=False)
    period = Column(String(10), nullable=False)
    window_start = Column(DateTime(timezone=True), nullable=False)
    metrics_json = Column(Text, nullable=False)
    score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class QualityAlert(Base):
    """Significant quality degradation alerts (deduplicated by fingerprint)."""

    __tablename__ = "quality_alerts"
    __table_args__ = (
        Index("ix_quality_alerts_domain", "domain", "created_at"),
        UniqueConstraint("fingerprint", name="uq_quality_alert_fingerprint"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(40), nullable=False)
    severity = Column(String(10), nullable=False, default="MEDIUM")
    fingerprint = Column(String(64), nullable=False)
    message = Column(Text, nullable=False)
    workspace_id = Column(Integer, nullable=True)
    resolved = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Retrieval / RAG self-improvement
# ---------------------------------------------------------------------------

class RetrievalFailure(Base):
    """Classified retrieval failure events."""

    __tablename__ = "retrieval_failures"
    __table_args__ = (
        Index("ix_retrieval_failures_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    query_hash = Column(String(64), nullable=False)
    query_preview = Column(String(300), nullable=True)
    failure_class = Column(String(40), nullable=False)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RetrievalRecommendation(Base):
    """Retrieval improvement recommendation (evaluation required)."""

    __tablename__ = "retrieval_recommendations"
    __table_args__ = (
        Index("ix_retrieval_recos_ws", "workspace_id", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    category = Column(String(40), nullable=False)  # chunk_size/overlap/weights/rerank/metadata/synonyms/embedding
    suggestion = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="PROPOSED")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RagFailure(Base):
    """Claim-level RAG failure classification."""

    __tablename__ = "rag_failures"
    __table_args__ = (
        Index("ix_rag_failures_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    execution_id = Column(String(64), nullable=True)
    failure_class = Column(String(40), nullable=False)
    claim = Column(Text, nullable=True)
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RagEvaluationPipeline(Base):
    """Persisted offline evaluation pipeline runs for RAG candidates."""

    __tablename__ = "rag_evaluation_pipelines"
    id = Column(Integer, primary_key=True, autoincrement=True)
    proposal_id = Column(Integer, nullable=True)
    dataset_id = Column(Integer, nullable=True)
    config_json = Column(Text, nullable=False)
    metrics_json = Column(Text, nullable=True)
    gate_passed = Column(Boolean, nullable=True)
    status = Column(String(20), nullable=False, default="RUNNING")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Knowledge intelligence
# ---------------------------------------------------------------------------

class KnowledgeHealth(Base):
    """Per-scope explainable knowledge health snapshot."""

    __tablename__ = "knowledge_health"
    __table_args__ = (
        Index("ix_knowledge_health_scope", "scope_type", "scope_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope_type = Column(String(20), nullable=False)  # DOCUMENT/COLLECTION/WORKSPACE/ORGANIZATION
    scope_id = Column(Integer, nullable=False)
    health_json = Column(Text, nullable=False)
    score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class KnowledgeGapInsight(Base):
    """Repeated questions lacking strong evidence (knowledge gaps).

    Named distinctly from phase15.KnowledgeGap to avoid table collisions.
    """

    __tablename__ = "knowledge_gap_insights"
    __table_args__ = (
        Index("ix_kgap_insights_ws", "workspace_id", "attempts"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    query_hash = Column(String(64), nullable=False)
    query_preview = Column(String(300), nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    best_evidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class DocChangeEvent(Base):
    """Document change classification + impact."""

    __tablename__ = "doc_change_events"
    __table_args__ = (
        Index("ix_doc_change_events_doc", "document_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, nullable=False, index=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    version_from = Column(Integer, nullable=True)
    version_to = Column(Integer, nullable=True)
    change_class = Column(String(30), nullable=False)  # formatting/metadata/minor/major/policy/numerical/deadline/entity/relationship
    impact_json = Column(Text, nullable=True)  # affected summaries/embeddings/citations/workflows/graph/memories
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Policy intelligence
# ---------------------------------------------------------------------------

class PolicyVersion(Base):
    """Immutable versioned policy with diff + actor."""

    __tablename__ = "policy_versions"
    __table_args__ = (
        Index("ix_policy_versions_scope", "scope_type", "scope_id"),
        UniqueConstraint("scope_type", "scope_id", "version",
                         name="uq_policy_version"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope_type = Column(String(20), nullable=False)  # GLOBAL/ORGANIZATION/WORKSPACE
    scope_id = Column(Integer, nullable=True)
    version = Column(Integer, nullable=False)
    policy_json = Column(Text, nullable=False)
    diff_json = Column(Text, nullable=True)
    actor_user_id = Column(Integer, nullable=True)
    reason = Column(String(1000), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class PolicyImpact(Base):
    """Recorded policy-drift impact analysis."""

    __tablename__ = "policy_impacts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_version_id = Column(Integer, nullable=False, index=True)
    impact_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Model / provider optimization
# ---------------------------------------------------------------------------

class ProviderProfile(Base):
    """Per-provider/model performance profile over a period."""

    __tablename__ = "provider_profiles"
    __table_args__ = (
        Index("ix_provider_profiles_provider", "provider", "model"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(120), nullable=False)
    model = Column(String(120), nullable=False)
    period = Column(String(10), nullable=False, default="daily")
    metrics_json = Column(Text, nullable=False)  # latency/success/cost/quality/rate limits
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class RoutingRecommendation(Base):
    """Candidate routing improvement (never auto-applied)."""

    __tablename__ = "routing_recommendations"
    __table_args__ = (
        Index("ix_routing_recos_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(40), nullable=False)
    candidate = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    expected_gain = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="PROPOSED")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ProviderAnomaly(Base):
    """Detected provider anomalies (latency/error/cost/quality spikes)."""

    __tablename__ = "provider_anomalies"
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(120), nullable=False)
    model = Column(String(120), nullable=True)
    anomaly_type = Column(String(30), nullable=False)
    detail = Column(Text, nullable=True)
    severity = Column(String(10), nullable=False, default="MEDIUM")
    resolved = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Cost intelligence 5.0
# ---------------------------------------------------------------------------

class CostBaseline(Base):
    """Cost baselines per scope/model/provider/pipeline."""

    __tablename__ = "cost_baselines"
    __table_args__ = (
        Index("ix_cost_baselines_scope", "scope_type", "scope_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    scope_type = Column(String(20), nullable=False)  # WORKSPACE/ORGANIZATION/MODEL/PROVIDER/WORKFLOW/AGENT/DOCUMENT
    scope_id = Column(Integer, nullable=True)
    dimension = Column(String(60), nullable=True)  # model/provider/workflow/agent name
    baseline_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class TokenEfficiency(Base):
    """Token usage accounting for efficiency analysis."""

    __tablename__ = "token_efficiency"
    __table_args__ = (
        Index("ix_token_efficiency_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    execution_ref = Column(String(64), nullable=True)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    context_tokens = Column(Integer, nullable=False, default=0)
    repeated_tokens = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Agent / workflow / memory / graph intelligence
# ---------------------------------------------------------------------------

class AgentIntelligence(Base):
    """Per-execution agent success/failure analytics."""

    __tablename__ = "agent_intelligence"
    __table_args__ = (
        Index("ix_agent_intelligence_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    execution_id = Column(String(64), nullable=False, index=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    metrics_json = Column(Text, nullable=False)
    failure_class = Column(String(30), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class WorkflowIntelligence(Base):
    """Per-node workflow analytics (bottlenecks, failure hotspots)."""

    __tablename__ = "workflow_intelligence"
    __table_args__ = (
        Index("ix_workflow_intel_run", "run_id", "node_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, nullable=False, index=True)
    node_id = Column(String(120), nullable=False)
    workspace_id = Column(Integer, nullable=False, index=True)
    metrics_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class MemoryIntelligence(Base):
    """Memory quality metrics per memory."""

    __tablename__ = "memory_intelligence"
    __table_args__ = (
        Index("ix_memory_intel_memory", "memory_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_id = Column(Integer, nullable=False, index=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    metrics_json = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class GraphHealth(Base):
    """Knowledge-graph health snapshots per workspace."""

    __tablename__ = "graph_health"
    __table_args__ = (
        Index("ix_graph_health_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    metrics_json = Column(Text, nullable=False)
    score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class GraphRecommendation(Base):
    """Candidate graph entities/aliases/relationships (validation required)."""

    __tablename__ = "graph_recommendations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    kind = Column(String(20), nullable=False)  # entity/alias/relationship
    candidate = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="PROPOSED")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Search intelligence + unified feedback
# ---------------------------------------------------------------------------

class SearchQualityEvent(Base):
    """Search behavior events for quality analytics."""

    __tablename__ = "search_quality_events"
    __table_args__ = (
        Index("ix_search_quality_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    query_hash = Column(String(64), nullable=False)
    query_preview = Column(String(300), nullable=True)
    event_type = Column(String(30), nullable=False)  # zero_result/reformulated/clicked/abandoned
    result_count = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class FeedbackEvent(Base):
    """Unified feedback from search/RAG/citations/summaries/agents/workflows."""

    __tablename__ = "feedback_events"
    __table_args__ = (
        Index("ix_feedback_events_ws", "workspace_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, nullable=False, index=True)
    user_id = Column(Integer, nullable=True)
    source = Column(String(30), nullable=False)  # search/rag/citation/summary/extraction/agent/workflow
    target_id = Column(String(64), nullable=True)
    rating = Column(Integer, nullable=True)  # -1/0/+1
    comment = Column(Text, nullable=True)
    quality_flags = Column(Text, nullable=True)  # noisy/contradictory/duplicate/abuse
    status = Column(String(20), nullable=False, default="RECEIVED")
    # RECEIVED -> TRIAGED -> GOLDEN (approved into dataset) | DISCARDED
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Observability 5.0: incidents
# ---------------------------------------------------------------------------

class Incident(Base):
    """Correlated failures grouped into incidents."""

    __tablename__ = "incidents"
    __table_args__ = (
        Index("ix_incidents_status", "status", "severity"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(300), nullable=False)
    severity = Column(String(10), nullable=False, default="MEDIUM")  # LOW/MEDIUM/HIGH/CRITICAL
    status = Column(String(20), nullable=False, default="OPEN")
    # OPEN -> INVESTIGATING -> MITIGATED -> RESOLVED -> POSTMORTEM
    affected_systems = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    fingerprint = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class IncidentEvent(Base):
    """Persisted incident timeline."""

    __tablename__ = "incident_events"
    __table_args__ = (
        Index("ix_incident_events_incident", "incident_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id",
                                             ondelete="CASCADE"),
                         nullable=False, index=True)
    event_type = Column(String(30), nullable=False)  # detected/investigating/mitigated/resolved/postmortem/note
    detail = Column(Text, nullable=True)
    actor_user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# SLO 2.0: history + error budgets
# ---------------------------------------------------------------------------

class SloHistory(Base):
    """Historical SLO measurements."""

    __tablename__ = "slo_history"
    __table_args__ = (
        Index("ix_slo_history_name", "name", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    window = Column(String(10), nullable=False, default="daily")
    measured_value = Column(Float, nullable=True)
    target = Column(Float, nullable=True)
    met = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ErrorBudget(Base):
    """Error budget per SLO with consumed amount."""

    __tablename__ = "error_budgets"
    __table_args__ = (
        UniqueConstraint("name", "period", name="uq_error_budget_period"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    period = Column(String(10), nullable=False, default="monthly")
    budget = Column(Float, nullable=False, default=100.0)
    consumed = Column(Float, nullable=False, default=0.0)
    status = Column(String(20), nullable=False, default="OK")  # OK/WARNING/EXHAUSTED
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ReliabilityScore(Base):
    """Service reliability scoring snapshots."""

    __tablename__ = "reliability_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    service = Column(String(60), nullable=False, index=True)
    score = Column(Float, nullable=False)
    detail_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Notifications / alerts / preferences
# ---------------------------------------------------------------------------

class OpsAlertEvent(Base):
    """Prioritized, deduplicated operational alert events.

    Named distinctly from phase17.AlertEvent to avoid table collisions.
    """

    __tablename__ = "ops_alert_events"
    __table_args__ = (
        Index("ix_ops_alert_events_status", "status", "severity"),
        UniqueConstraint("fingerprint", name="uq_ops_alert_event_fingerprint"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(40), nullable=False)
    severity = Column(String(10), nullable=False)  # informational/low/medium/high/critical
    fingerprint = Column(String(64), nullable=False)
    message = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="OPEN")  # OPEN/ACKNOWLEDGED/RESOLVED
    occurrence_count = Column(Integer, nullable=False, default=1)
    workspace_id = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class NotificationPreference(Base):
    """Per-user/workspace notification category preferences."""

    __tablename__ = "notification_preferences"
    __table_args__ = (
        UniqueConstraint("user_id", "workspace_id", "category",
                         name="uq_notification_pref"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False, index=True)
    workspace_id = Column(Integer, nullable=True)
    category = Column(String(40), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# Reports + retention
# ---------------------------------------------------------------------------

class ReportVersion(Base):
    """Immutable versioned exportable reports."""

    __tablename__ = "report_versions"
    __table_args__ = (
        Index("ix_report_versions_kind", "kind", "scope_type"),
        UniqueConstraint("kind", "scope_type", "scope_id", "version",
                         name="uq_report_version"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(30), nullable=False)  # ai_quality/knowledge_health/governance/reliability
    scope_type = Column(String(20), nullable=False, default="WORKSPACE")
    scope_id = Column(Integer, nullable=True)
    version = Column(Integer, nullable=False)
    content_json = Column(Text, nullable=False)
    generated_by = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class ImprovementRetention(Base):
    """Phase 20 retention for improvement/experiment/incident/feedback data.

    Named distinctly from retention.RetentionPolicy to avoid collisions.
    """

    __tablename__ = "improvement_retention"
    __table_args__ = (
        UniqueConstraint("category", name="uq_improvement_retention_category"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(40), nullable=False)  # experiments/evaluation_runs/traces/incidents/feedback
    retention_days = Column(Integer, nullable=False, default=90)
    enabled = Column(Boolean, nullable=False, default=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


# ---------------------------------------------------------------------------
# API / database health metrics
# ---------------------------------------------------------------------------

class ApiHealthMetric(Base):
    """Per-endpoint request/latency/error aggregates."""

    __tablename__ = "api_health_metrics"
    __table_args__ = (
        Index("ix_api_health_metric_ep", "endpoint", "period"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    endpoint = Column(String(300), nullable=False, index=True)
    period = Column(String(10), nullable=False, default="hourly")
    requests = Column(Integer, nullable=False, default=0)
    errors = Column(Integer, nullable=False, default=0)
    latency_p50_ms = Column(Float, nullable=True)
    latency_p95_ms = Column(Float, nullable=True)
    auth_failures = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)


class DbHealthMetric(Base):
    """Database health snapshots (connections, slow queries, growth)."""

    __tablename__ = "db_health_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric = Column(String(60), nullable=False, index=True)
    value = Column(Float, nullable=True)
    detail_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)