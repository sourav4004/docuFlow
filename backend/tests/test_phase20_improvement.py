"""Phase 20 tests — improvement control plane + experiment platform.

Proposal lifecycle validation and audit, immutable experiment configs,
golden/synthetic datasets, persisted runs, deterministic comparison
verdicts with minimum sample sizes, and promotion gates.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.models.phase20 import (  # noqa: E402
    ImprovementProposal, ImprovementAudit, ImprovementTransition,
    ExperimentDataset, Experiment, ExperimentRun, ExperimentComparison,
    QualityScorecard, QualityTrend, QualityAlert,
    RetrievalFailure, RetrievalRecommendation, RagFailure,
    RagEvaluationPipeline, KnowledgeHealth, KnowledgeGapInsight,
    DocChangeEvent, PolicyVersion, PolicyImpact, ProviderProfile,
    RoutingRecommendation, ProviderAnomaly, CostBaseline, TokenEfficiency,
    AgentIntelligence, WorkflowIntelligence, MemoryIntelligence,
    GraphHealth, GraphRecommendation, SearchQualityEvent, FeedbackEvent,
    Incident, IncidentEvent, SloHistory, ErrorBudget, ReliabilityScore,
    OpsAlertEvent, NotificationPreference, ReportVersion,
    ImprovementRetention, ApiHealthMetric, DbHealthMetric,
)
from app.models.phase15 import AIMemory  # noqa: E402
from app.services import improvement_platform as imp
from app.services import quality3 as q3
from app.services import retrieval_intel as reti
from app.services import rag_intel as ragi
from app.services import knowledge_intel as ki
from app.services import policy_intel as pi
from app.services import provider_intel as pri
from app.services import cost_intel as ci
from app.services import agent_intel as ai2
from app.services import workflow_intel2 as wi
from app.services import memory_intel as mi
from app.services import graph_intel as gi
from app.services import search_intel as si
from app.services import feedback_intel as fi
from app.services import governance5 as g5
from app.services import safety9 as s9
from app.services import observability5 as ob5
from app.services import slo2
from app.services import notifications_intel as ni
from app.services import reporting3 as rep
from app.services import api_intel as apii
from app.services import db_intel as dbi

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    return TestClient(app)


PH20_TABLES = [
    ImprovementProposal, ImprovementAudit, ImprovementTransition,
    ExperimentDataset, Experiment, ExperimentRun, ExperimentComparison,
    QualityScorecard, QualityTrend, QualityAlert, RetrievalFailure,
    RetrievalRecommendation, RagFailure, RagEvaluationPipeline,
    KnowledgeHealth, KnowledgeGapInsight, DocChangeEvent, PolicyVersion,
    PolicyImpact, ProviderProfile, RoutingRecommendation, ProviderAnomaly,
    CostBaseline, TokenEfficiency, AgentIntelligence, WorkflowIntelligence,
    MemoryIntelligence, GraphHealth, GraphRecommendation,
    SearchQualityEvent, FeedbackEvent, Incident, IncidentEvent, SloHistory,
    ErrorBudget, ReliabilityScore, OpsAlertEvent, NotificationPreference,
    ReportVersion, ImprovementRetention, ApiHealthMetric, DbHealthMetric,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in PH20_TABLES:
        db_session.query(model).delete()
    db_session.query(AIMemory).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.commit()
    yield


def _definition(trigger="document.ready", nodes=None):
    return {"trigger": trigger,
            "nodes": nodes if nodes is not None else
            [{"id": "s", "type": "summarize", "name": "S", "inputs": []}]}


# ===========================================================================
# Improvement proposals / lifecycle
# ===========================================================================

class TestProposals:
    def test_create_proposal_default_status(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T",
            problem="zero results", proposed_change="change weights")
        assert p.status == "PROPOSED"
        assert p.domain == "retrieval"

    def test_create_proposal_writes_audit(self, db_session):
        p = imp.create_proposal(
            db_session, domain="rag", title="T", problem="hallucinations",
            proposed_change="gate evidence")
        audit = db_session.query(ImprovementAudit).filter_by(
            proposal_id=p.id).all()
        assert len(audit) == 1
        assert audit[0].new_state == "PROPOSED"
        assert audit[0].previous_state is None

    def test_unknown_domain_rejected(self, db_session):
        with pytest.raises(ValueError):
            imp.create_proposal(db_session, domain="teleport", title="T",
                                problem="p", proposed_change="c")

    def test_ai_proposal_stays_proposed(self, db_session):
        p = imp.create_proposal(
            db_session, domain="model_routing", title="AI suggestion",
            problem="cost", proposed_change="use smaller model",
            author_source="ai")
        assert p.status == "PROPOSED"  # never auto-active

    def test_valid_lifecycle_to_active(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        imp.transition(db_session, p.id, "EVALUATING", reason="evaluate")
        imp.transition(db_session, p.id, "APPROVAL_REQUIRED",
                       reason="eval done")
        imp.transition(db_session, p.id, "APPROVED", reason="approved")
        imp.transition(db_session, p.id, "STAGED", reason="staged")
        result = imp.transition(db_session, p.id, "ACTIVE", reason="go")
        assert result["new_state"] == "ACTIVE"
        assert result["approved"] is True

    def test_cannot_skip_to_active(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        with pytest.raises(ValueError):
            imp.transition(db_session, p.id, "ACTIVE")

    def test_activation_records_approval_audit(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        for s in ("EVALUATING", "APPROVAL_REQUIRED", "APPROVED",
                  "STAGED"):
            imp.transition(db_session, p.id, s, reason="fwd")
        imp.transition(db_session, p.id, "ACTIVE", reason="go")
        states = [a["new_state"] for a in
                  imp.proposal_audit_trail(db_session, p.id)]
        assert "APPROVED" in states
        assert "ACTIVE" in states

    def test_cannot_activate_directly_from_approved(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        imp.transition(db_session, p.id, "EVALUATING")
        imp.transition(db_session, p.id, "APPROVAL_REQUIRED")
        imp.transition(db_session, p.id, "APPROVED")
        with pytest.raises(ValueError):
            imp.transition(db_session, p.id, "ACTIVE")  # needs STAGED first

    def test_illegal_transition_rejected(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        with pytest.raises(ValueError):
            imp.transition(db_session, p.id, "APPROVED")  # skip states

    def test_self_transition_rejected(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        with pytest.raises(ValueError):
            imp.transition(db_session, p.id, "PROPOSED")

    def test_rejection_path(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        imp.transition(db_session, p.id, "REJECTED", reason="not worth it")
        assert db_session.get(ImprovementProposal, p.id).status == "REJECTED"

    def test_rollback_after_active(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        for s in ("EVALUATING", "APPROVAL_REQUIRED", "APPROVED", "STAGED"):
            imp.transition(db_session, p.id, s, reason="fwd")
        imp.transition(db_session, p.id, "ACTIVE", reason="go")
        imp.transition(db_session, p.id, "ROLLED_BACK", reason="bad")
        assert db_session.get(ImprovementProposal,
                              p.id).status == "ROLLED_BACK"

    def test_audit_trail_records_actor_and_reason(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c", author_user_id=7)
        imp.transition(db_session, p.id, "EVALUATING", actor_user_id=3,
                       reason="start eval")
        trail = imp.proposal_audit_trail(db_session, p.id)
        assert len(trail) == 2
        assert trail[0]["actor_user_id"] == 3
        assert trail[0]["previous_state"] == "PROPOSED"
        assert trail[0]["reason"] == "start eval"

    def test_transition_unknown_proposal(self, db_session):
        with pytest.raises(KeyError):
            imp.transition(db_session, 999999, "EVALUATING")

    def test_list_proposals_filtered(self, db_session):
        imp.create_proposal(db_session, domain="retrieval", title="A",
                            problem="p", proposed_change="c")
        imp.create_proposal(db_session, domain="rag", title="B",
                            problem="p", proposed_change="c")
        rows = imp.list_proposals(db_session, domain="retrieval")
        assert len(rows) == 1
        assert rows[0]["title"] == "A"

    def test_transition_table_custom_row_wins(self, db_session):
        db_session.add(ImprovementTransition(
            from_state="PROPOSED", to_state="APPROVED",
            requires_approval=True))
        db_session.commit()
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        result = imp.transition(db_session, p.id, "APPROVED")
        assert result["new_state"] == "APPROVED"

    def test_proposal_domain_validation_all_domains(self, db_session):
        for domain in ("retrieval", "rag", "model_routing", "provider_routing",
                       "prompt", "workflow", "ingestion", "search", "cost",
                       "latency", "knowledge_quality", "agent", "memory",
                       "graph"):
            p = imp.create_proposal(db_session, domain=domain, title=domain,
                                    problem="p", proposed_change="c")
            assert p.status == "PROPOSED"


# ===========================================================================
# Experiment datasets + experiments
# ===========================================================================

class TestExperiments:
    def test_create_dataset_golden(self, db_session):
        ds = imp.create_dataset(
            db_session, name="golden_retrieval", kind="golden",
            items=[{"query": "q", "relevant": ["d1"]}])
        assert ds.kind == "golden"

    def test_dataset_name_unique(self, db_session):
        imp.create_dataset(db_session, name="dup", kind="golden", items=[])
        with pytest.raises(ValueError):
            imp.create_dataset(db_session, name="dup", kind="golden",
                               items=[])

    def test_dataset_kinds(self, db_session):
        for kind in ("golden", "synthetic", "anonymized", "curated"):
            ds = imp.create_dataset(db_session, name=f"ds_{kind}", kind=kind,
                                    items=[])
            assert ds.kind == kind

    def test_bad_dataset_kind(self, db_session):
        with pytest.raises(ValueError):
            imp.create_dataset(db_session, name="x", kind="random", items=[])

    def test_create_experiment_immutable_fingerprint(self, db_session):
        exp = imp.create_experiment(db_session, name="e1", domain="retrieval",
                                    config={"chunk_size": 500})
        fp1 = exp.config_fingerprint
        exp2 = imp.create_experiment(db_session, name="e2", domain="retrieval",
                                     config={"chunk_size": 500})
        assert fp1 == exp2.config_fingerprint  # deterministic

    def test_experiment_config_immutable_on_disk(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="rag",
                                    config={"k": 5})
        cfg = imp.get_experiment_config(db_session, exp.id)
        cfg["k"] = 999  # mutating the copy must not change the persisted one
        again = imp.get_experiment_config(db_session, exp.id)
        assert again["k"] == 5

    def test_experiment_domains(self, db_session):
        for domain in ("retrieval", "reranking", "model_routing", "prompt",
                       "chunking", "embeddings", "search_ranking", "rag"):
            exp = imp.create_experiment(db_session, name=f"e_{domain}",
                                        domain=domain, config={})
            assert exp.domain == domain

    def test_record_run_persists_metrics(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        run = imp.record_run(db_session, experiment_id=exp.id,
                             metrics={"quality_score": 0.91,
                                      "sample_size": 20,
                                      "precision": 0.8},
                             cost=0.01, latency_ms=120.0)
        assert run.status == "DONE"
        stored = json.loads(run.metrics_json)
        assert stored["quality_score"] == 0.91

    def test_comparison_insufficient_sample(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        r1 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"quality_score": 0.8, "sample_size": 3})
        r2 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"quality_score": 0.9, "sample_size": 3})
        result = imp.compare_runs(db_session, experiment_id=exp.id,
                                  baseline_run_id=r1.id,
                                  candidate_run_id=r2.id,
                                  metrics=["quality_score"])
        assert result["verdict"] == "INCONCLUSIVE"

    def test_comparison_candidate_better(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        r1 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"quality_score": 0.7, "sample_size": 20})
        r2 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"quality_score": 0.9, "sample_size": 20})
        result = imp.compare_runs(db_session, experiment_id=exp.id,
                                  baseline_run_id=r1.id,
                                  candidate_run_id=r2.id,
                                  metrics=["quality_score"])
        assert result["verdict"] == "CANDIDATE_BETTER"
        assert result["wins"] == 1 and result["losses"] == 0

    def test_comparison_baseline_better(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        r1 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"quality_score": 0.9, "sample_size": 20})
        r2 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"quality_score": 0.6, "sample_size": 20})
        result = imp.compare_runs(db_session, experiment_id=exp.id,
                                  baseline_run_id=r1.id,
                                  candidate_run_id=r2.id,
                                  metrics=["quality_score"])
        assert result["verdict"] == "BASELINE_BETTER"

    def test_comparison_multi_metric_wins(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        r1 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"precision": 0.8, "recall": 0.8,
                                     "sample_size": 20})
        r2 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"precision": 0.9, "recall": 0.95,
                                     "sample_size": 20})
        result = imp.compare_runs(db_session, experiment_id=exp.id,
                                  baseline_run_id=r1.id,
                                  candidate_run_id=r2.id,
                                  metrics=["precision", "recall"])
        assert result["verdict"] == "CANDIDATE_BETTER"
        assert result["wins"] == 2

    def test_comparison_persisted(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        r1 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"quality_score": 0.5, "sample_size": 20})
        r2 = imp.record_run(db_session, experiment_id=exp.id,
                            metrics={"quality_score": 0.8, "sample_size": 20})
        imp.compare_runs(db_session, experiment_id=exp.id,
                         baseline_run_id=r1.id, candidate_run_id=r2.id,
                         metrics=["quality_score"])
        cmp_row = db_session.query(ExperimentComparison).first()
        assert cmp_row.verdict == "CANDIDATE_BETTER"

    def test_promotion_gate_requires_runs(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        result = imp.promotion_eligible(db_session, exp.id)
        assert result["eligible"] is False

    def test_promotion_gate_quality_fail(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        imp.record_run(db_session, experiment_id=exp.id,
                       metrics={"quality_score": 0.5, "sample_size": 10})
        result = imp.promotion_eligible(db_session, exp.id,
                                        min_quality=0.8)
        assert result["eligible"] is False
        assert any("quality" in r for r in result["reasons"])

    def test_promotion_gate_quality_pass(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        imp.record_run(db_session, experiment_id=exp.id,
                       metrics={"quality_score": 0.9, "sample_size": 10},
                       cost=0.01, latency_ms=50.0)
        result = imp.promotion_eligible(db_session, exp.id, min_quality=0.8,
                                        max_cost=0.5, max_latency_ms=1000)
        assert result["eligible"] is True

    def test_promotion_gate_cost_fail(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        imp.record_run(db_session, experiment_id=exp.id,
                       metrics={"quality_score": 0.95}, cost=2.0)
        result = imp.promotion_eligible(db_session, exp.id, min_quality=0.8,
                                        max_cost=1.0)
        assert result["eligible"] is False

    def test_promotion_gate_latency_fail(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={"k": 3})
        imp.record_run(db_session, experiment_id=exp.id,
                       metrics={"quality_score": 0.95}, latency_ms=5000.0)
        result = imp.promotion_eligible(db_session, exp.id, min_quality=0.8,
                                        max_latency_ms=1000.0)
        assert result["eligible"] is False

    def test_list_runs_newest_first(self, db_session):
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={})
        imp.record_run(db_session, experiment_id=exp.id,
                       metrics={"quality_score": 0.7})
        imp.record_run(db_session, experiment_id=exp.id,
                       metrics={"quality_score": 0.9})
        runs = imp.list_runs(db_session, exp.id)
        assert len(runs) == 2
        assert runs[0]["id"] > runs[1]["id"]

    def test_experiment_links_to_proposal(self, db_session):
        p = imp.create_proposal(
            db_session, domain="retrieval", title="T", problem="p",
            proposed_change="c")
        exp = imp.create_experiment(db_session, name="e", domain="retrieval",
                                    config={}, proposal_id=p.id)
        assert exp.proposal_id == p.id

    def test_experiment_isolation_workspace_scoped(self, db_session):
        imp.create_experiment(db_session, name="ws1", domain="retrieval",
                              config={}, workspace_id=1)
        imp.create_experiment(db_session, name="ws2", domain="retrieval",
                              config={}, workspace_id=2)
        rows = imp.list_experiments(db_session, workspace_id=1)
        assert len(rows) == 1 and rows[0]["workspace_id"] == 1


# ===========================================================================
# Search-intel ranking experiment integration
# ===========================================================================

class TestSearchRankingExperiments:
    def test_ranking_experiment_created(self, db_session):
        exp = si.ranking_experiment(db_session, name="rank_exp",
                                    config={"alpha": 0.7})
        assert exp["status"] == "DRAFT"
        assert exp["config_fingerprint"]

    def test_ranking_experiment_is_immutable(self, db_session):
        si.ranking_experiment(db_session, name="r2", config={"w": 1})
        rows = db_session.query(Experiment).filter_by(name="r2").all()
        assert rows[0].config_fingerprint  # fingerprint = immutable marker

    def test_ranking_experiment_domain(self, db_session):
        si.ranking_experiment(db_session, name="r3", config={"x": 1})
        assert db_session.query(Experiment).filter_by(
            name="r3").first().domain == "search_ranking"