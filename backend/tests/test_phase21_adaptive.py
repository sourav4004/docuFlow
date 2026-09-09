"""Phase 21 tests — adaptive knowledge, ingestion, retrieval, RAG.

Knowledge health dimensions + incident detection + governed recovery plans;
ingestion quality sampling, anomaly detection, adaptation candidates;
retrieval continuous evaluation + drift + bounded tuning candidates; RAG
quality dimensions, drift, failure clustering; candidate evaluation gates
and governed promotion (never without autonomy ALLOWED); model/provider
drift + routing simulation + cost autopilot (forecast, anomalies, guard,
audited optimizations, cost-aware scheduling with fairness).
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
from app.models.phase20 import RagFailure  # noqa: E402
from app.models.phase21 import (  # noqa: E402
    AdaptiveCandidate, AutonomousOperation, CostForecast, IngestionAnomaly,
    IngestionQualitySample, KnowledgeRecoveryPlan, ModelDriftEvent,
    ModelPerformanceSample, RetrievalDriftSnapshot, RoutingSimulation,
    CostGuardDecision, CostOptimizationEvent, CostAwareQueueDecision,
    RagDriftSnapshot, RagFailureCluster,
)
from app.models.phase21 import CostAnomaly as CostAnomalyP21  # noqa: E402
from app.services import autonomy as au  # noqa: E402
from app.services import knowledge_heal as kh  # noqa: E402
from app.services import adaptive_ai as ad  # noqa: E402
from app.services import autopilot as ap  # noqa: E402

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


P21_ADAPTIVE_TABLES = [
    AutonomousOperation, AdaptiveCandidate, IngestionAnomaly,
    CostForecast, KnowledgeRecoveryPlan, ModelDriftEvent,
    ModelPerformanceSample, RetrievalDriftSnapshot, RagFailure,
    IngestionQualitySample, RoutingSimulation, CostGuardDecision,
    CostOptimizationEvent, CostAnomalyP21, CostAwareQueueDecision,
    RagDriftSnapshot, RagFailureCluster,
]

# Emergency stops from earlier suites must never veto this suite's
# policy-allowed assertions (shared in-memory DB).
from app.models.phase21 import EmergencyStop as _EmergencyStop  # noqa: E402
from app.models.phase21 import PlatformEvent as _PlatformEvent  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in P21_ADAPTIVE_TABLES:
        db_session.query(model).delete()
    db_session.query(_EmergencyStop).delete()
    db_session.query(_PlatformEvent).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def fresh_workspace(db_session):
    _counter[0] += 1
    user = User(name=f"owner{_counter[0]}",
                email=f"owner{_counter[0]}@p21adapt.example",
                password_hash="x")
    db_session.add(user)
    db_session.flush()
    ws = Workspace(name=f"ws{_counter[0]}", owner_id=user.id)
    db_session.add(ws)
    db_session.commit()
    return ws.id


def allow_low_risk(db_session, ws, operation_type="recovery"):
    au.create_policy(db_session, ws, operation_type, risk_level="LOW",
                     autonomy_level="AUTO_LOW_RISK",
                     requires_approval=False)


# ===========================================================================
# Knowledge health + recovery
# ===========================================================================

class TestKnowledgeHealth:
    def test_full_health(self, db_session):
        ws = fresh_workspace(db_session)
        result = kh.knowledge_health(db_session, ws, {})
        assert result["overall"] == 100.0

    def test_dimension_scores(self, db_session):
        ws = fresh_workspace(db_session)
        result = kh.knowledge_health(db_session, ws, {
            "freshness_score": 80, "embedding_coverage": 60})
        assert result["dimensions"]["freshness"] == 80.0
        assert result["dimensions"]["embeddings"] == 60.0
        assert result["overall"] == 60.0  # min wins

    def test_clamped_to_100(self, db_session):
        ws = fresh_workspace(db_session)
        result = kh.knowledge_health(db_session, ws,
                                     {"freshness_score": 250})
        assert result["dimensions"]["freshness"] == 100.0

    def test_incident_detection_all_kinds(self):
        incidents = kh.detect_knowledge_incidents({
            "ingestion_failures": 2, "stale_document_count": 3,
            "embedding_coverage": 50, "graph_corruption": True,
            "memory_conflicts": 1, "connector_drift": True})
        kinds = {i["kind"] for i in incidents}
        assert kinds == set(kh.KNOWLEDGE_ISSUES)

    def test_healthy_signals_no_incidents(self):
        assert kh.detect_knowledge_incidents({
            "embedding_coverage": 99}) == []


class TestKnowledgeRecovery:
    def test_plan_created_as_proposal(self, db_session):
        ws = fresh_workspace(db_session)
        plan = kh.create_recovery_plan(db_session, ws, "stale_document",
                                       "document", target_id=5)
        assert plan.status == "PROPOSED"
        assert json.loads(plan.plan) == ["reprocess_document"]

    def test_unknown_issue_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            kh.create_recovery_plan(db_session, ws, "teleport_docs", "doc")

    def test_execute_blocked_by_default_policy(self, db_session):
        ws = fresh_workspace(db_session)
        plan = kh.create_recovery_plan(db_session, ws, "embedding_gap",
                                       "embedding")
        result = kh.execute_recovery_plan(db_session, plan)
        assert result["decision"] == "REQUIRES_APPROVAL"
        assert plan.status == "PROPOSED"

    def test_execute_allowed_with_policy(self, db_session):
        ws = fresh_workspace(db_session)
        allow_low_risk(db_session, ws, "knowledge.recovery.embedding_gap")
        plan = kh.create_recovery_plan(db_session, ws, "embedding_gap",
                                       "embedding")
        result = kh.execute_recovery_plan(db_session, plan)
        assert result["decision"] == "ALLOWED"
        assert plan.status == "COMPLETED"
        assert plan.executed_at is not None

    def test_execute_idempotent_per_plan(self, db_session):
        ws = fresh_workspace(db_session)
        allow_low_risk(db_session, ws, "knowledge.recovery.retry_ingestion")
        plan = kh.create_recovery_plan(db_session, ws, "ingestion_failure",
                                       "document")
        kh.execute_recovery_plan(db_session, plan)
        ops = db_session.query(AutonomousOperation).filter_by(
            workspace_id=ws).all()
        assert len(ops) == 1  # same idempotency key reused

    def test_high_risk_plan_requires_approval(self, db_session):
        ws = fresh_workspace(db_session)
        plan = kh.create_recovery_plan(db_session, ws, "graph_corruption",
                                       "graph")
        assert plan.risk_level == "MEDIUM"
        result = kh.execute_recovery_plan(db_session, plan)
        assert result["decision"] == "REQUIRES_APPROVAL"

    def test_recovery_audit_event(self, db_session):
        from app.models.phase21 import PlatformEvent
        ws = fresh_workspace(db_session)
        kh.audit_recovery(db_session, ws, 42, "ALLOWED")
        row = db_session.query(PlatformEvent).filter_by(
            workspace_id=ws).first()
        assert row.event_kind == "ops.knowledge_recovery"


# ===========================================================================
# Adaptive ingestion
# ===========================================================================

class TestIngestion:
    def test_sample_recorded(self, db_session):
        ws = fresh_workspace(db_session)
        row = kh.record_ingestion_sample(db_session, ws,
                                         extraction_quality=0.9)
        assert row.id is not None

    def test_anomaly_on_drop(self, db_session):
        ws = fresh_workspace(db_session)
        for _ in range(5):
            kh.record_ingestion_sample(db_session, ws, ocr_quality=0.9)
        kh.record_ingestion_sample(db_session, ws, ocr_quality=0.4)
        anomalies = kh.detect_ingestion_anomalies(db_session, ws)
        assert len(anomalies) == 1
        assert anomalies[0].metric == "ocr_quality"
        assert anomalies[0].drop_percent >= 25

    def test_no_anomaly_steady(self, db_session):
        ws = fresh_workspace(db_session)
        for _ in range(6):
            kh.record_ingestion_sample(db_session, ws, ocr_quality=0.9)
        assert kh.detect_ingestion_anomalies(db_session, ws) == []

    def test_too_few_samples_skips(self, db_session):
        ws = fresh_workspace(db_session)
        kh.record_ingestion_sample(db_session, ws, ocr_quality=0.1)
        assert kh.detect_ingestion_anomalies(db_session, ws) == []

    def test_adaptation_candidate_created(self, db_session):
        ws = fresh_workspace(db_session)
        for _ in range(5):
            kh.record_ingestion_sample(db_session, ws, chunk_quality=0.9)
        kh.record_ingestion_sample(db_session, ws, chunk_quality=0.3)
        anomalies = kh.detect_ingestion_anomalies(db_session, ws)
        candidate = kh.propose_ingestion_adaptation(db_session, ws,
                                                    anomalies[0])
        assert candidate.domain == "ingestion"
        assert candidate.status == "CANDIDATE"


# ===========================================================================
# Adaptive retrieval / RAG
# ===========================================================================

class TestRetrievalEvaluation:
    def test_metrics_batch(self):
        cases = [
            {"query": "a", "hit": True, "hit_rank": 1},
            {"query": "b", "hit": True, "hit_rank": 3},
            {"query": "c", "zero_result": True},
        ]
        metrics = ad.record_retrieval_evaluation(None, 0, cases)
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == pytest.approx((1.0 + 1 / 3) / 2, abs=1e-3)
        assert metrics["zero_result_rate"] == pytest.approx(1 / 3, abs=1e-3)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            ad.record_retrieval_evaluation(None, 0, [])

    def test_rejects_oversized(self):
        with pytest.raises(ValueError):
            ad.record_retrieval_evaluation(
                None, 0, [{"query": "x"}] * 501)

    def test_drift_persisted(self, db_session):
        ws = fresh_workspace(db_session)
        row = ad.record_retrieval_drift(db_session, ws, 0.8, 0.6)
        assert row.drift_percent == 25.0
        assert row.degraded is True

    def test_drift_not_degraded_small(self, db_session):
        ws = fresh_workspace(db_session)
        row = ad.record_retrieval_drift(db_session, ws, 0.8, 0.78)
        assert row.degraded is False

    def test_candidate_unknown_tunable_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            ad.propose_retrieval_candidate(db_session, ws, "teleport_weight",
                                           1.0, "why not")


class TestRagQuality:
    def test_full_quality(self):
        answers = [{"grounded": True, "citation_count": 2,
                    "citations_correct": True, "complete": True,
                    "temporal_ok": True}] * 4
        metrics = ad.record_rag_quality(None, 0, answers)
        assert metrics["groundedness"] == 1.0
        assert metrics["citation_correctness"] == 1.0
        assert metrics["temporal_correctness"] == 1.0

    def test_refusal_accuracy(self):
        answers = [{"refused": True, "refusal_correct": True},
                   {"refused": True, "refusal_correct": False},
                   {"refused": False, "grounded": True}]
        metrics = ad.record_rag_quality(None, 0, answers)
        assert metrics["refusal_accuracy"] == 0.5
        assert metrics["refusal_rate"] == pytest.approx(2 / 3, abs=1e-3)

    def test_rag_drift(self, db_session):
        ws = fresh_workspace(db_session)
        row = ad.record_rag_drift(db_session, ws, 0.9, 0.6)
        assert row.degraded is True
        assert row.drift_percent == pytest.approx(33.3, abs=0.1)

    def test_failure_clustering(self, db_session):
        ws = fresh_workspace(db_session)
        for claim in ("c1", "c2", "c3"):
            db_session.add(RagFailure(workspace_id=ws,
                                      failure_class="missing_context",
                                      claim=claim))
        db_session.add(RagFailure(workspace_id=ws,
                                  failure_class="citation_missing",
                                  claim="c4"))
        db_session.commit()
        clusters = ad.cluster_rag_failures(db_session, ws)
        assert clusters[0].cause == "missing_context"
        assert clusters[0].sample_count == 3
        assert "retrieval" in clusters[0].recommendation

    def test_rag_candidate_allowed_kinds(self, db_session):
        ws = fresh_workspace(db_session)
        cand = ad.propose_rag_candidate(db_session, ws, "evidence_threshold",
                                        0.7, "raise grounding")
        assert cand.domain == "rag"
        with pytest.raises(ValueError):
            ad.propose_rag_candidate(db_session, ws, "prompt", {}, "x")


# ===========================================================================
# Candidate evaluation gates + promotion
# ===========================================================================

class TestCandidateGates:
    def _candidate(self, db_session, ws):
        return kh.propose_ingestion_adaptation(
            db_session, ws,
            kh.detect_ingestion_anomalies(db_session, ws)[0])

    def _anomaly(self, db_session, ws):
        for _ in range(5):
            kh.record_ingestion_sample(db_session, ws, chunk_quality=0.9)
        kh.record_ingestion_sample(db_session, ws, chunk_quality=0.3)
        return kh.detect_ingestion_anomalies(db_session, ws)[0]

    def test_passing_gates(self, db_session):
        ws = fresh_workspace(db_session)
        cand = kh.propose_ingestion_adaptation(db_session, ws,
                                               self._anomaly(db_session, ws))
        row = kh.evaluate_candidate(db_session, cand, 0.9, 0.5)
        assert row.gates_passed is True
        assert row.status == "EVALUATED"

    def test_failing_quality_gate(self, db_session):
        ws = fresh_workspace(db_session)
        cand = kh.propose_ingestion_adaptation(db_session, ws,
                                               self._anomaly(db_session, ws))
        row = kh.evaluate_candidate(db_session, cand, 0.3, 0.5)
        assert row.gates_passed is False

    def test_regression_gate(self, db_session):
        ws = fresh_workspace(db_session)
        cand = kh.propose_ingestion_adaptation(db_session, ws,
                                               self._anomaly(db_session, ws))
        row = kh.evaluate_candidate(db_session, cand, 0.2, 0.9)
        assert row.gates_passed is False

    def test_promotion_requires_gates(self, db_session):
        ws = fresh_workspace(db_session)
        cand = kh.propose_ingestion_adaptation(db_session, ws,
                                               self._anomaly(db_session, ws))
        kh.evaluate_candidate(db_session, cand, 0.2, 0.9)
        result = ad.promote_candidate(db_session, cand)
        assert result["decision"] == "BLOCKED"

    def test_promotion_requires_autonomy(self, db_session):
        ws = fresh_workspace(db_session)
        cand = kh.propose_ingestion_adaptation(db_session, ws,
                                               self._anomaly(db_session, ws))
        kh.evaluate_candidate(db_session, cand, 0.95, 0.5)
        result = ad.promote_candidate(db_session, cand)
        assert result["decision"] == "REQUIRES_APPROVAL"
        assert cand.promoted is False

    def test_promotion_with_full_governance(self, db_session):
        ws = fresh_workspace(db_session)
        allow_low_risk(db_session, ws,
                       "adaptation.promote.ingestion")
        cand = kh.propose_ingestion_adaptation(db_session, ws,
                                               self._anomaly(db_session, ws))
        kh.evaluate_candidate(db_session, cand, 0.95, 0.5)
        result = ad.promote_candidate(db_session, cand)
        assert result["decision"] == "ALLOWED"
        assert cand.status == "PROMOTED"
        assert cand.promoted is True


# ===========================================================================
# Model / provider autopilot
# ===========================================================================

class TestModelAutopilot:
    def test_sample_recorded(self, db_session):
        ws = fresh_workspace(db_session)
        row = ap.record_model_sample(db_session, ws, "gpt-test", "fake",
                                     quality_score=0.9)
        assert row.id is not None

    def test_model_drift_detected(self, db_session):
        ws = fresh_workspace(db_session)
        for _ in range(5):
            ap.record_model_sample(db_session, ws, "m1", "fake",
                                   quality_score=0.9)
        ap.record_model_sample(db_session, ws, "m1", "fake",
                               quality_score=0.5)
        event = ap.detect_model_drift(db_session, ws, "m1", "fake")
        assert event is not None
        assert event.direction == "DEGRADED"
        assert event.drop_percent >= 15

    def test_no_drift_steady(self, db_session):
        ws = fresh_workspace(db_session)
        for _ in range(6):
            ap.record_model_sample(db_session, ws, "m2", "fake",
                                   quality_score=0.9)
        assert ap.detect_model_drift(db_session, ws, "m2", "fake") is None

    def test_provider_drift_detected(self, db_session):
        ws = fresh_workspace(db_session)
        for _ in range(5):
            ap.record_model_sample(db_session, ws, "m3", "provA",
                                   availability=0.99)
        ap.record_model_sample(db_session, ws, "m3", "provA",
                               availability=0.5)
        event = ap.detect_provider_drift(db_session, ws, "provA")
        assert event is not None
        assert event.kind == "PROVIDER"

    def test_routing_simulation_safe(self, db_session):
        ws = fresh_workspace(db_session)
        workload = [{"task": "summarize",
                     "model_perf": {"fast": {"cost_usd": 0.01,
                                             "latency_ms": 200,
                                             "quality": 0.8}}}] * 10
        sim = ap.simulate_routing(db_session, ws, "use fast model", workload,
                                  {"summarize": "slow"},
                                  {"summarize": "fast"})
        assert sim.safe is True
        assert sim.executed is False
        assert sim.id is not None

    def test_routing_simulation_budget_unsafe(self, db_session):
        ws = fresh_workspace(db_session)
        workload = [{"task": "t", "model_perf": {
            "expensive": {"cost_usd": 1.0, "latency_ms": 100,
                          "quality": 0.99}}}] * 5
        sim = ap.simulate_routing(db_session, ws, "expensive switch",
                                  workload, {"t": "expensive"},
                                  {"t": "expensive"},
                                  constraints={"budget_multiplier": 0.5})
        assert sim.safe is False

    def test_routing_sensitivity_block(self, db_session):
        ws = fresh_workspace(db_session)
        workload = [{"task": "t", "model_perf": {
            "a": {"cost_usd": 0.01, "latency_ms": 10, "quality": 0.9}}}]
        sim = ap.simulate_routing(db_session, ws, "sensitive", workload,
                                  {"t": "a"}, {"t": "a"},
                                  constraints={
                                      "sensitive_task_uses_unapproved_model":
                                      True})
        assert sim.safe is False

    def test_routing_promotion_governed(self, db_session):
        ws = fresh_workspace(db_session)
        workload = [{"task": "t", "model_perf": {
            "b": {"cost_usd": 0.01, "latency_ms": 10, "quality": 0.9}}}]
        sim = ap.simulate_routing(db_session, ws, "candidate", workload,
                                  {"t": "a"}, {"t": "b"})
        cand = ap.propose_routing_candidate(db_session, ws, "candidate",
                                            workload)
        result = ap.promote_routing(db_session, sim, cand)
        assert result["decision"] == "REQUIRES_APPROVAL"

    def test_routing_promotion_blocked_unsafe(self, db_session):
        ws = fresh_workspace(db_session)
        workload = [{"task": "t", "model_perf": {
            "a": {"cost_usd": 0.01, "latency_ms": 10, "quality": 0.9}}}]
        sim = ap.simulate_routing(db_session, ws, "bad", workload,
                                  {"t": "a"}, {"t": "a"},
                                  constraints={"policy_violation": True})
        cand = ap.propose_routing_candidate(db_session, ws, "bad", workload)
        result = ap.promote_routing(db_session, sim, cand)
        assert result["decision"] == "BLOCKED"


# ===========================================================================
# Cost autopilot
# ===========================================================================

class TestCostAutopilot:
    def test_forecast_over_budget(self, db_session):
        ws = fresh_workspace(db_session)
        row = ap.record_cost_snapshot(db_session, ws, 100.0,
                                      daily_run_rate_usd=10.0,
                                      period_days=30, budget_usd=250.0)
        assert row.forecast_usd == 400.0
        assert row.over_budget is True

    def test_forecast_within_budget(self, db_session):
        ws = fresh_workspace(db_session)
        row = ap.record_cost_snapshot(db_session, ws, 100.0,
                                      daily_run_rate_usd=1.0,
                                      budget_usd=500.0)
        assert row.over_budget is False

    def test_cost_anomaly_detected(self, db_session):
        ws = fresh_workspace(db_session)
        row = ap.detect_cost_anomaly(db_session, ws, 100.0, 300.0)
        assert row is not None
        assert row.increase_percent == 200.0
        assert row.severity == "HIGH"

    def test_cost_anomaly_not_detected(self, db_session):
        ws = fresh_workspace(db_session)
        assert ap.detect_cost_anomaly(db_session, ws, 100.0, 110.0) is None

    def test_cost_guard_blocks_over_budget(self, db_session):
        ws = fresh_workspace(db_session)
        row = ap.cost_guard(db_session, ws, "big.op", 500.0,
                            remaining_budget_usd=100.0)
        assert row.decision == "BLOCKED"

    def test_cost_guard_allows_within(self, db_session):
        ws = fresh_workspace(db_session)
        row = ap.cost_guard(db_session, ws, "small.op", 5.0,
                            remaining_budget_usd=100.0)
        assert row.decision == "ALLOWED"

    def test_cost_guard_idempotent(self, db_session):
        ws = fresh_workspace(db_session)
        key = f"cg-{uuid.uuid4()}"
        a = ap.cost_guard(db_session, ws, "op.x", 1.0,
                          remaining_budget_usd=10.0, idempotency_key=key)
        b = ap.cost_guard(db_session, ws, "op.x", 1.0,
                          remaining_budget_usd=10.0, idempotency_key=key)
        assert a.id == b.id

    def test_unapproved_optimization_blocked(self, db_session):
        ws = fresh_workspace(db_session)
        result = ap.apply_cost_optimization(db_session, ws, "delete_backups")
        assert result["decision"] == "BLOCKED"

    def test_preapproved_optimization_governed(self, db_session):
        ws = fresh_workspace(db_session)
        result = ap.apply_cost_optimization(db_session, ws, "cache_reuse")
        # Without an allowing policy, the guard must not apply it.
        assert result["applied"] is False
        assert result["decision"] in ("REQUIRES_APPROVAL", "BLOCKED")

    def test_optimization_applied_with_policy(self, db_session):
        ws = fresh_workspace(db_session)
        allow_low_risk(db_session, ws, "cost.optimize.cache_reuse")
        result = ap.apply_cost_optimization(db_session, ws, "cache_reuse",
                                            estimated_savings_usd=12.5)
        assert result["applied"] is True
        assert result["decision"] == "ALLOWED"

    def test_optimization_audit_recorded(self, db_session):
        from app.models.phase21 import CostOptimizationEvent
        ws = fresh_workspace(db_session)
        allow_low_risk(db_session, ws, "cost.optimize.batching")
        ap.apply_cost_optimization(db_session, ws, "batching")
        rows = db_session.query(CostOptimizationEvent).filter_by(
            workspace_id=ws).all()
        assert len(rows) == 1
        assert rows[0].applied is True
        assert rows[0].reason  # audit: why it occurred


class TestCostAwareScheduling:
    def test_urgency_prioritization(self, db_session):
        ws = fresh_workspace(db_session)
        critical = ap.schedule_job(db_session, ws, urgency="CRITICAL")
        normal = ap.schedule_job(db_session, ws, urgency="NORMAL")
        assert critical.priority < normal.priority

    def test_budget_tight_lowers_priority_not_starves(self, db_session):
        ws = fresh_workspace(db_session)
        row = ap.schedule_job(db_session, ws, urgency="NORMAL",
                              estimated_cost_usd=95.0,
                              remaining_budget_usd=100.0)
        assert row.budget_state == "TIGHT"
        assert row.fairness_preserved is True

    def test_budget_exceeded_requires_approval_with_floor(self, db_session):
        ws = fresh_workspace(db_session)
        row = ap.schedule_job(db_session, ws, urgency="LOW",
                              estimated_cost_usd=500.0,
                              remaining_budget_usd=100.0)
        assert row.budget_state == "EXCEEDED"
        assert row.guard == "REQUIRES_APPROVAL"
        assert row.priority <= 9  # never fully starved (fairness floor)
        assert row.fairness_preserved is True

    def test_wealthy_tenant_cannot_monopolize(self, db_session):
        ws = fresh_workspace(db_session)
        priorities = [ap.schedule_job(
            db_session, ws, urgency="NORMAL",
            estimated_cost_usd=90.0 + i,
            remaining_budget_usd=100.0).priority for i in range(10)]
        # Guard caps priority (no monopolization) and keeps a fairness floor.
        assert all(p >= 5 for p in priorities)
