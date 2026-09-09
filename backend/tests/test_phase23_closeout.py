"""Phase 23 tests — closeout parametrized matrices.

Deterministic matrices across routing sensitivity levels, admission gates,
reconciliation classifications, residency classifications, dependency
components, and stream kinds. Locks in the governed-autonomy invariants of
the Phase 23 platform.
"""

import pytest

from app.main import app  # noqa: F401  (register routes/models)
from tests.shared_db import TestingSessionLocal

from app.models.phase23 import (
    ProviderReadinessScore, ProviderRoutingDecision, ResidencyDecisionLog,
    DependencyEdge, KnowledgeFreshnessState, ConsistencyCheckRun,
    RequestDedupRecord, SchedulerTaskRun, WebhookDeliveryAttempt,
    WebhookEndpointHealth, ReviewDecision, StorageMigrationPlan,
    StorageMigrationObject, RegionDrainOperation, EmbeddingModelVersion,
    DocumentEmbeddingStatus, SearchShadowComparison,
)
from app.models.phase22 import CostReconciliationRun, OpsStreamEvent
from app.models.phase19 import ResidencyRule
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.services import provider_gate as pg
from app.services import global_control as gc
from app.services import region_control as rc
from app.services import security_eval3 as se3
from app.services import data_consistency as dc

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


TABLES = [
    ProviderReadinessScore, ProviderRoutingDecision, ResidencyDecisionLog,
    DependencyEdge, KnowledgeFreshnessState, ConsistencyCheckRun,
    RequestDedupRecord, SchedulerTaskRun, WebhookDeliveryAttempt,
    WebhookEndpointHealth, ReviewDecision, StorageMigrationPlan,
    StorageMigrationObject, RegionDrainOperation, EmbeddingModelVersion,
    DocumentEmbeddingStatus, SearchShadowComparison, CostReconciliationRun,
    OpsStreamEvent,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in TABLES:
        db_session.query(model).delete()
    db_session.query(ResidencyRule).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"co{n}@p23co.example", name="co", password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-co-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Routing across sensitivity levels (Step 7)
# ===========================================================================

ROUTING_MATRIX = [
    ("PUBLIC", True),
    ("INTERNAL", True),
    ("CONFIDENTIAL", True),
    ("RESTRICTED", True),
]


class TestRoutingMatrix:
    @pytest.mark.parametrize("sensitivity,routable", ROUTING_MATRIX)
    def test_default_policy_routes(self, db_session, sensitivity, routable):
        ws = _mkws(db_session)
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion",
                                   sensitivity=sensitivity)
        assert (result["decision"] == "ROUTED") is routable

    def test_restricted_with_blanket_prohibition_rejected(self, db_session):
        ws = _mkws(db_session)
        policy = {"restricted_providers": list(pg.PROVIDER_BASE_SCORES),
                  "max_context_chars": 100_000,
                  "emergency_fallback": "fake"}
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion",
                                   sensitivity="RESTRICTED", policy=policy)
        assert result["decision"] == "REJECTED"
        assert result["reasons"]  # audit evidence of every prohibition

    def test_all_decisions_persisted(self, db_session):
        ws = _mkws(db_session)
        for sensitivity, _ in ROUTING_MATRIX:
            pg.route_provider(db_session, workspace_id=ws.id,
                              operation="completion",
                              sensitivity=sensitivity)
        count = db_session.query(ProviderRoutingDecision).filter_by(
            workspace_id=ws.id).count()
        assert count == len(ROUTING_MATRIX)


# ===========================================================================
# Admission gate matrix (Step 8)
# ===========================================================================

class TestAdmissionMatrix:
    @pytest.mark.parametrize("cost,budget,admitted", [
        (0.0, None, True),
        (0.5, 1.0, True),
        (1.0, 1.0, True),
        (1.5, 1.0, False),
    ])
    def test_budget_gate(self, db_session, cost, budget, admitted):
        ws = _mkws(db_session)
        result = pg.admission_check(db_session, workspace_id=ws.id,
                                    operation="completion",
                                    estimated_cost=cost,
                                    budget_remaining=budget)
        assert result["admitted"] is admitted

    @pytest.mark.parametrize("context,admitted", [
        (0, True), (50_000, True), (100_000, True), (100_001, False),
    ])
    def test_context_gate(self, db_session, context, admitted):
        ws = _mkws(db_session)
        result = pg.admission_check(db_session, workspace_id=ws.id,
                                    operation="completion",
                                    context_chars=context)
        assert result["admitted"] is admitted


# ===========================================================================
# Reconciliation classification matrix (Step 9)
# ===========================================================================

class TestReconciliationMatrix:
    @pytest.mark.parametrize("est,actual,expected", [
        (0.01, 0.0100, "OK"),
        (0.01, 0.0101, "OK"),
        (0.01, 0.0500, "OVERBILLING_RISK"),
        (0.05, 0.0100, "UNDERBILLING_RISK"),
    ])
    def test_cost_classification(self, db_session, est, actual, expected):
        ws = _mkws(db_session)
        row = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai", model="m",
            estimated_tokens=100, actual_input_tokens=95,
            actual_output_tokens=5, estimated_cost=est, actual_cost=actual)
        assert row["classification"] == expected


# ===========================================================================
# Dependency impact matrix (Step 19)
# ===========================================================================

DEPENDENCY_MATRIX = [
    ("database", "api"), ("broker", "worker"), ("storage", "ingestion"),
]


class TestDependencyMatrix:
    @pytest.mark.parametrize("component,dependent", DEPENDENCY_MATRIX)
    def test_upstream_impact(self, db_session, component, dependent):
        impact = gc.dependency_impact(db_session, component)
        assert dependent in impact["upstream_hard"]

    def test_every_graph_component_resolves(self, db_session):
        for component in gc.known_components(db_session):
            impact = gc.dependency_impact(db_session, component)
            assert impact["component"] == component

    def test_unknown_component_raises(self, db_session):
        with pytest.raises(ValueError):
            gc.dependency_impact(db_session, "definitely-not-a-component")


# ===========================================================================
# Residency classification matrix (Step 14)
# ===========================================================================

class TestResidencyMatrix:
    @pytest.mark.parametrize("classification,dest,allowed", [
        ("PUBLIC", "eu", True),
        ("INTERNAL", "eu", True),
        ("RESTRICTED", "us", True),
        ("RESTRICTED", "eu", False),
    ])
    def test_residency_rules(self, db_session, classification, dest,
                             allowed):
        ws = _mkws(db_session)
        db_session.add(ResidencyRule(
            classification="RESTRICTED", allowed_regions_json='["us"]'))
        db_session.commit()
        result = rc.evaluate_residency(
            db_session, workspace_id=ws.id, operation="storage_write",
            source_region="us", destination_region=dest,
            classification=classification)
        assert result["allowed"] is allowed


# ===========================================================================
# Stream kind matrix (Step 62)
# ===========================================================================

STREAM_KINDS = ["worker", "provider", "incident", "ops", "broker"]


class TestStreamMatrix:
    @pytest.mark.parametrize("stream", STREAM_KINDS)
    def test_known_streams_accepted(self, db_session, stream):
        ws = _mkws(db_session)
        result = se3.emit_stream_event(db_session, workspace_id=ws.id,
                                       stream=stream, kind="event",
                                       payload={})
        assert result["seq"] >= 1

    @pytest.mark.parametrize("stream", ["nope", "", "ALL_CAPS"])
    def test_unknown_streams_rejected(self, db_session, stream):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            se3.emit_stream_event(db_session, workspace_id=ws.id,
                                  stream=stream, kind="event", payload={})


# ===========================================================================
# Consistency check matrix (Step 43)
# ===========================================================================

class TestConsistencyMatrix:
    @pytest.mark.parametrize("check", ["document_chunks",
                                       "cross_tenant_integrity",
                                       "memory_integrity",
                                       "execution_integrity"])
    def test_check_runs(self, db_session, check):
        ws = _mkws(db_session)
        result = dc.run_consistency_checks(db_session, workspace_id=ws.id,
                                           persist=False)
        assert check in result["by_check"]
        assert isinstance(result["by_check"][check], int)
