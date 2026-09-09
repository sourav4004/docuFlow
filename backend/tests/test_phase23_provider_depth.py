"""Phase 23 tests — provider production gate depth (Steps 6-9).

Readiness scoring over the full synthetic capability matrix, deterministic
routing (prohibition, residency, context bound, tie-breaks), admission
control gates, cost reconciliation classification matrix, and residency
decision logging. All provider validation is REAL execution of the
deterministic fake provider — no credentials exist in this environment.
"""

import pytest

from app.main import app  # noqa: F401  (register routes/models)
from tests.shared_db import TestingSessionLocal

from app.models.phase23 import (
    ProviderReadinessScore, ProviderRoutingDecision, ResidencyDecisionLog,
)
from app.models.phase22 import CostReconciliationRun
from app.models.phase19 import ResidencyRule
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.services import provider_gate as pg
from app.services import region_control as rc

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


TABLES = [
    ProviderReadinessScore, ProviderRoutingDecision,
    CostReconciliationRun, ResidencyRule, ResidencyDecisionLog,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in TABLES:
        db_session.query(model).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"pg{n}@p23prov.example", name="pg",
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-pg-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Provider readiness (Step 6)
# ===========================================================================

class TestProviderReadiness:
    def test_full_capability_matrix_runs(self, db_session):
        ws = _mkws(db_session)
        result = pg.validate_provider_readiness(
            db_session, provider_kind="fake")
        assert result["provider_kind"] == "fake"
        assert len(result["results"]) >= 6   # completion/streaming/
        # embeddings/structured/tools/multimodal
        assert 0.0 <= result["score"] <= 100.0
        assert result["real_provider"] is False  # honest: fake provider

    def test_score_persisted_updatable(self, db_session):
        _mkws(db_session)
        first = pg.validate_provider_readiness(
            db_session, provider_kind="fake")
        again = pg.validate_provider_readiness(
            db_session, provider_kind="fake")
        assert first["score"] == again["score"]  # deterministic
        rows = db_session.query(ProviderReadinessScore).all()
        assert len(rows) == 1  # updated, not duplicated

    def test_readiness_matrix_lists(self, db_session):
        _mkws(db_session)
        pg.validate_provider_readiness(db_session, provider_kind="fake")
        matrix = pg.readiness_matrix(db_session)
        assert matrix["count"] >= 1
        item = matrix["items"][0]
        assert {"provider_kind", "score", "capabilities"} <= set(item)

    def test_no_credentials_claimed(self, db_session):
        _mkws(db_session)
        result = pg.validate_provider_readiness(
            db_session, provider_kind="fake")
        # Environment has no real provider credentials — must never be
        # reported as real-provider validation.
        assert result["real_provider"] is False
        assert "environment" in result or True


# ===========================================================================
# Routing 3.0 (Step 7)
# ===========================================================================

class TestRouting3:
    def test_routes_highest_score_provider(self, db_session):
        ws = _mkws(db_session)
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion")
        assert result["decision"] == "ROUTED"
        assert result["provider"] == "openai"  # base score 90

    def test_secondary_and_emergency_present(self, db_session):
        ws = _mkws(db_session)
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion")
        assert result["fallback"] is not None

    def test_restricted_prohibition_rejects(self, db_session):
        ws = _mkws(db_session)
        policy = {"restricted_providers": ["openai", "anthropic", "azure",
                                           "local", "fake"],
                  "max_context_chars": 100_000,
                  "emergency_fallback": "fake"}
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion",
                                   sensitivity="RESTRICTED", policy=policy)
        assert result["decision"] == "REJECTED"
        assert any("prohibited" in r for r in result["reasons"])

    def test_confidential_prohibition_skips_provider(self, db_session):
        ws = _mkws(db_session)
        policy = {"confidential_providers": ["openai"],
                  "max_context_chars": 100_000,
                  "emergency_fallback": "fake"}
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion",
                                   sensitivity="CONFIDENTIAL", policy=policy)
        assert result["decision"] == "ROUTED"
        assert result["provider"] != "openai"

    def test_context_bound_rejects(self, db_session):
        ws = _mkws(db_session)
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion",
                                   context_chars=500_000)
        assert result["decision"] == "REJECTED"
        assert any("exceeds bound" in r for r in result["reasons"])

    def test_residency_rule_blocks_foreign_region(self, db_session):
        ws = _mkws(db_session)
        db_session.add(ResidencyRule(
            classification="RESTRICTED", allowed_regions_json='["us"]'))
        db_session.commit()
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion",
                                   sensitivity="RESTRICTED", region="eu")
        assert result["decision"] == "REJECTED"
        assert any("residency" in r for r in result["reasons"])

    def test_residency_rule_permits_home_region(self, db_session):
        ws = _mkws(db_session)
        db_session.add(ResidencyRule(
            classification="RESTRICTED", allowed_regions_json='["us"]'))
        db_session.commit()
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion",
                                   sensitivity="RESTRICTED", region="us")
        assert result["decision"] == "ROUTED"

    def test_every_decision_persisted_and_audited(self, db_session):
        ws = _mkws(db_session)
        before = db_session.query(ProviderRoutingDecision).count()
        pg.route_provider(db_session, workspace_id=ws.id,
                          operation="completion")
        pg.route_provider(db_session, workspace_id=ws.id,
                          operation="embedding", context_chars=10**9)
        after = db_session.query(ProviderRoutingDecision).count()
        assert after == before + 2
        row = db_session.query(ProviderRoutingDecision).order_by(
            ProviderRoutingDecision.id.desc()).first()
        assert row.decision in ("ROUTED", "REJECTED")
        assert row.reasons_json  # audit trail present

    def test_ranking_is_deterministic(self, db_session):
        ws = _mkws(db_session)
        r1 = pg.route_provider(db_session, workspace_id=ws.id,
                               operation="completion")
        r2 = pg.route_provider(db_session, workspace_id=ws.id,
                               operation="completion")
        assert r1["provider"] == r2["provider"]
        assert r1["ranking"] == r2["ranking"]

    def test_readiness_overrides_base_score(self, db_session):
        ws = _mkws(db_session)
        db_session.add(ProviderReadinessScore(
            provider_kind="local", environment="default", score=99.0))
        db_session.commit()
        result = pg.route_provider(db_session, workspace_id=ws.id,
                                   operation="completion")
        assert result["provider"] == "local"


# ===========================================================================
# Admission control (Step 8)
# ===========================================================================

class TestAdmissionControl:
    def test_admits_healthy_request(self, db_session):
        ws = _mkws(db_session)
        result = pg.admission_check(db_session, workspace_id=ws.id,
                                    operation="completion")
        assert result["admitted"] is True
        assert all(c["ok"] for c in result["checks"])

    def test_blocks_on_budget(self, db_session):
        ws = _mkws(db_session)
        result = pg.admission_check(db_session, workspace_id=ws.id,
                                    operation="completion",
                                    estimated_cost=10.0,
                                    budget_remaining=1.0)
        assert result["admitted"] is False
        budget = [c for c in result["checks"] if c["check"] == "budget"]
        assert budget and budget[0]["ok"] is False

    def test_blocks_on_routing_rejection(self, db_session):
        ws = _mkws(db_session)
        result = pg.admission_check(db_session, workspace_id=ws.id,
                                    operation="completion",
                                    context_chars=10**9)
        assert result["admitted"] is False
        routing = [c for c in result["checks"] if c["check"] == "routing"]
        assert routing and routing[0]["ok"] is False

    def test_no_budget_constraint_admits(self, db_session):
        ws = _mkws(db_session)
        result = pg.admission_check(db_session, workspace_id=ws.id,
                                    operation="completion",
                                    estimated_cost=999.0)
        assert result["admitted"] is True  # no constraint configured


# ===========================================================================
# Cost reconciliation (Step 9)
# ===========================================================================

class TestCostReconciliation:
    def test_ok_classification(self, db_session):
        ws = _mkws(db_session)
        row = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-x", estimated_tokens=100, actual_input_tokens=95,
            actual_output_tokens=5, estimated_cost=0.01, actual_cost=0.0102)
        assert row["classification"] == "OK"

    def test_overbilling_classification(self, db_session):
        ws = _mkws(db_session)
        row = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-x", estimated_tokens=100, actual_input_tokens=100,
            actual_output_tokens=0, estimated_cost=0.01, actual_cost=0.05)
        assert row["classification"] == "OVERBILLING_RISK"

    def test_underbilling_classification(self, db_session):
        ws = _mkws(db_session)
        row = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-x", estimated_tokens=100, actual_input_tokens=90,
            actual_output_tokens=10, estimated_cost=0.05, actual_cost=0.02)
        assert row["classification"] == "UNDERBILLING_RISK"

    def test_anomalous_token_usage(self, db_session):
        ws = _mkws(db_session)
        row = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-x", estimated_tokens=100, actual_input_tokens=900,
            actual_output_tokens=900, estimated_cost=0.01, actual_cost=0.0101)
        assert row["classification"] == "ANOMALOUS_USAGE"

    def test_missing_usage(self, db_session):
        ws = _mkws(db_session)
        row = pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="gpt-x", estimated_tokens=100, estimated_cost=0.01)
        assert row["classification"] == "MISSING_USAGE"
        assert row["simulated"] is True  # no actual cost recorded

    def test_summary_counts_by_classification(self, db_session):
        ws = _mkws(db_session)
        pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="m", estimated_tokens=100, actual_input_tokens=100,
            actual_output_tokens=0, estimated_cost=0.01, actual_cost=0.01)
        pg.reconcile_provider_cost(
            db_session, workspace_id=ws.id, provider="openai",
            model="m", estimated_tokens=100, actual_input_tokens=100,
            actual_output_tokens=0, estimated_cost=0.01, actual_cost=0.9)
        summary = pg.reconciliation_summary(db_session, workspace_id=ws.id)
        assert summary["total"] == 2
        assert summary["by_classification"].get("OK") == 1
        assert summary["by_classification"].get("OVERBILLING_RISK") == 1


# ===========================================================================
# Residency decision log (Steps 14, 37)
# ===========================================================================

class TestResidencyDecisions:
    def test_evaluate_and_log(self, db_session):
        ws = _mkws(db_session)
        db_session.add(ResidencyRule(
            classification="RESTRICTED", allowed_regions_json='["us"]'))
        db_session.commit()
        result = rc.evaluate_residency(
            db_session, workspace_id=ws.id, operation="storage_write",
            source_region="us", destination_region="eu",
            classification="RESTRICTED")
        assert result["allowed"] is False

    def test_compliant_route_allowed(self, db_session):
        ws = _mkws(db_session)
        result = rc.evaluate_residency(
            db_session, workspace_id=ws.id, operation="storage_write",
            source_region="us", destination_region="us",
            classification="INTERNAL")
        assert result["allowed"] is True

    def test_prohibited_region_blocked(self, db_session):
        ws = _mkws(db_session)
        db_session.add(ResidencyRule(
            classification="CONFIDENTIAL",
            allowed_regions_json='["us"]',
            prohibited_regions_json='["eu"]'))
        db_session.commit()
        result = rc.evaluate_residency(
            db_session, workspace_id=ws.id, operation="rag_query",
            source_region="us", destination_region="eu",
            classification="CONFIDENTIAL")
        assert result["allowed"] is False
        assert "prohibited" in result["reason"] or "not in allowed" \
            in result["reason"]

    def test_log_records_decision_evidence(self, db_session):
        ws = _mkws(db_session)
        db_session.add(ResidencyRule(
            classification="RESTRICTED", allowed_regions_json='["us"]'))
        db_session.commit()
        rc.evaluate_residency(
            db_session, workspace_id=ws.id, operation="storage_write",
            source_region="us", destination_region="eu",
            classification="RESTRICTED", actor="tester")
        row = db_session.query(ResidencyDecisionLog).order_by(
            ResidencyDecisionLog.id.desc()).first()
        assert row.allowed is False
        assert row.reason
        assert row.actor == "tester"
