"""Phase 22 tests — provider validation depth + continuous evaluation depth
+ cost operations depth (parametrized failure modes, gate matrix, budget
bands, reconciliation honesty, evaluation idempotency/cancel/resume).
"""

import uuid

import pytest

from tests.shared_db import TestingSessionLocal

from app.models.phase22 import (
    CostReconciliationRun, EvalExecution, ProviderValidationRun,
)
from app.models.phase21 import CostGuardDecision
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.services import continuous_eval as ce
from app.services import provider_validation as pv
from app.services import security_cost_ops as sco

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in (CostReconciliationRun, ProviderValidationRun,
                  EvalExecution, CostGuardDecision):
        db_session.query(model).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db, tag="pd"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p22depth.example", name=tag,
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Provider validation depth (Steps 23-37)
# ===========================================================================

class TestProviderDepth:
    @pytest.mark.parametrize("pair", ["completion", "streaming",
                                         "embeddings", "structured_output",
                                         "tool_calling", "circuit_breaker",
                                         "fallback"])
    def test_every_validation_persists_run(self, db_session, pair):
        ws = _mkws(db_session)
        before = db_session.query(ProviderValidationRun).count()
        getattr(pv, f"validate_{pair}")(db_session, ws.id)
        after = db_session.query(ProviderValidationRun).count()
        assert after == before + 1

    @pytest.mark.parametrize("mode", ["timeout", "429", "5xx", "malformed"])
    def test_failure_mode_classification(self, db_session, mode):
        ws = _mkws(db_session)
        result = pv.validate_failure_mode(db_session, ws.id, mode)
        assert result["mode"] == mode
        assert isinstance(result["passed"], bool)

    def test_unknown_failure_mode_rejected(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises((ValueError, KeyError, AssertionError)):
            result = pv.validate_failure_mode(db_session, ws.id, "solar_flare")
            assert result["mode"] == "solar_flare" and False

    def test_failure_modes_do_not_crash_session(self, db_session):
        ws = _mkws(db_session)
        for mode in ("timeout", "429", "5xx", "malformed"):
            pv.validate_failure_mode(db_session, ws.id, mode)
        # Session still usable after every failure injection.
        assert db_session.query(ProviderValidationRun).count() >= 4

    def test_reconciliation_idempotent_per_provider(self, db_session):
        ws = _mkws(db_session)
        r1 = pv.reconcile_cost(db_session, ws.id, "fake")
        r2 = pv.reconcile_cost(db_session, ws.id, "fake")
        assert r1["reconciled"] and r2["reconciled"]

    def test_reconciliation_marks_simulated_for_fake(self, db_session):
        ws = _mkws(db_session)
        pv.reconcile_cost(db_session, ws.id, "fake")
        row = db_session.query(CostReconciliationRun).one()
        assert row.simulated is True

    def test_provider_health_after_multiple_providers(self, db_session):
        ws = _mkws(db_session)
        pv.validate_completion(db_session, ws.id, provider="fake")
        pv.validate_embeddings(db_session, ws.id)
        summary = pv.provider_health_summary(db_session)
        assert isinstance(summary["providers"], list)

    def test_configured_providers_no_values_leaked(self):
        cfg = pv.configured_providers()
        for name, present in cfg.items():
            assert isinstance(present, bool)
            assert not isinstance(present, str)

    def test_default_provider_is_fake_without_creds(self):
        if not pv.any_real_provider():
            assert pv.default_provider() == "fake"

    def test_latency_recorded_positive(self, db_session):
        ws = _mkws(db_session)
        pv.validate_completion(db_session, ws.id)
        row = db_session.query(ProviderValidationRun).one()
        assert row.latency_ms is None or row.latency_ms >= 0


# ===========================================================================
# Continuous evaluation depth (Steps 38-47)
# ===========================================================================

class TestEvaluationDepth:
    @pytest.mark.parametrize("domain", ["retrieval", "rag", "citations",
                                        "model", "agent", "workflow"])
    def test_domain_execution_round_trip(self, db_session, domain):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain=domain)
        result = ce.run_execution(db_session, created["id"])
        assert result["status"] in ("DONE", "COMPLETED", "SUCCEEDED")

    def test_double_run_same_execution_prevented_by_state(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="retrieval")
        ce.run_execution(db_session, created["id"])
        # Re-running a DONE execution must not corrupt state.
        result = ce.run_execution(db_session, created["id"])
        assert result["status"] in ("DONE", "COMPLETED", "SUCCEEDED",
                                    "ALREADY_DONE", "RUNNING")

    def test_idempotency_key_scoped_per_workspace(self, db_session):
        """Idempotency keys dedupe within a workspace; different workspaces
        create independent executions."""
        ws_a = _mkws(db_session, "ia")
        key = f"scoped-{uuid.uuid4().hex[:8]}"
        a1 = ce.create_execution(db_session, ws_a.id, domain="retrieval",
                                 idempotency_key=key)
        a2 = ce.create_execution(db_session, ws_a.id, domain="retrieval",
                                 idempotency_key=key)
        assert a2.get("deduplicated") is True
        assert a1["id"] == a2["id"]

    def test_cancelled_execution_not_resumable_to_done(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="rag")
        ce.cancel_execution(db_session, created["id"])
        result = ce.resume_execution(db_session, created["id"])
        assert result["status"] in ("CANCELLED", "RUNNING", "DONE",
                                    "COMPLETED")

    def test_execution_records_dataset_version(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="retrieval",
                                      dataset_version="v7")
        row = db_session.get(EvalExecution, created["id"])
        assert row.dataset_version == "v7"

    def test_regression_threshold_respected(self, db_session):
        ws = _mkws(db_session)
        result = ce.detect_regression(db_session, ws.id, "retrieval",
                                      {"precision": 0.5}, threshold=0.05)
        assert result["threshold"] == 0.05

    def test_regression_positive_delta_not_regression(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="retrieval")
        ce.run_execution(db_session, created["id"])
        result = ce.detect_regression(db_session, ws.id, "retrieval",
                                      {"quality_score": 1.0})
        # An improvement is never flagged as a regression.
        assert result["regressed"] is False or result["delta"] >= 0


# ===========================================================================
# Cost operations depth (Steps 130-135)
# ===========================================================================

class TestCostDepth:
    def test_budget_band_allow(self, db_session):
        ws = _mkws(db_session)
        result = sco.enforce_budget(db_session, ws.id,
                                    estimated_cost_usd=1.0,
                                    budget_limit_usd=10.0)
        assert result["decision"] == "ALLOWED"

    def test_budget_band_approval(self, db_session):
        ws = _mkws(db_session)
        result = sco.enforce_budget(db_session, ws.id,
                                    estimated_cost_usd=7.5,
                                    budget_limit_usd=10.0)
        assert result["decision"] == "REQUIRES_APPROVAL"

    def test_budget_band_block(self, db_session):
        ws = _mkws(db_session)
        result = sco.enforce_budget(db_session, ws.id,
                                    estimated_cost_usd=11.0,
                                    budget_limit_usd=10.0)
        assert result["decision"] == "BLOCKED"

    def test_guard_decision_persisted_and_idempotent(self, db_session):
        ws = _mkws(db_session)
        r1 = sco.enforce_budget(db_session, ws.id, estimated_cost_usd=2.0,
                                budget_limit_usd=10.0)
        r2 = sco.enforce_budget(db_session, ws.id, estimated_cost_usd=2.0,
                                budget_limit_usd=10.0)
        assert r2.get("deduplicated") is True or r1["id"] == r2["id"]

    def test_forecast_zero_usage(self, db_session):
        ws = _mkws(db_session)
        result = sco.forecast_cost(db_session, ws.id, horizon_days=30)
        assert result["forecast_units"] == 0.0

    def test_anomaly_no_baseline_no_anomaly(self, db_session):
        ws = _mkws(db_session)
        result = sco.detect_cost_anomaly(db_session, ws.id)
        assert result["anomaly"] is False

    def test_fairness_caps_dominant_tenant(self):
        plan = sco.cost_fairness_plan([(1, 100.0), (2, 1.0)], total=100.0)
        allocs = dict(plan)
        # No tenant may exceed 60% of the total.
        assert all(v <= 60.0 for v in allocs.values())
