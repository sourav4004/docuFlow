"""Phase 19 tests — AI provider platform 2.0 + cost platform 4.0.

Provider: capability matrix, routing modes, admission control, load
shedding, fallback chain, usage accounting, shadow testing. Cost:
estimation labels, budget reservations/reconciliation, forecasting,
anomaly detection, optimization recommendations, attribution.
"""

import uuid
from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.ai_execution import AIExecution  # noqa: E402
from app.models.phase15 import ProviderHealth  # noqa: E402
from app.models.phase16 import ProviderCapability  # noqa: E402
from app.models.phase17 import AIPolicyRule, CostAnomaly  # noqa: E402
from app.models.phase19 import (  # noqa: E402
    CostReservation, CostRecommendation,
)
from app.services import provider_ops2 as po  # noqa: E402
from app.services import cost4  # noqa: E402

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
    db_session.query(CostRecommendation).delete()
    db_session.query(CostReservation).delete()
    db_session.query(CostAnomaly).delete()
    db_session.query(AIExecution).delete()
    db_session.query(ProviderHealth).delete()
    db_session.query(ProviderCapability).delete()
    db_session.query(AIPolicyRule).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p19pc"):
    _counter[0] += 1
    user = User(name=f"P19 PC {_counter[0]}",
                email=f"{tag}{_counter[0]}@p19-pc.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p19 pc ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def execution(db, ws, user, *, cost=0.1, tokens=100, status="COMPLETED",
              model="gpt-4", provider="openai", task="chat",
              retries=0, days_ago=0, request_hash=None):
    row = AIExecution(
        id=str(uuid.uuid4())[:32], workspace_id=ws.id, user_id=user.id,
        task_type=task, model=model, provider=provider,
        status=status, retry_count=retries, estimated_cost=cost,
        input_tokens=tokens, output_tokens=tokens,
        total_tokens=tokens * 2, request_hash=request_hash,
        created_at=datetime.now(timezone.utc)
        - timedelta(days=days_ago))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ===========================================================================
# Provider platform 2.0
# ===========================================================================

class TestCapabilityMatrix:
    def test_matrix_task_capabilities(self):
        matrix = po.capability_matrix()
        assert "embedding" in matrix["embeddings"]
        assert "tools" in matrix["tools"]
        assert "structured" in matrix["structured"]
        assert "long_context" in matrix["long_context"]
        assert "reasoning" in matrix["reasoning"]
        assert "embedding" not in matrix["vision"]
        assert "vision" in matrix["vision"]

    def test_register_and_update_capability(self, db_session):
        po.register_capability(db_session, provider="openai",
                               model="gpt-4o", supports_tools=True,
                               context_window=128000)
        po.register_capability(db_session, provider="openai",
                               model="gpt-4o", cost_per_1k_input=0.01)
        rows = db_session.query(ProviderCapability).all()
        assert len(rows) == 1
        assert rows[0].cost_per_1k_input == 0.01

    def test_list_capabilities_derives_flags(self, db_session):
        po.register_capability(db_session, provider="p", model="m",
                               supports_vision=True,
                               embedding_dimensions=768,
                               context_window=70000)
        rows = po.list_capabilities(db_session)
        assert len(rows) == 1
        caps = rows[0]["capabilities"]
        assert "vision" in caps
        assert "embedding" in caps
        assert "long_context" in caps

    def test_reasoning_like_model_flagged(self, db_session):
        po.register_capability(db_session, provider="openai",
                               model="o3-mini")
        caps = po.list_capabilities(db_session)[0]["capabilities"]
        assert "reasoning" in caps

    def test_list_filtered_by_provider(self, db_session):
        po.register_capability(db_session, provider="a", model="m1")
        po.register_capability(db_session, provider="b", model="m2")
        assert len(po.list_capabilities(db_session, provider="a")) == 1


class TestRouting:
    def _cands(self):
        return [
            {"provider": "openai", "model": "gpt-4o",
             "capabilities": ["text", "tools", "structured", "vision",
                              "long_context"],
             "cost_per_1k_input": 0.005, "cost_per_1k_output": 0.015,
             "latency_class": "medium"},
            {"provider": "fastco", "model": "f-1",
             "capabilities": ["text"],
             "cost_per_1k_input": 0.0005, "cost_per_1k_output": 0.0005,
             "latency_class": "fast"},
            {"provider": "deep", "model": "r-1",
             "capabilities": ["text", "reasoning"],
             "cost_per_1k_input": 0.02, "cost_per_1k_output": 0.06,
             "latency_class": "slow"},
        ]

    def test_route_filters_missing_capability(self, db_session):
        result = po.route_model(db_session, task="vision",
                                candidates=self._cands())
        assert result["selected"]["model"] == "gpt-4o"

    def test_route_requires_tool_capability(self, db_session):
        result = po.route_model(db_session, task="tools",
                                candidates=self._cands())
        assert result["selected"]["provider"] == "openai"

    def test_route_cheapest_mode(self, db_session):
        result = po.route_model(db_session, task="text",
                                candidates=self._cands(), mode="cheapest")
        assert result["selected"]["model"] == "f-1"

    def test_route_fastest_mode(self, db_session):
        result = po.route_model(db_session, task="text",
                                candidates=self._cands(), mode="fastest")
        assert result["selected"]["model"] == "f-1"

    def test_route_quality_mode_picks_reasoning(self, db_session):
        result = po.route_model(db_session, task="text",
                                candidates=self._cands(),
                                mode="highest_quality")
        assert result["selected"]["model"] == "r-1"

    def test_route_no_candidates_records_reason(self, db_session):
        result = po.route_model(db_session, task="embeddings",
                                candidates=self._cands())
        assert result["selected"] is None
        assert result["excluded"]

    def test_route_unknown_mode_raises(self, db_session):
        with pytest.raises(ValueError):
            po.route_model(db_session, task="text",
                           candidates=self._cands(), mode="weird")

    def test_route_respects_policy_deny(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        db_session.add(AIPolicyRule(rule_type="MODEL",
                                    workspace_id=ws.id,
                                    deny_json='["gpt-4o"]'))
        db_session.commit()
        result = po.route_model(db_session, task="vision",
                                candidates=self._cands(),
                                workspace_id=ws.id)
        assert result["selected"] is None

    def test_route_mode_score_deterministic(self):
        a = po._mode_score({"cost_per_1k_input": 1.0,
                            "cost_per_1k_output": 1.0,
                            "latency_class": "slow",
                            "capabilities": []}, "balanced")
        b = po._mode_score({"cost_per_1k_input": 0.001,
                            "cost_per_1k_output": 0.001,
                            "latency_class": "fast",
                            "capabilities": []}, "balanced")
        assert b > a


class TestAdmission:
    def test_admission_unknown_provider(self, db_session):
        result = po.admission_control(
            db_session, provider="nope", model="x", task="text",
            organization_id=None, workspace_id=1)
        assert result["allowed"] is False
        assert result["code"] == "PROVIDER_UNKNOWN"

    def test_admission_capability_gate(self, db_session):
        po.register_capability(db_session, provider="p", model="m",
                               supports_text=True)
        result = po.admission_control(
            db_session, provider="p", model="m", task="vision",
            organization_id=None, workspace_id=1)
        assert result["code"] == "CAPABILITY_MISSING"

    def test_admission_allowed(self, db_session):
        po.register_capability(db_session, provider="p", model="m",
                               supports_text=True)
        result = po.admission_control(
            db_session, provider="p", model="m", task="text",
            organization_id=None, workspace_id=1)
        assert result["allowed"] is True

    def test_admission_blocked_by_down_provider(self, db_session):
        po.register_capability(db_session, provider="p", model="m")
        db_session.add(ProviderHealth(provider="p", model="m",
                                      status="DOWN",
                                      circuit_state="OPEN"))
        db_session.commit()
        result = po.admission_control(
            db_session, provider="p", model="m", task="text",
            organization_id=None, workspace_id=1)
        assert result["code"] == "PROVIDER_UNHEALTHY"

    def test_admission_budget_gate(self, db_session):
        po.register_capability(db_session, provider="p", model="m")
        result = po.admission_control(
            db_session, provider="p", model="m", task="text",
            organization_id=None, workspace_id=1,
            estimated_cost=50.0, budget_remaining=10.0)
        assert result["code"] == "BUDGET_EXCEEDED"

    def test_admission_sensitivity_blocked(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        po.register_capability(db_session, provider="p", model="m")
        db_session.add(AIPolicyRule(rule_type="SENSITIVITY",
                                    workspace_id=ws.id,
                                    sensitivity_max="INTERNAL"))
        db_session.commit()
        result = po.admission_control(
            db_session, provider="p", model="m", task="text",
            organization_id=None, workspace_id=ws.id,
            sensitivity="RESTRICTED")
        assert result["code"] == "GOVERNANCE_BLOCKED"


class TestLoadShedding:
    def test_shed_low_priority_at_capacity(self, db_session):
        result = po.load_shedding(db_session, queue_depth=9000,
                                  provider_load=1.0, priority="LOW")
        assert result["shed"] is True

    def test_high_priority_admitted_under_pressure(self, db_session):
        result = po.load_shedding(db_session, queue_depth=9000,
                                  provider_load=1.0, priority="HIGH")
        assert result["shed"] is False

    def test_no_shed_when_available(self, db_session):
        result = po.load_shedding(db_session, queue_depth=10,
                                  provider_load=0.2, priority="LOW")
        assert result["shed"] is False


class TestFallback:
    def test_chain_order_normal(self, db_session):
        plan = po.fallback_plan(db_session, task="text",
                                primary={"provider": "a", "model": "m"},
                                secondary={"provider": "b", "model": "m2"})
        steps = [s["step"] for s in plan["chain"]]
        assert steps[0] == "primary"
        assert "retry" in steps
        assert "secondary" in steps
        assert steps[-1] == "final_failure"

    def test_circuit_open_skips_retries_on_primary(self, db_session):
        plan = po.fallback_plan(db_session, task="text",
                                primary={"provider": "a", "model": "m"},
                                secondary={"provider": "b", "model": "m2"},
                                circuit_open=True)
        primary = plan["chain"][0]
        retry_step = plan["chain"][1]
        assert primary["decision"] == "SKIP"
        assert retry_step["decision"] == "SKIP"

    def test_degraded_mode_present(self, db_session):
        plan = po.fallback_plan(db_session, task="text")
        assert any(s["step"] == "degraded" and s["decision"] == "TRY"
                   for s in plan["chain"])

    def test_degraded_disabled_when_disallowed(self, db_session):
        plan = po.fallback_plan(db_session, task="text",
                                degraded_allowed=False)
        assert all(s["step"] != "degraded" for s in plan["chain"])


class TestUsageShadow:
    def test_account_usage_records_tokens(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = execution(db_session, ws, user, cost=0)
        result = po.account_usage(db_session, execution_id=run.id,
                                  provider="openai", model="gpt-4o",
                                  input_tokens=10, output_tokens=5,
                                  estimated_cost=0.02)
        assert result["recorded"] is True
        assert result["tokens"] == 15

    def test_account_usage_missing_execution(self, db_session):
        result = po.account_usage(db_session, execution_id="nope",
                                  provider="x", model="y")
        assert result["recorded"] is False

    def test_shadow_sample_redacted_for_sensitive(self):
        sample = po.build_shadow_sample(sensitivity="RESTRICTED")
        assert "shadow" in sample["text"]
        assert "REDACTED" in sample["note"]

    def test_shadow_public_allows_query(self):
        sample = po.build_shadow_sample(query="refund?")
        assert "refund" in sample["text"]

    def test_compare_shadow_results(self):
        result = po.compare_shadow_results(
            {"text": "hello", "latency_ms": 100},
            {"text": "hello", "latency_ms": 150})
        assert result["agreement"] is True
        assert result["latency_delta_ms"] == 50.0

    def test_compare_shadow_disagreement(self):
        result = po.compare_shadow_results({"text": "a", "latency_ms": 1},
                                           {"text": "b", "latency_ms": 2})
        assert result["agreement"] is False


# ===========================================================================
# Cost platform 4.0
# ===========================================================================

class TestEstimateReserve:
    def test_estimate_labeled(self):
        estimate = cost4.estimate_cost("openai", "gpt-4o",
                                       input_tokens=1000,
                                       output_tokens=1000)
        assert estimate["is_estimate"] is True
        assert estimate["estimated_cost_usd"] > 0

    def test_estimate_zero_tokens_free(self):
        estimate = cost4.estimate_cost("p", "m")
        assert estimate["estimated_cost_usd"] == 0.0

    def test_reserve_budget(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = cost4.reserve_budget(db_session, organization_id=None,
                                      workspace_id=ws.id,
                                      execution_ref="e1", feature="rag",
                                      estimated_cost=1.5)
        row = db_session.query(CostReservation).get(result["reservation_id"])
        assert row.status == "RESERVED"
        assert row.reserved_amount == 1.5

    def test_reserve_negative_raises(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            cost4.reserve_budget(db_session, organization_id=None,
                                 workspace_id=ws.id, execution_ref=None,
                                 feature="x", estimated_cost=-1)

    def test_release_reconciles(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = cost4.reserve_budget(db_session, organization_id=None,
                                      workspace_id=ws.id,
                                      execution_ref=None, feature="x",
                                      estimated_cost=2.0)
        released = cost4.release_reservation(
            db_session, reservation_id=result["reservation_id"],
            actual_cost=0.5)
        assert released["status"] == "RECONCILED"
        assert released["released"] == 1.5

    def test_release_idempotent(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = cost4.reserve_budget(db_session, organization_id=None,
                                      workspace_id=ws.id,
                                      execution_ref=None, feature="x",
                                      estimated_cost=1.0)
        cost4.release_reservation(db_session,
                                  reservation_id=result["reservation_id"],
                                  actual_cost=0.1)
        again = cost4.release_reservation(
            db_session, reservation_id=result["reservation_id"],
            actual_cost=0.1)
        assert again["status"] == "ALREADY_RECONCILED"

    def test_reconcile_detects_discrepancy(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = cost4.reserve_budget(db_session, organization_id=None,
                                      workspace_id=ws.id,
                                      execution_ref=None, feature="x",
                                      estimated_cost=1.0)
        reconciled = cost4.reconcile(db_session,
                                     reservation_id=result[
                                         "reservation_id"],
                                     estimated_cost=1.0, actual_cost=5.0)
        assert reconciled["discrepancy"] is True

    def test_reconcile_within_tolerance(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = cost4.reserve_budget(db_session, organization_id=None,
                                      workspace_id=ws.id,
                                      execution_ref=None, feature="x",
                                      estimated_cost=1.0)
        reconciled = cost4.reconcile(db_session,
                                     reservation_id=result[
                                         "reservation_id"],
                                     estimated_cost=1.0, actual_cost=1.05)
        assert reconciled["discrepancy"] is False


class TestForecastAnomaly:
    def test_forecast_labeled_and_daily(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for day in range(10):
            execution(db_session, ws, user, cost=0.1, days_ago=day)
        forecast = cost4.forecast2(db_session, organization_id=None,
                                   granularity="daily")
        assert forecast["is_estimate"] is True
        assert forecast["observed_30d"] > 0

    def test_forecast_invalid_granularity(self, db_session):
        with pytest.raises(ValueError):
            cost4.forecast2(db_session, organization_id=1,
                            granularity="yearly")

    def test_anomaly_detected_on_spike(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for day in range(1, 15):
            execution(db_session, ws, user, cost=1.0, days_ago=day)
        execution(db_session, ws, user, cost=100.0, days_ago=0)
        result = cost4.detect_anomalies(db_session, organization_id=None)
        metrics = {item["metric"] for item in result["detected"]}
        assert "cost_usd" in metrics

    def test_anomaly_persists_rows(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for day in range(1, 15):
            execution(db_session, ws, user, cost=1.0, days_ago=day)
        execution(db_session, ws, user, cost=90.0, days_ago=0)
        cost4.detect_anomalies(db_session, organization_id=None)
        assert db_session.query(CostAnomaly).count() >= 1

    def test_repeated_failure_cost_reported(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution(db_session, ws, user, cost=2.0, status="FAILED",
                  retries=3)
        result = cost4.detect_anomalies(db_session, organization_id=None)
        assert result["repeated_failure_cost"]["actual"] >= 2.0

    def test_no_anomaly_on_flat_spend(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for day in range(10):
            execution(db_session, ws, user, cost=0.5, days_ago=day)
        result = cost4.detect_anomalies(db_session, organization_id=None)
        assert result["detected"] == []


class TestRecommendationsAttribution:
    def test_caching_recommendation_on_repeated_hash(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for _ in range(4):
            execution(db_session, ws, user, cost=0.1,
                      request_hash="same-request")
        result = cost4.optimization_recommendations(
            db_session, organization_id=None)
        assert any(r["kind"] == "CACHING"
                   for r in result["recommendations"])
        assert db_session.query(CostRecommendation).filter(
            CostRecommendation.kind == "CACHING").count() == 1

    def test_recommendations_idempotent_persist(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for _ in range(4):
            execution(db_session, ws, user, cost=0.1,
                      request_hash="dup")
        cost4.optimization_recommendations(db_session,
                                           organization_id=None)
        cost4.optimization_recommendations(db_session,
                                           organization_id=None)
        assert db_session.query(CostRecommendation).count() == 1

    def test_recommendations_never_applied_flag(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for _ in range(4):
            execution(db_session, ws, user, request_hash="h1")
        result = cost4.optimization_recommendations(
            db_session, organization_id=None)
        assert result["recommendations"]
        assert "never auto-applied" in result["note"]

    def test_attribution_dimensions(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution(db_session, ws, user, cost=1.0, model="m1",
                  provider="p1", task="chat")
        result = cost4.attribution2(db_session, organization_id=None)
        assert result["total_cost"] >= 1.0
        assert result["by_model"]
        assert result["by_provider"]
        assert result["by_feature"]

    def test_attribution_empty(self, db_session):
        result = cost4.attribution2(db_session, organization_id=None)
        assert result["total_cost"] == 0.0
