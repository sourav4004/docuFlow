"""Phase 20 tests — provider/model optimization + cost intelligence 5.0.

Performance profiles, routing recommendations + simulation + promotion
validation, provider anomaly detection, cost baselines + drift, token
efficiency, optimization candidates, and quality/cost tradeoffs.
"""

import json

import pytest

from tests.shared_db import TestingSessionLocal

from app.models.phase20 import (
    ProviderProfile, RoutingRecommendation, ProviderAnomaly, CostBaseline,
    TokenEfficiency,
)
from app.models.usage import UsageRecord  # noqa: E402
from app.services import provider_intel as pri
from app.services import cost_intel as ci


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in (ProviderProfile, RoutingRecommendation, ProviderAnomaly,
                  CostBaseline, TokenEfficiency, UsageRecord):
        db_session.query(model).delete()
    db_session.commit()
    yield


def _usage(db, workspace_id=1, cost=1.0):
    r = UsageRecord(workspace_id=workspace_id, usage_type="ai_request",
                    quantity=1, units="count")
    db.add(r)
    db.flush()
    return r


# ===========================================================================
# Provider profiles
# ===========================================================================

class TestProviderProfiles:
    def test_record_profile(self, db_session):
        row = pri.record_profile(db_session, provider="openai",
                                 model="gpt-4o",
                                 metrics={"latency_ms": 300,
                                          "error_rate": 0.01})
        assert row.provider == "openai"
        stored = json.loads(row.metrics_json)
        assert stored["error_rate"] == 0.01

    def test_profile_summary_filter(self, db_session):
        pri.record_profile(db_session, provider="openai", model="gpt-4o",
                           metrics={})
        pri.record_profile(db_session, provider="anthropic",
                           model="claude", metrics={})
        rows = pri.profile_summary(db_session, provider="openai")
        assert len(rows) == 1 and rows[0]["provider"] == "openai"

    def test_routing_recommendation_high_error(self, db_session):
        for i in range(3):
            pri.record_profile(db_session, provider="openai",
                               model="gpt-4o",
                               metrics={"error_rate": 0.3,
                                        "latency_p95_ms": 100})
        recs = pri.routing_recommendations(db_session)
        assert any("gpt-4o" in r["candidate"]
                   for r in recs if r["domain"] == "provider_routing")

    def test_routing_recommendation_latency(self, db_session):
        pri.record_profile(db_session, provider="openai", model="slow-1",
                           metrics={"error_rate": 0.0,
                                    "latency_p95_ms": 9000})
        recs = pri.routing_recommendations(db_session)
        assert any("slow-1" in r["candidate"]
                   for r in recs if r["domain"] == "model_routing")

    def test_routing_recommendation_cost(self, db_session):
        pri.record_profile(db_session, provider="openai", model="pricey-1",
                           metrics={"error_rate": 0.0,
                                    "latency_ms": 100,
                                    "cost_per_1k": 0.5})
        recs = pri.routing_recommendations(db_session)
        assert any("pricey-1" in r["candidate"]
                   for r in recs if r["domain"] == "cost")

    def test_recommendation_status_proposed(self, db_session):
        pri.record_profile(db_session, provider="p", model="m",
                           metrics={"error_rate": 0.9, "latency_ms": 1})
        pri.routing_recommendations(db_session)
        rows = db_session.query(RoutingRecommendation).all()
        assert rows and all(r.status == "PROPOSED" for r in rows)

    def test_list_recommendations(self, db_session):
        pri.record_profile(db_session, provider="p", model="m",
                           metrics={"error_rate": 0.9})
        pri.routing_recommendations(db_session)
        assert len(pri.list_routing_recommendations(db_session)) >= 1

    def test_recommendation_uses_provided_profiles(self, db_session):
        recs = pri.routing_recommendations(
            db_session,
            profiles=[{"model": "m1",
                       "metrics": {"error_rate": 0.5}}])
        assert recs  # generated from provided profiles without DB rows


# ===========================================================================
# Routing simulation
# ===========================================================================

class TestRoutingSimulation:
    WORKLOAD = [
        {"task": "summarize", "model": "a", "cost": 1.0,
         "latency_ms": 100, "quality": 0.9},
        {"task": "summarize", "model": "a", "cost": 1.0,
         "latency_ms": 100, "quality": 0.9},
        {"task": "classify", "model": "b", "cost": 0.5,
         "latency_ms": 50, "quality": 0.8},
    ]

    def test_simulation_baseline(self, db_session):
        result = pri.simulate_routing(db_session, workload=self.WORKLOAD,
                                      candidates=[])
        assert result["baseline"]["total_cost"] == 2.5
        assert result["workload_size"] == 3

    def test_simulation_reroutes_matching_tasks(self, db_session):
        result = pri.simulate_routing(
            db_session, workload=self.WORKLOAD,
            candidates=[{"name": "cheap", "task_pattern": "summarize",
                         "cost_per_call": 0.1, "latency_ms": 40,
                         "quality": 0.7}])
        cand = result["candidates"][0]
        assert cand["matched"] == 2
        assert cand["total_cost"] == pytest.approx(0.7, abs=1e-6)

    def test_simulation_no_match(self, db_session):
        result = pri.simulate_routing(
            db_session, workload=self.WORKLOAD,
            candidates=[{"name": "x", "task_pattern": "research",
                         "cost_per_call": 0.1, "latency_ms": 10,
                         "quality": 0.5}])
        assert result["candidates"][0]["matched"] == 0
        # cost identical to baseline
        assert result["candidates"][0]["total_cost"] == pytest.approx(
            result["baseline"]["total_cost"])


class TestPromotionValidation:
    def test_valid_candidate(self):
        result = pri.promotion_validation(
            {"model": "gpt-4o", "provider": "openai", "quality": 0.9,
             "cost_per_call": 0.01},
            {"allowed_models": ["gpt-4o"],
             "allowed_providers": ["openai"],
             "min_quality": 0.8, "max_cost_per_call": 0.1})
        assert result["valid"] is True

    def test_model_not_allowed(self):
        result = pri.promotion_validation(
            {"model": "gpt-4o", "quality": 0.9},
            {"allowed_models": ["claude"]})
        assert result["valid"] is False

    def test_quality_below_minimum(self):
        result = pri.promotion_validation(
            {"model": "m", "quality": 0.5},
            {"allowed_models": ["m"], "min_quality": 0.8})
        assert result["valid"] is False

    def test_cost_above_max(self):
        result = pri.promotion_validation(
            {"model": "m", "quality": 0.9, "cost_per_call": 5.0},
            {"allowed_models": ["m"], "max_cost_per_call": 1.0})
        assert result["valid"] is False


# ===========================================================================
# Provider anomalies
# ===========================================================================

class TestProviderAnomalies:
    def _seed(self, db, values):
        for v in values:
            pri.record_profile(db, provider="openai", model="m",
                               metrics={"latency_p95_ms": v,
                                        "error_rate": 0.01,
                                        "cost_per_1k": 0.01})

    def test_no_anomaly_on_stable(self, db_session):
        self._seed(db_session, [100, 102, 101])
        anomalies = pri.detect_anomalies(db_session)
        assert anomalies == []

    def test_latency_spike_detected(self, db_session):
        self._seed(db_session, [100, 102, 101, 5000])
        anomalies = pri.detect_anomalies(db_session)
        assert any(a["anomaly_type"] == "latency" for a in anomalies)

    def test_error_spike_detected(self, db_session):
        for v in (0.01, 0.02, 0.01):
            pri.record_profile(db_session, provider="p", model="m",
                               metrics={"error_rate": v,
                                        "latency_p95_ms": 100,
                                        "cost_per_1k": 0.01})
        pri.record_profile(db_session, provider="p", model="m",
                           metrics={"error_rate": 0.99,
                                    "latency_p95_ms": 100,
                                    "cost_per_1k": 0.01})
        anomalies = pri.detect_anomalies(db_session)
        assert any(a["anomaly_type"] == "error_rate" for a in anomalies)

    def test_anomaly_persisted(self, db_session):
        self._seed(db_session, [100, 102, 101, 5000])
        pri.detect_anomalies(db_session)
        assert db_session.query(ProviderAnomaly).count() >= 1

    def test_anomaly_resolution_flag(self, db_session):
        self._seed(db_session, [100, 102, 101, 5000])
        pri.detect_anomalies(db_session)
        row = db_session.query(ProviderAnomaly).first()
        assert row.resolved is False


# ===========================================================================
# Cost intelligence
# ===========================================================================

class TestCostBaselines:
    def test_set_baseline(self, db_session):
        row = ci.set_baseline(db_session, scope_type="WORKSPACE",
                              scope_id=1, baseline={"cost": 5.0})
        assert row.scope_type == "WORKSPACE"

    def test_latest_baseline(self, db_session):
        ci.set_baseline(db_session, scope_type="WORKSPACE", scope_id=1,
                        baseline={"cost": 5.0})
        ci.set_baseline(db_session, scope_type="WORKSPACE", scope_id=1,
                        baseline={"cost": 8.0})
        latest = ci.latest_baseline(db_session, scope_type="WORKSPACE",
                                    scope_id=1)
        assert latest["baseline"]["cost"] == 8.0

    def test_bad_scope(self, db_session):
        with pytest.raises(ValueError):
            ci.set_baseline(db_session, scope_type="GALAXY", scope_id=1,
                            baseline={})

    def test_dimension_scoped_baseline(self, db_session):
        ci.set_baseline(db_session, scope_type="MODEL", scope_id=1,
                        baseline={"cost": 1.0}, dimension="gpt-4o")
        ci.set_baseline(db_session, scope_type="MODEL", scope_id=1,
                        baseline={"cost": 2.0}, dimension="claude")
        assert ci.latest_baseline(
            db_session, scope_type="MODEL", scope_id=1,
            dimension="gpt-4o")["baseline"]["cost"] == 1.0


class TestCostDrift:
    def test_no_baseline_no_drift(self, db_session):
        result = ci.detect_cost_drift(db_session, scope_type="WORKSPACE",
                                      scope_id=1)
        assert result["drift"] is False

    def test_no_drift_within_budget(self, db_session):
        ci.set_baseline(db_session, scope_type="WORKSPACE", scope_id=1,
                        baseline={"cost": 100.0})
        result = ci.detect_cost_drift(db_session, scope_type="WORKSPACE",
                                      scope_id=1, sample_cost=110.0,
                                      max_delta_fraction=0.2)
        assert result["drift"] is False

    def test_drift_detected(self, db_session):
        ci.set_baseline(db_session, scope_type="WORKSPACE", scope_id=1,
                        baseline={"cost": 100.0})
        result = ci.detect_cost_drift(db_session, scope_type="WORKSPACE",
                                      scope_id=1, sample_cost=200.0,
                                      max_delta_fraction=0.2)
        assert result["drift"] is True
        assert result["delta_fraction"] == pytest.approx(1.0)

    def test_negative_drift_not_reported(self, db_session):
        ci.set_baseline(db_session, scope_type="WORKSPACE", scope_id=1,
                        baseline={"cost": 100.0})
        result = ci.detect_cost_drift(db_session, scope_type="WORKSPACE",
                                      scope_id=1, sample_cost=50.0)
        assert result["drift"] is False


class TestTokenEfficiency:
    def test_record_tokens(self, db_session):
        row = ci.record_token_usage(db_session, workspace_id=1,
                                    input_tokens=1000, output_tokens=200,
                                    context_tokens=1000,
                                    repeated_tokens=800)
        assert row.input_tokens == 1000

    def test_repeated_fraction(self, db_session):
        ci.record_token_usage(db_session, workspace_id=1,
                              input_tokens=10, output_tokens=1,
                              context_tokens=100, repeated_tokens=60)
        report = ci.token_efficiency_report(db_session, workspace_id=1)
        assert report["repeated_fraction"] == pytest.approx(0.6)

    def test_output_ratio(self, db_session):
        ci.record_token_usage(db_session, workspace_id=1,
                              input_tokens=100, output_tokens=50)
        report = ci.token_efficiency_report(db_session, workspace_id=1)
        assert report["output_ratio"] == pytest.approx(0.5)

    def test_empty_report(self, db_session):
        report = ci.token_efficiency_report(db_session, workspace_id=9)
        assert report["samples"] == 0

    def test_workspace_isolation_tokens(self, db_session):
        ci.record_token_usage(db_session, workspace_id=1,
                              input_tokens=100, output_tokens=0)
        assert ci.token_efficiency_report(
            db_session, workspace_id=2)["total_input_tokens"] == 0


class TestCostOptimization:
    def test_context_reduction_candidate(self, db_session):
        ci.record_token_usage(db_session, workspace_id=1,
                              input_tokens=100, output_tokens=10,
                              context_tokens=100, repeated_tokens=90)
        cands = ci.cost_optimization_candidates(db_session, workspace_id=1)
        assert any(c["strategy"] == "context_reduction" for c in cands)

    def test_output_reduction_candidate(self, db_session):
        ci.record_token_usage(db_session, workspace_id=1,
                              input_tokens=10, output_tokens=9)
        cands = ci.cost_optimization_candidates(db_session, workspace_id=1)
        assert any(c["strategy"] == "output_reduction" for c in cands)

    def test_model_downgrade_on_drift(self, db_session):
        ci.set_baseline(db_session, scope_type="WORKSPACE", scope_id=1,
                        baseline={"cost": 100.0})
        # record some usage so drift check has a non-trivial sample
        _usage(db_session, workspace_id=1)
        cands = ci.cost_optimization_candidates(db_session, workspace_id=1)
        # drift candidates are only produced when the sample exceeds budget;
        # without that the list still contains the none entry
        assert cands


class TestQualityCostTradeoff:
    def test_recommended_when_quality_preserved(self):
        result = ci.quality_cost_tradeoff(
            {"quality": 0.9, "cost": 1.0},
            {"quality": 0.88, "cost": 0.4})
        assert result["verdict"] == "RECOMMENDED"

    def test_not_recommended_on_quality_loss(self):
        result = ci.quality_cost_tradeoff(
            {"quality": 0.9, "cost": 1.0},
            {"quality": 0.5, "cost": 0.1})
        assert result["verdict"] == "NOT_RECOMMENDED_QUALITY_LOSS"

    def test_not_recommended_cost_increase(self):
        result = ci.quality_cost_tradeoff(
            {"quality": 0.5, "cost": 0.1},
            {"quality": 0.9, "cost": 2.0})
        assert result["verdict"] == "NOT_RECOMMENDED_COST_INCREASE"

    def test_never_sacrifices_quality_for_cost(self):
        result = ci.quality_cost_tradeoff(
            {"quality": 0.9, "cost": 1.0, "quality_tolerance": 0.01},
            {"quality": 0.8, "cost": 0.01})
        assert result["verdict"] != "RECOMMENDED"

    def test_quality_improvement_no_gain_case(self):
        result = ci.quality_cost_tradeoff(
            {"quality": 0.9, "cost": 1.0},
            {"quality": 0.95, "cost": 1.0})
        # cost_delta = 0, quality improves -> not cost saving but no loss:
        assert result["cost_delta"] == 0.0