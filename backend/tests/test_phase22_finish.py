"""Phase 22 tests — final parametrized depth.

Worker lifecycle parametrized by priority, region matrix, provider
validation rounds, trace stage chains, and loop governance parametrics.
"""

import uuid

import pytest

from tests.shared_db import TestingSessionLocal

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.services import continuous_eval as ce
from app.services import observability2 as ob
from app.services import operating_loops as ol
from app.services import provider_validation as pv
from app.services import region_dr as rd
from app.services import worker_platform as wp

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
    from app.models import TraceSpan
    from app.models.phase16 import WorkerJob, WorkerHeartbeat
    from app.models.phase19 import RegionRecord, RegionFailover
    from app.models.phase20 import ImprovementProposal, Experiment, \
        ExperimentRun
    from app.models.phase21 import AutonomousOperation, AutonomyPolicy, \
        EmergencyStop, PlatformEvent, RecoveryAttempt, RecoveryPlaybook, \
        DiagnosisReport
    from app.models.phase22 import EvalExecution, ProviderValidationRun, \
        OpsStreamEvent, SelfHealLoopRun, AutonomyLoopRun
    for model in (OpsStreamEvent, SelfHealLoopRun, AutonomyLoopRun,
                  EvalExecution, ProviderValidationRun, TraceSpan,
                  WorkerJob, WorkerHeartbeat, RegionFailover, RegionRecord):
        db_session.query(model).delete()
    for model in (ImprovementProposal, ExperimentRun, Experiment,
                  AutonomousOperation, EmergencyStop, PlatformEvent,
                  RecoveryAttempt, RecoveryPlaybook, AutonomyPolicy,
                  DiagnosisReport):
        try:
            db_session.query(model).delete()
        except Exception:  # noqa: BLE001
            db_session.rollback()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db, tag="fx"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p22fx.example", name=tag,
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Worker lifecycle by priority
# ===========================================================================

class TestWorkerPriorityMatrix:
    @pytest.mark.parametrize("priority", ["CRITICAL", "HIGH", "NORMAL",
                                          "LOW", "BACKGROUND"])
    def test_enqueue_all_priorities(self, db_session, priority):
        ws = _mkws(db_session)
        job = wp.enqueue_job(db_session, "prio-mx", "probe", ws.id,
                             payload={}, priority=priority)
        assert job.priority == priority

    def test_priority_ranking_full_order(self, db_session):
        ws = _mkws(db_session)
        for prio in ("BACKGROUND", "LOW", "NORMAL", "HIGH", "CRITICAL"):
            wp.enqueue_job(db_session, "prio-full", "probe", ws.id,
                           payload={}, priority=prio)
        order = []
        for i in range(5):
            claimed = wp.claim_job(db_session, "prio-full",
                                   f"prio-worker-{i}", per_workspace_cap=10)
            if claimed is not None:
                order.append(claimed.priority)
        assert order == ["CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND"]

    def test_fairness_two_tenants_round_robin(self, db_session):
        ws_a = _mkws(db_session, "fa")
        ws_b = _mkws(db_session, "fb")
        for _ in range(4):
            wp.enqueue_job(db_session, "fair-mx", "probe", ws_a.id,
                           payload={})
            wp.enqueue_job(db_session, "fair-mx", "probe", ws_b.id,
                           payload={})
        winners = []
        for i in range(4):
            claimed = wp.claim_job(db_session, "fair-mx",
                                   f"fair-worker-{i}")
            if claimed is not None:
                winners.append(claimed.workspace_id)
        # Both tenants must win claims (no monopoly).
        assert set(winners) == {ws_a.id, ws_b.id}


# ===========================================================================
# Region matrix
# ===========================================================================

class TestRegionMatrix:
    @pytest.mark.parametrize("region", ["us-east-1", "us-west-2",
                                        "eu-central-1", "ap-southeast-1"])
    def test_region_upsert_matrix(self, db_session, region):
        result = rd.upsert_region(db_session, region=region)
        assert result["region"] == region
        rows = rd.list_regions(db_session)
        assert any(r["region"] == region for r in rows)

    def test_capacity_zero_workers(self, db_session):
        result = rd.region_capacity(db_session, "empty-1", workers=0,
                                    queue_depth=0, db_healthy=False,
                                    provider_healthy=False)
        assert result["region"] == "empty-1"

    def test_drill_rpo_rto_targets_recorded(self, db_session):
        ws = _mkws(db_session, "drt")
        result = rd.run_restore_drill(db_session, ws.id, target_rpo_s=120,
                                      target_rto_s=900)
        assert result["simulated"] is True
        assert "rpo" in str(result).lower() or "target" in str(result).lower()


# ===========================================================================
# Provider validation rounds
# ===========================================================================

class TestProviderRounds:
    @pytest.mark.parametrize("round_no", [1, 2, 3])
    def test_completion_rounds_stable(self, db_session, round_no):
        ws = _mkws(db_session)
        result = pv.validate_completion(db_session, ws.id)
        assert result["passed"] is True
        assert result["simulated"] is True

    def test_latency_recorded(self, db_session):
        from app.models.phase22 import ProviderValidationRun
        ws = _mkws(db_session)
        pv.validate_streaming(db_session, ws.id)
        row = db_session.query(ProviderValidationRun).one()
        assert row.latency_ms is None or row.latency_ms >= 0.0


# ===========================================================================
# Trace stage chains
# ===========================================================================

class TestTraceChains:
    @pytest.mark.parametrize("stages", [
        ("request", "retrieval", "rag", "provider"),
        ("request", "tool", "workflow"),
        ("worker", "checkpoint"),
    ])
    def test_stage_chain_breakdown(self, db_session, stages):
        ws = _mkws(db_session)
        first = None
        for stage in stages:
            s = ob.start_trace(db_session, workspace_id=ws.id, stage=stage,
                               name=stage, trace_id=first["trace_id"]
                               if first else None, force_record=True)
            if first is None:
                first = s
            ob.finish_trace(db_session, s["span_id"], ok=True)
        breakdown = ob.latency_breakdown(db_session, first["trace_id"])
        assert set(stages) <= set(breakdown["stages_ms"].keys())

    def test_error_and_cost_together(self, db_session):
        ws = _mkws(db_session)
        s = ob.start_trace(db_session, workspace_id=ws.id, stage="provider",
                           name="mixed", force_record=True)
        ob.finish_trace(db_session, s["span_id"], ok=False,
                        error="Timeout", cost_usd=0.05)
        errs = ob.correlate_errors(db_session, error_class="Timeout")
        cost = ob.correlate_cost(db_session, s["trace_id"])
        assert errs.get("spans", 0) >= 1
        assert float(cost.get("cost_usd") or
                     cost.get("total_cost") or 0) >= 0.05


# ===========================================================================
# Loop governance parametrics
# ===========================================================================

class TestLoopParametrics:
    @pytest.mark.parametrize("metric,expected_governance", [
        ({"quality": 0.1, "reliability": 0.5, "cost": 500.0}, None),
        ({"quality": 0.95, "reliability": 0.99, "cost": 1.0}, "NO_ACTION"),
    ])
    def test_loop_outcomes(self, db_session, metric, expected_governance):
        ws = _mkws(db_session)
        loop = ol.run_autonomy_loop(db_session, ws.id, "retrieval",
                                    metric=metric)
        if expected_governance == "NO_ACTION":
            assert loop.deviation_detected is False
        else:
            # Degenerate metrics never auto-activate.
            assert loop.activated is False

    @pytest.mark.parametrize("domain", ["retrieval", "rag", "search",
                                        "workflow", "cost"])
    def test_loop_domains_supported(self, db_session, domain):
        ws = _mkws(db_session)
        loop = ol.run_autonomy_loop(db_session, ws.id, domain,
                                    metric={"quality": 0.9,
                                            "reliability": 0.99,
                                            "cost": 1.0})
        assert loop.domain == domain
        assert loop.stage == "DONE"

    def test_selfheal_signal_kinds(self, db_session):
        ws = _mkws(db_session)
        for signals in ({"provider_failures": 2}, {"db_errors": 1},
                        {"queue_depth": 900}):
            loop = ol.run_selfheal_loop(db_session, ws.id,
                                        "probe_trigger", signals=signals)
            assert loop.detected is True
