"""Phase 22 tests — contract depth.

Parametrized infrastructure detection across components, vector platform
edge cases, multi-round evaluation stability, stream audit invariants, and
cost/retention bounds.
"""

import uuid

import pytest

from tests.shared_db import TestingSessionLocal

from app.models.phase22 import (
    CostReconciliationRun, EvalExecution, InfraCapability,
    ProviderValidationRun, RetentionExecution, VectorBenchmarkRun,
)
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.services import capabilities as cap
from app.services import continuous_eval as ce
from app.services import observability2 as ob
from app.services import operating_loops as ol
from app.services import provider_validation as pv
from app.services import security_cost_ops as sco
from app.services import vector_production as vp

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
    from app.models.phase21 import EmergencyStop, PlatformEvent, \
        AutonomousOperation, AutonomyPolicy, RecoveryAttempt, \
        RecoveryPlaybook
    from app.models.phase20 import ImprovementProposal, Experiment, \
        ExperimentRun
    for model in (VectorBenchmarkRun, CostReconciliationRun,
                  ProviderValidationRun, EvalExecution,
                  RetentionExecution, InfraCapability, TraceSpan):
        db_session.query(model).delete()
    for model in (ImprovementProposal, ExperimentRun, Experiment,
                  AutonomousOperation, EmergencyStop, PlatformEvent,
                  RecoveryAttempt, RecoveryPlaybook, AutonomyPolicy):
        try:
            db_session.query(model).delete()
        except Exception:  # noqa: BLE001
            db_session.rollback()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db, tag="ct"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p22ct.example", name=tag,
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Capability registry depth
# ===========================================================================

class TestCapabilityDepth:
    @pytest.mark.parametrize("component", [
        "postgres", "pgvector", "redis", "object_storage", "provider",
        "smtp", "webhook", "container"])
    def test_component_detect_contract(self, component):
        item = cap.detect_component(component)
        assert item["component"] == component
        assert item["state"] in ("AVAILABLE", "UNAVAILABLE", "DEGRADED",
                                 "NOT_CONFIGURED", "UNKNOWN")
        assert isinstance(item["detail"], str)

    @pytest.mark.parametrize("bad", ["", "redis2", "POSTGRES", "../etc"])
    def test_component_invalid_rejected(self, bad):
        with pytest.raises(ValueError):
            cap.detect_component(bad)

    def test_detect_all_count(self):
        assert len(cap.detect_all()) == 8

    def test_persist_upserts_not_duplicates(self, db_session):
        cap.persist_capabilities(db_session)
        cap.persist_capabilities(db_session)
        assert db_session.query(InfraCapability).count() == 8

    def test_redis_config_pool_bounds(self):
        cfg = cap.redis_production_config()
        assert 1 <= cfg["pool"]["max_connections"] <= 1000
        assert 0 < cfg["visibility_timeout_s"] <= 3600

    def test_failover_plan_safety_contract(self):
        plan = cap.broker_failover_plan()
        safety = " ".join(plan["safety"])
        assert "idempotent" in safety or "dedupe" in safety
        assert "no silent switch" in safety


# ===========================================================================
# Vector platform depth
# ===========================================================================

class TestVectorDepth:
    def test_coverage_zero_workspace(self, db_session):
        snap = vp.coverage_snapshot(db_session, 987654)
        assert snap["total_chunks"] == 0
        assert snap["coverage_pct"] == 0.0

    def test_drift_zero_workspace(self, db_session):
        snap = vp.drift_snapshot(db_session, 987654)
        assert snap["drift_pct"] == 0.0

    def test_benchmark_query_bounds(self, db_session):
        ws = _mkws(db_session, "vq")
        big = vp.benchmark(db_session, ws.id, queries=10 ** 9)
        assert big["queries"] == 100  # clamped
        small = vp.benchmark(db_session, ws.id, queries=0)
        assert small["queries"] == 1

    def test_benchmark_p50_p95_ordered(self, db_session):
        ws = _mkws(db_session, "vp")
        result = vp.benchmark(db_session, ws.id, queries=10)
        if result["p50_ms"] is not None and result["p95_ms"] is not None:
            assert result["p50_ms"] <= result["p95_ms"]

    def test_rebuild_missing_document(self, db_session):
        ws = _mkws(db_session, "vr")
        result = vp.rebuild_document_vectors(db_session, ws.id, 424242)
        assert result["ok"] is True and result["chunks"] == 0

    def test_coverage_snapshot_rows_increase(self, db_session):
        from app.models.phase19 import VectorCoverageSnapshot
        ws = _mkws(db_session, "vcs")
        before = db_session.query(VectorCoverageSnapshot).count()
        vp.coverage_snapshot(db_session, ws.id)
        after = db_session.query(VectorCoverageSnapshot).count()
        assert after == before + 1


# ===========================================================================
# Continuous evaluation stability
# ===========================================================================

class TestEvalStability:
    @pytest.mark.parametrize("rounds", [1, 2, 3])
    def test_repeated_executions_stable(self, db_session, rounds):
        ws = _mkws(db_session)
        results = []
        for _ in range(rounds):
            created = ce.create_execution(db_session, ws.id,
                                          domain="retrieval")
            results.append(ce.run_execution(db_session, created["id"]))
        assert all(r["status"] in ("DONE", "COMPLETED", "SUCCEEDED")
                   for r in results)

    def test_max_cases_clamped(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="rag")
        result = ce.run_execution(db_session, created["id"],
                                  max_cases=10 ** 9)
        assert result["status"] in ("DONE", "COMPLETED", "SUCCEEDED")

    def test_list_executions_offset(self, db_session):
        ws = _mkws(db_session)
        for _ in range(3):
            created = ce.create_execution(db_session, ws.id,
                                          domain="retrieval")
            ce.run_execution(db_session, created["id"])
        page1 = ce.list_executions(db_session, ws.id, limit=2, offset=0)
        page2 = ce.list_executions(db_session, ws.id, limit=2, offset=2)
        ids1 = {r.id for r in page1}
        ids2 = {r.id for r in page2}
        assert not ids1 & ids2  # no overlap

    def test_regression_missing_previous_handled(self, db_session):
        ws = _mkws(db_session)
        result = ce.detect_regression(db_session, ws.id, "workflow",
                                      {"success_rate": 0.9})
        assert isinstance(result, dict)
        assert result.get("regressed") is False or result.get("previous") is None


# ===========================================================================
# Provider/cost/retention bounds
# ===========================================================================

class TestBoundsDepth:
    def test_reconciliation_zero_usage(self, db_session):
        ws = _mkws(db_session)
        result = pv.reconcile_cost(db_session, ws.id, "fake")
        assert result["delta_pct"] == 0.0

    def test_retention_future_cutoff(self, db_session):
        result = sco.run_retention(db_session, "traces", older_than_days=0)
        assert result["dry_run"] is True

    def test_retention_huge_window_safe(self, db_session):
        result = sco.run_retention(db_session, "events", older_than_days=9999)
        assert isinstance(result, dict)

    def test_cost_fairness_empty(self):
        assert sco.cost_fairness_plan([], total=100.0) == []

    def test_cost_fairness_zero_total(self):
        assert sco.cost_fairness_plan([(1, 1.0)], total=0.0) == []

    def test_provider_health_never_leaks_secrets(self, db_session):
        ws = _mkws(db_session)
        pv.validate_completion(db_session, ws.id)
        blob = str(pv.provider_health_summary(db_session))
        assert "sk-" not in blob and "Bearer" not in blob


# ===========================================================================
# Stream + trace audit invariants
# ===========================================================================

class TestAuditInvariants:
    def test_every_loop_run_emits_exactly_one_event(self, db_session):
        ws = _mkws(db_session)
        loop = ol.run_selfheal_loop(db_session, ws.id, "worker_stall")
        events = [e for e in ol.stream_events(db_session, "incident",
                                              workspace_id=ws.id)
                  if e.dedup_key == f"selfheal:{ws.id}:{loop.id}"]
        assert len(events) == 1

    def test_trace_finish_idempotent_safe(self, db_session):
        ws = _mkws(db_session)
        started = ob.start_trace(db_session, workspace_id=ws.id,
                                 stage="api", name="idem", force_record=True)
        ob.finish_trace(db_session, started["span_id"], ok=True)
        result = ob.finish_trace(db_session, started["span_id"], ok=False,
                                 error="late")
        assert "ok" in result

    def test_streams_ordered_across_types(self, db_session):
        ws = _mkws(db_session)
        for stream in ("execution", "worker", "provider", "incident"):
            ol.emit_stream(db_session, ws.id, stream, "tick", {})
        for stream in ("execution", "worker", "provider", "incident"):
            seqs = [e.seq for e in ol.stream_events(db_session, stream)]
            assert seqs == sorted(seqs)

    def test_stream_read_isolation_between_streams(self, db_session):
        ws = _mkws(db_session)
        ol.emit_stream(db_session, ws.id, "execution", "exec_kind", {})
        ol.emit_stream(db_session, ws.id, "provider", "prov_kind", {})
        execs = ol.stream_events(db_session, "execution", workspace_id=ws.id)
        assert all(e.kind != "prov_kind" for e in execs)

    def test_loop_timeline_json_loadable(self, db_session):
        import json
        ws = _mkws(db_session)
        loop = ol.run_autonomy_loop(db_session, ws.id, "retrieval",
                                    metric={"quality": 0.9,
                                            "reliability": 0.99,
                                            "cost": 1.0})
        if loop.timeline:
            stages = json.loads(loop.timeline)
            assert isinstance(stages, str) or isinstance(stages, list)
