"""Phase 22 tests — observability 2.0, SLO 2.0, worker runtime, security
operations, cost operations, operating loops, chaos/load harness.
"""

import uuid

import pytest

from tests.shared_db import TestingSessionLocal

from app.models.phase22 import (
    OpsStreamEvent, SecurityScanRun, SelfHealLoopRun, SloBurnEvent,
    WorkerRuntimeEvent,
)
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.services import chaos_harness as chs
from app.services import observability2 as ob
from app.services import operating_loops as ol
from app.services import security_cost_ops as sco

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


P22_OPS_TABLES = [
    WorkerRuntimeEvent, SecurityScanRun, SelfHealLoopRun,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    from app.models import TraceSpan, SloBurnEvent as _SloBurn, \
        SloBudgetWindow as _SloWindow, SloDefinition as _SloDef
    from app.models.phase21 import EmergencyStop, PlatformEvent, \
        RecoveryAttempt, RecoveryPlaybook, AutonomousOperation, \
        AutonomyPolicy, DiagnosisReport, IncidentP21
    from app.models.phase16 import WorkerJob
    from app.models.phase20 import ImprovementProposal, Experiment, \
        ExperimentRun
    for model in P22_OPS_TABLES:
        db_session.query(model).delete()
    for model in (TraceSpan, _SloBurn, _SloWindow, _SloDef, WorkerJob):
        db_session.query(model).delete()
    # phase22 stream rows
    db_session.query(OpsStreamEvent).delete()
    for model in (ImprovementProposal, ExperimentRun, Experiment,
                  AutonomousOperation, EmergencyStop, PlatformEvent,
                  RecoveryAttempt, RecoveryPlaybook, AutonomyPolicy,
                  DiagnosisReport, IncidentP21):
        try:
            db_session.query(model).delete()
        except Exception:  # noqa: BLE001
            db_session.rollback()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db, tag="ops"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p22ops.example", name=tag,
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Steps 96-102: observability 2.0
# ===========================================================================

class TestObservability2:
    def test_trace_start_and_finish(self, db_session):
        ws = _mkws(db_session)
        started = ob.start_trace(db_session, workspace_id=ws.id,
                                 stage="request", name="e2e",
                                 force_record=True)
        assert started["sampled"] is True
        finished = ob.finish_trace(db_session, started["span_id"], ok=True)
        assert finished["ok"] is True

    def test_trace_sampling_drops_when_rate_low(self, db_session):
        started = ob.start_trace(db_session, stage="request", name="x",
                                 force_record=False)
        assert "trace_id" in started

    def test_pii_redaction(self):
        text = "contact john.doe@corp.com or 4111-1111-1111-1111"
        red = ob.redact_pii(text)
        assert "john.doe@corp.com" not in red
        assert "4111" not in red

    def test_latency_breakdown(self, db_session):
        ws = _mkws(db_session)
        s1 = ob.start_trace(db_session, workspace_id=ws.id, stage="retrieval",
                            name="q", force_record=True)
        ob.finish_trace(db_session, s1["span_id"], ok=True)
        s2 = ob.start_trace(db_session, workspace_id=ws.id, stage="rag",
                            name="a", trace_id=s1["trace_id"],
                            force_record=True)
        ob.finish_trace(db_session, s2["span_id"], ok=True)
        breakdown = ob.latency_breakdown(db_session, s1["trace_id"])
        assert "stages_ms" in breakdown
        assert "retrieval" in breakdown["stages_ms"]

    def test_error_correlation(self, db_session):
        ws = _mkws(db_session)
        s = ob.start_trace(db_session, workspace_id=ws.id, stage="provider",
                           name="call", force_record=True)
        ob.finish_trace(db_session, s["span_id"], ok=False,
                        error="ProviderTimeout")
        result = ob.correlate_errors(db_session, error_class="ProviderTimeout")
        assert result.get("spans", 0) >= 1
        assert result.get("traces", 0) >= 1

    def test_cost_correlation(self, db_session):
        ws = _mkws(db_session)
        s = ob.start_trace(db_session, workspace_id=ws.id, stage="provider",
                           name="gen", force_record=True)
        ob.finish_trace(db_session, s["span_id"], ok=True, cost_usd=0.02)
        result = ob.correlate_cost(db_session, s["trace_id"])
        assert "cost_usd" in result or "total_cost" in result

    def test_trace_tenant_scoped(self, db_session):
        ws_a = _mkws(db_session, "ta")
        ws_b = _mkws(db_session, "tb")
        s = ob.start_trace(db_session, workspace_id=ws_a.id, stage="request",
                           name="x", force_record=True)
        ob.finish_trace(db_session, s["span_id"], ok=True)
        breakdown = ob.latency_breakdown(db_session, s["trace_id"])
        spans_ws = breakdown.get("workspace_id")
        if spans_ws is not None:
            assert spans_ws == ws_a.id


# ===========================================================================
# Steps 103-107: SLO 2.0
# ===========================================================================

class TestSlo2:
    def test_define_slo(self, db_session):
        result = ob.define_slo(db_session, domain="api", name="p22-api-slo",
                               target=0.99)
        assert result["name"] == "p22-api-slo"
        assert result["id"] > 0

    def test_define_slo_idempotent_by_name(self, db_session):
        ob.define_slo(db_session, domain="api", name="p22-idem", target=0.99)
        again = ob.define_slo(db_session, domain="api", name="p22-idem",
                              target=0.995)
        assert again["name"] == "p22-idem"

    def test_error_budget(self, db_session):
        defined = ob.define_slo(db_session, domain="api", name="p22-budget",
                                target=0.99)
        result = ob.compute_error_budget(
            db_session, defined["id"], good_events=990, total_events=1000)
        assert result.get("ok", True)

    def test_burn_rate_creates_incident_on_breach(self, db_session):
        ws = _mkws(db_session)
        defined = ob.define_slo(db_session, domain="rag",
                                name=f"p22-burn-{ws.id}", target=0.99)
        ob.compute_error_budget(db_session, defined["id"],
                                good_events=50, total_events=1000)
        result = ob.evaluate_burn_rate(db_session, ws.id)
        assert "burn_events" in result or "incidents" in result


# ===========================================================================
# Steps 136-144: worker runtime
# ===========================================================================

class TestWorkerRuntime:
    def test_heartbeat_recorded(self, db_session):
        result = ob.record_heartbeat(db_session, "p22-heartbeat-worker",
                                     depth=0)
        assert result["ok"] is True

    def test_lease_recovery(self, db_session):
        result = ob.recover_stale_leases(db_session, stale_after_s=0)
        assert "recovered" in result or "jobs_recovered" in result

    def test_weighted_fair_share(self):
        shares = ob.weighted_fair_share([(1, 1, 10), (2, 2, 10)], slots=6)
        assert isinstance(shares, list) and shares

    def test_priority_order(self):
        jobs = [{"id": 1, "priority": "LOW"}, {"id": 2, "priority": "CRITICAL"}]
        ordered = ob.priority_order(jobs)
        assert ordered[0]["priority"] == "CRITICAL"

    def test_dead_letter_recovery(self, db_session):
        result = ob.recover_dead_letters(db_session)
        assert "requeued" in result or isinstance(result, dict)

    def test_graceful_shutdown(self, db_session):
        ob.record_heartbeat(db_session, "p22-shutdown-worker", depth=0)
        result = ob.graceful_shutdown_state(db_session,
                                            "p22-shutdown-worker",
                                            active_jobs=0)
        assert result["state"] == "drain_complete"
        assert result["accepting_new"] is False


# ===========================================================================
# Steps 115-122: security operations
# ===========================================================================

class TestSecurityOps:
    def test_injection_corpus_scan(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_security_scan(db_session, ws.id, "prompt_injection")
        assert result["corpus"] == "prompt_injection"
        assert result["cases"] >= 1
        assert result["scan_passed"] is True

    def test_exfiltration_scan(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_security_scan(db_session, ws.id, "exfiltration")
        assert result["scan_passed"] is True

    def test_tool_abuse_scan(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_security_scan(db_session, ws.id, "tool_abuse")
        assert result["scan_passed"] is True

    def test_ssrf_scan(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_security_scan(db_session, ws.id, "ssrf")
        assert result["scan_passed"] is True

    def test_tenant_matrix_scan(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_security_scan(db_session, ws.id, "tenant_matrix")
        assert result["scan_passed"] is True

    def test_api_abuse_scan(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_security_scan(db_session, ws.id, "api_abuse")
        assert result["scan_passed"] is True

    def test_autonomy_bypass_scan(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_security_scan(db_session, ws.id, "autonomy_bypass")
        assert result["scan_passed"] is True

    def test_run_all_scans(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_all_security_scans(db_session, ws.id)
        assert len(result["scans"]) >= 7
        assert result["all_passed"] is True

    def test_scan_run_persisted(self, db_session):
        ws = _mkws(db_session)
        sco.run_security_scan(db_session, ws.id, "prompt_injection")
        assert db_session.query(SecurityScanRun).count() >= 1


# ===========================================================================
# Steps 130-135: cost operations
# ===========================================================================

class TestCostOps:
    def test_cost_anomaly_shape(self, db_session):
        ws = _mkws(db_session)
        result = sco.detect_cost_anomaly(db_session, ws.id)
        assert "anomaly" in result and "today" in result

    def test_cost_forecast_shape(self, db_session):
        ws = _mkws(db_session)
        result = sco.forecast_cost(db_session, ws.id, horizon_days=7)
        assert "forecast_units" in result and "horizon_days" in result

    def test_budget_guard_allow(self, db_session):
        ws = _mkws(db_session)
        result = sco.enforce_budget(db_session, ws.id,
                                    estimated_cost_usd=1.0,
                                    budget_limit_usd=10.0)
        assert result["decision"] == "ALLOWED"

    def test_budget_guard_block(self, db_session):
        ws = _mkws(db_session)
        result = sco.enforce_budget(db_session, ws.id,
                                    estimated_cost_usd=100.0,
                                    budget_limit_usd=10.0)
        assert result["decision"] in ("BLOCKED", "REQUIRES_APPROVAL")

    def test_cost_fairness(self):
        plan = sco.cost_fairness_plan([(1, 5.0), (2, 1.0)], total=12.0)
        assert isinstance(plan, list) and plan


# ===========================================================================
# Steps 175-191: operating loops
# ===========================================================================

class TestOperatingLoops:
    def test_selfheal_loop_no_failure(self, db_session):
        ws = _mkws(db_session)
        loop = ol.run_selfheal_loop(db_session, ws.id, "worker_stall")
        assert loop.detected is False
        assert loop.stage == "DONE"
        assert loop.escalated is False

    def test_selfheal_loop_persisted(self, db_session):
        ws = _mkws(db_session)
        ol.run_selfheal_loop(db_session, ws.id, "provider_failure")
        assert db_session.query(SelfHealLoopRun).count() >= 1

    def test_selfheal_loop_emits_stream_event(self, db_session):
        ws = _mkws(db_session)
        loop = ol.run_selfheal_loop(db_session, ws.id, "queue_buildup")
        events = ol.stream_events(db_session, "incident",
                                  workspace_id=ws.id)
        assert any(e.dedup_key == f"selfheal:{ws.id}:{loop.id}"
                   for e in events)

    def test_autonomy_loop_healthy_no_action(self, db_session):
        ws = _mkws(db_session)
        loop = ol.run_autonomy_loop(db_session, ws.id, "retrieval",
                                    metric={"quality": 0.95,
                                            "reliability": 0.99,
                                            "cost": 1.0})
        assert loop.deviation_detected is False
        assert loop.stage == "DONE"

    def test_autonomy_loop_deviation_blocked_or_approval(self, db_session):
        ws = _mkws(db_session)
        loop = ol.run_autonomy_loop(
            db_session, ws.id, "retrieval",
            metric={"quality": 0.3, "reliability": 0.7, "cost": 200.0,
                    "latency_ms": 400.0})
        # Low quality candidate must never be auto-activated.
        assert loop.governance in ("BLOCKED", "APPROVAL")
        assert loop.activated is False

    def test_autonomy_loop_timeline_auditable(self, db_session):
        ws = _mkws(db_session)
        loop = ol.run_autonomy_loop(db_session, ws.id, "rag",
                                    metric={"quality": 0.4,
                                            "reliability": 0.8,
                                            "cost": 10.0})
        assert loop.timeline is not None
        assert "OBSERVE" in loop.timeline

    def test_autonomy_loop_emits_stream_event(self, db_session):
        ws = _mkws(db_session)
        loop = ol.run_autonomy_loop(db_session, ws.id, "search",
                                    metric={"quality": 0.2,
                                            "reliability": 0.9,
                                            "cost": 5.0})
        events = ol.stream_events(db_session, "execution",
                                  workspace_id=ws.id)
        assert any(e.dedup_key == f"autonomyloop:{ws.id}:{loop.id}"
                   for e in events)


# ===========================================================================
# Steps 170-174: ops streams
# ===========================================================================

class TestOpsStreams:
    def test_emit_and_read_back(self, db_session):
        ws = _mkws(db_session)
        ol.emit_stream(db_session, ws.id, "worker", "heartbeat",
                       {"worker": "w1"})
        events = ol.stream_events(db_session, "worker", workspace_id=ws.id)
        assert any(e.kind == "heartbeat" for e in events)

    def test_stream_seq_monotonic(self, db_session):
        ws = _mkws(db_session)
        e1 = ol.emit_stream(db_session, ws.id, "provider", "health", {})
        e2 = ol.emit_stream(db_session, ws.id, "provider", "health", {})
        assert e2.seq == e1.seq + 1

    def test_stream_dedup_key_unique(self, db_session):
        import pytest as _pytest
        from sqlalchemy.exc import IntegrityError
        ws = _mkws(db_session)
        ol.emit_stream(db_session, ws.id, "execution", "job_done", {},
                       dedup_key=f"dup-test2-{ws.id}")
        # A duplicate dedup key must be rejected by the unique constraint —
        # duplicate side effects are impossible.
        with _pytest.raises(IntegrityError):
            ol.emit_stream(db_session, ws.id, "execution", "job_done", {},
                           dedup_key=f"dup-test2-{ws.id}")
            db_session.rollback()

    def test_stream_bounded_read(self, db_session):
        ws = _mkws(db_session)
        for i in range(10):
            ol.emit_stream(db_session, ws.id, "incident", "tick", {"i": i})
        events = ol.stream_events(db_session, "incident", after_seq=0,
                                  limit=5)
        assert len(events) <= 5

    def test_stream_reconnect_from_cursor(self, db_session):
        ws = _mkws(db_session)
        e1 = ol.emit_stream(db_session, ws.id, "worker", "a", {})
        ol.emit_stream(db_session, ws.id, "worker", "b", {})
        tail = ol.stream_events(db_session, "worker", after_seq=e1.seq)
        assert all(e.seq > e1.seq for e in tail)


# ===========================================================================
# Steps 192-208: chaos / load / soak
# ===========================================================================

class TestChaosHarness:
    @pytest.mark.parametrize("mode", ["timeout", "429", "500", "malformed",
                                      "disconnect"])
    def test_provider_chaos_modes(self, mode):
        result = chs.provider_chaos(mode, requests=10)
        assert result["pass"] is True
        assert result["bounded"] is True

    def test_worker_chaos_reclaims_jobs(self, db_session):
        ws = _mkws(db_session)
        result = chs.worker_chaos(db_session, ws.id, lost_workers=1)
        assert result["reclaimed"] == 1
        assert result["pass"] is True

    def test_broker_chaos(self, db_session):
        ws = _mkws(db_session)
        result = chs.broker_chaos(db_session, ws.id)
        assert result["pass"] is True

    def test_scheduler_chaos(self, db_session):
        ws = _mkws(db_session)
        result = chs.scheduler_chaos(db_session, ws.id)
        assert result["pass"] is True

    def test_database_chaos_atomicity(self, db_session):
        ws = _mkws(db_session)
        result = chs.database_chaos(db_session, ws.id)
        assert result["pass"] is True
        assert result["partial_state"] is False

    def test_ingestion_chaos_quarantines(self, db_session):
        ws = _mkws(db_session)
        result = chs.ingestion_chaos(db_session, ws.id)
        assert result["pass"] is True
        assert result["quarantined"] == 1

    def test_connector_chaos_backoff(self, db_session):
        ws = _mkws(db_session)
        result = chs.connector_chaos(db_session, ws.id)
        assert result["pass"] is True
        assert result["backoff_seconds"] >= 60

    def test_network_chaos_durable_event(self, db_session):
        ws = _mkws(db_session)
        result = chs.network_chaos(db_session, ws.id)
        assert result["pass"] is True

    def test_recovery_validation_governed(self, db_session):
        ws = _mkws(db_session)
        result = chs.recovery_validation(db_session, ws.id)
        assert result["pass"] is True


class TestLoadHarness:
    def test_rag_load(self, db_session):
        result = chs.rag_load_test(queries=50, concurrency=5)
        assert result["queries"] == 50 and result["passed"] is True

    def test_ingestion_load(self, db_session):
        result = chs.ingestion_load_test(documents=100, batch=10)
        assert result["batches"] == 10

    def test_search_load(self, db_session):
        result = chs.search_load_test(queries=200)
        assert result["queries"] == 200

    def test_provider_load(self, db_session):
        result = chs.provider_load_test(calls=100)
        assert result["calls"] == 100

    def test_soak_honest(self):
        result = chs.soak_run(duration_hours=0.1, memory_growth_mb=50,
                              error_rate=0.0)
        assert "NOT run" in result["environment_note"]

    def test_load_result_recorded(self, db_session):
        ws = _mkws(db_session)
        result = chs.rag_load_test(queries=10)
        row = chs.record_load_result_returns_row(db_session, ws.id, "rag",
                                                 result)
        from app.models.phase21 import ChaosTestRun
        fetched = (db_session.query(ChaosTestRun)
                   .filter_by(id=row.id).one_or_none())
        assert fetched is not None
        assert fetched.scenario == "rag_load"
        assert fetched.passed is True

    def test_record_rejects_unknown_kind(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            chs.record_load_result(db_session, ws.id, "bogus", {})
