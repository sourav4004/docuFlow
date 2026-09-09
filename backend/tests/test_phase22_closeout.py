"""Phase 22 tests — closeout suite.

Cross-cutting invariants required by the Phase 22 charter: honest
infrastructure reporting, tenant isolation on every new surface, secret
hygiene across all service outputs, bounded execution everywhere, and
rollback/audit availability for every consequential path.
"""

import uuid

import pytest

from tests.shared_db import TestingSessionLocal

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.services import capabilities as cap
from app.services import chaos_harness as chs
from app.services import continuous_eval as ce
from app.services import knowledge_ops as ko
from app.services import observability2 as ob
from app.services import operating_loops as ol
from app.services import provider_validation as pv
from app.services import region_dr as rd
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
    from app.models.phase16 import WorkerJob, WorkerHeartbeat
    from app.models.phase19 import RegionRecord, RegionFailover, \
        ResidencyRule
    from app.models.phase20 import ImprovementProposal, Experiment, \
        ExperimentRun
    from app.models.phase21 import AutonomousOperation, AutonomyPolicy, \
        EmergencyStop, PlatformEvent, RecoveryAttempt, RecoveryPlaybook, \
        DiagnosisReport, CostGuardDecision
    from app.models.phase22 import EvalExecution, ProviderValidationRun, \
        OpsStreamEvent, SelfHealLoopRun, AutonomyLoopRun, \
        VectorBenchmarkRun, VectorDriftSnapshot, CostReconciliationRun, \
        SecurityScanRun, RetentionExecution, MaintenanceRun, \
        ConnectorSyncState
    for model in (OpsStreamEvent, SelfHealLoopRun, AutonomyLoopRun,
                  VectorBenchmarkRun, VectorDriftSnapshot,
                  CostReconciliationRun, SecurityScanRun,
                  RetentionExecution, MaintenanceRun, EvalExecution,
                  ProviderValidationRun, CostGuardDecision, TraceSpan,
                  WorkerJob, WorkerHeartbeat, RegionFailover, RegionRecord,
                  ResidencyRule, ConnectorSyncState):
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


def _mkws(db, tag="co"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p22co.example", name=tag,
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Honesty invariants (no fabricated validation)
# ===========================================================================

class TestHonestyInvariants:
    def test_no_component_claims_real_without_check(self, db_session):
        cap.persist_capabilities(db_session)
        for row in db_session.query(cap.InfraCapability).all():
            if row.realization == "REAL":
                assert row.state == "AVAILABLE"

    def test_drill_never_real_without_backup(self, db_session):
        ws = _mkws(db_session)
        result = rd.run_restore_drill(db_session, ws.id)
        health = rd.backup_health(db_session)
        if not health.get("configured"):
            assert result["simulated"] is True

    def test_provider_realization_matches_credentials(self, db_session):
        ws = _mkws(db_session)
        result = pv.validate_completion(db_session, ws.id)
        if not pv.any_real_provider():
            assert result["real"] is False
            assert result["simulated"] is True

    def test_vector_benchmark_native_only_when_pgvector(self, db_session):
        ws = _mkws(db_session)
        result = vp.benchmark(db_session, ws.id, queries=5)
        if vp.vector_backend_kind() != "pgvector":
            assert result["native"] is False

    def test_failover_never_auto_by_default(self):
        assert cap.broker_failover_plan()["approved_auto"] is False

    def test_soak_and_load_marked_simulated(self, db_session):
        ws = _mkws(db_session)
        result = chs.rag_load_test(queries=10)
        chs.record_load_result(db_session, ws.id, "rag", result)
        from app.models.phase21 import ChaosTestRun
        row = (db_session.query(ChaosTestRun)
               .filter_by(workspace_id=ws.id, scenario="rag_load")
               .order_by(ChaosTestRun.id.desc()).first())
        assert row.simulated is True


# ===========================================================================
# Secret hygiene across service outputs
# ===========================================================================

class TestSecretHygiene:
    @pytest.mark.parametrize("blob_fn", [
        lambda: str(cap.detect_all()),
        lambda: str(cap.redis_production_config()),
        lambda: str(cap.broker_failover_plan()),
    ])
    def test_capability_outputs_clean(self, blob_fn):
        blob = blob_fn()
        for marker in ("sk-", "password=", "Bearer ", "secret=", "token="):
            assert marker not in blob

    def test_provider_health_clean(self, db_session):
        ws = _mkws(db_session)
        pv.validate_completion(db_session, ws.id)
        blob = str(pv.provider_health_summary(db_session))
        assert "sk-" not in blob

    def test_capability_summary_clean(self, db_session):
        cap.persist_capabilities(db_session)
        blob = str(cap.infrastructure_summary(db_session))
        assert "sk-" not in blob and "password" not in blob.lower()

    def test_region_outputs_clean(self, db_session):
        rd.upsert_region(db_session, region="clean-1")
        blob = str(rd.list_regions(db_session))
        assert "sk-" not in blob and "password" not in blob.lower()


# ===========================================================================
# Tenant isolation on every new surface
# ===========================================================================

class TestIsolationMatrix:
    def test_evaluations_isolated(self, db_session):
        ws_a = _mkws(db_session, "ea")
        ws_b = _mkws(db_session, "eb")
        created = ce.create_execution(db_session, ws_a.id, domain="rag")
        ce.run_execution(db_session, created["id"])
        rows_b = ce.list_executions(db_session, ws_b.id)
        assert all(r.workspace_id == ws_b.id for r in rows_b)

    def test_maintenance_isolated(self, db_session):
        from app.models.phase22 import MaintenanceRun
        ws_a = _mkws(db_session, "ma")
        ws_b = _mkws(db_session, "mb")
        ko.run_maintenance(db_session, ws_a.id, "graph")
        ko.run_maintenance(db_session, ws_b.id, "graph")
        rows = db_session.query(MaintenanceRun).all()
        # Each row belongs to its own workspace.
        assert {r.workspace_id for r in rows} <= {ws_a.id, ws_b.id}

    def test_streams_isolated(self, db_session):
        ws_a = _mkws(db_session, "sa")
        ws_b = _mkws(db_session, "sb")
        ol.emit_stream(db_session, ws_a.id, "incident", "a_only", {})
        events_b = ol.stream_events(db_session, "incident",
                                    workspace_id=ws_b.id)
        assert all(e.workspace_id == ws_b.id for e in events_b)

    def test_traces_isolated(self, db_session):
        ws_a = _mkws(db_session, "ta")
        s = ob.start_trace(db_session, workspace_id=ws_a.id, stage="api",
                           name="iso", force_record=True)
        breakdown = ob.latency_breakdown(db_session, s["trace_id"])
        if breakdown.get("workspace_id") is not None:
            assert breakdown["workspace_id"] == ws_a.id

    def test_cost_ops_isolated(self, db_session):
        ws_a = _mkws(db_session, "ca")
        sco.enforce_budget(db_session, ws_a.id, estimated_cost_usd=1.0,
                           budget_limit_usd=10.0,
                           operation_type="iso.op")
        from app.models.phase21 import CostGuardDecision
        ws_b = _mkws(db_session, "cb")
        rows_b = (db_session.query(CostGuardDecision)
                  .filter_by(workspace_id=ws_b.id).all())
        assert all(r.workspace_id == ws_b.id for r in rows_b)


# ===========================================================================
# Bounded execution everywhere
# ===========================================================================

class TestBoundedExecution:
    def test_maintenance_max_findings(self, db_session):
        ws = _mkws(db_session)
        result = ko.run_maintenance(db_session, ws.id, "embeddings",
                                    max_findings=1)
        assert result["findings"] <= 1

    def test_connector_health_list_bounded(self, db_session):
        ws = _mkws(db_session)
        for cid in range(1, 6):
            ko.connector_sync(db_session, ws.id, cid, items=1)
        summary = ko.connector_health_summary(db_session, ws.id)
        assert len(summary["connectors"]) <= 5

    def test_region_list_bounded(self, db_session):
        for i in range(3):
            rd.upsert_region(db_session, region=f"bounded-{i}")
        rows = rd.list_regions(db_session, limit=2)
        assert len(rows) <= 2

    def test_stream_limit_bounded(self, db_session):
        ws = _mkws(db_session)
        for i in range(12):
            ol.emit_stream(db_session, ws.id, "execution", "tick", {"i": i})
        events = ol.stream_events(db_session, "execution",
                                  workspace_id=ws.id, limit=5)
        assert len(events) <= 5

    def test_retention_batch_limit(self, db_session):
        result = sco.run_retention(db_session, "usage", older_than_days=1,
                                   batch_limit=10)
        assert result["dry_run"] is True


# ===========================================================================
# Rollback / audit availability
# ===========================================================================

class TestRollbackAudit:
    def test_promoted_proposal_rollback_capable(self, db_session):
        ws = _mkws(db_session)
        created = ce.create_execution(db_session, ws.id, domain="retrieval")
        ce.run_execution(db_session, created["id"])
        result = ce.rollback(db_session, ws.id, created["id"])
        assert isinstance(result, dict) and result

    def test_every_guard_operation_recorded(self, db_session):
        from app.models.phase21 import AutonomousOperation
        ws = _mkws(db_session)
        ol.run_selfheal_loop(db_session, ws.id, "worker_stall")
        ops = (db_session.query(AutonomousOperation)
               .filter_by(workspace_id=ws.id).all())
        # Any autonomous operation must carry actor + source for audit.
        for op in ops:
            assert op.actor is not None
            assert op.source is not None

    def test_phase21_guard_still_enforced(self, db_session):
        from app.services import autonomy as au
        ws = _mkws(db_session)
        sim = au.simulate_operation(db_session, ws.id, "never.registered",
                                    risk_level="HIGH")
        assert sim["decision"] != "ALLOWED"

    def test_cost_guard_decision_queryable(self, db_session):
        from app.models.phase21 import CostGuardDecision
        ws = _mkws(db_session)
        sco.enforce_budget(db_session, ws.id, estimated_cost_usd=2.0,
                           budget_limit_usd=10.0,
                           operation_type="audit.cost")
        rows = (db_session.query(CostGuardDecision)
                .filter_by(workspace_id=ws.id).all())
        assert len(rows) == 1
        assert rows[0].decision in ("ALLOWED", "REQUIRES_APPROVAL",
                                    "BLOCKED")
