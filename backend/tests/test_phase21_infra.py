"""Phase 21 tests — infrastructure self-healing, regions, incidents,
maintenance, DR, chaos/load, personal AI, agent/workflow autonomy.

Region health + residency guard + failover simulation; worker health/
quarantine/recovery/capacity with fairness; broker/scheduler healing with
duplicate protection and schedule dedup; slow queries (recommendations-only);
cache health + tenant-safe invalidation; graph repair governance; memory
autonomy; incident severity model + auto-creation + timeline + postmortem
draft/finalize + learning; maintenance dry-run + approval + legal holds;
backup/DR honesty; bounded chaos/load probes; personal autonomy clamping,
memory controls, activity feed; agent plan risk/simulation/optimization/
recovery/dead letters; workflow risk engine.
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
from app.models.phase19 import AgentDeadLetter  # noqa: E402
from app.models.phase21 import (  # noqa: E402
    AutonomousOperation, AutonomyPolicy, RegionHealthP21,
    FailoverSimulation, ResidencyGuardEvent, WorkerHealthScore,
    CapacityRecommendation, BrokerHealthSnapshot, SchedulerHealthSnapshot,
    SlowQueryRecord, CacheHealthSnapshot, GraphRepairProposal,
    MemoryAutonomyEvent, PersonalAutonomySetting, AIActivityItem,
    IncidentP21, MaintenancePlan, BackupHealthRecord, ChaosTestRun,
    AgentPlanRisk, AgentRecoveryEvent, WorkflowRiskAssessment,
)
from app.services import autonomy as au  # noqa: E402
from app.services import infra_heal as ih  # noqa: E402
from app.services import incident_ops as io_  # noqa: E402
from app.services import personal_ai as pa  # noqa: E402
from app.services import agent_autonomy as aa  # noqa: E402

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


P21_INFRA_TABLES = [
    ChaosTestRun, BackupHealthRecord, MaintenancePlan, IncidentP21,
    AIActivityItem, PersonalAutonomySetting, MemoryAutonomyEvent,
    GraphRepairProposal, CacheHealthSnapshot, SlowQueryRecord,
    SchedulerHealthSnapshot, BrokerHealthSnapshot, CapacityRecommendation,
    WorkerHealthScore, ResidencyGuardEvent, FailoverSimulation,
    RegionHealthP21, WorkflowRiskAssessment, AgentRecoveryEvent,
    AgentPlanRisk, AutonomousOperation, AutonomyPolicy, AgentDeadLetter,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in P21_INFRA_TABLES:
        db_session.query(model).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def fresh_workspace(db_session, org_id=None):
    _counter[0] += 1
    user = User(name=f"owner{_counter[0]}",
                email=f"owner{_counter[0]}@p21infra.example",
                password_hash="x")
    db_session.add(user)
    db_session.flush()
    ws = Workspace(name=f"ws{_counter[0]}", owner_id=user.id,
                   organization_id=org_id)
    db_session.add(ws)
    db_session.commit()
    return ws.id


def fresh_org(db_session):
    from app.models.organization import Organization
    _counter[0] += 1
    org = Organization(name=f"org{_counter[0]}",
                       slug=f"org-{_counter[0]}-{uuid.uuid4().hex[:8]}",
                       owner_id=1)
    db_session.add(org)
    db_session.commit()
    return org.id


def allow_low_risk(db_session, ws, operation_type):
    au.create_policy(db_session, ws, operation_type, risk_level="LOW",
                     autonomy_level="AUTO_LOW_RISK", requires_approval=False)


# ===========================================================================
# Multi-region operations
# ===========================================================================

class TestRegions:
    def test_region_health_recorded(self, db_session):
        org = fresh_org(db_session)
        row = ih.record_region_health(db_session, org, "us-east",
                                      workers=4, queue_depth=10)
        assert row.state == "HEALTHY"
        assert row.failover_ready is True

    def test_unhealthy_region_not_failover_ready(self, db_session):
        org = fresh_org(db_session)
        row = ih.record_region_health(db_session, org, "us-west",
                                      state="DEGRADED", workers=2)
        assert row.failover_ready is False

    def test_residency_guard_blocks(self, db_session):
        org = fresh_org(db_session)
        result = ih.check_residency(db_session, org, "eu-west",
                                    {"__allowed__": ["us-east"]})
        assert result["blocked"] is True
        row = db_session.query(ResidencyGuardEvent).filter_by(
            organization_id=org).first()
        assert row.requested_region == "eu-west"

    def test_residency_allows_permitted(self, db_session):
        org = fresh_org(db_session)
        result = ih.check_residency(db_session, org, "us-east",
                                    {"__allowed__": ["us-east"]})
        assert result["blocked"] is False

    def test_failover_simulation_ready(self, db_session):
        org = fresh_org(db_session)
        target = ih.record_region_health(db_session, org, "us-east",
                                         workers=4)
        sim = ih.simulate_failover(db_session, org, "us-west", "us-east",
                                   target)
        assert sim.ready is True
        assert sim.simulated is True
        assert sim.executed is False

    def test_failover_blocked_no_capacity(self, db_session):
        org = fresh_org(db_session)
        target = ih.record_region_health(db_session, org, "us-east",
                                         workers=0)
        sim = ih.simulate_failover(db_session, org, "us-west", "us-east",
                                   target)
        assert sim.ready is False

    def test_failover_blocked_residency(self, db_session):
        org = fresh_org(db_session)
        target = ih.record_region_health(db_session, org, "eu-west",
                                         workers=4)
        sim = ih.simulate_failover(db_session, org, "us-west", "eu-west",
                                   target, {"__allowed__": ["us-east"]})
        assert sim.ready is False
        assert sim.residency_ok is False

    def test_failover_execution_governed(self, db_session):
        org = fresh_org(db_session)
        target = ih.record_region_health(db_session, org, "us-east",
                                         workers=4)
        sim = ih.simulate_failover(db_session, org, "us-west", "us-east",
                                   target)
        result = ih.execute_failover(db_session, sim)
        # HIGH-risk operation under default policy -> requires approval.
        assert result["decision"] == "REQUIRES_APPROVAL"

    def test_failover_execution_blocked_when_not_ready(self, db_session):
        org = fresh_org(db_session)
        target = ih.record_region_health(db_session, org, "us-east",
                                         workers=0)
        sim = ih.simulate_failover(db_session, org, "us-west", "us-east",
                                   target)
        result = ih.execute_failover(db_session, sim)
        assert result["decision"] == "BLOCKED"


# ===========================================================================
# Worker self-healing 2.0
# ===========================================================================

class TestWorkerHealing:
    def test_healthy_worker(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_worker_health(db_session, ws, "w1",
                                      heartbeat_age_seconds=5,
                                      throughput=10.0)
        assert row.score >= 90
        assert row.state == "HEALTHY"

    def test_stale_heartbeat_degraded_or_quarantined(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_worker_health(db_session, ws, "w2",
                                      heartbeat_age_seconds=500)
        # 500s stale costs 40 points -> 60 = DEGRADED; with any queue latency
        # or failure it crosses into QUARANTINED.
        row2 = ih.record_worker_health(db_session, ws, "w2b",
                                       heartbeat_age_seconds=500,
                                       failure_count=5)
        assert row.state in ("DEGRADED", "QUARANTINED")
        assert row2.state == "QUARANTINED"
        assert row2.quarantined is True

    def test_quarantine_safe_auto(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_worker_health(db_session, ws, "w3",
                                      heartbeat_age_seconds=500,
                                      failure_count=5)
        ih.quarantine_worker(db_session, row)
        assert row.quarantined is True

    def test_recovery_restores_worker(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_worker_health(db_session, ws, "w4",
                                      heartbeat_age_seconds=500)
        recovered = ih.recover_worker(db_session, row)
        assert recovered.state == "RECOVERED"
        assert recovered.quarantined is False
        assert recovered.heartbeat_age_seconds == 0

    def test_capacity_scale_up(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.recommend_capacity(db_session, ws, current_workers=2,
                                    queue_depth=100)
        # Growth cap: min(needed=10, 2x current=4) -> 4, still above floor.
        assert row.recommended_workers == 4
        assert row.fairness_preserved is True

    def test_capacity_scale_up_uncapped_small_fleet(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.recommend_capacity(db_session, ws, current_workers=1,
                                    queue_depth=50, avg_worker_throughput=10)
        # Growth cap 2x binds even for a 1-worker fleet (conservative).
        assert row.recommended_workers == 2

    def test_capacity_fairness_floor(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.recommend_capacity(db_session, ws, current_workers=10,
                                    queue_depth=1)
        # Never recommends below half the current fleet.
        assert row.recommended_workers >= 5
        assert row.fairness_preserved is True

    def test_capacity_growth_cap(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.recommend_capacity(db_session, ws, current_workers=4,
                                    queue_depth=100000)
        assert row.recommended_workers <= 8  # capped at 2x current


# ===========================================================================
# Broker / scheduler / database / cache healing
# ===========================================================================

class TestBrokerHealing:
    def test_healthy_broker(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_broker_health(db_session, ws)
        assert row.state == "HEALTHY"
        assert row.degraded is False

    def test_degraded_on_reconnects(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_broker_health(db_session, ws, reconnects=5)
        assert row.degraded is True

    def test_recovery_plan_safe_actions(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_broker_health(db_session, ws, reconnects=5,
                                      visibility_timeouts=9)
        plan = ih.broker_recovery_plan(row)
        assert "reconnect_broker" in plan["safe_auto"]
        assert "requeue_safe_jobs" in plan["safe_auto"]
        assert "drop_data" not in plan["actions"]

    def test_duplicate_protection(self):
        result = ih.verify_duplicate_protection(["a", "b", "a"])
        assert result["duplicates_detected"] == 1
        assert result["protected"] is True


class TestSchedulerHealing:
    def test_no_leader_degraded(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_scheduler_health(db_session, ws, has_leader=False)
        assert row.state == "DEGRADED"

    def test_missed_schedules_recovered_bounded(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_scheduler_health(db_session, ws, has_leader=True,
                                         missed_schedules=500)
        assert row.recovered_schedules == 50  # bounded

    def test_schedule_dedup(self, db_session):
        ws = fresh_workspace(db_session)
        result = ih.dedupe_schedule(db_session, ws, "nightly",
                                    active_leaders=["l1", "l2"])
        assert result["fired"] is False
        assert "dedup" in result["reason"].lower()

    def test_single_leader_fires(self, db_session):
        ws = fresh_workspace(db_session)
        result = ih.dedupe_schedule(db_session, ws, "nightly",
                                    active_leaders=["l1"])
        assert result["fired"] is True

    def test_no_leader_no_fire(self, db_session):
        ws = fresh_workspace(db_session)
        result = ih.dedupe_schedule(db_session, ws, "nightly",
                                    active_leaders=[])
        assert result["fired"] is False


class TestDatabaseMonitoring:
    def test_slow_query_recorded(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_slow_query(db_session, ws, "SELECT * FROM docs WHERE x = 1",
                                   2500.0)
        assert row is not None
        assert "index" in row.recommendation

    def test_fast_query_skipped(self, db_session):
        ws = fresh_workspace(db_session)
        assert ih.record_slow_query(db_session, ws, "SELECT 1", 5.0) is None

    def test_recommendations_only_never_auto(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_slow_query(db_session, ws, "SELECT 1", 5000.0)
        assert row.auto_applied is False

    def test_db_anomalies(self):
        anomalies = ih.db_anomalies([
            {"latency_ms": 2000, "locks": 15, "failures": 2}])
        kinds = {a["kind"] for a in anomalies}
        assert kinds == {"high_latency", "lock_contention", "failures"}


class TestCacheIntelligence:
    def test_healthy_cache(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_cache_health(db_session, ws, "embedding_cache",
                                     hits=90, misses=10)
        assert row.hit_rate == 0.9
        assert row.anomaly is False

    def test_low_hit_rate_anomaly(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.record_cache_health(db_session, ws, "embedding_cache",
                                     hits=5, misses=95)
        assert row.anomaly is True
        assert "cardinality" in row.recommendation

    def test_tenant_safe_invalidation(self):
        keys = ["ws:1:doc:1", "ws:1:doc:2", "ws:2:doc:1"]
        result = ih.invalidate_cache_tenant_safe(keys, workspace_id=1)
        assert result["invalidated"] == ["ws:1:doc:1", "ws:1:doc:2"]
        assert result["skipped_foreign"] == 1
        assert result["tenant_safe"] is True


# ===========================================================================
# Graph + memory autonomy
# ===========================================================================

class TestGraphMemoryAutonomy:
    def test_repair_proposal_created(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.propose_graph_repair(db_session, ws, "stale_edge_refresh",
                                      target_entity_id=3)
        assert row.status == "PROPOSED"
        assert row.risk_level == "LOW"

    def test_unknown_repair_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            ih.propose_graph_repair(db_session, ws, "rewrite_history")

    def test_repair_execution_governed(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.propose_graph_repair(db_session, ws, "stale_edge_refresh")
        result = ih.execute_graph_repair(db_session, row)
        assert result["decision"] == "REQUIRES_APPROVAL"

    def test_repair_execution_allowed_with_policy(self, db_session):
        ws = fresh_workspace(db_session)
        allow_low_risk(db_session, ws, "graph.repair.stale_edge_refresh")
        row = ih.propose_graph_repair(db_session, ws, "stale_edge_refresh")
        result = ih.execute_graph_repair(db_session, row)
        assert result["decision"] == "ALLOWED"
        assert row.status == "COMPLETED"

    def test_memory_health(self):
        result = ih.memory_health_check([
            {"stale": False, "confidence": 0.9, "provenance": True},
            {"stale": True, "confidence": 0.1, "provenance": False}])
        assert result["stale"] == 1
        assert result["low_confidence"] == 1
        assert result["overall"] < 100

    def test_suppression_requires_policy(self, db_session):
        ws = fresh_workspace(db_session)
        result = ih.suppress_memory(db_session, ws, 1, policy_allows=False,
                                    reason="old")
        assert result["suppressed"] is False

    def test_suppression_with_policy_audited(self, db_session):
        ws = fresh_workspace(db_session)
        result = ih.suppress_memory(db_session, ws, 1, policy_allows=True,
                                    reason="expired", automatic=True)
        assert result["suppressed"] is True
        row = db_session.query(MemoryAutonomyEvent).filter_by(
            workspace_id=ws).first()
        assert row.event_kind == "suppressed"
        assert row.automatic is True

    def test_conflict_routed_to_review(self, db_session):
        ws = fresh_workspace(db_session)
        row = ih.route_memory_conflict(db_session, ws, 2, "conflicting facts")
        assert row.event_kind == "conflict_review"
        assert row.automatic is False


# ===========================================================================
# Incident management 2.0
# ===========================================================================

class TestIncidents2:
    def test_severity_model(self, db_session):
        ws = fresh_workspace(db_session)
        for sev in ("SEV0", "SEV1", "SEV2", "SEV3", "SEV4"):
            row = io_.create_incident(db_session, ws, f"inc-{sev}",
                                      severity=sev)
            assert row.severity == sev

    def test_invalid_severity_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            io_.create_incident(db_session, ws, "x", severity="SEV9")

    def test_auto_incident_creation(self, db_session):
        ws = fresh_workspace(db_session)
        rows = io_.evaluate_incident_rules(db_session, ws,
                                           {"broker_failure": 2})
        assert len(rows) == 1
        assert rows[0].severity == "SEV1"

    def test_no_incident_below_threshold(self, db_session):
        ws = fresh_workspace(db_session)
        rows = io_.evaluate_incident_rules(db_session, ws,
                                           {"provider_failure": 1})
        assert rows == []

    def test_timeline_appended(self, db_session):
        ws = fresh_workspace(db_session)
        inc = io_.create_incident(db_session, ws, "outage")
        io_.add_timeline_event(db_session, inc, "diagnosis", "db down")
        io_.add_timeline_event(db_session, inc, "recovery", "failover")
        timeline = json.loads(inc.timeline)
        kinds = [t["kind"] for t in timeline]
        assert kinds == ["detection", "diagnosis", "recovery"]

    def test_resolution(self, db_session):
        ws = fresh_workspace(db_session)
        inc = io_.create_incident(db_session, ws, "outage")
        io_.resolve_incident(db_session, inc)
        assert inc.status == "RESOLVED"
        assert inc.resolved_at is not None

    def test_postmortem_draft_requires_review(self, db_session):
        ws = fresh_workspace(db_session)
        inc = io_.create_incident(db_session, ws, "outage")
        io_.resolve_incident(db_session, inc)
        io_.draft_postmortem(db_session, inc, "db connection pool exhausted",
                             ["traffic spike"])
        assert inc.postmortem_finalized is False
        pm = json.loads(inc.postmortem)
        assert pm["status"] == "DRAFT_REQUIRES_HUMAN_REVIEW"
        assert pm["diagnosis"] == "db connection pool exhausted"

    def test_postmortem_finalize_human_only(self, db_session):
        ws = fresh_workspace(db_session)
        inc = io_.create_incident(db_session, ws, "outage")
        io_.draft_postmortem(db_session, inc, "cause", [])
        io_.finalize_postmortem(db_session, inc, "alice@example.com",
                                action_items=[{"action": "add alert"}])
        assert inc.postmortem_finalized is True
        pm = json.loads(inc.postmortem)
        assert "alice" in pm["status"]

    def test_finalize_without_draft_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        inc = io_.create_incident(db_session, ws, "outage")
        with pytest.raises(ValueError):
            io_.finalize_postmortem(db_session, inc, "alice")

    def test_learning_requires_finalized_postmortem(self, db_session):
        ws = fresh_workspace(db_session)
        inc = io_.create_incident(db_session, ws, "outage")
        io_.draft_postmortem(db_session, inc, "cause", [])
        with pytest.raises(ValueError):
            io_.record_incident_learning(db_session, inc,
                                         [{"kind": "test", "detail": "t"}])

    def test_learning_recorded(self, db_session):
        ws = fresh_workspace(db_session)
        inc = io_.create_incident(db_session, ws, "outage")
        io_.draft_postmortem(db_session, inc, "cause", [])
        io_.finalize_postmortem(db_session, inc, "alice")
        io_.record_incident_learning(db_session, inc, [
            {"kind": "test", "detail": "regression test for pool"},
            {"kind": "monitoring_rule", "detail": "alert on pool>80%"},
            {"kind": "recovery_playbook", "detail": "pool recycle"},
            {"kind": "evaluation_case", "detail": "load case"}])
        learnings = json.loads(inc.learnings)
        assert len(learnings) == 4

    def test_learning_filters_unknown_kinds(self, db_session):
        ws = fresh_workspace(db_session)
        inc = io_.create_incident(db_session, ws, "outage")
        io_.draft_postmortem(db_session, inc, "cause", [])
        io_.finalize_postmortem(db_session, inc, "alice")
        io_.record_incident_learning(db_session, inc, [
            {"kind": "test", "detail": "t"},
            {"kind": "grant_admin", "detail": "nope"}])
        learnings = json.loads(inc.learnings)
        assert len(learnings) == 1


# ===========================================================================
# Maintenance / DR / chaos
# ===========================================================================

class TestMaintenance:
    def test_plan_proposed(self, db_session):
        ws = fresh_workspace(db_session)
        row = io_.create_maintenance_plan(db_session, ws, "cleanup")
        assert row.status == "PROPOSED"
        assert row.destructive is False

    def test_unknown_kind_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            io_.create_maintenance_plan(db_session, ws, "delete_everything")

    def test_dry_run_respects_legal_holds(self, db_session):
        ws = fresh_workspace(db_session)
        plan = io_.create_maintenance_plan(
            db_session, ws, "stale_artifacts",
            targets=[{"id": 1}, {"id": 2}, {"id": 3}])
        result = io_.maintenance_dry_run(db_session, plan,
                                         protected_ids={2})
        assert result["would_delete"] == 2
        assert result["blocked_by_legal_hold"] == 1
        assert result["executed"] is False

    def test_destructive_requires_approval(self, db_session):
        ws = fresh_workspace(db_session)
        plan = io_.create_maintenance_plan(db_session, ws, "cleanup",
                                           targets=[{"id": 1}])
        result = io_.execute_maintenance(db_session, plan)
        assert result["executed"] is False
        assert "approval" in result["reason"]

    def test_destructive_requires_dry_run(self, db_session):
        ws = fresh_workspace(db_session)
        plan = io_.create_maintenance_plan(db_session, ws, "cleanup",
                                           targets=[{"id": 1}])
        io_.approve_maintenance(db_session, plan)
        result = io_.execute_maintenance(db_session, plan)
        assert result["executed"] is False
        assert "dry run" in result["reason"]

    def test_approved_destructive_executes_bounded(self, db_session):
        ws = fresh_workspace(db_session)
        plan = io_.create_maintenance_plan(db_session, ws, "cleanup",
                                           targets=[{"id": 1}])
        io_.maintenance_dry_run(db_session, plan)
        io_.approve_maintenance(db_session, plan)
        result = io_.execute_maintenance(db_session, plan)
        assert result["executed"] is True


class TestDR:
    def test_backup_unknown_honest(self, db_session):
        ws = fresh_workspace(db_session)
        row = io_.record_backup_health(db_session, ws, None)
        assert row.status == "UNKNOWN"

    def test_backup_stale(self, db_session):
        from datetime import timedelta
        ws = fresh_workspace(db_session)
        old = io_._now() - timedelta(hours=72)
        row = io_.record_backup_health(db_session, ws, old)
        assert row.status == "STALE"

    def test_restore_validation_simulated_flagged(self, db_session):
        ws = fresh_workspace(db_session)
        row = io_.record_backup_health(db_session, ws, io_._now())
        io_.validate_restore(db_session, ws, row, tenant_isolation_ok=True,
                             rto_minutes=30, rpo_minutes=5,
                             simulated=True)
        assert row.restore_validated is True
        assert row.simulated is True
        assert "SIMULATED" in row.detail

    def test_rto_rpo_targets_vs_observed(self, db_session):
        ws = fresh_workspace(db_session)
        row = io_.record_backup_health(db_session, ws, io_._now())
        io_.validate_restore(db_session, ws, row, True, rto_minutes=90,
                             rpo_minutes=20, rto_target=60, rpo_target=15)
        assert row.rto_observed_minutes > row.rto_target_minutes
        assert row.rpo_observed_minutes > row.rpo_target_minutes

    def test_dr_simulation_honest(self, db_session):
        ws = fresh_workspace(db_session)
        result = io_.dr_simulation(db_session, ws)
        assert result["simulated"] is True
        assert "no physical recovery" in result["honesty_note"]
        assert result["dr_ready"] is False  # nothing validated yet


class TestChaosLoad:
    def test_chaos_run_recorded(self, db_session):
        ws = fresh_workspace(db_session)
        row = io_.record_chaos_run(db_session, ws, "provider_timeout",
                                   passed=True, simulated=True)
        assert row.simulated is True
        assert row.bounded is True

    def test_unknown_scenario_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            io_.record_chaos_run(db_session, ws, "meteor", passed=True)

    def test_provider_chaos_scenarios_all_valid(self, db_session):
        ws = fresh_workspace(db_session)
        for scenario in ("provider_timeout", "provider_429", "provider_5xx",
                         "provider_malformed"):
            row = io_.record_chaos_run(db_session, ws, scenario, True)
            assert row.id is not None

    def test_api_load_bounded(self, db_session):
        ws = fresh_workspace(db_session)
        result = io_.api_load_test(200, 8)
        assert result["requests"] == 200
        assert result["simulated"] is True
        assert result["passed"] is True

    def test_api_load_hard_cap(self, db_session):
        ws = fresh_workspace(db_session)
        result = io_.api_load_test(100000, 64)
        assert result["requests"] == 1000
        assert result["concurrency"] == 32

    def test_worker_load(self, db_session):
        ws = fresh_workspace(db_session)
        result = io_.worker_load_test(5000, 4)
        assert result["queue_depth"] == 5000
        assert result["estimated_drain_seconds"] > 0

    def test_noisy_neighbor_fairness(self, db_session):
        ws = fresh_workspace(db_session)
        result = io_.noisy_neighbor_test([
            {"id": "big", "job_weight": 100},
            {"id": "small1", "job_weight": 1},
            {"id": "small2", "job_weight": 1}])
        assert result["fairness_violated"] is False

    def test_soak_verdict(self, db_session):
        result = io_.soak_result(24, 100, 0.001)
        assert result["stable"] is True
        result_bad = io_.soak_result(24, 900, 0.05)
        assert result_bad["stable"] is False


# ===========================================================================
# Personal AI control
# ===========================================================================

class TestPersonalAI:
    def test_default_settings_restrictive(self, db_session):
        ws = fresh_workspace(db_session)
        row = pa.get_personal_settings(db_session, ws, user_id=1)
        assert row.autonomy_level == "RECOMMEND"
        assert row.allow_memory_delete is False

    def test_personal_autonomy_clamped_to_policy(self, db_session):
        ws = fresh_workspace(db_session)
        result = pa.set_personal_autonomy(db_session, ws, 1, "AUTO_LOW_RISK",
                                          workspace_policy_level="OBSERVE")
        assert result["effective_level"] == "OBSERVE"
        assert result["clamped_to_policy"] is True

    def test_personal_autonomy_within_policy(self, db_session):
        ws = fresh_workspace(db_session)
        result = pa.set_personal_autonomy(db_session, ws, 1, "OBSERVE",
                                          workspace_policy_level="RECOMMEND")
        assert result["effective_level"] == "OBSERVE"
        assert result["clamped_to_policy"] is False

    def test_invalid_personal_level_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            pa.set_personal_autonomy(db_session, ws, 1, "AUTO_APPROVAL",
                                     workspace_policy_level="AUTO_APPROVAL")

    def test_memory_inspect_always_allowed(self, db_session):
        ws = fresh_workspace(db_session)
        result = pa.memory_control(db_session, ws, 1, "inspect")
        assert result["performed"] is True

    def test_memory_delete_blocked_by_default(self, db_session):
        ws = fresh_workspace(db_session)
        result = pa.memory_control(db_session, ws, 1, "delete", memory_id=5)
        assert result["performed"] is False

    def test_memory_suppress_permitted_audited(self, db_session):
        ws = fresh_workspace(db_session)
        pa.get_personal_settings(db_session, ws, user_id=1)
        result = pa.memory_control(db_session, ws, 1, "suppress",
                                   memory_id=5)
        assert result["performed"] is True
        rows = db_session.query(MemoryAutonomyEvent).filter_by(
            workspace_id=ws).all()
        assert len(rows) == 1

    def test_activity_feed(self, db_session):
        ws = fresh_workspace(db_session)
        pa.record_activity(db_session, ws, 1, "SUGGESTION", "try filters",
                           explanation="evidence: 3 searches with 0 results")
        pa.record_activity(db_session, ws, 1, "RECOVERY", "reconnected broker")
        items = pa.activity_feed(db_session, ws, 1)
        assert len(items) == 2
        kinds = {i.item_kind for i in items}
        assert kinds == {"SUGGESTION", "RECOVERY"}

    def test_activity_invalid_kind_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            pa.record_activity(db_session, ws, 1, "MIND_CONTROL", "x")

    def test_explanations_policy_only(self, db_session):
        # Explanation text is stored verbatim but the API contract is
        # evidence/policy: the module has no chain-of-thought storage field.
        import inspect
        src = inspect.getsource(pa)
        assert "chain_of_thought" not in src
        assert "hidden_reasoning" not in src


# ===========================================================================
# Agent autonomy 2.0 + workflow autonomy
# ===========================================================================

class TestAgentAutonomy:
    def _plan(self, **step_overrides):
        step = {"tool": "retrieve", "top_k": 5}
        step.update(step_overrides)
        return {"summary": "s", "steps": [step]}

    def test_low_risk_plan(self, db_session):
        ws = fresh_workspace(db_session)
        row = aa.assess_plan(db_session, ws, self._plan())
        assert row.risk_level == "LOW"

    def test_high_risk_plan_external(self, db_session):
        ws = fresh_workspace(db_session)
        row = aa.assess_plan(db_session, ws, self._plan(
            tool="send_email", communication=True))
        assert row.risk_level == "HIGH"
        assert row.handed_off is True  # RECOMMEND default -> handoff

    def test_critical_plan_destructive(self, db_session):
        ws = fresh_workspace(db_session)
        row = aa.assess_plan(db_session, ws, self._plan(destructive=True))
        assert row.risk_level == "CRITICAL"

    def test_sensitive_data_flagged(self, db_session):
        ws = fresh_workspace(db_session)
        result = aa.classify_plan_risk(self._plan(reads_sensitive=True))
        assert "reads_sensitive_data" in result["risk_factors"]
        assert result["risk_level"] == "MEDIUM"

    def test_plan_simulation(self):
        plan = {"steps": [{"tool": "retrieve"},
                          {"tool": "send_email", "communication": True,
                           "condition": False}]}
        sim = aa.simulate_plan(plan)
        assert sim["would_execute"][0]["tool"] == "retrieve"
        assert len(sim["skipped"]) == 1
        assert sim["external_steps"] == 0  # email step skipped by condition

    def test_plan_optimization_finds_redundancy(self):
        plan = {"steps": [{"tool": "retrieve"},
                          {"tool": "retrieve"},
                          {"tool": "retrieve", "top_k": 50}]}
        result = aa.optimize_plan(plan)
        kinds = {o["kind"] for o in result["optimizations"]}
        assert "redundant_step" in kinds
        assert "expensive_call" in kinds
        assert result["count"] >= 2

    def test_optimization_governed(self, db_session):
        ws = fresh_workspace(db_session)
        row = aa.assess_plan(db_session, ws, self._plan())
        result = aa.apply_plan_optimizations(
            db_session, row,
            {"steps": [{"tool": "retrieve"}, {"tool": "retrieve"}]})
        # Default RECOMMEND policy: nothing applied without approval.
        assert row.applied_optimizations is None

    def test_handoff_creates_event(self, db_session):
        ws = fresh_workspace(db_session)
        row = aa.assess_plan(db_session, ws, self._plan(destructive=True))
        event = aa.handoff_to_human(db_session, ws, row, "CRITICAL risk")
        assert event.recovery_kind == "HANDOFF"
        assert row.handed_off is True

    def test_agent_recovery_idempotent(self, db_session):
        ws = fresh_workspace(db_session)
        a = aa.recover_agent_run(db_session, ws, 7, "RETRY")
        b = aa.recover_agent_run(db_session, ws, 7, "RETRY")
        assert a.id == b.id
        c = aa.recover_agent_run(db_session, ws, 7, "RESUME")
        assert c.id != a.id

    def test_invalid_recovery_kind_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            aa.recover_agent_run(db_session, ws, 7, "TIME_TRAVEL")

    def test_dead_letter_recorded(self, db_session):
        ws = fresh_workspace(db_session)
        letter = aa.dead_letter_run(db_session, ws, "exec-123",
                                    "provider timeout after 3 retries")
        assert letter.status == "OPEN"
        assert letter.execution_id == "exec-123"


class TestWorkflowAutonomy:
    def test_low_risk_workflow_governed(self, db_session):
        ws = fresh_workspace(db_session)
        row = aa.assess_workflow_risk(db_session, ws, {"steps": []})
        assert row.risk_level == "LOW"
        # Under the default RECOMMEND policy, execution requires approval —
        # workflows are never auto-executed without a policy allowing it.
        assert row.requires_approval is True
        assert row.autonomy_decision in ("REQUIRES_APPROVAL", "BLOCKED")

    def test_low_risk_workflow_auto_with_policy(self, db_session):
        ws = fresh_workspace(db_session)
        allow_low_risk(db_session, ws, "workflow.execute")
        row = aa.assess_workflow_risk(db_session, ws, {"steps": []})
        assert row.risk_level == "LOW"
        assert row.requires_approval is False
        assert row.autonomy_decision == "ALLOWED"

    def test_destructive_workflow_critical(self, db_session):
        ws = fresh_workspace(db_session)
        row = aa.assess_workflow_risk(
            db_session, ws, {"steps": [], "destructive": True})
        assert row.risk_level == "CRITICAL"
        assert row.requires_approval is True

    def test_external_workflow_high(self, db_session):
        ws = fresh_workspace(db_session)
        row = aa.assess_workflow_risk(
            db_session, ws, {"steps": [], "has_external_side_effects": True})
        assert row.risk_level == "HIGH"

    def test_financial_workflow_critical(self, db_session):
        ws = fresh_workspace(db_session)
        row = aa.assess_workflow_risk(
            db_session, ws, {"steps": [], "financial_impact": True})
        assert row.risk_level == "CRITICAL"

    def test_recurring_failure_detection(self, db_session):
        events = [{"step": "extract"}] * 5 + [{"step": "load"}]
        stats = aa.detect_recurring_workflow_failures(db_session, 1, 9,
                                                      events)
        assert stats["hotspots"][0]["step"] == "extract"
        assert stats["hotspots"][0]["count"] == 5

    def test_optimization_recommendations(self):
        stats = {"hotspots": [{"step": "extract", "count": 5}]}
        recs = aa.recommend_workflow_optimizations(
            stats, {"slowest_step": "load", "slowest_step_ms": 9000})
        kinds = {r["kind"] for r in recs}
        assert "add_retry" in kinds
        assert "parallelize" in kinds

    def test_workflow_simulation(self, db_session):
        ws = fresh_workspace(db_session)
        sim = aa.simulate_workflow(db_session, ws, {
            "steps": [{"name": "extract"}, {"name": "load", "external": True}]})
        assert sim["steps_executed"] == 2
        assert sim["external_steps"] == 1
        assert sim["simulated"] is True
