"""Phase 19 tests — enterprise control plane + distributed workers.

Covers: versioned configuration snapshots with rollback, region/residency/
failover semantics, worker quarantine/self-healing/load/placement/migration/
backpressure, scheduler leader election + job dedupe, broker guarantees/
recovery, and autoscaling signals.
"""

from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.organization import Organization  # noqa: E402
from app.models.phase16 import WorkerJob, WorkerHeartbeat  # noqa: E402
from app.models.phase17 import JobLease, AlertRule  # noqa: E402
from app.models.phase19 import (  # noqa: E402
    ControlPlaneSnapshot, RegionRecord, ResidencyRule, RegionFailover,
    SchedulerLeader, WorkerQuarantine, JobMigrationRecord,
)
from app.services import control_plane as cp  # noqa: E402
from app.services import worker_ops2 as wo  # noqa: E402
from app.services import broker2  # noqa: E402
from app.services import broker  # noqa: E402

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
    db_session.query(JobMigrationRecord).delete()
    db_session.query(WorkerQuarantine).delete()
    db_session.query(SchedulerLeader).delete()
    db_session.query(RegionFailover).delete()
    db_session.query(ResidencyRule).delete()
    db_session.query(RegionRecord).delete()
    db_session.query(ControlPlaneSnapshot).delete()
    db_session.query(JobLease).delete()
    db_session.query(WorkerJob).delete()
    db_session.query(WorkerHeartbeat).delete()
    db_session.commit()
    yield
    broker.reset_broker_cache()


def fresh_user(db, tag="p19c"):
    _counter[0] += 1
    user = User(name=f"P19 Control {_counter[0]}",
                email=f"{tag}{_counter[0]}@p19-control.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user, org=None):
    _counter[0] += 1
    ws = Workspace(name=f"p19 ws {_counter[0]}", owner_id=user.id,
                   organization_id=org.id if org else None)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def fresh_org(db, user):
    _counter[0] += 1
    org = Organization(name=f"p19 org {_counter[0]}",
                       slug=f"p19-org-{_counter[0]}", owner_id=user.id)
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


def hb(db, worker_id, *, status="RUNNING", active=0, assignments="",
       age_s=0):
    now = datetime.now(timezone.utc)
    row = WorkerHeartbeat(worker_id=worker_id, status=status,
                          active_jobs=active,
                          queue_assignments=assignments,
                          last_heartbeat=now - timedelta(seconds=age_s))
    db.add(row)
    db.commit()
    return row


def job(db, ws, *, queue="AI_EXECUTIONS", status="QUEUED", claimed_by=None,
        attempts=0):
    now = datetime.now(timezone.utc)
    row = WorkerJob(queue_name=queue, job_type="test.job",
                    workspace_id=ws.id, status=status,
                    claimed_by=claimed_by, attempt=attempts,
                    lease_expires_at=now + timedelta(minutes=5),
                    created_at=now)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ===========================================================================
# Control-plane snapshots
# ===========================================================================

class TestSnapshots:
    def test_snapshot_versions_are_immutable_and_sequential(self, db_session):
        cp.snapshot_config(db_session, scope_type="ORGANIZATION",
                           scope_id=1, config={"budget": 100},
                           reason="initial")
        cp.snapshot_config(db_session, scope_type="ORGANIZATION",
                           scope_id=1, config={"budget": 150},
                           reason="raise")
        rows = db_session.query(ControlPlaneSnapshot).all()
        assert len(rows) == 2
        assert {r.version for r in rows} == {1, 2}
        assert all(not r.is_active for r in rows)

    def test_activate_snapshot_is_idempotent(self, db_session):
        snap = cp.snapshot_config(db_session, scope_type="WORKSPACE",
                                  scope_id=7, config={"x": 1})
        first = cp.activate_snapshot(db_session, snap.id)
        second = cp.activate_snapshot(db_session, snap.id)
        assert first["status"] == "ACTIVATED"
        assert second["status"] == "ALREADY_ACTIVE"

    def test_activation_deactivates_peer(self, db_session):
        s1 = cp.snapshot_config(db_session, scope_type="ORGANIZATION",
                                scope_id=3, config={"mode": "a"})
        s2 = cp.snapshot_config(db_session, scope_type="ORGANIZATION",
                                scope_id=3, config={"mode": "b"})
        cp.activate_snapshot(db_session, s1.id)
        cp.activate_snapshot(db_session, s2.id)
        db_session.flush()
        active = cp.latest_snapshot(db_session, scope_type="ORGANIZATION",
                                    scope_id=3, only_active=True)
        assert active.id == s2.id
        assert db_session.query(ControlPlaneSnapshot).get(
            s1.id).is_active is False

    def test_effective_config_returns_active(self, db_session):
        snap = cp.snapshot_config(db_session, scope_type="WORKSPACE",
                                  scope_id=9, config={"flag": True})
        cp.activate_snapshot(db_session, snap.id)
        assert cp.effective_config(db_session, scope_type="WORKSPACE",
                                   scope_id=9) == {"flag": True}

    def test_rollback_creates_new_snapshot_and_restores(self, db_session):
        cp.snapshot_config(db_session, scope_type="ORGANIZATION",
                           scope_id=2, config={"policy": "v1"})
        s2 = cp.snapshot_config(db_session, scope_type="ORGANIZATION",
                                scope_id=2, config={"policy": "v2"})
        cp.activate_snapshot(db_session, s2.id)
        result = cp.rollback_snapshot(db_session, scope_type="ORGANIZATION",
                                      scope_id=2, target_version=1,
                                      reason="bad change")
        assert result["rollback"] is True
        assert result["restores_version"] == 1
        assert cp.effective_config(db_session, scope_type="ORGANIZATION",
                                   scope_id=2) == {"policy": "v1"}
        # immutable history: rollback created a NEW snapshot
        assert cp.latest_snapshot(db_session, scope_type="ORGANIZATION",
                                  scope_id=2).version == 3

    def test_rollback_to_active_version_is_noop(self, db_session):
        snap = cp.snapshot_config(db_session, scope_type="WORKSPACE",
                                  scope_id=4, config={"k": 1})
        cp.activate_snapshot(db_session, snap.id)
        result = cp.rollback_snapshot(db_session, scope_type="WORKSPACE",
                                      scope_id=4, target_version=1)
        assert result["status"] == "NOOP"

    def test_rollback_unknown_version_raises(self, db_session):
        with pytest.raises(ValueError):
            cp.rollback_snapshot(db_session, scope_type="WORKSPACE",
                                 scope_id=4, target_version=99)

    def test_diff_configs_detects_add_remove_change(self):
        diff = cp.diff_configs({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert diff["removed"] == {"a": 1}
        assert diff["added"] == {"c": 4}
        assert diff["changed"]["b"] == {"from": 2, "to": 3}

    def test_snapshots_scoped_independently(self, db_session):
        cp.snapshot_config(db_session, scope_type="WORKSPACE", scope_id=1,
                           config={"a": 1})
        cp.snapshot_config(db_session, scope_type="WORKSPACE", scope_id=2,
                           config={"a": 2})
        assert cp.latest_snapshot(db_session, scope_type="WORKSPACE",
                                  scope_id=1).version == 1
        assert cp.latest_snapshot(db_session, scope_type="WORKSPACE",
                                  scope_id=2).version == 1


# ===========================================================================
# Regions / residency / failover
# ===========================================================================

class TestRegions:
    def test_upsert_region_then_update(self, db_session):
        cp.upsert_region(db_session, organization_id=None,
                         region_id="us-east-1", status="HEALTHY")
        cp.upsert_region(db_session, organization_id=None,
                         region_id="us-east-1", status="DEGRADED",
                         health_score=0.6)
        rows = cp.list_regions(db_session)
        assert len(rows) == 1
        assert rows[0].status == "DEGRADED"
        assert rows[0].health_score == 0.6

    def test_invalid_region_status_rejected(self, db_session):
        with pytest.raises(ValueError):
            cp.upsert_region(db_session, organization_id=None,
                             region_id="r", status="BOGUS")

    def test_residency_prohibits_region(self, db_session):
        org = fresh_org(db_session, fresh_user(db_session))
        cp.set_residency_rule(db_session, organization_id=org.id,
                              classification="RESTRICTED",
                              prohibited_regions=["eu-central-1"],
                              allowed_regions=["us-east-1"])
        blocked = cp.evaluate_residency(db_session, organization_id=org.id,
                                        classification="RESTRICTED",
                                        region_id="eu-central-1")
        allowed = cp.evaluate_residency(db_session, organization_id=org.id,
                                        classification="RESTRICTED",
                                        region_id="us-east-1")
        assert blocked["allowed"] is False
        assert allowed["allowed"] is True

    def test_residency_default_allows_when_no_rule(self, db_session):
        assert cp.evaluate_residency(db_session, organization_id=None,
                                     classification="INTERNAL",
                                     region_id="eu-1")["allowed"] is True

    def test_residency_violations_scanner(self, db_session):
        org = fresh_org(db_session, fresh_user(db_session))
        cp.set_residency_rule(db_session, organization_id=org.id,
                              classification="CONFIDENTIAL",
                              allowed_regions=["us-east-1"])
        violations = cp.residency_violations(
            db_session, organization_id=org.id,
            samples=[{"classification": "CONFIDENTIAL", "region":
                      "eu-west-1"},
                     {"classification": "CONFIDENTIAL", "region":
                      "us-east-1"}])
        assert len(violations) == 1
        assert violations[0]["region"] == "eu-west-1"

    def test_region_selection_prefers_healthy_and_capability(self,
                                                             db_session):
        org = fresh_org(db_session, fresh_user(db_session))
        cp.upsert_region(db_session, organization_id=org.id,
                         region_id="eu-1", health_score=0.9,
                         provider_availability={"openai": True})
        cp.upsert_region(db_session, organization_id=org.id,
                         region_id="us-1", health_score=1.0,
                         provider_availability={"openai": True})
        cp.set_residency_rule(db_session, organization_id=org.id,
                              classification="INTERNAL",
                              allowed_regions=["eu-1", "us-1"])
        decision = cp.select_region(db_session, organization_id=org.id,
                                    classification="INTERNAL",
                                    required_capability="openai")
        assert decision["selected"] == "us-1"

    def test_region_selection_blocks_non_resident(self, db_session):
        org = fresh_org(db_session, fresh_user(db_session))
        cp.upsert_region(db_session, organization_id=org.id,
                         region_id="eu-1", health_score=1.0,
                         provider_availability={"openai": True})
        cp.set_residency_rule(db_session, organization_id=org.id,
                              classification="CONFIDENTIAL",
                              allowed_regions=["us-1"])
        decision = cp.select_region(db_session, organization_id=org.id,
                                    classification="CONFIDENTIAL",
                                    required_capability="openai")
        assert decision["selected"] is None

    def test_region_selection_preference_order(self, db_session):
        cp.upsert_region(db_session, organization_id=None, region_id="b",
                         health_score=1.0)
        cp.upsert_region(db_session, organization_id=None, region_id="a",
                         health_score=1.0)
        decision = cp.select_region(db_session, organization_id=None,
                                    classification="PUBLIC",
                                    preferred=["a", "b"])
        assert decision["selected"] == "a"

    def test_failover_requires_registered_target(self, db_session):
        org = fresh_org(db_session, fresh_user(db_session))
        cp.upsert_region(db_session, organization_id=org.id,
                         region_id="a")
        with pytest.raises(ValueError):
            cp.initiate_failover(db_session, organization_id=org.id,
                                 region_from="a", region_to="ghost")

    def test_failover_into_outage_blocked(self, db_session):
        org = fresh_org(db_session, fresh_user(db_session))
        cp.upsert_region(db_session, organization_id=org.id, region_id="a")
        cp.upsert_region(db_session, organization_id=org.id, region_id="b",
                         status="OUTAGE")
        with pytest.raises(ValueError):
            cp.initiate_failover(db_session, organization_id=org.id,
                                 region_from="a", region_to="b")

    def test_failover_complete_and_revert(self, db_session):
        org = fresh_org(db_session, fresh_user(db_session))
        cp.upsert_region(db_session, organization_id=org.id, region_id="a")
        cp.upsert_region(db_session, organization_id=org.id, region_id="b")
        result = cp.initiate_failover(db_session, organization_id=org.id,
                                      region_from="a", region_to="b",
                                      reason="drill")
        cp.complete_failover(db_session, result["failover_id"])
        rows = cp.list_regions(db_session, organization_id=org.id)
        source = next(r for r in rows if r.region_id == "a")
        assert source.status == "FAILED_OVER"
        assert source.failover_to == "b"
        cp.revert_failover(db_session, result["failover_id"])
        rows = cp.list_regions(db_session, organization_id=org.id)
        source = next(r for r in rows if r.region_id == "a")
        assert source.status == "HEALTHY"
        assert source.failover_to is None

    def test_failover_complete_idempotent(self, db_session):
        org = fresh_org(db_session, fresh_user(db_session))
        cp.upsert_region(db_session, organization_id=org.id, region_id="a")
        cp.upsert_region(db_session, organization_id=org.id, region_id="b")
        result = cp.initiate_failover(db_session, organization_id=org.id,
                                      region_from="a", region_to="b")
        cp.complete_failover(db_session, result["failover_id"])
        again = cp.complete_failover(db_session, result["failover_id"])
        assert again["status"] == "ALREADY_COMPLETED"

    def test_global_health_shape(self, db_session):
        report = cp.global_health(db_session)
        assert "workers" in report
        assert "scheduler_leader" in report
        assert "regions" in report


# ===========================================================================
# Worker quarantine / health / placement
# ===========================================================================

class TestWorkerOps:
    def test_quarantine_worker_idempotent(self, db_session):
        wo.quarantine_worker(db_session, worker_id="w1", reason="fail")
        wo.quarantine_worker(db_session, worker_id="w1", reason="fail2",
                             failure_count=5)
        rows = wo.list_quarantines(db_session)
        assert len(rows) == 1
        assert rows[0].failure_count == 5

    def test_quarantine_status_blocks_work(self, db_session):
        wo.quarantine_worker(db_session, worker_id="w1")
        assert wo.quarantine_status(db_session, "w1")["quarantined"] is True

    def test_operator_override_releases(self, db_session):
        wo.quarantine_worker(db_session, worker_id="w1")
        wo.operator_override(db_session, worker_id="w1", operator_user_id=1)
        assert wo.quarantine_status(db_session, "w1")["quarantined"] is False
        assert wo.quarantine_status(db_session, "w1")[
            "status"] == "OPERATOR_OVERRIDE"

    def test_auto_recover_after_due(self, db_session):
        wo.quarantine_worker(db_session, worker_id="w1",
                             auto_recover_after_minutes=0)
        result = wo.auto_recover_quarantines(db_session)
        assert result["recovered"] == 1
        assert wo.quarantine_status(db_session, "w1")["quarantined"] is False

    def test_auto_recover_skips_future(self, db_session):
        wo.quarantine_worker(db_session, worker_id="w1",
                             auto_recover_after_minutes=60)
        result = wo.auto_recover_quarantines(db_session)
        assert result["recovered"] == 0

    def test_health_score_zero_without_heartbeat(self, db_session):
        assert wo.worker_health_score(db_session, "ghost") == 0.0

    def test_health_score_live_worker(self, db_session):
        hb(db_session, "w1")
        assert wo.worker_health_score(db_session, "w1") == 1.0

    def test_load_score_dead_worker(self, db_session):
        assert wo.worker_load_score(db_session, "gone")["live"] is False

    def test_load_score_live_worker(self, db_session):
        hb(db_session, "w1", active=0)
        score = wo.worker_load_score(db_session, "w1")
        assert score["live"] is True
        assert score["score"] == 0.0

    def test_load_score_utilization(self, db_session):
        hb(db_session, "w1", active=4)
        score = wo.worker_load_score(db_session, "w1")
        assert 0.0 < score["score"] <= 1.0

    def test_capacity_model_shape(self, db_session):
        model = wo.capacity_model(db_session)
        assert "global_concurrency" in model
        assert model["global_concurrency"] >= 1
        assert "live_workers" in model

    def test_placement_defers_when_no_workers(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        decision = wo.placement_decision(db_session, workspace_id=ws.id,
                                         queue_name="AI_EXECUTIONS")
        assert decision["decision"] == "DEFER"

    def test_placement_skips_quarantined(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        hb(db_session, "w1")
        wo.quarantine_worker(db_session, worker_id="w1")
        decision = wo.placement_decision(db_session, workspace_id=ws.id,
                                         queue_name="AI_EXECUTIONS")
        assert decision["decision"] == "DEFER"

    def test_placement_assigns_least_loaded(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        hb(db_session, "busy", active=3)
        hb(db_session, "idle", active=0)
        decision = wo.placement_decision(db_session, workspace_id=ws.id,
                                         queue_name="AI_EXECUTIONS")
        assert decision["decision"] == "ASSIGN"
        assert decision["worker_id"] == "idle"

    def test_placement_respects_tenant_limit(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for _ in range(50):
            job(db_session, ws, status="RUNNING", claimed_by="w1")
        hb(db_session, "w1", active=0)
        decision = wo.placement_decision(db_session, workspace_id=ws.id,
                                         queue_name="AI_EXECUTIONS")
        assert decision["decision"] == "DEFER"
        assert "tenant" in decision["reason"]

    def test_migrate_job_requeues_and_audits(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = job(db_session, ws, status="CLAIMED", claimed_by="w1")
        lease = JobLease(job_id=row.id, worker_id="w1",
                         lease_token="tok",
                         expires_at=datetime.now(timezone.utc)
                         + timedelta(minutes=5))
        db_session.add(lease)
        db_session.commit()
        result = wo.migrate_job(db_session, job_id=row.id, reason="unhealthy")
        db_session.refresh(row)
        assert result["migrated"] is True
        assert row.status == "QUEUED"
        assert row.claimed_by is None
        audit = db_session.query(JobMigrationRecord).all()
        assert len(audit) == 1
        assert audit[0].from_worker == "w1"

    def test_migrate_job_to_target_worker(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        row = job(db_session, ws, status="CLAIMED", claimed_by="w1")
        result = wo.migrate_job(db_session, job_id=row.id,
                                reason="rebalance", target_worker="w2")
        db_session.refresh(row)
        assert row.claimed_by == "w2"
        assert result["to_worker"] == "w2"

    def test_backpressure_accepts_with_capacity(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        hb(db_session, "w1")
        decision = wo.backpressure_decision(db_session, workspace_id=ws.id,
                                            queue_name="AI_EXECUTIONS")
        assert decision["decision"] == "ACCEPT"

    def test_backpressure_defers_without_workers(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        decision = wo.backpressure_decision(db_session, workspace_id=ws.id,
                                            queue_name="AI_EXECUTIONS")
        assert decision["decision"] == "DEFER"

    def test_backpressure_rejects_exhausted_budget(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        hb(db_session, "w1")
        decision = wo.backpressure_decision(db_session, workspace_id=ws.id,
                                            queue_name="AI_EXECUTIONS",
                                            budget_remaining=-1.0)
        assert decision["decision"] == "REJECT"
        assert decision["code"] == "BUDGET_EXHAUSTED"

    def test_autoscale_signals(self, db_session):
        signals = wo.autoscale_signals(db_session)
        assert set(signals) >= {"queue_depth", "running_jobs",
                                "live_workers", "throughput_5m",
                                "failure_rate_5m"}

    def test_mark_inactive_via_deregister(self, db_session):
        hb(db_session, "w1")
        from app.services.runtime import mark_worker_inactive
        mark_worker_inactive(db_session, "w1")
        db_session.flush()
        row = db_session.query(WorkerHeartbeat).filter(
            WorkerHeartbeat.worker_id == "w1").first()
        assert row.status == "STOPPED"
        assert row.stopped_at is not None


# ===========================================================================
# Scheduler 2.0
# ===========================================================================

class TestScheduler:
    def test_acquire_leadership_first(self, db_session):
        result = wo.acquire_leadership(db_session, leader_id="sched-1")
        assert result["status"] == "LEADER"

    def test_second_scheduler_stays_standby(self, db_session):
        wo.acquire_leadership(db_session, leader_id="sched-1")
        result = wo.acquire_leadership(db_session, leader_id="sched-2")
        assert result["status"] == "STANDBY"
        assert result["leader"] == "sched-1"

    def test_takeover_after_expiry(self, db_session):
        wo.acquire_leadership(db_session, leader_id="sched-1")
        row = db_session.query(SchedulerLeader).filter(
            SchedulerLeader.leader_id == "sched-1").first()
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        db_session.commit()
        result = wo.acquire_leadership(db_session, leader_id="sched-2")
        assert result["status"] == "LEADER"
        assert result["takeover"] is True
        # old leader is now stale
        old = db_session.query(SchedulerLeader).filter(
            SchedulerLeader.leader_id == "sched-1").first()
        assert old.status == "STALE"

    def test_renew_leadership(self, db_session):
        wo.acquire_leadership(db_session, leader_id="sched-1",
                              lease_seconds=10)
        assert wo.renew_leadership(db_session, leader_id="sched-1",
                                   lease_seconds=10) is True
        assert wo.renew_leadership(db_session, leader_id="other") is False

    def test_release_leadership(self, db_session):
        wo.acquire_leadership(db_session, leader_id="sched-1")
        result = wo.release_leadership(db_session, leader_id="sched-1")
        assert result["status"] == "RELEASED"
        assert wo.release_leadership(db_session,
                                     leader_id="sched-1")["status"] == \
            "NOT_LEADER"

    def test_recover_stale_leadership(self, db_session):
        wo.acquire_leadership(db_session, leader_id="sched-1")
        row = db_session.query(SchedulerLeader).filter(
            SchedulerLeader.leader_id == "sched-1").first()
        row.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        db_session.commit()
        result = wo.recover_stale_leadership(db_session)
        assert result["stale_marked"] == 1

    def test_scheduler_dedupe_key_deterministic(self):
        assert wo.scheduler_dedupe_key("retention", "ws:1", "2026-09-04") == \
            wo.scheduler_dedupe_key("retention", "ws:1", "2026-09-04")
        assert wo.scheduler_dedupe_key("a") != wo.scheduler_dedupe_key("b")

    def test_ensure_scheduled_job_deduplicates(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        key = wo.scheduler_dedupe_key("retention", ws.id)
        first = wo.ensure_scheduled_job(db_session, workspace_id=ws.id,
                                        organization_id=None,
                                        queue_name="RETENTION",
                                        job_type="retention.run",
                                        dedupe_key=key)
        second = wo.ensure_scheduled_job(db_session, workspace_id=ws.id,
                                         organization_id=None,
                                         queue_name="RETENTION",
                                         job_type="retention.run",
                                         dedupe_key=key)
        assert first.id == second.id
        jobs = db_session.query(WorkerJob).count()
        assert jobs == 1

    def test_dedupe_key_conflicts_across_types(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        from app.services.worker_platform import DuplicateJobError
        with pytest.raises(DuplicateJobError):
            wo.ensure_scheduled_job(db_session, workspace_id=ws.id,
                                    organization_id=None,
                                    queue_name="X", job_type="a.type",
                                    dedupe_key="k1")
            wo.ensure_scheduled_job(db_session, workspace_id=ws.id,
                                    organization_id=None,
                                    queue_name="X", job_type="b.type",
                                    dedupe_key="k1")


# ===========================================================================
# Broker 2.0
# ===========================================================================

class TestBroker2:
    def test_delivery_guarantees_at_least_once(self):
        guarantees = broker2.delivery_guarantees()
        assert guarantees["semantics"] == "at-least-once"

    def test_handler_idempotent_by_default(self):
        assert broker2.handler_idempotent("documents.embed")["idempotent"]

    def test_non_idempotent_handler_policy(self):
        policy = broker2.handler_idempotent("external.send",
                                            declared=False)
        assert policy["idempotent"] is False
        assert "no automatic replay" in policy["policy"]

    def test_broker_config_default_postgres(self):
        assert broker2.broker_config()["backend"] in ("postgres", "redis")

    def test_redis_optional_never_crashes(self):
        # no redis server in dev/test → available is False, not an error
        assert broker2.redis_available() in (True, False)

    def test_outage_policy_no_silent_loss(self):
        policy = broker2.outage_policy()
        assert "never loses queued jobs" in policy["durability"]

    def test_broker_recovery_check_counts(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        old = datetime.now(timezone.utc) - timedelta(minutes=10)
        row = WorkerJob(queue_name="Q", job_type="t", workspace_id=ws.id,
                        status="CLAIMED", claimed_by="dead",
                        lease_expires_at=old, created_at=old)
        db_session.add(row)
        db_session.commit()
        check = broker2.broker_recovery_check(db_session)
        assert check["recoverable"] >= 1
        assert check["expired_job_claims"] >= 1

    def test_pool_config_bounded(self):
        config = broker2.pool_config()
        assert config["pool_pre_ping"] is True
        assert config["pool_size"] >= 1

    def test_redis_backend_interface_conformance(self):
        backend = broker.PostgresBroker()
        contract = __import__("app.services.broker_ops",
                              fromlist=["BrokerContract"]).BrokerContract
        assert contract.verify(backend) == []
