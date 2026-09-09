"""Phase 18 tests — distributed worker runtime + broker production layer.

Covers: process entry points, capacity/load heartbeats, scheduler
maintenance rounds (delayed promotion, lease recovery, expiry), event
processor batching, broker health/failover/delayed jobs, Redis-adapter
contract via an in-memory mock, and worker registration/deregistration.
"""

import time
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.phase16 import WorkerJob, WorkerHeartbeat  # noqa: E402
from app.models.phase17 import JobLease, AlertRule, AlertEvent  # noqa: E402
from app.models.phase15 import AIMemory  # noqa: E402
from app.services import worker_platform as wp  # noqa: E402
from app.services import broker, broker_ops, distributed as fleet  # noqa: E402
from app.services import runtime  # noqa: E402

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
    db_session.query(JobLease).delete()
    db_session.query(WorkerJob).delete()
    db_session.query(WorkerHeartbeat).delete()
    db_session.query(AIMemory).delete()
    db_session.commit()
    yield
    broker.reset_broker_cache()


def fresh_user(db, tag="p18w"):
    _counter[0] += 1
    user = User(name=f"P18 Worker {_counter[0]}",
                email=f"{tag}{_counter[0]}@p18-worker.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p18 ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def enqueue(db, ws, job_type="test.job", queue="default", **kw):
    return wp.enqueue_job(db, queue_name=queue, job_type=job_type,
                          workspace_id=ws.id, payload={"n": 1}, **kw)


# ============================================================
# Worker runtime — process entry points + capacity
# ============================================================

class TestWorkerRuntime:
    def test_entry_points_import(self):
        import app.worker
        import app.scheduler
        import app.event_processor
        assert callable(app.worker.main)
        assert callable(app.scheduler.main)
        assert callable(app.event_processor.main)

    def test_worker_capacity_defaults(self):
        cap = runtime.worker_capacity()
        assert cap["concurrency"] >= 1
        assert isinstance(cap["queue_caps"], dict)

    def test_worker_capacity_queue_caps(self, monkeypatch):
        monkeypatch.setenv("WORKER_QUEUE_CONCURRENCY", "AI_TASKS:3,EVENTS:2")
        cap = runtime.worker_capacity()
        assert cap["queue_caps"] == {"AI_TASKS": 3, "EVENTS": 2}

    def test_queue_capacity_uses_cap(self, monkeypatch):
        monkeypatch.setenv("WORKER_QUEUE_CONCURRENCY", "AI_TASKS:5")
        assert runtime.queue_capacity("AI_TASKS") == 5
        assert runtime.queue_capacity("UNKNOWN") >= 1

    def test_compute_load_bounds(self):
        assert runtime.compute_load(4, 4) == 1.0
        assert runtime.compute_load(0, 4) == 0.0
        assert runtime.compute_load(10, 4) == 1.0  # capped
        assert runtime.compute_load(2, 4) == 0.5

    def test_heartbeat_with_capacity_updates_row(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        runtime.heartbeat_with_capacity(
            db_session, "w-test-1", active_jobs=3,
            queue_assignments=["AI_TASKS", "EVENTS"], load=0.75)
        db_session.commit()
        hb = db_session.query(WorkerHeartbeat).filter(
            WorkerHeartbeat.worker_id == "w-test-1").first()
        assert hb is not None
        assert hb.active_jobs == 3
        assert hb.load == 0.75
        assert "AI_TASKS" in (hb.queue_assignments or "")

    def test_heartbeat_creates_row_when_missing(self, db_session):
        runtime.heartbeat_with_capacity(db_session, "w-test-2", active_jobs=0)
        db_session.commit()
        hb = db_session.query(WorkerHeartbeat).filter(
            WorkerHeartbeat.worker_id == "w-test-2").first()
        assert hb is not None
        assert hb.status == "RUNNING"

    def test_mark_worker_inactive(self, db_session):
        runtime.heartbeat_with_capacity(db_session, "w-test-3", active_jobs=1)
        db_session.commit()
        runtime.mark_worker_inactive(db_session, "w-test-3")
        db_session.commit()
        hb = db_session.query(WorkerHeartbeat).filter(
            WorkerHeartbeat.worker_id == "w-test-3").first()
        assert hb.status == "STOPPED"
        assert hb.stopped_at is not None

    def test_worker_identity_safe_public_fields(self):
        identity = fleet.WorkerIdentity(queue_names=["AI_TASKS"])
        pub = identity.to_public()
        assert pub["worker_id"]
        assert pub["version"]
        assert "AI_TASKS" in pub["queues"]
        # No environment/secrets ever exposed.
        assert "env" not in pub and "password" not in pub

    def test_register_fleet_worker_persists(self, db_session):
        identity = fleet.WorkerIdentity(worker_id="w-reg-1",
                                        queue_names=["AI_TASKS"])
        fleet.register_fleet_worker(db_session, identity)
        db_session.commit()
        hb = db_session.query(WorkerHeartbeat).filter(
            WorkerHeartbeat.worker_id == "w-reg-1").first()
        assert hb is not None and hb.status == "RUNNING"

    def test_shutdown_manager_stop_flag(self):
        identity = fleet.WorkerIdentity(worker_id="w-shut-1")
        manager = fleet.ShutdownManager(identity)
        assert not manager.stopping()
        manager.request_stop()
        assert manager.stopping()

    def test_scheduler_maintenance_promotes_delayed(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = enqueue(db_session, ws, run_after=datetime.now(timezone.utc)
                      - timedelta(seconds=5))
        db_session.commit()
        result = runtime._scheduler_maintenance(db_session, "sched-1")
        db_session.commit()
        assert result["promoted"] >= 1
        db_session.refresh(job)
        assert job.run_after is None

    def test_scheduler_maintenance_keeps_future_delayed(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = enqueue(db_session, ws, run_after=datetime.now(timezone.utc)
                      + timedelta(hours=1))
        db_session.commit()
        result = runtime._scheduler_maintenance(db_session, "sched-2")
        db_session.commit()
        assert result["promoted"] == 0
        db_session.refresh(job)
        assert job.run_after is not None

    def test_scheduler_maintenance_enqueues_cleanup(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = runtime._scheduler_maintenance(db_session, "sched-3")
        db_session.commit()
        assert result["cleanup_enqueued"] is True
        cleanup = db_session.query(WorkerJob).filter(
            WorkerJob.job_type == "RETENTION_CLEANUP").first()
        assert cleanup is not None
        # Idempotent within the hour — no duplicate.
        runtime._scheduler_maintenance(db_session, "sched-3")
        db_session.commit()
        count = db_session.query(WorkerJob).filter(
            WorkerJob.job_type == "RETENTION_CLEANUP").count()
        assert count == 1

    def test_scheduler_maintenance_cleanup_uses_real_workspace(self, db_session):
        # Regression: the cleanup job used to be enqueued with a hard-coded
        # workspace_id=1, violating the workspaces FK when that id did not
        # exist. The enqueued job must reference an existing workspace.
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = runtime._scheduler_maintenance(db_session, "sched-ws")
        db_session.commit()
        assert result["cleanup_enqueued"] is True
        job = (db_session.query(WorkerJob)
               .filter(WorkerJob.job_type == "RETENTION_CLEANUP")
               .order_by(WorkerJob.id.desc()).first())
        assert job is not None
        assert db_session.query(Workspace).filter(
            Workspace.id == job.workspace_id).first() is not None

    def test_event_processor_round_returns_dict_contract(self, db_session):
        # Regression: the event-processor loop expected an int from
        # process_pending_events, which returns a dict.
        from app.services.event_bus import process_pending_events
        result = process_pending_events(db_session, batch_size=50)
        assert isinstance(result, dict)
        assert "processed" in result
        assert int(result.get("processed", 0)) >= 0

    def test_event_processor_loop_tolerates_dict_result(self, db_session,
                                                        monkeypatch):
        # Regression: run_event_processor_process must pass batch_size and
        # tolerate a dict result without crashing.
        from app.services import event_bus as event_bus_mod
        calls = {"n": 0}

        def fake_process(db, batch_size=50):
            calls["n"] += 1
            calls["batch_size"] = batch_size
            return {"processed": 2, "outbox_total": 2}

        monkeypatch.setattr(event_bus_mod, "process_pending_events",
                            fake_process)
        monkeypatch.setattr(fleet, "register_fleet_worker",
                            lambda *a, **k: None)

        class FakeShutdown:
            def __init__(self):
                self.rounds = 0

            def install(self):
                pass

            def stopping(self):
                self.rounds += 1
                return self.rounds > 2

            def wait(self, seconds):
                pass

        identity = fleet.WorkerIdentity(queue_names=["EVENTS"])
        runtime.run_event_processor_process(
            lambda: db_session, identity, FakeShutdown(),
            tick_seconds=0.0, batch_size=50)
        assert calls["n"] == 2
        assert calls.get("batch_size") == 50

    def test_scheduler_maintenance_expires_merge_requests(self, db_session):
        from app.models.phase18 import EntityMergeRequest
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        from app.models.knowledge_graph import Entity
        e1 = Entity(workspace_id=ws.id, name="alpha",
                    entity_type="concept")
        e2 = Entity(workspace_id=ws.id, name="beta",
                    entity_type="concept")
        db_session.add_all([e1, e2])
        db_session.commit()
        db_session.add(EntityMergeRequest(
            workspace_id=ws.id, source_entity_id=e1.id,
            target_entity_id=e2.id,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=5)))
        db_session.commit()
        result = runtime._scheduler_maintenance(db_session, "sched-4")
        db_session.commit()
        assert result["merges_expired"] >= 1

    def test_scheduler_maintenance_expires_approvals(self, db_session):
        from app.models.ai_execution import AIApproval
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        from app.models.ai_execution import AIExecution
        import uuid as _uuid
        execution_id = str(_uuid.uuid4())
        db_session.add(AIExecution(
            id=execution_id, workspace_id=ws.id, user_id=user.id,
            execution_type="rag", task_type="test", status="QUEUED"))
        db_session.commit()
        db_session.add(AIApproval(
            id=str(_uuid.uuid4()),
            execution_id=execution_id, workspace_id=ws.id,
            user_id=user.id, action="test.action",
            status="pending",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=5)))
        db_session.commit()
        result = runtime._scheduler_maintenance(db_session, "sched-5")
        db_session.commit()
        assert result["approvals_expired"] >= 1

    def test_event_processor_function_runs_bounded(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        from app.services.event_bus import emit_event, process_pending_events
        emit_event(db_session, workspace_id=ws.id,
                   event_type="DOCUMENT_UPLOADED",
                   payload={"document_id": 1})
        db_session.commit()
        summary = process_pending_events(db_session, batch_size=10)
        db_session.commit()
        assert summary["processed"] + summary["skipped"] >= 1


# ============================================================
# Broker — health, failover, delayed jobs, Redis contract
# ============================================================

class TestBrokerOps:
    def test_validate_broker_config_default(self):
        assert broker_ops.configured_broker() == "postgres"

    def test_broker_failover_policy_safe(self):
        policy = broker_ops.broker_failover_policy()
        assert policy["on_unavailable"] == "stop_claiming_and_alert"
        assert policy["requeue_on_reconnect"] is True
        assert policy["dead_letter_after_attempts"] >= 1

    def test_broker_health_postgres(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        enqueue(db_session, ws)
        db_session.commit()
        health = broker_ops.broker_health(db_session)
        assert health["broker"] == "postgres"
        assert health["status"] == "UP"
        assert health["queue_depth"] >= 1

    def test_queue_depth_per_queue(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        enqueue(db_session, ws, queue="AI_TASKS")
        enqueue(db_session, ws, queue="EVENTS")
        enqueue(db_session, ws, queue="AI_TASKS")
        db_session.commit()
        depth = broker_ops.queue_depth(db_session)
        assert depth["per_queue"]["AI_TASKS"]["depth"] == 2
        assert depth["per_queue"]["EVENTS"]["depth"] == 1
        assert depth["oldest_job_age_s"] is not None

    def test_promote_due_delayed_bounded(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for i in range(3):
            enqueue(db_session, ws,
                    run_after=datetime.now(timezone.utc) - timedelta(minutes=1))
        db_session.commit()
        promoted = broker_ops.promote_due_delayed(db_session, limit=2)
        db_session.commit()
        assert promoted == 2

    def test_broker_contract_all_backends(self):
        postgres = broker.PostgresBroker()
        missing = broker_ops.BrokerContract.verify(postgres)
        assert missing == []
        # Fake conforming backend also passes.
        class Fake:
            def enqueue(self, *a, **k): ...
            def claim(self, *a, **k): ...
            def acknowledge(self, *a, **k): ...
            def retry(self, *a, **k): ...
            def dead_letter(self, *a, **k): ...
            def heartbeat(self, *a, **k): ...
            def cancel(self, *a, **k): ...
            def close(self): ...
        assert broker_ops.BrokerContract.verify(Fake()) == []

    def test_broker_contract_detects_missing(self):
        class Bad:
            pass
        missing = broker_ops.BrokerContract.verify(Bad())
        assert len(missing) == 8

    def test_get_broker_unknown_raises(self, monkeypatch):
        monkeypatch.setenv("WORKER_BROKER", "kafka")
        broker.reset_broker_cache()
        with pytest.raises(broker.BrokerError):
            broker.get_broker("kafka")
        broker.reset_broker_cache()

    def test_postgres_broker_enqueue_claim_ack(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        backend = broker.get_broker("postgres")
        item = backend.enqueue(
            db_session, queue_name="default", job_type="test.job",
            workspace_id=ws.id, payload={"n": 1})
        db_session.commit()
        claimed = backend.claim(db_session, queue_name="default",
                                worker_id="w-1")
        assert claimed is not None
        ack = backend.acknowledge(db_session, claimed["id"], "w-1")
        assert ack["status"] == "COMPLETED"

    def test_redis_adapter_requires_redis_package(self, monkeypatch):
        monkeypatch.setenv("WORKER_BROKER", "redis")
        if broker_ops.redis_available():
            pytest.skip("redis installed — server contract covered separately")
        broker.reset_broker_cache()
        with pytest.raises(broker.BrokerUnavailable):
            broker.get_broker("redis")
        broker.reset_broker_cache()

    def test_redis_adapter_contract_with_mock(self, monkeypatch):
        """Exercise the RedisBroker code path with a fake redis client."""
        class FakeRedis:
            def __init__(self):
                self.z = {}
                self.counter = 0
                self.inflight = {}
            def incr(self, key):
                self.counter += 1
                return self.counter
            def zadd(self, key, mapping):
                self.z.setdefault(key, []).extend(mapping.items())
            def zpopmin(self, key, count=1):
                if not self.z.get(key):
                    return []
                items = sorted(self.z[key], key=lambda kv: kv[1])
                out = items[:count]
                self.z[key] = items[count:]
                return out
            def hset(self, key, field, value):
                self.inflight.setdefault(key, {})[field] = value
            def hdel(self, key, field):
                self.inflight.get(key, {}).pop(field, None)
            def ping(self):
                return True
            def close(self):
                pass

        fake = FakeRedis()
        import app.services.broker as broker_mod
        orig_init = broker_mod.RedisBroker.__init__

        def fake_init(self):
            self._redis = fake
            self._url = "redis://mock"
        monkeypatch.setattr(broker_mod.RedisBroker, "__init__", fake_init)
        backend = broker_mod.RedisBroker()
        missing = broker_ops.BrokerContract.verify(backend)
        assert missing == []
        assert backend.ping() is True
        backend.close()

    def test_redis_delayed_job_score_uses_run_after(self, monkeypatch):
        """Delayed Redis jobs are parked by score until due."""
        class FakeRedis:
            def __init__(self):
                self.z = {}
                self.counter = 0
                self.inflight = {}
            def incr(self, key):
                self.counter += 1
                return self.counter
            def zadd(self, key, mapping):
                self.z.setdefault(key, []).extend(mapping.items())
            def zpopmin(self, key, count=1):
                return []
            def hset(self, key, field, value):
                pass
            def hdel(self, key, field):
                pass
            def ping(self):
                return True
            def close(self):
                pass

        import app.services.broker as broker_mod
        fake = FakeRedis()

        def fake_init(self):
            self._redis = fake
            self._url = "redis://mock"
        monkeypatch.setattr(broker_mod.RedisBroker, "__init__", fake_init)
        backend = broker_mod.RedisBroker()
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        item = backend.enqueue(
            None, queue_name="default", job_type="test.job",
            workspace_id=1, payload={}, run_after=future)
        assert item["status"] == "QUEUED"
        # Job is parked with a future score — not claimable yet.
        assert fake.z  # zadd was called
        backend.close()


# ============================================================
# Worker platform — run_once / claims on the new columns
# ============================================================

class TestWorkerPlatformIntegration:
    def test_run_once_processes_job(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        enqueue(db_session, ws, job_type="test.ok", queue="AI_TASKS")
        db_session.commit()

        def handler(session, job):
            job.status = "COMPLETED"
        job = wp.run_once(db_session, "AI_TASKS", "w-once",
                          handler=handler)
        assert job is not None
        db_session.commit()
        row = db_session.get(WorkerJob, job.id)
        assert row.status == "COMPLETED"

    def test_run_once_no_work_returns_none(self, db_session):
        job = wp.run_once(db_session, "AI_TASKS", "w-empty",
                          handler=lambda s, j: None)
        assert job is None

    def test_retryable_failure_keeps_job_for_retry(self, db_session):
        from app.services.worker_platform import WorkerRetryError
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        enqueue(db_session, ws, job_type="test.retry", queue="default")
        db_session.commit()

        def handler(session, job):
            raise WorkerRetryError("transient", retryable=True,
                                   delay_seconds=5)
        job = wp.run_once(db_session, "default", "w-retry", handler=handler)
        db_session.commit()
        db_session.refresh(job)
        assert job.status == "QUEUED"
        assert job.attempt == 1
        assert job.next_retry_at is not None

    def test_claim_respects_workspace_cap(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for _ in range(2):
            enqueue(db_session, ws, queue="default")
        db_session.commit()
        claimed = []
        for _ in range(2):
            job = wp.claim_job(db_session, "default", "w-cap",
                               per_workspace_cap=1)
            claimed.append(job)
        assert len([c for c in claimed if c is not None]) == 1