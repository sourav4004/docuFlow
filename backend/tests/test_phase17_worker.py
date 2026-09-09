"""Phase 17 tests — distributed worker platform.

Broker abstraction (postgres backend), explicit job leases, worker identity,
expired-lease recovery, weighted fairness, dead-letter administration,
autoscaling signals, graceful-shutdown semantics, and vector backfill.
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
from app.models.phase16 import WorkerJob, WorkerHeartbeat  # noqa: E402
from app.models.phase17 import (  # noqa: E402
    JobLease, VectorBackfillRun,
)
from app.services import worker_platform as wp  # noqa: E402
from app.services import broker, distributed as fleet  # noqa: E402

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
    db_session.query(VectorBackfillRun).delete()
    db_session.commit()
    yield
    broker.reset_broker_cache()


def fresh_user(db, tag="p17wu"):
    _counter[0] += 1
    user = User(name=f"P17 Worker {_counter[0]}",
                email=f"{tag}{_counter[0]}@p17-worker.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p17 ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def enqueue(db, ws, job_type="test.job", queue="default", **kw):
    return wp.enqueue_job(db, queue_name=queue, job_type=job_type,
                          workspace_id=ws.id, payload={"n": 1}, **kw)


# ============================================================
# Broker abstraction
# ============================================================

class TestBroker:
    def test_default_backend_is_postgres(self):
        b = broker.get_broker()
        assert b.name == "postgres"

    def test_unknown_backend_rejected(self):
        with pytest.raises(broker.BrokerError):
            broker.get_broker("kafka")

    def test_redis_without_server_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        try:
            broker.get_broker("redis")
            raise AssertionError("expected BrokerUnavailable")
        except broker.BrokerUnavailable:
            pass  # redis client/server unavailable in the test env

    def test_enqueue_and_claim_roundtrip(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        b = broker.get_broker()
        item = b.enqueue(db_session, queue_name="default",
                         job_type="test.job", workspace_id=ws.id,
                         payload={"x": 1})
        assert item["status"] == "QUEUED"
        claimed = b.claim(db_session, queue_name="default",
                          worker_id="w1")
        assert claimed is not None
        assert claimed["id"] == item["id"]
        assert claimed["status"] == "CLAIMED"

    def test_broker_acknowledge(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        b = broker.get_broker()
        item = b.enqueue(db_session, queue_name="default",
                         job_type="test.job", workspace_id=ws.id,
                         payload={})
        claimed = b.claim(db_session, queue_name="default", worker_id="w1")
        done = b.acknowledge(db_session, claimed["id"], "w1")
        assert done["status"] == "COMPLETED"

    def test_broker_retry_dead_letter(self, db_session):
        from datetime import timedelta as _td
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        b = broker.get_broker()
        item = b.enqueue(db_session, queue_name="default",
                         job_type="test.job", workspace_id=ws.id,
                         payload={})
        claimed = b.claim(db_session, queue_name="default", worker_id="w1")
        assert claimed["id"] == item["id"]
        b.retry(db_session, claimed["id"], "w1", "boom")
        job = db_session.query(WorkerJob).filter(
            WorkerJob.id == item["id"]).first()
        assert job.status == "QUEUED"
        assert job.next_retry_at is not None
        # exhaust attempts → dead letter (advance past the backoff each time)
        for _ in range(job.max_attempts + 1):
            job = db_session.query(WorkerJob).filter(
                WorkerJob.id == item["id"]).first()
            if job.status != "QUEUED":
                break
            job.next_retry_at = datetime.now(timezone.utc) - _td(seconds=1)
            db_session.commit()
            c = b.claim(db_session, queue_name="default", worker_id="w2")
            if c is None:
                break
            b.retry(db_session, c["id"], "w2", "boom")
        job = db_session.query(WorkerJob).filter(
            WorkerJob.id == item["id"]).first()
        assert job.status == "DEAD_LETTERED"

    def test_dedupe_same_key_conflict(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        b = broker.get_broker()
        b.enqueue(db_session, queue_name="q", job_type="t",
                  workspace_id=ws.id, payload={}, dedupe_key="k1")
        b.enqueue(db_session, queue_name="q", job_type="t",
                  workspace_id=ws.id, payload={}, dedupe_key="k1")
        jobs = db_session.query(WorkerJob).filter(
            WorkerJob.dedupe_key == "k1").all()
        assert len(jobs) == 1


# ============================================================
# Leases
# ============================================================

class TestLeases:
    def test_claim_creates_lease(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = enqueue(db_session, ws)
        claimed = fleet.claim_weighted(db_session, "default", "w1")
        assert claimed is not None
        lease = db_session.query(JobLease).filter(
            JobLease.job_id == claimed.id).first()
        assert lease is not None
        assert lease.worker_id == "w1"
        assert claimed.lease_token == lease.lease_token
        assert claimed.lease_expires_at is not None

    def test_lease_extension_ownership(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = enqueue(db_session, ws)
        fleet.claim_weighted(db_session, "default", "w1")
        job = db_session.query(WorkerJob).filter(
            WorkerJob.id == job.id).first()
        assert fleet.extend_lease(db_session, job, "w1") is True
        assert fleet.extend_lease(db_session, job, "intruder") is False

    def test_recover_expired_lease_requeues(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = enqueue(db_session, ws)
        identity = fleet.WorkerIdentity(worker_id="w1")
        fleet.register_fleet_worker(db_session, identity)
        fleet.claim_weighted(db_session, "default", "w1")
        now = datetime.now(timezone.utc)
        # freeze the lease as expired and mark the worker dead
        lease = db_session.query(JobLease).filter(
            JobLease.job_id == job.id).first()
        lease.expires_at = now - timedelta(seconds=5)
        hb = db_session.query(WorkerHeartbeat).filter(
            WorkerHeartbeat.worker_id == "w1").first()
        hb.status = "DEAD"
        db_session.commit()
        result = fleet.recover_expired_leases(db_session, now=now)
        assert result["recovered"] == 1
        job = db_session.query(WorkerJob).filter(
            WorkerJob.id == job.id).first()
        assert job.status == "QUEUED"
        assert job.claimed_by is None
        assert db_session.query(JobLease).filter(
            JobLease.job_id == job.id).first() is None

    def test_alive_worker_lease_not_recovered(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = enqueue(db_session, ws)
        fleet.claim_weighted(db_session, "default", "w1")
        now = datetime.now(timezone.utc)
        lease = db_session.query(JobLease).filter(
            JobLease.job_id == job.id).first()
        lease.expires_at = now - timedelta(seconds=1)
        db_session.commit()
        result = fleet.recover_expired_leases(db_session, now=now)
        assert result["recovered"] == 0

    def test_terminal_job_lease_cleaned(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = enqueue(db_session, ws)
        fleet.claim_weighted(db_session, "default", "w1")
        wp.complete_job(db_session, job)
        db_session.commit()
        now = datetime.now(timezone.utc)
        lease = db_session.query(JobLease).filter(
            JobLease.job_id == job.id).first()
        lease.expires_at = now - timedelta(seconds=5)
        db_session.commit()
        result = fleet.recover_expired_leases(db_session, now=now)
        assert result["recovered"] == 0
        assert result["skipped"] >= 1
        assert db_session.query(JobLease).filter(
            JobLease.job_id == job.id).first() is None


# ============================================================
# Weighted fairness
# ============================================================

class TestFairness:
    def _three_tenants(self, db_session):
        users = [fresh_user(db_session, f"p17f{i}") for i in range(3)]
        return [fresh_workspace(db_session, u) for u in users]

    def test_critical_preempts_normal(self, db_session):
        ws_a, ws_b, _ = self._three_tenants(db_session)
        enqueue(db_session, ws_a, priority="NORMAL")
        enqueue(db_session, ws_b, priority="CRITICAL")
        first = fleet.claim_weighted(db_session, "default", "w1")
        assert first.workspace_id == ws_b.id

    def test_tenant_round_robin_after_cap(self, db_session):
        ws_a, ws_b, ws_c = self._three_tenants(db_session)
        for _ in range(4):
            enqueue(db_session, ws_a)
        for _ in range(2):
            enqueue(db_session, ws_b)
        for _ in range(2):
            enqueue(db_session, ws_c)
        claimed_ws = []
        for _ in range(5):
            job = fleet.claim_weighted(db_session, "default", "w1",
                                       per_workspace_cap=2)
            if job is None:
                break
            claimed_ws.append(job.workspace_id)
        # tenant A is capped at 2 of the first 5 while B and C get theirs
        assert claimed_ws.count(ws_a.id) <= 2
        assert ws_b.id in claimed_ws and ws_c.id in claimed_ws

    def test_anti_starvation_oldest_eventually_claimed(self, db_session):
        ws_a, ws_b, _ = self._three_tenants(db_session)
        old = enqueue(db_session, ws_a)
        old.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        db_session.commit()
        for _ in range(4):
            enqueue(db_session, ws_b)
        # even with cap=3 for B, A's very old job gets a turn
        seen = set()
        for _ in range(6):
            job = fleet.claim_weighted(db_session, "default", "w1",
                                       per_workspace_cap=3)
            if job is None:
                break
            seen.add(job.workspace_id)
        assert ws_a.id in seen

    def test_skip_workspace(self, db_session):
        ws_a, _, _ = self._three_tenants(db_session)
        enqueue(db_session, ws_a)
        job = fleet.claim_weighted(db_session, "default", "w1",
                                   skip_workspace_ids={ws_a.id})
        assert job is None


# ============================================================
# Identity + heartbeat + autoscaling
# ============================================================

class TestFleet:
    def test_register_and_touch(self, db_session):
        identity = fleet.WorkerIdentity(queue_names=["default"])
        fleet.register_fleet_worker(db_session, identity)
        hb = db_session.query(WorkerHeartbeat).filter(
            WorkerHeartbeat.worker_id == identity.worker_id).first()
        assert hb is not None and hb.status == "RUNNING"
        assert hb.hostname and hb.pid and hb.version
        wp.touch_heartbeat(db_session, identity.worker_id)
        db_session.commit()
        hb2 = db_session.query(WorkerHeartbeat).filter(
            WorkerHeartbeat.worker_id == identity.worker_id).first()
        assert hb2.last_heartbeat >= hb.last_heartbeat

    def test_public_identity_no_secrets(self):
        identity = fleet.WorkerIdentity()
        pub = identity.to_public()
        joined = " ".join(str(v) for v in pub.values())
        assert "KEY" not in joined.upper() or "apikey" not in joined.lower()
        assert "secret" not in joined.lower()

    def test_autoscale_signals(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for i in range(3):
            wp.enqueue_job(db_session, queue_name="default",
                           job_type=f"j{i}", workspace_id=ws.id, payload={})
        signals = fleet.autoscale_signals(db_session)
        assert signals["queue_depth"] >= 3
        assert signals["oldest_job_age_seconds"] is not None
        assert signals["worker_count"] >= 0

    def test_recover_stale_worker_rows(self, db_session):
        identity = fleet.WorkerIdentity()
        fleet.register_fleet_worker(db_session, identity)
        stale = datetime.now(timezone.utc) - timedelta(minutes=10)
        hb = db_session.query(WorkerHeartbeat).filter(
            WorkerHeartbeat.worker_id == identity.worker_id).first()
        hb.last_heartbeat = stale
        db_session.commit()
        result = wp.recover_stale_workers(db_session, stale_seconds=60)
        assert isinstance(result, dict)


# ============================================================
# Dead-letter administration
# ============================================================

class TestDeadLetters:
    def _dead_letter(self, db_session, ws):
        job = enqueue(db_session, ws)
        wp.claim_job(db_session, "default", "w1")
        job = db_session.query(WorkerJob).filter(
            WorkerJob.id == job.id).first()
        wp.fail_job(db_session, job, "boom", retryable=False)
        db_session.commit()
        return job

    def test_list_and_requeue(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        self._dead_letter(db_session, ws)
        listing = fleet.list_dead_letters(db_session,
                                          workspace_id=ws.id)
        assert listing["total"] >= 1
        job_id = listing["items"][0]["id"]
        updated = fleet.requeue_dead_letter(db_session, job_id,
                                            operator_user_id=user.id)
        assert updated.status == "QUEUED"
        assert updated.attempt == 0

    def test_requeue_missing_job_raises(self, db_session):
        with pytest.raises(wp.JobNotFoundError):
            fleet.requeue_dead_letter(db_session, 999999,
                                      operator_user_id=1)

    def test_abandon_marks_cancelled(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        self._dead_letter(db_session, ws)
        listing = fleet.list_dead_letters(db_session, workspace_id=ws.id)
        job_id = listing["items"][0]["id"]
        updated = fleet.abandon_dead_letter(db_session, job_id,
                                            operator_user_id=user.id)
        assert updated.status == "CANCELLED"

    def test_listing_bounded(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        self._dead_letter(db_session, ws)
        listing = fleet.list_dead_letters(db_session, workspace_id=ws.id,
                                          limit=20000)
        assert listing["limit"] == 200


# ============================================================
# Graceful shutdown
# ============================================================

class TestShutdown:
    def test_shutdown_manager_stops(self):
        identity = fleet.WorkerIdentity()
        mgr = fleet.ShutdownManager(identity)
        assert not mgr.stopping()
        mgr.request_stop()
        assert mgr.stopping()

    def test_drain_releases_to_queue_on_failure(self, db_session, monkeypatch):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = enqueue(db_session, ws)
        fleet.claim_weighted(db_session, "default", "w1")
        wp.release_job(db_session, job, reason="worker shutdown")
        db_session.commit()
        job = db_session.query(WorkerJob).filter(
            WorkerJob.id == job.id).first()
        assert job.status == "QUEUED"


# ============================================================
# Vector backfill
# ============================================================

class TestVectorBackfill:
    def _doc(self, db_session, ws, user):
        from app.models.document import Document
        doc = Document(
            workspace_id=ws.id, user_id=user.id,
            original_filename="bf.pdf", mime_type="text/plain",
            file_size=10, status="READY",
            storage_key=f"bf-{uuid.uuid4().hex}")
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)
        return doc

    def _chunks(self, db_session, doc, n=3):
        from app.models.document_chunk import DocumentChunk
        chunks = []
        for i in range(n):
            c = DocumentChunk(document_id=doc.id, chunk_index=i,
                              text=f"chunk {i} of document about policies",
                              char_start=i * 10, char_end=i * 10 + 9)
            db_session.add(c)
            chunks.append(c)
        db_session.commit()
        return chunks

    def test_preview_is_dry(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = self._doc(db_session, ws, user)
        self._chunks(db_session, doc)
        preview = fleet_doc_check = None
        from app.services import vector_backfill as vb
        preview = vb.preview_backfill(db_session, workspace_id=ws.id)
        assert preview["dry_run"] is True
        assert preview["chunks_to_embed"] == 3
        assert isinstance(preview["native_pgvector"], bool)

    def test_backfill_embeds_chunks(self, db_session):
        from app.services import vector_backfill as vb
        from app.models.document_chunk import DocumentChunk
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = self._doc(db_session, ws, user)
        chunks = self._chunks(db_session, doc)
        run = vb.start_backfill(db_session, workspace_id=ws.id,
                                dry_run=False, created_by=user.id)
        assert run.status == "COMPLETED"
        assert run.processed == 3
        for chunk in db_session.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.id).all():
            assert chunk.embedding is not None

    def test_dry_run_run_record(self, db_session):
        from app.services import vector_backfill as vb
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = self._doc(db_session, ws, user)
        self._chunks(db_session, doc)
        run = vb.start_backfill(db_session, workspace_id=ws.id,
                                dry_run=True, created_by=user.id)
        assert run.dry_run is True
        assert run.status == "COMPLETED"
        assert run.processed == 0

    def test_rebuild_regenerates(self, db_session):
        from app.services import vector_backfill as vb
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = self._doc(db_session, ws, user)
        self._chunks(db_session, doc)
        result = vb.rebuild_vectors(db_session, workspace_id=ws.id,
                                    created_by=user.id)
        assert result["chunks_embedded"] == 3
        assert result["failed_documents"] == 0
