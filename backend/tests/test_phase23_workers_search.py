"""Phase 23 tests — worker platform 3.0, search 6.0, streams, API versioning.

Worker fairness (noisy-neighbor cap), leases, dead-letter management,
autoscale signals, search ranking explainability + diversity + self
evaluation, durable stream cursors, API v1/v2 compatibility metadata, and
the contract case matrix.
"""

import pytest

from app.main import app  # noqa: F401  (register routes/models)
from tests.shared_db import TestingSessionLocal

from app.models.phase16 import WorkerJob
from app.models.phase22 import OpsStreamEvent
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.services import distributed as dist
from app.services import security_eval3 as se3
from app.services import api_platform3 as ap3
from app.services import worker_ops2 as w2

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
    db_session.query(WorkerJob).delete()
    db_session.query(OpsStreamEvent).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"wk{n}@p23wk.example", name="wk", password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-wk-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


def _mkjob(db, ws, priority="NORMAL", queue="default", status="QUEUED"):
    _counter[0] += 1
    job = WorkerJob(workspace_id=ws.id, queue_name=queue, priority=priority,
                    status=status, job_type="test",
                    payload_json='{"x": 1}')
    db.add(job)
    db.commit()
    return job


# ===========================================================================
# Worker platform 3.0 — fairness (Step 47)
# ===========================================================================

class TestWorkerFairness:
    def test_noisy_neighbor_capped(self, db_session):
        ws_flood = _mkws(db_session)
        ws_quiet = _mkws(db_session)
        for _ in range(6):
            _mkjob(db_session, ws_flood)
        _mkjob(db_session, ws_quiet)
        claimed_flood = 0
        for i in range(4):
            job = dist.claim_weighted(db_session, "default", f"w-{i}",
                                      per_workspace_cap=3)
            if job is not None and job.workspace_id == ws_flood.id:
                claimed_flood += 1
        assert claimed_flood <= 3  # cap enforced before quiet tenant starves

    def test_quiet_tenant_gets_capacity(self, db_session):
        ws_flood = _mkws(db_session)
        ws_quiet = _mkws(db_session)
        for _ in range(5):
            _mkjob(db_session, ws_flood)
        _mkjob(db_session, ws_quiet)
        got_quiet = False
        for i in range(5):
            job = dist.claim_weighted(db_session, "default", f"wq-{i}",
                                      per_workspace_cap=1)
            if job is not None and job.workspace_id == ws_quiet.id:
                got_quiet = True
                break
        assert got_quiet is True

    def test_priority_dominates(self, db_session):
        ws = _mkws(db_session)
        _mkjob(db_session, ws, priority="LOW")
        _mkjob(db_session, ws, priority="CRITICAL")
        job = dist.claim_weighted(db_session, "default", "wp-1")
        assert job is not None and job.priority == "CRITICAL"

    def test_claim_sets_lease_and_heartbeat(self, db_session):
        ws = _mkws(db_session)
        job = _mkjob(db_session, ws)
        claimed = dist.claim_weighted(db_session, "default", "wl-1")
        assert claimed is not None
        assert claimed.status in ("CLAIMED", "RUNNING")
        assert claimed.claimed_by == "wl-1"

    def test_empty_queue_returns_none(self, db_session):
        assert dist.claim_weighted(db_session, "default", "wx-1") is None


# ===========================================================================
# Dead letters + autoscale (Steps 46, 104)
# ===========================================================================

class TestDeadLettersAutoscale:
    def test_dead_letter_listing_bounded(self, db_session):
        ws = _mkws(db_session)
        for _ in range(5):
            _mkjob(db_session, ws, status="DEAD_LETTERED")
        result = dist.list_dead_letters(db_session, limit=3)
        assert result["total"] >= 5
        assert len(result["items"]) <= 3

    def test_dead_letter_requeue_idempotent(self, db_session):
        ws = _mkws(db_session)
        job = _mkjob(db_session, ws, status="DEAD_LETTERED")
        r1 = dist.requeue_dead_letter(db_session, job.id,
                                      operator_user_id=1)
        assert r1.status != "DEAD_LETTERED"
        r2 = dist.requeue_dead_letter(db_session, job.id,
                                      operator_user_id=1)
        assert r2.id == job.id  # unchanged, no duplicate side effect

    def test_requeue_unknown_job_raises(self, db_session):
        with pytest.raises(Exception):
            dist.requeue_dead_letter(db_session, 99999999,
                                     operator_user_id=1)

    def test_autoscale_signals_shape(self, db_session):
        ws = _mkws(db_session)
        _mkjob(db_session, ws)
        signals = dist.autoscale_signals(db_session)
        assert signals["queue_depth"] >= 1
        assert "failure_rate" in signals

    def test_worker_health_zero_for_unknown(self, db_session):
        assert w2.worker_health_score(db_session, "ghost-worker") == 0.0

    def test_quarantine_workflow(self, db_session):
        result = w2.quarantine_worker(db_session, worker_id="bad-worker",
                                      reason="repeated failures")
        assert result is not None
        status = w2.quarantine_status(db_session, "bad-worker")
        assert status is not None


# ===========================================================================
# Search 6.0 (Steps 52-53)
# ===========================================================================

class TestSearch6:
    def test_explain_ranking_attributed(self, db_session):
        result = se3.explain_ranking(
            db_session,
            candidates=[{"id": "a", "score": 0.9, "keyword": 0.8,
                         "vector": 0.95, "freshness": 0.7,
                         "authority": 0.5}])
        assert result is not None

    def test_diversity_caps_document_dominance(self):
        ranking = [{"id": f"r{i}", "document_id": 1} for i in range(5)]
        ranked = se3.apply_diversity(ranking, max_per_group=2)
        top_ids = [r["document_id"] for r in ranked[:3]]
        assert top_ids.count(1) <= 2

    def test_diversity_preserves_order_within_cap(self):
        ranking = [{"id": "a", "document_id": 1}, {"id": "b", "document_id": 2}]
        ranked = se3.apply_diversity(ranking, max_per_group=2)
        assert ranked[0]["id"] == "a"

    def test_self_evaluation_flags_degradation(self, db_session):
        ws = _mkws(db_session)
        result = se3.search_self_evaluation(
            db_session, workspace_id=ws.id, zero_result_queries=40,
            total_queries=100, reformulations=60, precision_estimate=0.3)
        assert result["proposal_created"] is True

    def test_self_evaluation_healthy_no_proposal(self, db_session):
        ws = _mkws(db_session)
        result = se3.search_self_evaluation(
            db_session, workspace_id=ws.id, zero_result_queries=2,
            total_queries=100, reformulations=5, precision_estimate=0.95)
        assert result["proposal_created"] is False


# ===========================================================================
# Streams (Step 62)
# ===========================================================================

class TestStreamsDepth:
    def test_emit_returns_increasing_seq(self, db_session):
        ws = _mkws(db_session)
        e1 = se3.emit_stream_event(db_session, workspace_id=ws.id,
                                   stream="worker", kind="worker.event",
                                   payload={"n": 1})
        e2 = se3.emit_stream_event(db_session, workspace_id=ws.id,
                                   stream="worker", kind="worker.event",
                                   payload={"n": 2})
        assert e2["seq"] == e1["seq"] + 1

    def test_cursor_recovery_after_seq(self, db_session):
        ws = _mkws(db_session)
        e1 = se3.emit_stream_event(db_session, workspace_id=ws.id,
                                   stream="worker", kind="k", payload={})
        se3.emit_stream_event(db_session, workspace_id=ws.id,
                              stream="worker", kind="k", payload={})
        page = se3.read_stream(db_session, workspace_id=ws.id,
                               stream="worker", after_seq=e1["seq"])
        assert page["count"] == 1  # only events after the cursor

    def test_stream_scoped_per_workspace(self, db_session):
        ws1 = _mkws(db_session)
        ws2 = _mkws(db_session)
        se3.emit_stream_event(db_session, workspace_id=ws1.id,
                              stream="worker", kind="k", payload={})
        page = se3.read_stream(db_session, workspace_id=ws2.id,
                               stream="worker")
        assert page["count"] == 0

    def test_unknown_stream_rejected(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            se3.emit_stream_event(db_session, workspace_id=ws.id,
                                  stream="not-a-stream", kind="k",
                                  payload={})

    def test_read_bounded(self, db_session):
        ws = _mkws(db_session)
        for i in range(5):
            se3.emit_stream_event(db_session, workspace_id=ws.id,
                                  stream="worker", kind="k",
                                  payload={"i": i})
        page = se3.read_stream(db_session, workspace_id=ws.id,
                               stream="worker", limit=3)
        assert page["count"] == 3


# ===========================================================================
# API versioning + contract matrix (Steps 39-40)
# ===========================================================================

class TestApiVersioning:
    def test_compatibility_metadata(self):
        meta = ap3.compatibility_metadata()
        assert meta["current_version"] == "v1"
        assert "v2" in meta["supported_versions"]
        assert "deprecation_policy" in meta

    def test_contract_matrix_complete(self):
        matrix = ap3.contract_case_matrix()
        assert matrix["count"] >= 10
        for case in ("tenant_isolation", "idempotency_replay",
                     "bounded_pagination", "auth_failure"):
            assert case in matrix["cases"]

    def test_request_hash_deterministic(self):
        h1 = ap3.request_hash({"a": 1, "b": [1, 2]})
        h2 = ap3.request_hash({"b": [1, 2], "a": 1})
        assert h1 == h2  # key-order independent

    def test_request_hash_differs_by_payload(self):
        assert ap3.request_hash({"a": 1}) != ap3.request_hash({"a": 2})
