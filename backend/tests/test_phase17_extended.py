"""Phase 17 tests — extended deterministic coverage (edge behaviors).

Complements the six main Phase 17 suites with targeted tests for:

- worker handler registry dispatch + broker delay/heartbeat ownership semantics
- ingestion retry/resume bookkeeping and run pagination
- fingerprint metadata fallback, jaccard math, isolation
- connector source updates, deep sync idempotency, sync-state introspection
- memory supersession validation, expiration idempotency, user isolation
- RAG5 query classification edges, evidence scoring math, partial citation
  coverage, scaled-number conflict equivalence, refusal-aware evaluation
- agent budget enforcement, cancellation propagation, handoff lifecycle
- workflow pause/resume idempotency, dependency ordering, compensation flags,
  workflow timeout enforcement
- governance retention precedence, cleanup isolation + hold protection,
  feature-scoped policy resolution
- alert severity validation, cooldown floor, event listing, missing metrics
- cost export isolation/bounds, constant-baseline anomaly detection
- search freshness ranking, saved-search daily dedupe
"""

import json
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.phase16 import (  # noqa: E402
    WorkerJob, WorkerHeartbeat, TraceSpan,
)
from app.models.phase17 import (  # noqa: E402
    JobLease, IngestionRun, IngestionStage, DocumentFingerprint,
    DuplicateCandidate, ConnectorSource, ConnectorSync, ConnectorItem,
    MemorySupersession, AgentPlan, HumanHandoff, WorkflowRun,
    AIPolicyRule, LegalHold, HoldEntity, RetentionAssignment,
    AlertRule, AlertEvent,
)
from app.models.phase15 import (  # noqa: E402
    AIMemory, AIQualityMetric, AIExecutionCheckpoint,
    WorkflowNodeExecution, WorkflowCompensation,
)
from app.models.ai_execution import AIExecution  # noqa: E402
from app.models.usage import UsageRecord  # noqa: E402
from app.models.search_intel import SavedSearch  # noqa: E402
from app.services import worker_platform as wp  # noqa: E402
from app.services import broker, worker_handlers  # noqa: E402
from app.services import ingestion2 as ing  # noqa: E402
from app.services import fingerprint as fp  # noqa: E402
from app.services import federation as fed  # noqa: E402
from app.services import memory3, rag5, agent3  # noqa: E402
from app.services import workflow3 as wf3  # noqa: E402
from app.services import governance2 as gov  # noqa: E402
from app.services import alerts as alert_svc, cost2, search3  # noqa: E402

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
    """Remove every row this suite creates so tests are order-independent."""
    for model in (
        WorkflowCompensation, WorkflowNodeExecution, WorkflowRun,
        AgentPlan, HumanHandoff, AIExecutionCheckpoint, AIExecution,
        MemorySupersession, AIMemory,
        DuplicateCandidate, DocumentFingerprint,
        IngestionStage, IngestionRun,
        ConnectorItem, ConnectorSync, ConnectorSource,
        JobLease, WorkerJob, WorkerHeartbeat,
        HoldEntity, LegalHold, RetentionAssignment, AIPolicyRule,
        TraceSpan, AlertEvent, AlertRule, SavedSearch, UsageRecord,
        AIQualityMetric,
    ):
        db_session.query(model).delete()
    db_session.commit()


def fresh_user(db, tag="p17x"):
    _counter[0] += 1
    user = User(name=f"X User {_counter[0]}",
                email=f"{tag}{_counter[0]}@p17x.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p17x ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def fresh_doc(db, ws, user, name=None):
    doc = Document(workspace_id=ws.id, user_id=user.id,
                   original_filename=name or f"d-{uuid.uuid4().hex[:6]}.pdf",
                   mime_type="text/plain", file_size=10, status="READY",
                   storage_key=f"x-{uuid.uuid4().hex}")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def fresh_execution(db, ws, user, status="RUNNING", **kw):
    execution = AIExecution(
        id=str(uuid.uuid4()), workspace_id=ws.id, user_id=user.id,
        execution_type=kw.pop("execution_type", "agent"),
        task_type=kw.pop("task_type", "x"), status=status,
        priority="NORMAL", **kw)
    db.add(execution)
    db.commit()
    db.refresh(execution)
    return execution


# ============================================================
# Worker handler registry + broker delay/heartbeat semantics
# ============================================================

class TestWorkerHandlers:
    @pytest.mark.parametrize("job_type", [
        "EVENT_PROCESS", "INGESTION_ADVANCE", "CONNECTOR_SYNC",
        "VECTOR_BACKFILL_BATCH"])
    def test_builtin_handler_registered(self, job_type):
        handler = worker_handlers.get_handler(job_type)
        assert callable(handler)

    def test_unknown_handler_returns_none(self):
        assert worker_handlers.get_handler("NOT_A_REAL_TYPE") is None

    def test_registry_stable_across_calls(self):
        first = worker_handlers.get_handler("EVENT_PROCESS")
        second = worker_handlers.get_handler("EVENT_PROCESS")
        assert first is second


class TestBrokerDelayAndHeartbeat:
    def test_delayed_job_not_claimable_before_run_after(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        backend = broker.get_broker("postgres")
        backend.enqueue(
            db_session, queue_name="delay", job_type="j",
            workspace_id=ws.id, payload={},
            run_after=datetime.now(timezone.utc) + timedelta(hours=1))
        assert backend.claim(db_session, queue_name="delay",
                             worker_id="w1") is None

    def test_delayed_job_claimable_after_run_after(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        backend = broker.get_broker("postgres")
        record = backend.enqueue(
            db_session, queue_name="delay2", job_type="j",
            workspace_id=ws.id, payload={},
            run_after=datetime.now(timezone.utc) + timedelta(hours=1))
        job = db_session.query(WorkerJob).filter(
            WorkerJob.id == record["id"]).first()
        job.run_after = datetime.now(timezone.utc) - timedelta(seconds=1)
        db_session.commit()
        claimed = backend.claim(db_session, queue_name="delay2",
                                worker_id="w1")
        assert claimed is not None
        assert claimed["id"] == record["id"]
        assert claimed["status"] == "CLAIMED"

    def test_broker_cancel_prevents_claim(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        backend = broker.get_broker("postgres")
        record = backend.enqueue(db_session, queue_name="bc",
                                 job_type="j", workspace_id=ws.id,
                                 payload={})
        result = backend.cancel(db_session, record["id"])
        assert result["status"] == "CANCELLED"
        assert backend.claim(db_session, queue_name="bc",
                             worker_id="w1") is None

    def test_heartbeat_respects_ownership(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        backend = broker.get_broker("postgres")
        record = backend.enqueue(db_session, queue_name="hb",
                                 job_type="j", workspace_id=ws.id,
                                 payload={})
        job = backend.claim(db_session, queue_name="hb", worker_id="wA")
        before = db_session.query(WorkerJob).filter(
            WorkerJob.id == job["id"]).first().heartbeat_at
        # A different worker must not be able to heartbeat the job.
        backend.heartbeat(db_session, job["id"], "intruder")
        db_session.commit()
        row = db_session.query(WorkerJob).filter(
            WorkerJob.id == job["id"]).first()
        assert row.claimed_by == "wA"
        assert row.heartbeat_at == before
        # The owning worker can extend the heartbeat.
        backend.heartbeat(db_session, job["id"], "wA")
        db_session.commit()
        row = db_session.query(WorkerJob).filter(
            WorkerJob.id == job["id"]).first()
        assert row.heartbeat_at is not None

    def test_dedupe_enqueue_returns_existing(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        backend = broker.get_broker("postgres")
        first = backend.enqueue(db_session, queue_name="dd",
                                job_type="j", workspace_id=ws.id,
                                payload={"v": 1}, dedupe_key="same-key")
        second = backend.enqueue(db_session, queue_name="dd",
                                 job_type="j", workspace_id=ws.id,
                                 payload={"v": 1}, dedupe_key="same-key")
        assert first["id"] == second["id"]


# ============================================================
# Ingestion — retry bookkeeping, progress, pagination
# ============================================================

class TestIngestionRetryMechanics:
    def _run(self, db, ws, user, doc=None):
        doc = doc or fresh_doc(db, ws, user)
        return ing.start_ingestion(db, ws.id, doc.id, user_id=user.id)

    def test_retry_resets_only_failed_stage(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._run(db_session, ws, user)
        ing.advance_ingestion_run(db_session, run.id,
                                  simulate_failure="OCR")
        result = ing.retry_ingestion(db_session, run.id, user.id)
        assert result["reset_stages"] == ["OCR"]
        assert result["status"] == "RUNNING"
        stages = {s.stage: s for s in db_session.query(IngestionStage)
                  .filter(IngestionStage.run_id == run.id).all()}
        # Completed stages keep their status and never re-run.
        for stage in ("UPLOAD", "VALIDATE", "EXTRACT"):
            assert stages[stage].status == "COMPLETED"
        assert stages["OCR"].status == "PENDING"
        assert stages["OCR"].attempts == 0
        # Advancing after retry finishes the pipeline without touching
        # the completed stages again.
        final = ing.advance_ingestion_run(db_session, run.id)
        assert final["status"] == "COMPLETED"
        assert final["progress_pct"] == 100
        stages = {s.stage: s for s in db_session.query(IngestionStage)
                  .filter(IngestionStage.run_id == run.id).all()}
        assert stages["OCR"].attempts == 1  # reset then one fresh attempt
        assert stages["UPLOAD"].attempts == 0  # never re-run

    def test_retry_on_running_run_is_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._run(db_session, ws, user)
        result = ing.retry_ingestion(db_session, run.id, user.id)
        assert result["status"] == "RUNNING"
        assert "not in a retryable state" in result.get("error", "")

    def test_retry_on_completed_run_is_noop(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._run(db_session, ws, user)
        ing.advance_ingestion_run(db_session, run.id)
        result = ing.retry_ingestion(db_session, run.id, user.id)
        assert result["status"] == "COMPLETED"

    def test_advance_after_failure_does_not_rerun(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._run(db_session, ws, user)
        ing.advance_ingestion_run(db_session, run.id,
                                  simulate_failure="INDEX")
        # Advance again without retry: run stays FAILED (no partial rerun).
        again = ing.advance_ingestion_run(db_session, run.id)
        assert again["status"] == "FAILED"

    def test_missing_run_raises(self, db_session):
        with pytest.raises(wp.JobNotFoundError):
            ing.advance_ingestion_run(db_session, 999_999)

    def test_progress_reports_failure_detail(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._run(db_session, ws, user)
        ing.advance_ingestion_run(db_session, run.id,
                                  simulate_failure="EMBED")
        progress = ing.ingestion_progress(db_session, run.id)
        assert progress["status"] == "FAILED"
        assert progress["current_stage"] == "EMBED"
        assert "%" in progress["estimated_remaining"]
        embed = [s for s in progress["stages"] if s["stage"] == "EMBED"][0]
        assert embed["attempts"] == 1
        assert embed["error"] is not None

    def test_list_runs_offset_and_status_filter(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        runs = [self._run(db_session, ws, user) for _ in range(3)]
        page = ing.list_runs(db_session, ws.id, limit=2, offset=1)
        assert len(page["items"]) == 2
        assert page["total"] == 3
        ids = [i["run_id"] for i in page["items"]]
        assert runs[2].id not in ids  # desc order: offset drops newest
        failed = ing.list_runs(db_session, ws.id, status="RUNNING")
        assert failed["total"] == 3


# ============================================================
# Fingerprints — jaccard math, metadata fallback, isolation
# ============================================================

class TestFingerprintMath:
    def test_jaccard(self):
        assert fp.jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
        assert fp.jaccard({1, 2}, {3, 4}) == 0.0
        assert fp.jaccard(set(), {1}) == 0.0
        assert 0 < fp.jaccard({1, 2, 3}, {2, 3, 4}) < 1.0

    def test_metadata_fallback_stable(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_doc(db_session, ws, user, name="meta-doc.pdf")
        f1 = fp.compute_fingerprint(db_session, doc.id)  # no content
        f2 = fp.compute_fingerprint(db_session, doc.id)
        assert f1.content_hash == f2.content_hash
        assert f1.metadata_hash == f2.metadata_hash

    def test_classify_no_peers_returns_empty(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        doc = fresh_doc(db_session, ws, user)
        assert fp.classify_duplicates(db_session, ws.id, doc.id) == []

    def test_candidate_isolation_across_workspaces(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        d1 = fresh_doc(db_session, ws1, u1)
        d2 = fresh_doc(db_session, ws2, u2)
        text = "identical body content for fingerprint isolation " * 10
        fp.compute_fingerprint(db_session, d1.id, content=text)
        fp.compute_fingerprint(db_session, d2.id, content=text)
        results = fp.classify_duplicates(db_session, ws1.id, d1.id)
        assert results == []  # ws2 doc is invisible from ws1
        assert db_session.query(DuplicateCandidate).filter(
            DuplicateCandidate.document_id == d1.id).count() == 0


# ============================================================
# Federation — source updates, deep sync idempotency, sync state
# ============================================================

class TestFederationSource:
    def _source(self, db, ws, user, name="ExtSrc"):
        return fed.create_source(db, workspace_id=ws.id,
                                 organization_id=None, user_id=user.id,
                                 name=name, kind="knowledge_base")

    def test_update_source_fields(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = self._source(db_session, ws, user)
        updated = fed.update_source(db_session, src.id, ws.id,
                                    name="Renamed", enabled=False,
                                    credential_ref="vault://new")
        assert updated.name == "Renamed"
        assert updated.enabled is False
        assert updated.credential_ref == "vault://new"

    def test_update_foreign_workspace_denied(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        src = self._source(db_session, ws1, u1)
        with pytest.raises(wp.JobNotFoundError):
            fed.update_source(db_session, src.id, ws2.id, name="steal")

    def test_third_sync_is_noop(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = self._source(db_session, ws, user)
        fed.run_connector_sync(db_session, src.id)   # full: 2 added
        fed.run_connector_sync(db_session, src.id)   # delta: 1 change/1 delete
        third = fed.run_connector_sync(db_session, src.id)
        assert third["status"] == "COMPLETED"
        assert third["items_added"] == 0
        assert third["items_changed"] == 0
        assert third["items_deleted"] == 0
        assert db_session.query(ConnectorItem).filter(
            ConnectorItem.source_id == src.id).count() == 2  # never dupe

    def test_sync_state_recent_first_and_scoped(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        src = self._source(db_session, ws1, u1)
        fed.run_connector_sync(db_session, src.id)
        fed.run_connector_sync(db_session, src.id)
        state = fed.sync_state(db_session, src.id, ws1.id)
        assert len(state["items"]) == 2
        assert state["items"][0]["sync_id"] > state["items"][1]["sync_id"]
        with pytest.raises(wp.JobNotFoundError):
            fed.sync_state(db_session, src.id, ws2.id)

    def test_serialized_source_never_exposes_credential(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        src = fed.create_source(
            db_session, workspace_id=ws.id, organization_id=None,
            user_id=user.id, name="Sec", kind="cloud_storage",
            credential_ref="super-secret-credential-value")
        dumped = json.dumps(fed.source_dict(src), default=str)
        assert "super-secret-credential-value" not in dumped
        assert fed.source_dict(src)["has_credential_ref"] is True


# ============================================================
# Memory 3.0 — supersession validation, expiration, isolation
# ============================================================

class TestMemoryLifecycle:
    def _memory(self, db, ws, user=None, **kw):
        m = AIMemory(
            workspace_id=ws.id,
            user_id=user.id if user else None,
            memory_type=kw.get("memory_type", "WORKSPACE_FACT"),
            scope=kw.get("scope", "WORKSPACE"),
            content=kw.get("content", "a durable fact"),
            source=kw.get("source", "document"),
            confidence=kw.get("confidence", "HIGH"),
            expires_at=kw.get("expires_at"))
        db.add(m)
        db.commit()
        db.refresh(m)
        return m

    def test_supersede_rejects_foreign_workspace_memory(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        m = self._memory(db_session, ws1)
        with pytest.raises(ValueError):
            memory3.supersede(db_session, m.id, "new fact",
                              ws2.id, "WORKSPACE_FACT", "user")

    def test_expire_due_idempotent(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        past = datetime.now(timezone.utc) - timedelta(days=2)
        self._memory(db_session, ws, user, expires_at=past)
        first = memory3.expire_due(db_session)
        second = memory3.expire_due(db_session)
        assert first["expired"] >= 1
        assert second["expired"] == 0

    def test_expired_memory_not_retrieved(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        past = datetime.now(timezone.utc) - timedelta(days=2)
        self._memory(db_session, ws, user, expires_at=past)
        memory3.expire_due(db_session)
        result = memory3.retrieve_for_user(db_session, ws.id,
                                           user_id=user.id)
        assert result["items"] == []

    def test_user_scope_isolation(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws = fresh_workspace(db_session, u1)
        self._memory(db_session, ws, u1, scope="USER",
                     content="u1-private-fact",
                     memory_type="USER_PREFERENCE")
        visible_to_u2 = memory3.retrieve_for_user(
            db_session, ws.id, user_id=u2.id)
        visible_to_u1 = memory3.retrieve_for_user(
            db_session, ws.id, user_id=u1.id)
        assert all("u1-private-fact" not in i["content"]
                   for i in visible_to_u2["items"])
        assert any("u1-private-fact" in i["content"]
                   for i in visible_to_u1["items"])

    def test_confidence_conflict_penalty(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        m = self._memory(db_session, ws, user)
        clean = memory3.confidence(db_session, m)
        conflicted = memory3.confidence(db_session, m, "CONFLICT")
        assert conflicted < clean

    def test_consolidate_ignores_foreign_workspace(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        m1 = self._memory(db_session, ws1, u1, content="a")
        self._memory(db_session, ws2, u2, content="b")
        # only one memory in ws1 → consolidation must refuse
        with pytest.raises(ValueError):
            memory3.consolidate(db_session, [m1.id], ws1.id, u1.id)


# ============================================================
# RAG5 — classification edges, scoring math, coverage, conflicts
# ============================================================

class TestRag5Edges:
    def test_classify_empty_is_factual(self):
        assert rag5.classify_query("") == "factual"

    def test_classify_temporal_outranks_policy(self):
        assert rag5.classify_query("what was the approval policy "
                                   "as of 2023") == "temporal"

    def test_classify_procedural_and_entity(self):
        assert rag5.classify_query("how do I file an expense") == \
            "procedural"
        assert rag5.classify_query("who is the vendor for region x") == \
            "entity"

    def test_distance_to_relevance(self):
        chunk = {"text": "x" * 900, "distance": 0.25,
                 "document": {"mime_type": "text/plain"}}
        quality = rag5.score_evidence(chunk)
        assert quality["relevance"] == pytest.approx(0.75, abs=0.001)

    def test_empty_chunk_completeness_zero(self):
        chunk = {"text": "", "score": 0.9,
                 "document": {"mime_type": "application/pdf"}}
        quality = rag5.score_evidence(chunk)
        assert quality["completeness"] == 0.0
        assert 0.0 <= quality["overall"] <= 1.0

    def test_partial_citation_coverage_math(self):
        result = rag5.citation_coverage(
            ["c1", "c2", "c3", "c4"], [],
            supported=["SUPPORTED", "PARTIALLY_SUPPORTED",
                       "UNSUPPORTED", "UNSUPPORTED"])
        assert result["supported"] == 1
        assert result["partially_supported"] == 1
        assert result["unsupported"] == 2
        assert result["coverage_pct"] == 37.5

    def test_scaled_numbers_do_not_false_conflict(self):
        evidence = [
            {"chunk_index": 1, "text": "budget is $10k per quarter"},
            {"chunk_index": 2, "text": "budget is $10000 per quarter"},
        ]
        assert rag5.detect_evidence_conflicts(evidence) == []

    def test_distinct_numbers_are_conflicts(self):
        evidence = [
            {"chunk_index": 1, "text": "30 days net"},
            {"chunk_index": 2, "text": "60 days net"},
        ]
        conflicts = rag5.detect_evidence_conflicts(evidence)
        assert len(conflicts) == 1
        assert conflicts[0]["category"] == "numeric_contradiction"

    def test_refusal_evaluation_counts(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        dataset = [{
            "query": "internal-only detail",
            "expected": [],
            "expected_refusal": True,
        }]
        result = rag5.run_evaluation(db_session, ws.id, dataset)
        assert result["correct_refusals"] == 1
        assert db_session.query(AIQualityMetric).filter(
            AIQualityMetric.workspace_id == ws.id).count() == 1


# ============================================================
# Agent 3.0 — budget enforcement, cancellation, handoff
# ============================================================

class TestAgent3Edges:
    def test_budgets_for_defaults_without_plan(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        budgets = agent3.budgets_for(db_session, execution)
        assert budgets["token_budget"] == 100_000
        assert budgets["cost_budget_usd"] == 5.0

    def test_budget_exceeded_blocks_execution(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user,
                                    actual_cost=50.0)
        called = []

        def run_tool(db, tool, inputs):
            called.append(tool)
            return {"ok": True}

        with pytest.raises(RuntimeError):
            agent3.execute_step(
                db_session, execution_id=execution.id,
                workspace_id=ws.id, step_number=1, tool="noop",
                inputs={}, run_tool=run_tool)
        assert called == []  # tool never ran

    def test_cancelled_execution_rejects_steps(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        agent3.cancel_execution_durable(db_session, execution.id,
                                        ws.id, user.id)
        with pytest.raises(RuntimeError):
            agent3.execute_step(
                db_session, execution_id=execution.id,
                workspace_id=ws.id, step_number=1, tool="noop",
                inputs={})

    def test_cancel_closes_open_handoffs(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        agent3.request_handoff(db_session, workspace_id=ws.id,
                               execution_id=execution.id,
                               question="need review",
                               requested_by=user.id)
        result = agent3.cancel_execution_durable(
            db_session, execution.id, ws.id, user.id)
        assert result["open_handoffs_cancelled"] == 1
        handoff = db_session.query(HumanHandoff).first()
        assert handoff.status == "CANCELLED"

    def test_handoff_answer_twice_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        agent3.request_handoff(db_session, workspace_id=ws.id,
                               execution_id=execution.id,
                               question="approve?",
                               requested_by=user.id)
        handoff = db_session.query(HumanHandoff).first()
        agent3.answer_handoff(db_session, handoff_id=handoff.id,
                              workspace_id=ws.id, answer="yes",
                              answered_by=user.id)
        with pytest.raises(ValueError):
            agent3.answer_handoff(db_session, handoff_id=handoff.id,
                                  workspace_id=ws.id, answer="again",
                                  answered_by=user.id)

    def test_answer_handoff_foreign_workspace_rejected(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        execution = fresh_execution(db_session, ws1, u1)
        agent3.request_handoff(db_session, workspace_id=ws1.id,
                               execution_id=execution.id,
                               question="q", requested_by=u1.id)
        handoff = db_session.query(HumanHandoff).first()
        with pytest.raises(ValueError):
            agent3.answer_handoff(db_session, handoff_id=handoff.id,
                                  workspace_id=ws2.id, answer="yes",
                                  answered_by=u2.id)

    def test_invalid_plan_risk_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        plan = {"steps": [{"id": "1", "tool": "noop",
                           "dependencies": []}]}
        with pytest.raises(ValueError):
            agent3.create_agent_plan(
                db_session, workspace_id=ws.id,
                execution_id=execution.id, objective="o", plan=plan,
                risk="EXTREME")


# ============================================================
# Workflow 3.0 — pause idempotency, dependencies, compensation
# ============================================================

class TestWorkflow3Edges:
    def _start(self, db, ws, definition, **kw):
        return wf3.start_workflow_run(
            db, workspace_id=ws.id, organization_id=None,
            definition=definition, **kw)

    def test_pause_idempotent_and_resume(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._start(db_session, ws,
                          {"nodes": [{"id": "a", "type": "t",
                                      "inputs": {}}]})
        first = wf3.pause_run(db_session, run.id, ws.id, user.id)
        second = wf3.pause_run(db_session, run.id, ws.id, user.id)
        assert first["control_state"] == "PAUSED"
        assert second["control_state"] == "PAUSED"
        row = db_session.query(WorkflowRun).filter(
            WorkflowRun.id == run.id).first()
        assert row.idle_timeout_at is None  # paused → never idle-timed-out
        resumed = wf3.resume_run(db_session, run.id, ws.id, user.id)
        assert resumed["control_state"] == "ACTIVE"
        row = db_session.query(WorkflowRun).filter(
            WorkflowRun.id == run.id).first()
        assert row.idle_timeout_at is not None

    def test_resume_non_paused_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._start(db_session, ws,
                          {"nodes": [{"id": "a", "type": "t"}]})
        with pytest.raises(ValueError):
            wf3.resume_run(db_session, run.id, ws.id, user.id)

    def test_advance_honors_dependencies(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._start(db_session, ws, {"nodes": [
            {"id": "a", "type": "t", "inputs": {}},
            {"id": "b", "type": "t", "inputs": {},
             "depends_on": ["a"]},
        ]})
        first = wf3.advance_run(db_session, run.id, ws.id)
        assert first["node"] == "a"
        second = wf3.advance_run(db_session, run.id, ws.id)
        assert second["node"] == "b"
        third = wf3.advance_run(db_session, run.id, ws.id)
        assert third["status"] == "COMPLETED"

    def test_advance_completed_run_is_noop(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._start(db_session, ws,
                          {"nodes": [{"id": "a", "type": "t"}]})
        wf3.advance_run(db_session, run.id, ws.id)
        node_count = db_session.query(WorkflowNodeExecution).filter(
            WorkflowNodeExecution.workflow_execution_id ==
            f"wf-run-{run.id}").count()
        again = wf3.advance_run(db_session, run.id, ws.id)
        assert again["status"] == "COMPLETED"
        after = db_session.query(WorkflowNodeExecution).filter(
            WorkflowNodeExecution.workflow_execution_id ==
            f"wf-run-{run.id}").count()
        assert after == node_count  # no re-execution

    def test_advance_foreign_workspace_rejected(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        run = self._start(db_session, ws1,
                          {"nodes": [{"id": "a", "type": "t"}]})
        with pytest.raises(ValueError):
            wf3.advance_run(db_session, run.id, ws2.id)

    def test_workflow_timeout_seconds_enforced(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._start(db_session, ws,
                          {"nodes": [{"id": "a", "type": "t"}]},
                          timeout_seconds=3600)
        fixed_now = datetime.now(timezone.utc)
        # Simulate the workflow deadline passing (deterministic — no reliance
        # on wall-clock drift).
        row = db_session.query(WorkflowRun).filter(
            WorkflowRun.id == run.id).first()
        row.timeout_at = fixed_now - timedelta(minutes=1)
        db_session.commit()
        result = wf3.check_timeouts(db_session, now=fixed_now)
        assert result["timed_out"] >= 1
        row = db_session.query(WorkflowRun).filter(
            WorkflowRun.id == run.id).first()
        assert row.status == "FAILED"
        assert row.error == "workflow timeout exceeded"

    def test_compensation_reversible_flags(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self._start(db_session, ws,
                          {"nodes": [{"id": "a", "type": "t"}]})
        reversible = wf3.compensation_note(
            db_session, run=run, node_id="a",
            side_effect_type="email", reversible=True)
        irreversible = wf3.compensation_note(
            db_session, run=run, node_id="b",
            side_effect_type="payment", reversible=False)
        assert reversible.reversible is True
        assert irreversible.reversible is False
        assert reversible.status == "RECORDED"


# ============================================================
# Governance — retention precedence, cleanup, policy resolution
# ============================================================

class TestGovernance2Edges:
    def test_retention_default_90(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        assert gov.retention_days_for(db_session, "trace", ws.id, None) == 90

    def test_workspace_retention_overrides_org(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        gov.set_retention(db_session, entity_type="trace",
                          retention_days=30, organization_id=7)
        gov.set_retention(db_session, entity_type="trace",
                          retention_days=7, workspace_id=ws.id)
        assert gov.retention_days_for(db_session, "trace", ws.id, 7) == 7

    def test_org_retention_used_when_no_workspace_rule(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        gov.set_retention(db_session, entity_type="trace",
                          retention_days=30, organization_id=7)
        assert gov.retention_days_for(db_session, "trace", ws.id, 7) == 30

    def test_retention_invalid_range_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            gov.set_retention(db_session, entity_type="trace",
                              retention_days=-1, workspace_id=ws.id)

    def test_cleanup_unsupported_type_noop(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = gov.run_cleanup(db_session, entity_type="mystery",
                                 workspace_id=ws.id,
                                 organization_id=None)
        assert result["deleted"] == 0

    def _trace(self, db, ws, days_old=400, tag=None):
        span = TraceSpan(
            span_id=f"span-{uuid.uuid4().hex}",
            trace_id=f"tr-{uuid.uuid4().hex}",
            workspace_id=ws.id, span_type="request", status="OK",
            created_at=datetime.now(timezone.utc)
            - timedelta(days=days_old))
        db.add(span)
        db.commit()
        db.refresh(span)
        return span

    def test_cleanup_workspace_isolation(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        self._trace(db_session, ws1)
        other = self._trace(db_session, ws2)
        gov.set_retention(db_session, entity_type="trace",
                          retention_days=30, workspace_id=ws1.id)
        result = gov.run_cleanup(db_session, entity_type="trace",
                                 workspace_id=ws1.id,
                                 organization_id=None)
        assert result["deleted"] == 1
        assert db_session.query(TraceSpan).filter(
            TraceSpan.id == other.id).first() is not None

    def test_cleanup_hold_protects_entity(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        held = self._trace(db_session, ws)
        free = self._trace(db_session, ws)
        gov.place_hold(db_session, workspace_id=ws.id,
                       organization_id=None, name="litigation",
                       reason="review", entity_type="trace",
                       entity_ids=[held.id])
        gov.set_retention(db_session, entity_type="trace",
                          retention_days=0, workspace_id=ws.id)
        result = gov.run_cleanup(db_session, entity_type="trace",
                                 workspace_id=ws.id,
                                 organization_id=None)
        assert result["skipped_under_hold"] == 1
        assert result["deleted"] == 1
        assert db_session.query(TraceSpan).filter(
            TraceSpan.id == held.id).first() is not None
        assert db_session.query(TraceSpan).filter(
            TraceSpan.id == free.id).first() is None

    def test_release_hold_allows_cleanup(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        span = self._trace(db_session, ws)
        hold = gov.place_hold(db_session, workspace_id=ws.id,
                              organization_id=None, name="temp",
                              reason="x", entity_type="trace",
                              entity_ids=[span.id])
        gov.release_hold(db_session, hold.id, ws.id, user.id)
        assert gov.is_under_hold(db_session, ws.id, "trace", span.id) is False
        gov.set_retention(db_session, entity_type="trace",
                          retention_days=0, workspace_id=ws.id)
        result = gov.run_cleanup(db_session, entity_type="trace",
                                 workspace_id=ws.id,
                                 organization_id=None)
        assert result["deleted"] == 1

    def test_release_foreign_workspace_rejected(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        span = self._trace(db_session, ws1)
        hold = gov.place_hold(db_session, workspace_id=ws1.id,
                              organization_id=None, name="x",
                              reason="y", entity_type="trace",
                              entity_ids=[span.id])
        with pytest.raises(ValueError):
            gov.release_hold(db_session, hold.id, ws2.id, u2.id)

    def test_feature_scoped_policy_restricts_more(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        gov.create_rule(db_session, organization_id=None,
                        workspace_id=ws.id, feature=None,
                        rule_type="MODEL", allowlist=["gpt-4o", "claude"])
        gov.create_rule(db_session, organization_id=None,
                        workspace_id=ws.id, feature="summarize",
                        rule_type="MODEL", allowlist=["claude"])
        policy = gov.resolve_policy(db_session, organization_id=None,
                                    workspace_id=ws.id,
                                    feature="summarize",
                                    rule_type="MODEL")
        assert policy["allowlist"] == ["claude"]
        assert policy["restrictive"] is True


# ============================================================
# Alerts — validation, cooldown floor, event listing
# ============================================================

class TestAlertEdges:
    def test_invalid_severity_rejected(self, db_session):
        with pytest.raises(ValueError):
            alert_svc.create_rule(db_session, name="bad",
                                  metric="error_rate",
                                  operator=">", threshold=1,
                                  severity="FATAL")

    def test_cooldown_floor_is_one_minute(self, db_session):
        rule = alert_svc.create_rule(db_session, name="c",
                                     metric="queue_depth",
                                     operator=">", threshold=1,
                                     cooldown_minutes=0)
        assert rule.cooldown_minutes >= 1

    def test_missing_metric_never_fires(self, db_session):
        alert_svc.create_rule(db_session, name="m",
                              metric="provider_down",
                              operator=">", threshold=0)
        fired = alert_svc.evaluate(db_session, {"error_rate": 99})
        assert fired == []

    def test_list_events_after_fire(self, db_session):
        alert_svc.create_rule(db_session, name="q",
                              metric="queue_depth", operator=">",
                              threshold=10, severity="CRITICAL")
        alert_svc.evaluate(db_session, {"queue_depth": 500})
        listing = alert_svc.list_events(db_session)
        assert listing["total"] == 1
        event = listing["items"][0]
        assert event["metric_value"] == 500
        assert event["severity"] == "CRITICAL"
        assert "queue_depth" in event["message"]


# ============================================================
# Cost platform — export isolation, anomaly detection edges
# ============================================================

class TestCostEdges:
    def test_export_bounded_and_parseable(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        for i in range(5):
            db_session.add(UsageRecord(
                workspace_id=ws.id, user_id=user.id,
                usage_type="ai_request", quantity=100, units="tokens",
                created_at=datetime.now(timezone.utc)
                - timedelta(days=i)))
        db_session.commit()
        csv_text = cost2.export_usage_csv(db_session, organization_id=1,
                                          workspace_id=ws.id, days=9999)
        rows = [r for r in csv_text.splitlines() if r.strip()]
        assert rows[0] == "day,executions,tokens,cost_usd"
        assert len(rows) - 1 == 5  # one row per distinct day
        assert len(rows) <= 366  # bounded even with days=9999

    def test_export_workspace_isolation(self, db_session):
        u1 = fresh_user(db_session)
        u2 = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, u1)
        ws2 = fresh_workspace(db_session, u2)
        db_session.add(UsageRecord(workspace_id=ws1.id, user_id=u1.id,
                                   usage_type="ai_request", quantity=7,
                                   units="tokens"))
        db_session.add(UsageRecord(workspace_id=ws2.id, user_id=u2.id,
                                   usage_type="ai_request", quantity=999,
                                   units="tokens"))
        db_session.commit()
        csv_text = cost2.export_usage_csv(db_session, organization_id=1,
                                          workspace_id=ws1.id)
        assert "999" not in csv_text
        assert "7" in csv_text

    def test_anomaly_detected_on_constant_baseline(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        base = datetime.now(timezone.utc)
        for i in range(10):
            db_session.add(UsageRecord(
                workspace_id=ws.id, user_id=user.id,
                usage_type="ai_request", quantity=100, units="tokens",
                created_at=base - timedelta(days=10 - i)))
        # spike today (constant baseline → deviation-based detection)
        db_session.add(UsageRecord(workspace_id=ws.id, user_id=user.id,
                                   usage_type="ai_request", quantity=100_000,
                                   units="tokens", created_at=base))
        db_session.commit()
        anomalies = cost2.detect_anomalies(db_session, workspace_id=ws.id)
        token_anomalies = [a for a in anomalies if a["metric"] == "tokens"]
        assert token_anomalies
        assert "estimated anomaly" in token_anomalies[0]["label"]

    def test_flat_usage_no_anomaly(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        base = datetime.now(timezone.utc)
        for i in range(10):
            db_session.add(UsageRecord(
                workspace_id=ws.id, user_id=user.id,
                usage_type="ai_request", quantity=50, units="tokens",
                created_at=base - timedelta(days=9 - i)))
        db_session.commit()
        anomalies = cost2.detect_anomalies(db_session, workspace_id=ws.id)
        assert all(a["metric"] == "tokens" for a in anomalies) is False or \
            anomalies == []


# ============================================================
# Search platform — freshness, saved-search dedupe
# ============================================================

class TestSearchEdges:
    def test_freshness_rank_newer_first(self):
        now = datetime.now(timezone.utc)
        results = [
            {"document_id": 1, "score": 0.8,
             "created_at": (now - timedelta(days=900)).isoformat()},
            {"document_id": 2, "score": 0.8,
             "created_at": now.isoformat()},
        ]
        ranked = search3.freshness_rank(results)
        assert ranked[0]["document_id"] == 2

    def test_diversify_caps_per_document(self):
        results = [
            {"document_id": 1, "chunk": i} for i in range(5)
        ] + [{"document_id": 2, "chunk": 0}]
        diversified = search3.diversify(results, max_per_document=2)
        assert len(diversified) == 3
        assert [r["document_id"] for r in diversified] == [1, 1, 2]

    def test_saved_search_check_unchanged_no_job(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        record = SavedSearch(workspace_id=ws.id, owner_id=user.id,
                             name="watch", query="policy")
        db_session.add(record)
        db_session.commit()
        result = search3.run_saved_search_check(
            db_session, saved_search_id=record.id, workspace_id=ws.id,
            alert_evaluator=lambda db, ws_id, q: [])
        assert result["changed"] is False
        assert db_session.query(WorkerJob).filter(
            WorkerJob.queue_name == "NOTIFICATIONS").count() == 0

    def test_saved_search_daily_dedupe(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        record = SavedSearch(workspace_id=ws.id, owner_id=user.id,
                             name="watch", query="policy")
        db_session.add(record)
        db_session.commit()
        evaluator = lambda db, ws_id, q: [1]  # noqa: E731
        search3.run_saved_search_check(
            db_session, saved_search_id=record.id, workspace_id=ws.id,
            alert_evaluator=evaluator)
        search3.run_saved_search_check(
            db_session, saved_search_id=record.id, workspace_id=ws.id,
            alert_evaluator=evaluator)
        jobs = db_session.query(WorkerJob).filter(
            WorkerJob.queue_name == "NOTIFICATIONS").all()
        assert len(jobs) == 1  # same-day dedupe key collapsed both checks
