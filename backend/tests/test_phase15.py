"""Phase 15 Test Suite — Autonomous Knowledge Operating System (part 1).

Covers: AI execution orchestrator 2.0, idempotency, checkpoints, queue
fairness, event outbox, knowledge change/impact/policy/temporal/snapshots,
memory governance, claim validation, context budgeting, and review queue.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

_ws_counter = [1000]
_uid_counter = [1000]


def fresh_ws() -> int:
    _ws_counter[0] += 1
    return _ws_counter[0]


def fresh_user() -> int:
    _uid_counter[0] += 1
    return _uid_counter[0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(db, workspace_id, title="Policy Doc", filename=None, status="READY", user_id=None):
    from app.models.document import Document
    from app.models.document_tag import DocumentClassification, DocumentSummary
    from app.models.document_version import DocumentVersion
    _ws_counter[0] += 1
    filename = filename or f"phase15-{_ws_counter[0]}.pdf"
    user_id = user_id or fresh_user()
    doc = Document(
        user_id=user_id,
        workspace_id=workspace_id,
        original_filename=filename,
        storage_key=f"p15-{_ws_counter[0]}-{uuid.uuid4().hex[:8]}",
        mime_type="application/pdf",
        file_size=100,
        status=status,
    )
    db.add(doc)
    db.flush()
    version_meta = {"title": title}
    db.add(DocumentVersion(
        document_id=doc.id, version_number=1, created_by=user_id,
        storage_key=doc.storage_key, file_size=100, mime_type="application/pdf",
        is_current=True, processing_status="READY" if status == "READY" else "PENDING",
        metadata_json=json.dumps(version_meta),
    ))
    db.flush()
    return doc


def make_execution(db, workspace_id, user_id, status="QUEUED", priority="NORMAL",
                   task_type="rag", execution_type="rag", started_at=None,
                   payload=None, idempotency_key=None):
    from app.services.ai_execution_service import create_execution
    if idempotency_key is None:
        idempotency_key = None
    return create_execution(
        db, workspace_id=workspace_id, user_id=user_id,
        execution_type=execution_type, task_type=task_type,
        priority=priority, payload=payload, idempotency_key=idempotency_key,
    )


# ============================================================
# 1. AI Execution Orchestrator 2.0
# ============================================================

class TestExecutionLifecycle:
    def test_create_execution_defaults(self, db_session):
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        assert e.status == "QUEUED"
        assert e.priority == "NORMAL"
        assert e.trace_id
        assert e.estimated_cost == 0.0

    def test_create_execution_priority_validated(self, db_session):
        ws, uid = fresh_ws(), fresh_user()
        with pytest.raises(ValueError):
            make_execution(db_session, ws, uid, priority="URGENT")

    def test_critical_priority_accepted(self, db_session):
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid, priority="CRITICAL")
        assert e.priority == "CRITICAL"

    def test_status_set_is_exhaustive(self):
        from app.models.phase15 import EXECUTION_STATUSES
        assert set(EXECUTION_STATUSES) == {
            "QUEUED", "PLANNING", "RUNNING", "WAITING_APPROVAL", "WAITING_TOOL",
            "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "RETRYING",
        }

    def test_queue_to_running(self, db_session):
        from app.services.ai_execution_service import transition
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition(db_session, e, "RUNNING", uid)
        assert e.status == "RUNNING"
        assert e.started_at is not None

    def test_running_to_completed(self, db_session):
        from app.services.ai_execution_service import transition
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid, status="QUEUED")
        transition(db_session, e, "RUNNING", uid)
        transition(db_session, e, "COMPLETED", uid)
        assert e.status == "COMPLETED"
        assert e.completed_at is not None
        assert e.latency_ms is not None

    def test_invalid_transition_rejected(self, db_session):
        from app.services.ai_execution_service import transition, ExecutionStateError
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        with pytest.raises(ExecutionStateError):
            transition(db_session, e, "COMPLETED", uid)  # QUEUED -> COMPLETED invalid

    def test_terminal_immutable(self, db_session):
        from app.services.ai_execution_service import transition, ExecutionStateError
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition(db_session, e, "RUNNING", uid)
        transition(db_session, e, "COMPLETED", uid)
        with pytest.raises(ExecutionStateError):
            transition(db_session, e, "RETRYING", uid)

    def test_unknown_status_rejected(self, db_session):
        from app.services.ai_execution_service import transition, ExecutionStateError
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        with pytest.raises(ExecutionStateError):
            transition(db_session, e, "SOMEDAY", uid)

    def test_approval_flow(self, db_session):
        from app.services.ai_execution_service import transition
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition(db_session, e, "PLANNING", uid)
        transition(db_session, e, "WAITING_APPROVAL", uid)
        transition(db_session, e, "RUNNING", uid)
        assert e.status == "RUNNING"

    def test_tool_wait_flow(self, db_session):
        from app.services.ai_execution_service import transition
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition(db_session, e, "RUNNING", uid)
        transition(db_session, e, "WAITING_TOOL", uid)
        transition(db_session, e, "RUNNING", uid)
        assert e.status == "RUNNING"

    def test_complete_sets_output_and_tokens(self, db_session):
        from app.services.ai_execution_service import complete_execution
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition_running(db_session, e, uid)
        complete_execution(db_session, e, uid, output_reference="artifact:1",
                           actual_cost=0.05, input_tokens=100, output_tokens=50)
        assert e.output_reference == "artifact:1"
        assert e.actual_cost == 0.05
        assert e.total_tokens == 150

    def test_fail_retryable_goes_retrying(self, db_session):
        from app.services.ai_execution_service import fail_execution
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition_running(db_session, e, uid)
        fail_execution(db_session, e, uid, "temporary", retryable=True)
        assert e.status == "RETRYING"
        assert e.retry_count == 1

    def test_fail_after_max_retries_failed(self, db_session):
        from app.services.ai_execution_service import fail_execution
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        for _ in range(4):
            transition_running(db_session, e, uid)
            fail_execution(db_session, e, uid, "temporary", retryable=True)
        assert e.status == "FAILED"
        assert e.failure_reason == "temporary"

    def test_fail_three_retries_then_failed(self, db_session):
        from app.services.ai_execution_service import fail_execution
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        for _ in range(3):
            transition_running(db_session, e, uid)
            fail_execution(db_session, e, uid, "transient", retryable=True)
        assert e.status == "RETRYING"
        assert e.retry_count == 3
        transition_running(db_session, e, uid)
        fail_execution(db_session, e, uid, "transient", retryable=True)
        assert e.status == "FAILED"

    def test_fail_nonretryable_failed(self, db_session):
        from app.services.ai_execution_service import fail_execution
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition_running(db_session, e, uid)
        fail_execution(db_session, e, uid, "bad input", retryable=False)
        assert e.status == "FAILED"

    def test_cancel_running(self, db_session):
        from app.services.ai_execution_service import cancel_execution
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition_running(db_session, e, uid)
        cancel_execution(db_session, e, uid)
        assert e.status == "CANCELLED"


def transition_running(db, e, actor):
    from app.services.ai_execution_service import transition
    if e.status in ("QUEUED", "RETRYING"):
        transition(db, e, "RUNNING", actor)


class TestExecutionIdempotency:
    def test_same_key_same_payload_single_execution(self, db_session):
        ws, uid = fresh_ws(), fresh_user()
        key = f"idem-{uuid.uuid4().hex[:12]}"
        a = make_execution(db_session, ws, uid, payload={"q": 1}, idempotency_key=key)
        b = make_execution(db_session, ws, uid, payload={"q": 1}, idempotency_key=key)
        assert a.id == b.id

    def test_same_key_different_payload_conflict(self, db_session):
        from app.services.ai_execution_service import IdempotencyConflict
        ws, uid = fresh_ws(), fresh_user()
        key = f"idem-{uuid.uuid4().hex[:12]}"
        make_execution(db_session, ws, uid, payload={"q": 1}, idempotency_key=key)
        with pytest.raises(IdempotencyConflict):
            make_execution(db_session, ws, uid, payload={"q": 2}, idempotency_key=key)

    def test_same_key_different_tenant_ok(self, db_session):
        key = f"idem-{uuid.uuid4().hex[:12]}"
        ws1, ws2, uid = fresh_ws(), fresh_ws(), fresh_user()
        a = make_execution(db_session, ws1, uid, payload={"q": 1}, idempotency_key=key)
        b = make_execution(db_session, ws2, uid, payload={"q": 2}, idempotency_key=key)
        assert a.id != b.id

    def test_idempotency_record_persisted(self, db_session):
        from app.models.phase15 import AIExecutionIdempotency
        ws, uid = fresh_ws(), fresh_user()
        key = f"idem-{uuid.uuid4().hex[:12]}"
        e = make_execution(db_session, ws, uid, payload={}, idempotency_key=key)
        record = db_session.query(AIExecutionIdempotency).filter(
            AIExecutionIdempotency.idempotency_key == key
        ).first()
        assert record is not None
        assert record.execution_id == e.id
        assert record.expires_at is not None


class TestExecutionCheckpoints:
    def test_save_and_latest(self, db_session):
        from app.services.ai_execution_service import save_checkpoint, latest_checkpoint
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        save_checkpoint(db_session, e, 1, state={"step": 1})
        save_checkpoint(db_session, e, 2, state={"step": 2}, artifact_reference="art:1")
        latest = latest_checkpoint(db_session, e)
        assert latest.step_number == 2
        assert latest.artifact_reference == "art:1"

    def test_checksum_stable(self, db_session):
        from app.services.ai_execution_service import save_checkpoint
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        c1 = save_checkpoint(db_session, e, 1, state={"a": 1})
        c2 = save_checkpoint(db_session, e, 1, state={"a": 1})
        assert c1.checksum == c2.checksum

    def test_resume_from_retrying(self, db_session):
        from app.services.ai_execution_service import (
            save_checkpoint, resume_from_checkpoint, fail_execution,
        )
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition_running(db_session, e, uid)
        save_checkpoint(db_session, e, 1, state={"cursor": "page-3"})
        fail_execution(db_session, e, uid, "transient", retryable=True)
        e2, checkpoint = resume_from_checkpoint(db_session, e, uid)
        assert e2.status == "RUNNING"
        assert checkpoint is not None

    def test_resume_terminal_rejected(self, db_session):
        from app.services.ai_execution_service import resume_from_checkpoint, ExecutionStateError
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        transition_running(db_session, e, uid)
        from app.services.ai_execution_service import complete_execution
        complete_execution(db_session, e, uid)
        with pytest.raises(ExecutionStateError):
            resume_from_checkpoint(db_session, e, uid)


class TestExecutionQueue:
    def test_queue_orders_by_priority(self, db_session):
        ws, uid = fresh_ws(), fresh_user()
        low = make_execution(db_session, ws, uid, priority="LOW", task_type="a")
        critical = make_execution(db_session, ws, uid, priority="CRITICAL", task_type="b")
        normal = make_execution(db_session, ws, uid, priority="NORMAL", task_type="c")
        from app.services.ai_execution_service import queue_order
        order = queue_order(db_session, workspace_id=ws)
        ids = [e.id for e in order]
        assert ids.index(critical.id) < ids.index(normal.id) < ids.index(low.id)

    def test_queue_scoped_to_workspace(self, db_session):
        ws1, ws2, uid = fresh_ws(), fresh_ws(), fresh_user()
        make_execution(db_session, ws1, uid, task_type="x")
        make_execution(db_session, ws2, uid, task_type="y")
        from app.services.ai_execution_service import queue_order
        assert len(queue_order(db_session, workspace_id=ws1)) == 1

    def test_background_lowest(self, db_session):
        ws, uid = fresh_ws(), fresh_user()
        make_execution(db_session, ws, uid, priority="BACKGROUND", task_type="a")
        high = make_execution(db_session, ws, uid, priority="HIGH", task_type="b")
        from app.services.ai_execution_service import queue_order
        order = queue_order(db_session, workspace_id=ws)
        assert order[0].id == high.id

    def test_recover_stale(self, db_session):
        from app.services.ai_execution_service import recover_stale_executions
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        e.started_at = datetime.now(timezone.utc) - timedelta(hours=5)
        db_session.flush()
        recovered = recover_stale_executions(db_session, uid, stale_after=3600)
        assert len(recovered) == 1
        assert recovered[0].status == "RETRYING"

    def test_hard_timeout(self, db_session):
        from app.services.ai_execution_service import timeout_hard
        ws, uid = fresh_ws(), fresh_user()
        e = make_execution(db_session, ws, uid)
        e.started_at = datetime.now(timezone.utc) - timedelta(days=30)
        db_session.flush()
        overdue = timeout_hard(db_session, uid, timeout=86400)
        assert len(overdue) == 1
        assert overdue[0].status == "TIMED_OUT"


# ============================================================
# 2. Knowledge Event Outbox
# ============================================================

class TestEventOutbox:
    def test_emit_event(self, db_session):
        from app.services.event_bus import emit_event
        ws = fresh_ws()
        event = emit_event(db_session, ws, "DOCUMENT_READY", "document", 1,
                           payload={"document_id": 1})
        assert event.status == "PENDING"
        assert event.event_type == "DOCUMENT_READY"

    def test_unknown_event_type_rejected(self, db_session):
        from app.services.event_bus import emit_event, UnknownEventTypeError
        with pytest.raises(UnknownEventTypeError):
            emit_event(db_session, fresh_ws(), "ALIEN_LANDING")

    def test_emit_deduplicated(self, db_session):
        from app.services.event_bus import emit_event
        from app.models.phase15 import KnowledgeEvent
        ws = fresh_ws()
        a = emit_event(db_session, ws, "DOCUMENT_UPDATED", "document", 42, dedupe_key="k1")
        b = emit_event(db_session, ws, "DOCUMENT_UPDATED", "document", 42, dedupe_key="k1")
        assert a.id == b.id
        count = db_session.query(KnowledgeEvent).filter(
            KnowledgeEvent.event_type == "DOCUMENT_UPDATED",
            KnowledgeEvent.aggregate_id == 42,
            KnowledgeEvent.dedupe_key == "k1",
        ).count()
        assert count == 1

    def test_same_event_different_dedupe_ok(self, db_session):
        from app.services.event_bus import emit_event
        ws = fresh_ws()
        a = emit_event(db_session, ws, "DOCUMENT_UPDATED", "document", 42, dedupe_key="k1")
        b = emit_event(db_session, ws, "DOCUMENT_UPDATED", "document", 42, dedupe_key="k2")
        assert a.id != b.id

    def test_process_without_handlers_marks_processed(self, db_session):
        from app.services.event_bus import emit_event, process_pending_events
        ws = fresh_ws()
        emit_event(db_session, ws, "DOCUMENT_UPLOADED", "document", 7)
        summary = process_pending_events(db_session)
        assert summary["skipped"] == 1
        assert summary["processed"] == 0

    def test_event_types_cover_catalog(self):
        from app.models.phase15 import EVENT_TYPES
        assert "DOCUMENT_READY" in EVENT_TYPES
        assert "POLICY_CONFLICT_DETECTED" in EVENT_TYPES
        assert "DEADLINE_OVERDUE" in EVENT_TYPES

    def test_retry_dead_events(self, db_session):
        from app.models.phase15 import KnowledgeEvent
        from app.services.event_bus import emit_event, retry_dead_events
        ws = fresh_ws()
        event = emit_event(db_session, ws, "DOCUMENT_READY", "document", 99)
        event.status = "DEAD"
        event.retry_count = 5
        db_session.flush()
        count = retry_dead_events(db_session)
        assert count == 1
        db_session.refresh(event)
        assert event.status == "FAILED"
        assert event.retry_count == 0

    def test_event_summary(self, db_session):
        from app.services.event_bus import emit_event, event_summary
        ws = fresh_ws()
        emit_event(db_session, ws, "DOCUMENT_READY", "document", 1)
        emit_event(db_session, ws, "DOCUMENT_UPDATED", "document", 2)
        summary = event_summary(db_session, workspace_id=ws)
        assert summary["PENDING"] == 2

    def test_event_summary_tenant_scoped(self, db_session):
        from app.services.event_bus import emit_event, event_summary
        ws1, ws2 = fresh_ws(), fresh_ws()
        emit_event(db_session, ws1, "DOCUMENT_READY", "document", 1)
        emit_event(db_session, ws2, "DOCUMENT_READY", "document", 1)
        assert event_summary(db_session, workspace_id=ws1)["PENDING"] == 1


class TestEventHandlers:
    def test_document_ready_handler_notifies(self, db_session):
        from app.services.event_bus import emit_event, process_pending_events
        from app.services.event_handlers import register_default_handlers
        from app.models.notification import Notification
        register_default_handlers()
        ws, uid = fresh_ws(), fresh_user()
        emit_event(db_session, ws, "DOCUMENT_READY", "document", 1,
                   payload={"user_id": uid, "document_id": 1, "filename": "x.pdf"})
        summary = process_pending_events(db_session)
        assert summary["processed"] == 1
        note = db_session.query(Notification).filter(
            Notification.notification_type == "document_ready",
            Notification.user_id == uid,
        ).first()
        assert note is not None
        assert note.user_id == uid

    def test_handler_idempotent(self, db_session):
        from app.services.event_bus import emit_event, process_pending_events
        from app.services.event_handlers import register_default_handlers
        from app.models.notification import Notification
        register_default_handlers()
        ws, uid = fresh_ws(), fresh_user()
        emit_event(db_session, ws, "DOCUMENT_READY", "document", 11,
                   payload={"user_id": uid, "document_id": 11})
        process_pending_events(db_session)
        # Re-processing the same event must not create a second notification.
        from app.services.event_bus import process_pending_events as ppe2
        ppe2(db_session)
        notes = db_session.query(Notification).filter(
            Notification.notification_type == "document_ready",
            Notification.user_id == uid,
        ).all()
        assert len(notes) == 1

    def test_deadline_handler_notifies(self, db_session):
        from app.services.event_bus import emit_event, process_pending_events
        from app.services.event_handlers import register_default_handlers
        from app.models.notification import Notification
        register_default_handlers()
        ws, uid = fresh_ws(), fresh_user()
        emit_event(db_session, ws, "DEADLINE_APPROACHING", "deadline", 5,
                   payload={"user_id": uid, "title": "Renewal", "due_date": "2026-10-01"})
        process_pending_events(db_session)
        note = db_session.query(Notification).filter(
            Notification.notification_type == "deadline_reminder",
            Notification.user_id == uid,
        ).first()
        assert note is not None

    def test_policy_conflict_handler_creates_review(self, db_session):
        from app.services.event_bus import emit_event, process_pending_events
        from app.services.event_handlers import register_default_handlers
        from app.models.phase15 import ReviewItem
        register_default_handlers()
        ws = fresh_ws()
        emit_event(db_session, ws, "POLICY_CONFLICT_DETECTED", "policy_conflict", 3,
                   payload={"description": "Two policies conflict"})
        process_pending_events(db_session)
        item = db_session.query(ReviewItem).filter(
            ReviewItem.source_type == "policy_conflict"
        ).first()
        assert item is not None
        assert item.item_type == "POLICY_CONFLICT"
        assert item.status == "PENDING"

    def test_gap_handler_creates_suggestion(self, db_session):
        from app.services.event_bus import emit_event, process_pending_events
        from app.services.event_handlers import register_default_handlers
        from app.models.ai_action import AISuggestion
        register_default_handlers()
        ws, uid = fresh_ws(), fresh_user()
        emit_event(db_session, ws, "KNOWLEDGE_GAP_DETECTED", "knowledge_gap", 9,
                   payload={"user_id": uid, "title": "Missing policy owner"})
        process_pending_events(db_session)
        suggestion = db_session.query(AISuggestion).filter(
            AISuggestion.source_type == "knowledge_gap"
        ).first()
        assert suggestion is not None

    def test_failing_handler_retries_with_backoff(self, db_session, monkeypatch):
        from app.services.event_bus import emit_event, process_pending_events
        from app.services import event_bus as eb
        ws = fresh_ws()
        emit_event(db_session, ws, "DOCUMENT_READY", "document", 4, payload={"user_id": fresh_user()})

        def boom(db, event):
            raise RuntimeError("transient")

        monkeypatch.setattr(eb, "_handlers", {"DOCUMENT_READY": [boom]})
        summary = process_pending_events(db_session)
        assert summary["failed"] == 1
        from app.models.phase15 import KnowledgeEvent
        event = db_session.query(KnowledgeEvent).filter(
            KnowledgeEvent.aggregate_id == 4
        ).first()
        assert event.status == "FAILED"
        assert event.retry_count == 1
        assert event.next_retry_at is not None

    def test_handler_dead_lettered_after_retries(self, db_session, monkeypatch):
        from app.services.event_bus import emit_event, process_pending_events, MAX_RETRIES
        from app.services import event_bus as eb
        from app.models.phase15 import KnowledgeEvent
        ws = fresh_ws()
        event = emit_event(db_session, ws, "DOCUMENT_READY", "document", 5, payload={"user_id": fresh_user()})

        def boom(db, event):
            raise RuntimeError("always fails")

        monkeypatch.setattr(eb, "_handlers", {"DOCUMENT_READY": [boom]})
        for i in range(MAX_RETRIES + 1):
            # Advance the clock past the exponential backoff each round.
            process_pending_events(db_session, now=datetime.now(timezone.utc) + timedelta(minutes=10 * (i + 1)))
        db_session.refresh(event)
        assert event.status == "DEAD"


# ============================================================
# 3. Knowledge Engine (changes / impact / policy / temporal)
# ============================================================

class TestKnowledgeChangeDetection:
    def test_detect_change_records(self, db_session, monkeypatch):
        from app.services.knowledge_engine import detect_and_record_change
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, title="Change Doc", user_id=uid)
        from app.services import knowledge_engine as ke
        monkeypatch.setattr(ke, "_text_for_version", lambda db, d, v: (
            "Old policy text\n" if v == 1 else "Old policy text\nNew policy: employees must comply.\n"
        ))
        change = detect_and_record_change(db_session, ws, doc.id, 1, 2)
        assert change is not None
        assert change.change_type == "POLICY_CHANGE"
        assert change.severity in ("HIGH", "MEDIUM")
        assert change.old_evidence_json is not None
        assert change.new_evidence_json is not None

    def test_no_diff_no_change(self, db_session, monkeypatch):
        from app.services.knowledge_engine import detect_and_record_change
        from app.services import knowledge_engine as ke
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, user_id=uid)
        monkeypatch.setattr(ke, "_text_for_version", lambda db, d, v: "identical text\n")
        change = detect_and_record_change(db_session, ws, doc.id, 1, 2)
        assert change is None  # never fabricate a difference

    def test_change_type_enumeration(self):
        from app.services.knowledge_engine import CHANGE_TYPES
        for t in ("CONTENT_CHANGE", "POLICY_CHANGE", "NUMERIC_CHANGE", "ENTITY_CHANGE",
                  "STRUCTURAL_CHANGE", "DATE_CHANGE"):
            assert t in CHANGE_TYPES

    def test_list_changes_scoped(self, db_session, monkeypatch):
        from app.services.knowledge_engine import detect_and_record_change, list_changes
        from app.services import knowledge_engine as ke
        ws1, ws2, uid = fresh_ws(), fresh_ws(), fresh_user()
        doc1 = make_doc(db_session, ws1, user_id=uid)
        doc2 = make_doc(db_session, ws2, user_id=uid)
        monkeypatch.setattr(ke, "_text_for_version", lambda db, d, v: "a\n" if v == 1 else "a\nb\n")
        detect_and_record_change(db_session, ws1, doc1.id, 1, 2)
        detect_and_record_change(db_session, ws2, doc2.id, 1, 2)
        assert len(list_changes(db_session, ws1)) == 1


class TestImpactGraph:
    def test_build_impact_links_explicit_collection(self, db_session):
        from app.services.knowledge_engine import build_impact_links
        from app.models.collection import Collection, collection_documents
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, user_id=uid)
        collection = Collection(user_id=uid, workspace_id=ws, name="Policies")
        db_session.add(collection)
        db_session.flush()
        db_session.execute(collection_documents.insert().values(
            collection_id=collection.id, document_id=doc.id))
        result = build_impact_links(db_session, ws, doc.id)
        direct_types = [d["type"] for d in result["direct_impacts"]]
        assert "collection" in direct_types

    def test_impact_links_labeled(self, db_session):
        from app.services.knowledge_engine import build_impact_links
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, user_id=uid)
        result = build_impact_links(db_session, ws, doc.id)
        for key in ("direct_impacts", "indirect_impacts", "uncertain_impacts"):
            assert key in result


class TestPolicyIntelligence:
    def test_extract_policy_statements(self, db_session):
        from app.services.knowledge_engine import extract_policy_statements
        from app.models.document_content import DocumentContent
        from app.services import knowledge_engine as ke
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, user_id=uid)
        db_session.add(DocumentContent(document_id=doc.id, extracted_text=(
            "All travel above $500 requires approval. "
            "No approval is required for expenses below $200. "
            "Employees must submit timesheets weekly."
        )))
        db_session.flush()
        ke._text_for_version = lambda db, d, v: (
            "All travel above $500 requires approval. No approval is required "
            "for expenses below $200. Employees must submit timesheets weekly."
        )
        statements = extract_policy_statements(db_session, ws, doc.id)
        assert len(statements) >= 2
        types = {s.requirement_type for s in statements}
        assert "approval" in types or "obligation" in types

    def test_policy_conflict_detected(self, db_session, monkeypatch):
        from app.services.knowledge_engine import (
            extract_policy_statements, detect_policy_conflicts,
        )
        from app.services import knowledge_engine as ke
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, user_id=uid)
        monkeypatch.setattr(ke, "_text_for_version", lambda db, d, v: (
            "Approval is required for expenses above $500.\n"
            "No approval is required for expenses below $600.\n"
        ))
        extract_policy_statements(db_session, ws, doc.id)
        conflicts = detect_policy_conflicts(db_session, ws)
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == "NUMERIC_THRESHOLD"
        assert conflicts[0].severity == "HIGH"

    def test_no_false_conflict_when_consistent(self, db_session, monkeypatch):
        from app.services.knowledge_engine import (
            extract_policy_statements, detect_policy_conflicts,
        )
        from app.services import knowledge_engine as ke
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, user_id=uid)
        monkeypatch.setattr(ke, "_text_for_version", lambda db, d, v: (
            "Approval is required for expenses above $500.\n"
            "No approval is required for expenses below $200.\n"
        ))
        extract_policy_statements(db_session, ws, doc.id)
        assert detect_policy_conflicts(db_session, ws) == []  # no overlap — consistent

    def test_no_superficial_text_conflict(self, db_session, monkeypatch):
        from app.services.knowledge_engine import (
            extract_policy_statements, detect_policy_conflicts,
        )
        from app.services import knowledge_engine as ke
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, user_id=uid)
        # Same threshold — no conflict even though phrasing differs slightly.
        monkeypatch.setattr(ke, "_text_for_version", lambda db, d, v: (
            "Approval is required for expenses above $500.\n"
            "Approvals are mandatory above the $500 threshold.\n"
        ))
        extract_policy_statements(db_session, ws, doc.id)
        assert detect_policy_conflicts(db_session, ws) == []

    def test_conflict_deduplicated(self, db_session, monkeypatch):
        from app.services.knowledge_engine import (
            extract_policy_statements, detect_policy_conflicts,
        )
        from app.services import knowledge_engine as ke
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, user_id=uid)
        monkeypatch.setattr(ke, "_text_for_version", lambda db, d, v: (
            "Approval is required for expenses above $500.\n"
            "No approval is required for expenses below $600.\n"
        ))
        extract_policy_statements(db_session, ws, doc.id)
        first = detect_policy_conflicts(db_session, ws)
        assert len(first) == 1
        # Already-recorded conflicts are never duplicated.
        assert detect_policy_conflicts(db_session, ws) == []


class TestTemporalKnowledge:
    def test_record_current_fact(self, db_session):
        from app.services.knowledge_engine import record_temporal_fact, current_facts
        ws = fresh_ws()
        record_temporal_fact(db_session, ws, "policy_effective", "2026 policy",
                             datetime(2026, 1, 1, tzinfo=timezone.utc), source="doc:1")
        facts = current_facts(db_session, ws)
        assert len(facts) == 1
        assert facts[0].fact_value == "2026 policy"

    def test_supersede_old_fact(self, db_session):
        from app.services.knowledge_engine import record_temporal_fact, current_facts, facts_as_of
        ws = fresh_ws()
        old = record_temporal_fact(db_session, ws, "policy_effective", "2025 policy",
                                   datetime(2025, 1, 1, tzinfo=timezone.utc), source="doc:1")
        record_temporal_fact(db_session, ws, "policy_effective", "2026 policy",
                             datetime(2026, 1, 1, tzinfo=timezone.utc), source="doc:2")
        # Now: only the new fact is current.
        current = current_facts(db_session, ws)
        assert [f.fact_value for f in current] == ["2026 policy"]
        # In 2025 the old fact was valid.
        past = facts_as_of(db_session, ws, datetime(2025, 6, 1, tzinfo=timezone.utc))
        assert any(f.fact_value == "2025 policy" for f in past)

    def test_stale_never_current(self, db_session):
        from app.services.knowledge_engine import record_temporal_fact, current_facts
        ws = fresh_ws()
        record_temporal_fact(db_session, ws, "amount", "100",
                             datetime(2025, 1, 1, tzinfo=timezone.utc),
                             valid_until=datetime(2025, 12, 31, tzinfo=timezone.utc))
        assert current_facts(db_session, ws) == []

    def test_facts_scoped_to_workspace(self, db_session):
        from app.services.knowledge_engine import record_temporal_fact, current_facts
        ws1, ws2 = fresh_ws(), fresh_ws()
        record_temporal_fact(db_session, ws1, "amount", "10",
                             datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert current_facts(db_session, ws2) == []


class TestKnowledgeSnapshots:
    def test_create_snapshot(self, db_session):
        from app.services.knowledge_engine import create_snapshot, list_snapshots
        ws, uid = fresh_ws(), fresh_user()
        make_doc(db_session, ws, user_id=uid)
        snapshot = create_snapshot(db_session, ws, uid, name="weekly")
        assert snapshot.snapshot_type == "workspace"
        data = json.loads(snapshot.data_json)
        assert "health" in data
        assert "documents" in data
        assert data["documents"][0]["filename"]

    def test_snapshot_list_scoped(self, db_session):
        from app.services.knowledge_engine import create_snapshot, list_snapshots
        ws1, ws2, uid = fresh_ws(), fresh_ws(), fresh_user()
        create_snapshot(db_session, ws1, uid)
        create_snapshot(db_session, ws2, uid)
        assert len(list_snapshots(db_session, ws1)) == 1

    def test_snapshot_reproducible(self, db_session):
        from app.services.knowledge_engine import create_snapshot
        ws, uid = fresh_ws(), fresh_user()
        a = create_snapshot(db_session, ws, uid, name="s1")
        b = create_snapshot(db_session, ws, uid, name="s2")
        data_a = json.loads(a.data_json)
        data_b = json.loads(b.data_json)
        assert set(data_a.keys()) == set(data_b.keys())


# ============================================================
# 4. Memory Governance
# ============================================================

class TestMemoryService:
    def test_store_memory(self, db_session):
        from app.services.memory_service import store_memory
        ws, uid = fresh_ws(), fresh_user()
        m = store_memory(db_session, ws, uid, "WORKSPACE_FACT", "Remote work allowed",
                         scope="WORKSPACE", confidence="HIGH")
        assert m.scope == "WORKSPACE"
        assert m.confidence == "HIGH"

    def test_invalid_type_rejected(self, db_session):
        from app.services.memory_service import store_memory, MemoryValidationError
        with pytest.raises(MemoryValidationError):
            store_memory(db_session, fresh_ws(), fresh_user(), "ALIEN", "x")

    def test_empty_content_rejected(self, db_session):
        from app.services.memory_service import store_memory, MemoryValidationError
        with pytest.raises(MemoryValidationError):
            store_memory(db_session, fresh_ws(), fresh_user(), "WORKSPACE_FACT", "   ")

    def test_user_memory_requires_user_scope(self, db_session):
        from app.services.memory_service import store_memory
        ws, uid = fresh_ws(), fresh_user()
        m = store_memory(db_session, ws, uid, "USER_PREFERENCE", "concise answers", scope="USER")
        assert m.user_id == uid

    def test_dedupe_on_same_content(self, db_session):
        from app.services.memory_service import store_memory
        from app.models.phase15 import AIMemory
        ws, uid = fresh_ws(), fresh_user()
        store_memory(db_session, ws, uid, "WORKSPACE_FACT", "the sky is blue")
        store_memory(db_session, ws, uid, "WORKSPACE_FACT", "the sky is blue")
        count = db_session.query(AIMemory).filter(
            AIMemory.content == "the sky is blue"
        ).count()
        assert count == 1

    def test_list_memories_user_scope_isolation(self, db_session):
        from app.services.memory_service import store_memory, list_memories
        ws, uid_a, uid_b = fresh_ws(), fresh_user(), fresh_user()
        store_memory(db_session, ws, uid_a, "USER_PREFERENCE", "secret pref", scope="USER")
        # User B must not see user A's private memory.
        assert list_memories(db_session, ws, user_id=uid_b) == []
        # User A sees it.
        assert len(list_memories(db_session, ws, user_id=uid_a)) == 1

    def test_workspace_memory_visible_to_members(self, db_session):
        from app.services.memory_service import store_memory, list_memories
        ws, uid = fresh_ws(), fresh_user()
        store_memory(db_session, ws, uid, "WORKSPACE_FACT", "shared fact", scope="WORKSPACE")
        assert len(list_memories(db_session, ws, user_id=fresh_user())) == 1

    def test_delete_memory(self, db_session):
        from app.services.memory_service import store_memory, delete_memory
        ws, uid = fresh_ws(), fresh_user()
        m = store_memory(db_session, ws, uid, "WORKSPACE_FACT", "to delete")
        assert delete_memory(db_session, ws, m.id, uid) is True
        assert delete_memory(db_session, ws, m.id, uid) is False

    def test_expire_memories(self, db_session):
        from app.services.memory_service import store_memory, expire_memories
        ws, uid = fresh_ws(), fresh_user()
        store_memory(db_session, ws, uid, "WORKSPACE_FACT", "old", scope="WORKSPACE",
                     expires_at=datetime.now(timezone.utc) - timedelta(days=1))
        store_memory(db_session, ws, uid, "WORKSPACE_FACT", "fresh", scope="WORKSPACE",
                     expires_at=datetime.now(timezone.utc) + timedelta(days=1))
        assert expire_memories(db_session) == 1

    def test_export_memories_no_cross_tenant(self, db_session):
        from app.services.memory_service import store_memory, export_memories
        ws1, ws2, uid = fresh_ws(), fresh_ws(), fresh_user()
        store_memory(db_session, ws1, uid, "WORKSPACE_FACT", "private to ws1")
        store_memory(db_session, ws2, uid, "WORKSPACE_FACT", "private to ws2")
        exported = export_memories(db_session, ws1)
        assert len(exported) == 1
        assert exported[0]["content"] == "private to ws1"

    def test_reset_workspace_memory(self, db_session):
        from app.services.memory_service import store_memory, reset_workspace_memory
        ws, uid = fresh_ws(), fresh_user()
        store_memory(db_session, ws, uid, "WORKSPACE_FACT", "a")
        store_memory(db_session, ws, uid, "WORKSPACE_FACT", "b")
        assert reset_workspace_memory(db_session, ws, uid) == 2

    def test_memory_summary_counts(self, db_session):
        from app.services.memory_service import store_memory, memory_summary
        ws, uid = fresh_ws(), fresh_user()
        store_memory(db_session, ws, uid, "WORKSPACE_FACT", "a")
        store_memory(db_session, ws, uid, "DECISION", "approved x")
        summary = memory_summary(db_session, ws)
        assert summary["total"] == 2


# ============================================================
# 5. Claim Validator + Answer Repair
# ============================================================

class TestClaimValidator:
    def test_supported_claim(self):
        from app.services.claim_validator import validate_claims
        evidence = ["The budget for 2026 is exactly 50000 dollars per quarter."]
        result = validate_claims("The 2026 budget is 50000 dollars per quarter.", evidence)
        assert result.claims[0].status == "SUPPORTED"
        assert result.claims[0].confidence in ("HIGH", "MEDIUM")

    def test_unsupported_claim(self):
        from app.services.claim_validator import validate_claims
        evidence = ["The 2026 budget is 50000 dollars per quarter."]
        result = validate_claims("The CEO resigned in March 2025.", evidence)
        assert result.claims[0].status == "UNSUPPORTED"

    def test_partial_support(self):
        from app.services.claim_validator import validate_claims
        evidence = ["The 2026 budget is fifty thousand dollars."]
        result = validate_claims(
            "The 2026 budget is fifty thousand dollars however operations in Brazil "
            "were separately funded by the parent group.",
            evidence,
        )
        assert result.claims[0].status == "PARTIALLY_SUPPORTED"

    def test_contradicted_claim(self):
        from app.services.claim_validator import validate_claims
        evidence = ["No approval is required for expenses below 500 dollars."]
        result = validate_claims(
            "Employees are required to obtain approvals for every expense.",
            evidence,
        )
        assert result.claims[0].status == "CONTRADICTED"

    def test_claim_status_enum(self):
        from app.services.claim_validator import CLAIM_STATUSES
        assert set(CLAIM_STATUSES) == {
            "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNKNOWN",
        }

    def test_unsupported_not_presented_as_fact(self):
        from app.services.claim_validator import validate_claims, final_answer_text
        evidence = []
        result = validate_claims("The moon is made of cheese.", evidence)
        assert result.unsupported_count == 1
        rendered = final_answer_text(result)
        # The unsupported claim itself is never rendered as an established fact.
        assert "moon is made of cheese" not in rendered
        assert "could not be verified" in rendered

    def test_repair_removes_unsupported(self):
        from app.services.claim_validator import repair_answer
        evidence = ["The annual report was published on 2026-03-15."]
        result = repair_answer(
            "The annual report was published on 2026-03-15. The company opened offices in Mars.",
            evidence,
        )
        assert result.repaired is True
        assert result.repair_passes >= 1
        for claim in result.claims:
            assert claim.status != "SUPPORTED" or "Mars" not in claim.text

    def test_repair_bounded(self):
        from app.services.claim_validator import repair_answer, MAX_REPAIR_PASSES
        evidence = ["nothing relevant here at all."]
        result = repair_answer("Totally unrelated claim about submarines.", evidence)
        assert result.repair_passes <= MAX_REPAIR_PASSES

    def test_no_reasoning_exposed(self):
        from app.services.claim_validator import validate_claims
        result = validate_claims("The budget is 50000.", ["The budget is 50000."])
        for c in result.claims:
            assert "chain-of-thought" not in c.reason.lower()


# ============================================================
# 6. Context Builder
# ============================================================

class TestContextBuilder:
    def test_budget_drops_low_priority(self):
        from app.services.context_builder import ContextBuilder, ContextBlock
        builder = ContextBuilder(max_chars=50)
        builder.add(ContextBlock("direct_user_request", "user", "urgent request here"))
        builder.add(ContextBlock("historical_context", "history", "x" * 1000))
        assembled = builder.assemble()
        assert assembled["dropped_blocks"] == ["history"]
        assert any(b["label"] == "user" for b in assembled["blocks"])

    def test_priority_order(self):
        from app.services.context_builder import ContextBuilder, ContextBlock
        builder = ContextBuilder(max_chars=100000)
        builder.add(ContextBlock("approved_memory", "memory", "m"))
        builder.add(ContextBlock("direct_user_request", "user", "u"))
        assembled = builder.assemble()
        labels = [b["label"] for b in assembled["blocks"]]
        assert labels == ["user", "memory"]

    def test_token_estimate(self):
        from app.services.context_builder import ContextBuilder, ContextBlock
        builder = ContextBuilder()
        builder.add(ContextBlock("direct_user_request", "x", "word " * 400))
        assembled = builder.assemble()
        assert assembled["estimated_tokens"] > 0

    def test_blocks_carried_sources(self):
        from app.services.context_builder import ContextBlock
        block = ContextBlock("direct_user_request", "doc", "content", source_type="document", source_id=7)
        assert block.source_type == "document"
        assert block.source_id == 7


# ============================================================
# 7. Human Review Queue
# ============================================================

class TestReviewQueue:
    def test_create_review_item(self, db_session):
        from app.services.review_service import create_review_item
        ws, uid = fresh_ws(), fresh_user()
        item = create_review_item(db_session, ws, "POLICY_CONFLICT", "Review conflict", uid)
        assert item.status == "PENDING"
        assert item.due_at is not None

    def test_create_assignee_notified(self, db_session):
        from app.services.review_service import create_review_item
        from app.models.notification import Notification
        ws, assignee = fresh_ws(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "Approve action", fresh_user(),
                                  assignee_id=assignee)
        note = db_session.query(Notification).filter(
            Notification.resource_type == "review_item",
            Notification.resource_id == item.id,
        ).first()
        assert note is not None
        assert note.user_id == assignee

    def test_invalid_priority_rejected(self, db_session):
        from app.services.review_service import create_review_item, ReviewDecisionError
        with pytest.raises(ReviewDecisionError):
            create_review_item(db_session, fresh_ws(), "AI_ACTION", "x", fresh_user(), priority="MAX")

    def test_approve_decision(self, db_session):
        from app.services.review_service import create_review_item, decide_review_item
        ws, uid, decider = fresh_ws(), fresh_user(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "Approve", uid)
        decide_review_item(db_session, ws, item.id, "APPROVED", decider, note="ok")
        assert item.status == "APPROVED"
        assert item.decided_by == decider
        assert item.decided_at is not None

    def test_reject_with_note(self, db_session):
        from app.services.review_service import create_review_item, decide_review_item
        ws, uid, decider = fresh_ws(), fresh_user(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "Reject me", uid)
        decide_review_item(db_session, ws, item.id, "REJECTED", decider, note="unsafe")
        assert item.status == "REJECTED"
        assert item.decision_note == "unsafe"

    def test_double_decision_rejected(self, db_session):
        from app.services.review_service import create_review_item, decide_review_item, ReviewDecisionError
        ws, uid = fresh_ws(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "Once", uid)
        decide_review_item(db_session, ws, item.id, "APPROVED", uid)
        with pytest.raises(ReviewDecisionError):
            decide_review_item(db_session, ws, item.id, "REJECTED", uid)

    def test_invalid_decision_rejected(self, db_session):
        from app.services.review_service import create_review_item, decide_review_item, ReviewDecisionError
        ws, uid = fresh_ws(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "x", uid)
        with pytest.raises(ReviewDecisionError):
            decide_review_item(db_session, ws, item.id, "MAYBE", uid)

    def test_delegate_requires_assignee(self, db_session):
        from app.services.review_service import create_review_item, decide_review_item, ReviewDecisionError
        ws, uid = fresh_ws(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "Delegate", uid)
        with pytest.raises(ReviewDecisionError):
            decide_review_item(db_session, ws, item.id, "DELEGATED", uid)

    def test_delegate_reassigns(self, db_session):
        from app.services.review_service import create_review_item, decide_review_item
        ws, uid, new_owner = fresh_ws(), fresh_user(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "Delegate me", uid)
        decide_review_item(db_session, ws, item.id, "DELEGATED", uid, reassign_to=new_owner)
        assert item.assignee_id == new_owner
        assert item.status == "PENDING"

    def test_escalate_overdue_once(self, db_session):
        from app.services.review_service import create_review_item, escalate_overdue
        ws, uid = fresh_ws(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "Old", uid,
                                  due_at=datetime.now(timezone.utc) - timedelta(hours=2))
        escalated = escalate_overdue(db_session)
        assert len(escalated) == 1
        assert item.escalated is True
        # Never escalates twice.
        assert escalate_overdue(db_session) == []

    def test_sla_default_due(self, db_session):
        from app.services.review_service import create_review_item
        ws, uid = fresh_ws(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "x", uid, priority="CRITICAL")
        assert item.due_at < datetime.now(timezone.utc) + timedelta(hours=8)

    def test_review_summary(self, db_session):
        from app.services.review_service import create_review_item, review_summary
        ws, uid = fresh_ws(), fresh_user()
        create_review_item(db_session, ws, "AI_ACTION", "p1", uid)
        summary = review_summary(db_session, ws)
        assert summary["pending"] >= 1

    def test_decision_audited(self, db_session):
        from app.services.review_service import create_review_item, decide_review_item
        from app.models.audit_log import AuditLog
        ws, uid = fresh_ws(), fresh_user()
        item = create_review_item(db_session, ws, "AI_ACTION", "audit me", uid)
        decide_review_item(db_session, ws, item.id, "APPROVED", uid)
        log = db_session.query(AuditLog).filter(
            AuditLog.event_type == "review", AuditLog.event_action == "decision"
        ).first()
        assert log is not None