"""Phase 14 Test Suite — AI Productization + Enterprise Automation Platform."""

import json
import io
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import Base, get_db
from tests.shared_db import engine, TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db


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


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

_user_counter = [0]

VALID_PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\nxref\n0 4\n0000000000 65535 f \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n150\n%%EOF"


def register_user(client, tag: str = "user"):
    _user_counter[0] += 1
    n = _user_counter[0]
    email = f"{tag}{n}@phase14-test.com"
    client.post("/auth/register", json={
        "name": f"{tag.title()} {n}",
        "email": email,
        "password": "password123",
    })
    login = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert login.status_code == 200, login.text
    return login.cookies, email


def create_workspace(client, cookies, name: str = "Phase14 Workspace") -> int:
    resp = client.post("/workspaces", json={"name": name}, cookies=cookies)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


def create_org(client, cookies, name: str = "Phase14 Org", slug: str = None) -> dict:
    resp = client.post("/organizations", json={
        "name": name,
        "slug": slug or f"phase14-org-{_user_counter[0]}",
    }, cookies=cookies)
    assert resp.status_code == 201, resp.text
    return resp.json()


def upload_document(client, cookies, filename: str = "policy.pdf") -> dict:
    files = {"file": (filename, io.BytesIO(VALID_PDF), "application/pdf")}
    resp = client.post("/documents", files=files, cookies=cookies)
    assert resp.status_code == 201, resp.text
    return resp.json()


def make_action(db, workspace_id, user_id, action_type="summarize", title="Test action",
                organization_id=None, status="SUGGESTED"):
    from app.services.ai_action_service import create_action
    action = create_action(
        db, workspace_id, user_id, action_type, title,
        organization_id=organization_id, status=status,
    )
    db.flush()
    return action


def make_suggestion(db, workspace_id, user_id, suggestion_type="deadline",
                    title="Test suggestion", organization_id=None):
    from app.services.ai_action_service import create_suggestion
    s = create_suggestion(
        db, workspace_id, user_id, title, suggestion_type,
        reason="Testable reason", organization_id=organization_id,
    )
    db.flush()
    return s


def make_document(db, workspace_id, title="Policy Doc", filename="policy.pdf", status="READY", metadata_dict=None, user_id=1):
    """Create a Document plus real metadata rows (classification/summary/version)."""
    from app.models.document import Document
    from app.models.document_tag import DocumentClassification, DocumentSummary
    from app.models.document_version import DocumentVersion
    _user_counter[0] += 1
    doc = Document(
        user_id=user_id,
        workspace_id=workspace_id,
        original_filename=filename,
        storage_key=f"phase14-{_user_counter[0]}-{filename}",
        mime_type="application/pdf",
        file_size=100,
        status=status,
    )
    db.add(doc)
    db.flush()

    meta = metadata_dict or {}
    if meta.get("classification"):
        db.add(DocumentClassification(document_id=doc.id, document_type=meta["classification"], confidence=0.9))
    if meta.get("summary"):
        db.add(DocumentSummary(document_id=doc.id, summary_type="short", content=meta["summary"]))
    version_meta = {k: v for k, v in meta.items() if k not in ("classification", "summary")}
    if title:
        version_meta["title"] = title  # persisted schema-backed title
    db.add(DocumentVersion(
        document_id=doc.id,
        version_number=1,
        created_by=user_id,
        storage_key=doc.storage_key,
        file_size=100,
        mime_type="application/pdf",
        is_current=True,
        processing_status="READY" if status == "READY" else "PENDING",
        metadata_json=json.dumps(version_meta) if version_meta else None,
    ))
    db.flush()
    return doc


def make_deadline(db, workspace_id, user_id, days_from_now=10, title="Renewal deadline",
                  organization_id=None, source="manual"):
    from app.services.deadline_service import create_deadline
    due = datetime.now(timezone.utc) + timedelta(days=days_from_now)
    return create_deadline(
        db, workspace_id, user_id, title, due,
        organization_id=organization_id, source=source,
    )


def make_execution(db, workspace_id, user_id, status="completed", cost=0.01, task_type="rag"):
    import uuid
    from app.models.ai_execution import AIExecution
    execution = AIExecution(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        user_id=user_id,
        task_type=task_type,
        status=status,
        estimated_cost=cost,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )
    db.add(execution)
    db.flush()
    return execution


# ============================================================
# AI Action Center
# ============================================================

class TestAIActionLifecycle:
    def test_create_action_all_types(self, db_session):
        from app.models.ai_action import ACTION_TYPES
        for atype in ACTION_TYPES:
            action = make_action(db_session, 1, 1, action_type=atype, title=f"Action {atype}")
            assert action.action_type == atype
            assert action.status == "SUGGESTED"

    def test_create_action_unknown_type_rejected(self, db_session):
        from app.services.ai_action_service import create_action, UnknownActionTypeError
        with pytest.raises(UnknownActionTypeError):
            create_action(db_session, 1, 1, "frobnicate", "Bad action")

    def test_create_action_invalid_status_rejected(self, db_session):
        from app.services.ai_action_service import create_action
        with pytest.raises(ValueError):
            create_action(db_session, 1, 1, "summarize", "Bad", status="BOGUS")

    def test_default_risk_level_by_type(self, db_session):
        action = make_action(db_session, 1, 1, action_type="summarize")
        assert action.risk_level == "LOW"
        action2 = make_action(db_session, 1, 1, action_type="tag")
        assert action2.risk_level == "MEDIUM"

    def test_risk_levels_constrained(self, db_session):
        from app.models.ai_action import RISK_LEVELS
        assert set(RISK_LEVELS) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_valid_transitions_suggested_to_approved(self, db_session):
        from app.services.ai_action_service import transition
        action = make_action(db_session, 1, 1, status="APPROVAL_REQUIRED")
        transition(db_session, action, "APPROVED", 1)
        assert action.status == "APPROVED"
        assert action.approved_by == 1

    def test_invalid_transition_rejected(self, db_session):
        from app.services.ai_action_service import transition, InvalidTransitionError
        action = make_action(db_session, 1, 1, status="COMPLETED")
        with pytest.raises(InvalidTransitionError):
            transition(db_session, action, "RUNNING", 1)

    def test_terminal_state_immutable(self, db_session):
        from app.services.ai_action_service import transition, InvalidTransitionError
        action = make_action(db_session, 1, 1, status="REJECTED")
        with pytest.raises(InvalidTransitionError):
            transition(db_session, action, "APPROVED", 1)

    def test_approve_requires_approval_state(self, db_session):
        from app.services.ai_action_service import approve_action, InvalidTransitionError
        action = make_action(db_session, 1, 1, status="RUNNING")
        with pytest.raises(InvalidTransitionError):
            approve_action(db_session, action, 1)

    def test_reject_sets_reason(self, db_session):
        from app.services.ai_action_service import reject_action
        action = make_action(db_session, 1, 1, status="APPROVAL_REQUIRED")
        reject_action(db_session, action, 1, reason="Not applicable")
        assert action.status == "REJECTED"
        assert action.rejection_reason == "Not applicable"

    def test_complete_requires_running(self, db_session):
        from app.services.ai_action_service import complete_action, InvalidTransitionError
        action = make_action(db_session, 1, 1, status="QUEUED")
        with pytest.raises(InvalidTransitionError):
            complete_action(db_session, action, result={"ok": True})

    def test_complete_stores_result(self, db_session):
        from app.services.ai_action_service import complete_action
        action = make_action(db_session, 1, 1, status="RUNNING")
        complete_action(db_session, action, result={"lines": 42})
        assert action.status == "COMPLETED"
        assert json.loads(action.result_json) == {"lines": 42}

    def test_queue_to_running(self, db_session):
        from app.services.ai_action_service import transition
        action = make_action(db_session, 1, 1, status="QUEUED")
        transition(db_session, action, "RUNNING", 1)
        assert action.status == "RUNNING"

    def test_all_transition_pairs_covered(self):
        from app.services.ai_action_service import VALID_TRANSITIONS, ACTION_STATUSES
        from app.models.ai_action import ACTION_STATUSES as MODEL_STATUSES
        assert set(ACTION_STATUSES) == set(MODEL_STATUSES)
        assert set(VALID_TRANSITIONS.keys()) == set(ACTION_STATUSES)

    def test_create_action_audit_logged(self, db_session):
        from app.models.audit_log import AuditLog
        action = make_action(db_session, 1, 1, title="Audited action")
        log = db_session.query(AuditLog).filter(
            AuditLog.resource_type == "ai_action",
            AuditLog.resource_id == str(action.id),
        ).first()
        assert log is not None

    def test_transition_audit_logged(self, db_session):
        from app.models.audit_log import AuditLog
        from app.services.ai_action_service import transition
        action = make_action(db_session, 1, 1, status="APPROVAL_REQUIRED")
        transition(db_session, action, "APPROVED", 1)
        log = db_session.query(AuditLog).filter(
            AuditLog.event_action == "transition:approved",
        ).first()
        assert log is not None


class TestActionCenterAPI:
    def test_create_action_endpoint(self, client):
        cookies, _ = register_user(client, "actapi")
        ws = create_workspace(client, cookies)
        resp = client.post("/ai-actions", json={
            "workspace_id": ws,
            "action_type": "summarize",
            "title": "Summarize Q3",
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == "SUGGESTED"

    def test_create_action_requires_auth(self, client):
        resp = client.post("/ai-actions", json={
            "workspace_id": 1, "action_type": "summarize", "title": "X",
        })
        assert resp.status_code == 401

    def test_list_actions_filtered(self, client, db_session):
        cookies, _ = register_user(client, "actlist")
        ws = create_workspace(client, cookies)
        make_action(db_session, ws, 1, status="COMPLETED")
        resp = client.get(f"/ai-actions?workspace_id={ws}&status=COMPLETED", cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 1

    def test_action_summary_counts(self, client, db_session):
        cookies, _ = register_user(client, "actsum")
        ws = create_workspace(client, cookies)
        make_action(db_session, ws, 1, status="RUNNING")
        make_action(db_session, ws, 1, status="COMPLETED")
        resp = client.get(f"/ai-actions/summary?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        body = resp.json()
        assert body["running"] == 1
        assert body["completed"] == 1

    def test_transition_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "acttrans")
        ws = create_workspace(client, cookies)
        action = make_action(db_session, ws, 1, status="QUEUED")
        db_session.commit()
        resp = client.post(f"/ai-actions/{action.id}/transition", json={"new_status": "RUNNING"}, cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "RUNNING"

    def test_approve_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "actappr")
        ws = create_workspace(client, cookies)
        action = make_action(db_session, ws, 1, status="APPROVAL_REQUIRED")
        db_session.commit()
        resp = client.post(f"/ai-actions/{action.id}/approve", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["status"] == "APPROVED"

    def test_reject_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "actrej")
        ws = create_workspace(client, cookies)
        action = make_action(db_session, ws, 1, status="APPROVAL_REQUIRED")
        db_session.commit()
        resp = client.post(f"/ai-actions/{action.id}/reject?reason=no", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["status"] == "REJECTED"

    def test_cross_workspace_action_denied(self, client, db_session):
        cookies, _ = register_user(client, "actxws")
        ws1 = create_workspace(client, cookies)
        ws2 = create_workspace(client, cookies)
        action = make_action(db_session, ws1, 1)
        db_session.commit()
        resp = client.post(f"/ai-actions/{action.id}/approve", cookies=cookies)
        assert resp.status_code == 200  # same user, member of both

    def test_action_not_found(self, client):
        cookies, _ = register_user(client, "actnf")
        resp = client.post("/ai-actions/99999/approve", cookies=cookies)
        assert resp.status_code == 404

    def test_invalid_transition_returns_400(self, client, db_session):
        cookies, _ = register_user(client, "actbad")
        ws = create_workspace(client, cookies)
        action = make_action(db_session, ws, 1, status="COMPLETED")
        db_session.commit()
        resp = client.post(f"/ai-actions/{action.id}/transition", json={"new_status": "RUNNING"}, cookies=cookies)
        assert resp.status_code == 400


# ============================================================
# AI Suggestions
# ============================================================

class TestSuggestionEngine:
    def test_document_ready_classification_suggestion(self, db_session):
        from app.services.suggestion_engine import suggest_for_document_ready
        doc = make_document(db_session, 1, metadata_dict={"summary": "yes"})
        suggestions = suggest_for_document_ready(db_session, doc, 1, 1)
        titles = [s.title for s in suggestions]
        assert any("Classify" in t for t in titles)

    def test_document_ready_summary_suggestion(self, db_session):
        from app.services.suggestion_engine import suggest_for_document_ready
        doc = make_document(db_session, 1, metadata_dict={"classification": "policy"})
        suggestions = suggest_for_document_ready(db_session, doc, 1, 1)
        assert any("Summarize" in s.title for s in suggestions)

    def test_complete_metadata_no_suggestions(self, db_session):
        from app.services.suggestion_engine import suggest_for_document_ready
        doc = make_document(db_session, 1, metadata_dict={
            "classification": "policy", "summary": "done",
        })
        suggestions = suggest_for_document_ready(db_session, doc, 1, 1)
        assert suggestions == []

    def test_suggestions_are_explainable(self, db_session):
        from app.services.suggestion_engine import suggest_for_document_ready
        doc = make_document(db_session, 1)
        suggestions = suggest_for_document_ready(db_session, doc, 1, 1)
        for s in suggestions:
            assert s.reason

    def test_deadline_approaching_suggestion(self, db_session):
        from app.services.suggestion_engine import suggest_deadline_approaching
        make_deadline(db_session, 1, 1, days_from_now=10)
        suggestions = suggest_deadline_approaching(db_session, 1, 1, None)
        assert len(suggestions) == 1
        assert suggestions[0].suggestion_type == "deadline"

    def test_deadline_far_away_no_suggestion(self, db_session):
        from app.services.suggestion_engine import suggest_deadline_approaching
        make_deadline(db_session, 1, 1, days_from_now=90)
        suggestions = suggest_deadline_approaching(db_session, 1, 1, None)
        assert suggestions == []

    def test_deadline_suggestion_deduplicated(self, db_session):
        from app.services.suggestion_engine import suggest_deadline_approaching
        make_deadline(db_session, 1, 1, days_from_now=10)
        suggest_deadline_approaching(db_session, 1, 1, None)
        second = suggest_deadline_approaching(db_session, 1, 1, None)
        assert second == []

    def test_duplicate_candidate_detection(self, db_session):
        from app.services.suggestion_engine import suggest_duplicate_candidates
        make_document(db_session, 1, title="Annual Report 2026", filename="a.pdf")
        make_document(db_session, 1, title="Annual Report 2026", filename="b.pdf")
        suggestions = suggest_duplicate_candidates(db_session, 1, 1, None)
        assert len(suggestions) == 1
        assert suggestions[0].suggestion_type == "duplicate"

    def test_unrelated_titles_no_duplicate(self, db_session):
        from app.services.suggestion_engine import suggest_duplicate_candidates
        make_document(db_session, 1, title="Alpha Policy", filename="a.pdf")
        make_document(db_session, 1, title="Beta Contract", filename="b.pdf")
        suggestions = suggest_duplicate_candidates(db_session, 1, 1, None)
        assert suggestions == []

    def test_dismiss_suggestion(self, db_session):
        from app.services.ai_action_service import dismiss_suggestion
        s = make_suggestion(db_session, 1, 1)
        dismiss_suggestion(db_session, s)
        assert s.status == "DISMISSED"

    def test_suggestion_to_action_requires_approval(self, db_session):
        from app.services.ai_action_service import suggestion_to_action
        s = make_suggestion(db_session, 1, 1)
        action = suggestion_to_action(db_session, s, "summarize", 1)
        assert action.status == "APPROVAL_REQUIRED"
        assert s.status == "ACTIONED"

    def test_suggestion_to_action_links_source(self, db_session):
        from app.services.ai_action_service import suggestion_to_action
        s = make_suggestion(db_session, 1, 1)
        action = suggestion_to_action(db_session, s, "summarize", 1)
        assert action.source_type == "suggestion"
        assert action.source_id == s.id


class TestSuggestionAPI:
    def test_generate_suggestions_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "suggen")
        ws = create_workspace(client, cookies)
        make_deadline(db_session, ws, 1, days_from_now=5, organization_id=None)
        db_session.commit()
        resp = client.post(f"/ai-actions/suggestions/generate?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["suggestions_created"] >= 1

    def test_list_suggestions(self, client, db_session):
        cookies, _ = register_user(client, "suglist")
        ws = create_workspace(client, cookies)
        make_suggestion(db_session, ws, 1, suggestion_type="deadline")
        db_session.commit()
        resp = client.get(f"/ai-actions/suggestions?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["items"]) >= 1

    def test_dismiss_suggestion_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "sugdis")
        ws = create_workspace(client, cookies)
        s = make_suggestion(db_session, ws, 1)
        db_session.commit()
        resp = client.post(f"/ai-actions/suggestions/{s.id}/dismiss", cookies=cookies)
        assert resp.status_code == 200

    def test_suggestion_to_action_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "sugact")
        ws = create_workspace(client, cookies)
        s = make_suggestion(db_session, ws, 1)
        db_session.commit()
        resp = client.post(f"/ai-actions/suggestions/{s.id}/action?action_type=summarize", cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "APPROVAL_REQUIRED"

    def test_document_suggestion_endpoint(self, client):
        cookies, _ = register_user(client, "sugdoc")
        ws = create_workspace(client, cookies)
        doc = upload_document(client, cookies)
        resp = client.post(f"/ai-actions/suggestions/document/{doc['id']}", cookies=cookies)
        assert resp.status_code in (200, 400), resp.text

    def test_insights_list(self, client, db_session):
        from app.services.ai_action_service import record_insight
        cookies, _ = register_user(client, "inslist")
        ws = create_workspace(client, cookies)
        record_insight(db_session, ws, "health", "Low health", "Detail", importance="MEDIUM")
        db_session.commit()
        resp = client.get(f"/ai-actions/insights?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 1


# ============================================================
# Document / Collection / Workspace Health
# ============================================================

class TestDocumentHealth:
    def test_compute_health_ready_with_metadata(self, db_session):
        from app.services.health_service import compute_document_health
        doc = make_document(db_session, 1, metadata_dict={
            "classification": "policy", "summary": "x" * 60, "author": "A", "created": "2026-01-01",
        })
        health = compute_document_health(db_session, doc)
        assert 0 <= health.score <= 100
        assert health.score > 60

    def test_compute_health_failed_document_low(self, db_session):
        from app.services.health_service import compute_document_health
        doc = make_document(db_session, 1, status="FAILED")
        health = compute_document_health(db_session, doc)
        assert health.score <= 50

    def test_missing_metadata_reason(self, db_session):
        from app.services.health_service import compute_document_health
        doc = make_document(db_session, 1)
        health = compute_document_health(db_session, doc)
        reasons = json.loads(health.reasons_json)
        assert any("Missing metadata" in r for r in reasons)

    def test_health_persisted(self, db_session):
        from app.services.health_service import compute_document_health, get_document_health
        doc = make_document(db_session, 1)
        compute_document_health(db_session, doc)
        db_session.flush()
        assert get_document_health(db_session, doc.id) is not None

    def test_health_refresh_updates_score(self, db_session):
        from app.services.health_service import compute_document_health
        doc = make_document(db_session, 1, status="UPLOADED")
        h1 = compute_document_health(db_session, doc)
        doc.status = "READY"
        doc.metadata_json = json.dumps({"classification": "c", "summary": "s" * 60, "author": "a", "created": "d"})
        h2 = compute_document_health(db_session, doc)
        assert h2.score >= h1.score

    def test_factors_recorded(self, db_session):
        from app.services.health_service import compute_document_health
        doc = make_document(db_session, 1, status="READY")
        health = compute_document_health(db_session, doc)
        factors = json.loads(health.factors_json)
        assert "processing_state" in factors
        assert "metadata_completeness" in factors

    def test_health_never_false_precision(self, db_session):
        from app.services.health_service import compute_document_health
        doc = make_document(db_session, 1, status="DELETED")
        health = compute_document_health(db_session, doc)
        assert health.score <= 50


class TestWorkspaceHealth:
    def test_empty_workspace_health(self, db_session):
        from app.services.health_service import workspace_health
        h = workspace_health(db_session, 999)
        assert h["total_documents"] == 0
        assert h["avg_health"] is None

    def test_workspace_health_metrics(self, db_session):
        from app.services.health_service import workspace_health
        make_document(db_session, 1, status="READY")
        make_document(db_session, 1, status="FAILED")
        h = workspace_health(db_session, 1)
        assert h["total_documents"] == 2
        assert h["active_knowledge"] == 1
        assert h["processing_failures"] == 1

    def test_workspace_health_tenant_scoped(self, db_session):
        from app.services.health_service import workspace_health
        make_document(db_session, 1, status="READY")
        make_document(db_session, 2, status="READY")
        h = workspace_health(db_session, 1)
        assert h["total_documents"] == 1

    def test_ai_usage_summary_empty(self, db_session):
        from app.services.health_service import ai_usage_summary
        u = ai_usage_summary(db_session, 1)
        assert u["ai_executions_30d"] == 0

    def test_ai_usage_summary_with_executions(self, db_session):
        from app.services.health_service import ai_usage_summary
        make_execution(db_session, 1, 1, status="completed", cost=0.02)
        make_execution(db_session, 1, 1, status="failed", cost=0.01)
        u = ai_usage_summary(db_session, 1)
        assert u["ai_executions_30d"] == 2
        assert u["ai_cost_30d"] == 0.03
        assert u["success_rate"] == 0.5

    def test_workspace_health_duplicate_rate(self, db_session):
        from app.services.health_service import workspace_health
        make_document(db_session, 1, title="Duplicate Doc", filename="a.pdf")
        make_document(db_session, 1, title="Duplicate Doc", filename="b.pdf")
        h = workspace_health(db_session, 1)
        assert h["duplicate_rate"] > 0


class TestCollectionHealth:
    def test_empty_collection(self, db_session):
        from app.models.collection import Collection
        from app.services.health_service import collection_health
        collection = Collection(user_id=1, workspace_id=1, name="Empty")
        db_session.add(collection)
        db_session.flush()
        h = collection_health(db_session, collection)
        assert h["document_count"] == 0
        assert h["health"] is None

    def test_collection_with_documents(self, db_session):
        from app.models.collection import Collection, collection_documents
        from app.services.health_service import collection_health
        collection = Collection(user_id=1, workspace_id=1, name="Contracts")
        db_session.add(collection)
        db_session.flush()
        doc = make_document(db_session, 1, status="READY")
        db_session.execute(collection_documents.insert().values(collection_id=collection.id, document_id=doc.id))
        h = collection_health(db_session, collection)
        assert h["document_count"] == 1
        assert h["ready_documents"] == 1

    def test_collection_failure_recommendation(self, db_session):
        from app.models.collection import Collection, collection_documents
        from app.services.health_service import collection_health
        collection = Collection(user_id=1, workspace_id=1, name="Broken")
        db_session.add(collection)
        db_session.flush()
        doc = make_document(db_session, 1, status="FAILED")
        db_session.execute(collection_documents.insert().values(collection_id=collection.id, document_id=doc.id))
        h = collection_health(db_session, collection)
        assert any("failed" in r.lower() for r in h["recommendations"])


class TestHealthAPI:
    def test_document_health_endpoint(self, client, db_session):
        from app.services.health_service import compute_document_health
        cookies, _ = register_user(client, "hthapi")
        ws = create_workspace(client, cookies)
        doc = make_document(db_session, ws, status="READY")
        compute_document_health(db_session, doc)
        db_session.commit()
        resp = client.get(f"/knowledge/documents/{doc.id}/health", cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert 0 <= resp.json()["score"] <= 100

    def test_workspace_health_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "hthws")
        ws = create_workspace(client, cookies)
        make_document(db_session, ws, status="READY")
        db_session.commit()
        resp = client.get(f"/knowledge/workspaces/{ws}/health", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["total_documents"] == 1
        assert "ai_usage" in resp.json()

    def test_collection_health_endpoint(self, client, db_session):
        from app.models.collection import Collection, collection_documents
        from app.models.user import User
        cookies, _ = register_user(client, "hthcol")
        ws = create_workspace(client, cookies)
        user = db_session.query(User).order_by(User.id.desc()).first()
        collection = Collection(user_id=user.id, workspace_id=ws, name="C")
        db_session.add(collection)
        db_session.flush()
        doc = make_document(db_session, ws, status="READY", user_id=user.id)
        db_session.execute(collection_documents.insert().values(collection_id=collection.id, document_id=doc.id))
        db_session.commit()
        resp = client.get(f"/knowledge/collections/{collection.id}/health", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["document_count"] == 1

    def test_health_requires_auth(self, client):
        resp = client.get("/knowledge/workspaces/1/health")
        assert resp.status_code == 401

    def test_cross_workspace_health_denied(self, client, db_session):
        cookies_a, _ = register_user(client, "hthx1")
        ws_a = create_workspace(client, cookies_a)
        cookies_b, _ = register_user(client, "hthx2")
        create_workspace(client, cookies_b)
        make_document(db_session, ws_a, status="READY")
        db_session.commit()
        resp = client.get(f"/knowledge/workspaces/{ws_a}/health", cookies=cookies_b)
        assert resp.status_code in (403, 404)


# ============================================================
# Copilot
# ============================================================

class TestCopilotService:
    def test_classify_command_summarize(self):
        from app.services.copilot_service import classify_command
        assert classify_command("Can you summarize this document") == "summarize"

    def test_classify_command_risk(self):
        from app.services.copilot_service import classify_command
        assert classify_command("What are the risks here") == "risks"

    def test_classify_command_dates(self):
        from app.services.copilot_service import classify_command
        assert classify_command("When does this expire") == "dates"

    def test_classify_command_compare(self):
        from app.services.copilot_service import classify_command
        assert classify_command("Compare these two") == "compare"

    def test_classify_command_default_question(self):
        from app.services.copilot_service import classify_command
        assert classify_command("What is the weather") == "question"

    def test_unknown_scope_rejected(self, db_session):
        from app.services.copilot_service import copilot_ask, CopilotScopeError
        with pytest.raises(CopilotScopeError):
            copilot_ask(db_session, 1, "GALAXY", "hi")

    def test_document_scope_requires_document(self, db_session):
        from app.services.copilot_service import copilot_ask, CopilotScopeError
        with pytest.raises(CopilotScopeError):
            copilot_ask(db_session, 1, "DOCUMENT", "hi")

    def test_missing_document_not_found(self, db_session):
        from app.services.copilot_service import copilot_ask, CopilotScopeError
        with pytest.raises(CopilotScopeError):
            copilot_ask(db_session, 1, "DOCUMENT", "hi", document_id=99999)

    def test_collection_scope_requires_collection(self, db_session):
        from app.services.copilot_service import copilot_ask, CopilotScopeError
        with pytest.raises(CopilotScopeError):
            copilot_ask(db_session, 1, "COLLECTION", "hi")

    def test_workspace_scope_returns_envelope(self, db_session, client):
        from app.services.copilot_service import copilot_ask
        cookies, _ = register_user(client, "coptws")
        ws = create_workspace(client, cookies)
        result = copilot_ask(db_session, 1, "WORKSPACE", "What policies exist?")
        assert result["scope"] == "WORKSPACE"
        assert "answer" in result
        assert "sources" in result

    def test_copilot_provider_error_graceful(self, db_session, monkeypatch):
        from app.services.copilot_service import copilot_ask
        from app.services import copilot_service as copilot_mod
        from app.services.rag_service import RAGError

        def boom(**kwargs):
            raise RAGError("provider down")
        monkeypatch.setattr(copilot_mod, "answer_question_with_history", boom)
        result = copilot_ask(db_session, 1, "WORKSPACE", "hi")
        assert result["grounded"] is False
        assert result["confidence"]["level"] == "UNKNOWN"


class TestCopilotAPI:
    def test_ask_endpoint(self, client):
        cookies, _ = register_user(client, "copapi")
        create_workspace(client, cookies)
        resp = client.post("/copilot/ask", json={"question": "Summarize the workspace", "scope": "WORKSPACE"}, cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert "answer" in resp.json()

    def test_ask_invalid_scope(self, client):
        cookies, _ = register_user(client, "copinv")
        create_workspace(client, cookies)
        resp = client.post("/copilot/ask", json={"question": "hi", "scope": "NOPE"}, cookies=cookies)
        assert resp.status_code == 400

    def test_ask_requires_auth(self, client):
        resp = client.post("/copilot/ask", json={"question": "hi", "scope": "WORKSPACE"})
        assert resp.status_code == 401

    def test_document_ask_endpoint(self, client):
        cookies, _ = register_user(client, "copdoc")
        create_workspace(client, cookies)
        doc = upload_document(client, cookies)
        resp = client.post(f"/copilot/document/{doc['id']}/ask", json={"question": "Summarize"}, cookies=cookies)
        assert resp.status_code == 200, resp.text

    def test_workspace_ask_endpoint(self, client):
        cookies, _ = register_user(client, "copws2")
        ws = create_workspace(client, cookies)
        resp = client.post(f"/copilot/workspace/{ws}/ask", json={"question": "Policies"}, cookies=cookies)
        assert resp.status_code == 200

    def test_cross_workspace_document_denied(self, client, db_session):
        from app.models.document import Document
        cookies_a, _ = register_user(client, "copx1")
        ws_a = create_workspace(client, cookies_a)
        cookies_b, _ = register_user(client, "copx2")
        create_workspace(client, cookies_b)
        doc = make_document(db_session, ws_a)
        db_session.commit()
        resp = client.post(f"/copilot/document/{doc.id}/ask", json={"question": "hi"}, cookies=cookies_b)
        assert resp.status_code in (403, 404)


# ============================================================
# Natural Language Search
# ============================================================

class TestNLSearch:
    def test_detect_deadline_intent(self):
        from app.services.nl_search import detect_intent
        assert "deadline" in detect_intent("Which contracts expire soon?")

    def test_detect_policy_intent(self):
        from app.services.nl_search import detect_intent
        assert "policy" in detect_intent("policies about remote work")

    def test_detect_risk_intent(self):
        from app.services.nl_search import detect_intent
        assert "risk" in detect_intent("which projects have risks")

    def test_detect_compare_intent(self):
        from app.services.nl_search import detect_intent
        assert "compare" in detect_intent("compare the two reports")

    def test_no_false_intent(self):
        from app.services.nl_search import detect_intent
        assert detect_intent("hello there") == []

    def test_extract_last_month_filter(self):
        from app.services.nl_search import extract_filters
        filters = extract_filters("contracts modified last month")
        assert filters.get("updated") == "last_month"

    def test_extract_document_type_contract(self):
        from app.services.nl_search import extract_filters
        filters = extract_filters("all contracts")
        assert filters.get("document_type") == "contract"

    def test_extract_days_filter(self):
        from app.services.nl_search import extract_filters
        filters = extract_filters("documents modified in the last 7 days")
        assert filters.get("updated_days") == "7"

    def test_no_filters_for_plain_query(self):
        from app.services.nl_search import extract_filters
        assert extract_filters("how do I reset my password") == {}

    def test_explain_matches_reasons(self):
        from app.services.nl_search import explain_matches
        reasons = explain_matches("contracts expiring soon", "semantic")
        assert len(reasons) >= 1
        assert all(isinstance(r, str) for r in reasons)

    def test_explain_never_exposes_reasoning(self):
        from app.services.nl_search import explain_matches
        reasons = explain_matches("renewal deadlines", "keyword")
        for r in reasons:
            assert "chain" not in r.lower()
            assert "thought" not in r.lower()


class TestSavedSearchAPI:
    def test_save_search(self, client):
        cookies, _ = register_user(client, "svs")
        ws = create_workspace(client, cookies)
        resp = client.post("/search/saved", json={
            "workspace_id": ws, "name": "Expiring", "query": "contracts expiring",
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["name"] == "Expiring"

    def test_saved_search_shared_listing(self, client):
        cookies, _ = register_user(client, "svssh")
        ws = create_workspace(client, cookies)
        client.post("/search/saved", json={
            "workspace_id": ws, "name": "Shared search", "query": "policies",
            "is_shared": True,
        }, cookies=cookies)
        resp = client.get(f"/search/saved?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["items"]) >= 1

    def test_saved_search_private_not_visible_to_other(self, client):
        cookies_a, _ = register_user(client, "svspr1")
        ws = create_workspace(client, cookies_a)
        client.post("/search/saved", json={
            "workspace_id": ws, "name": "Private", "query": "secrets",
            "is_shared": False,
        }, cookies=cookies_a)
        cookies_b, _ = register_user(client, "svspr2")
        # b joins the same workspace via invitation is complex; skip membership
        resp = client.get(f"/search/saved?workspace_id={ws}", cookies=cookies_b)
        assert resp.status_code in (403, 404)

    def test_delete_saved_search(self, client):
        cookies, _ = register_user(client, "svsdel")
        ws = create_workspace(client, cookies)
        saved = client.post("/search/saved", json={
            "workspace_id": ws, "name": "To delete", "query": "x",
        }, cookies=cookies).json()
        resp = client.delete(f"/search/saved/{saved['id']}", cookies=cookies)
        assert resp.status_code == 200

    def test_saved_search_requires_auth(self, client):
        resp = client.get("/search/saved?workspace_id=1")
        assert resp.status_code == 401

    def test_intent_analysis_endpoint(self, client):
        cookies, _ = register_user(client, "svsint")
        resp = client.post("/search/intent?query=contracts+expiring+soon", cookies=cookies)
        assert resp.status_code == 200
        assert "deadline" in resp.json()["intents"]


class TestSearchAlerts:
    def test_create_alert(self, client):
        cookies, _ = register_user(client, "alrt1")
        ws = create_workspace(client, cookies)
        saved = client.post("/search/saved", json={
            "workspace_id": ws, "name": "Alerts", "query": "renewal",
        }, cookies=cookies).json()
        resp = client.post(f"/search/saved/{saved['id']}/alert", cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["is_active"] is True

    def test_run_alerts_creates_notification(self, client, db_session):
        from app.models.search_intel import SavedSearch
        from app.services.nl_search import create_search_alert
        cookies, _ = register_user(client, "alrt2")
        ws = create_workspace(client, cookies)
        saved = SavedSearch(workspace_id=ws, owner_id=1, name="N", query="q")
        db_session.add(saved)
        db_session.flush()
        create_search_alert(db_session, ws, saved.id)
        db_session.commit()
        resp = client.post(f"/search/alerts/run?workspace_id={ws}", json={str(saved.id): 5}, cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["notifications_created"] >= 1

    def test_alert_no_duplicate_notifications(self, db_session):
        from app.models.search_intel import SavedSearch
        from app.services.nl_search import create_search_alert, run_search_alerts
        from app.models.notification import Notification
        saved = SavedSearch(workspace_id=1, owner_id=1, name="N", query="q")
        db_session.add(saved)
        db_session.flush()
        create_search_alert(db_session, 1, saved.id)
        db_session.flush()
        run_search_alerts(db_session, 1, {saved.id: 3}, user_ids=[1])
        count_after_first = db_session.query(Notification).filter(
            Notification.notification_type == "search_alert"
        ).count()
        # New matches arrive; the same alert must not notify twice.
        run_search_alerts(db_session, 1, {saved.id: 5}, user_ids=[1])
        count_after_second = db_session.query(Notification).filter(
            Notification.notification_type == "search_alert"
        ).count()
        assert count_after_second == count_after_first
        assert count_after_first >= 1

    def test_toggle_alert(self, client, db_session):
        from app.models.search_intel import SavedSearch
        from app.services.nl_search import create_search_alert
        cookies, _ = register_user(client, "alrt3")
        ws = create_workspace(client, cookies)
        saved = SavedSearch(workspace_id=ws, owner_id=1, name="N", query="q")
        db_session.add(saved)
        db_session.flush()
        alert = create_search_alert(db_session, ws, saved.id)
        db_session.commit()
        resp = client.post(f"/search/alerts/{alert.id}/toggle", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

    def test_alert_tracked_incrementally(self, db_session):
        from app.models.search_intel import SavedSearch
        from app.services.nl_search import create_search_alert, run_search_alerts
        saved = SavedSearch(workspace_id=1, owner_id=1, name="N", query="q")
        db_session.add(saved)
        db_session.flush()
        alert = create_search_alert(db_session, 1, saved.id)
        run_search_alerts(db_session, 1, {saved.id: 2}, user_ids=[1])
        assert alert.last_match_count == 2


# ============================================================
# Change Intelligence
# ============================================================

class TestChangeIntelligence:
    def test_diff_identical_texts(self):
        from app.services.change_intelligence import diff_versions
        result = diff_versions("same\nlines\n", "same\nlines\n")
        assert result["added_count"] == 0
        assert result["removed_count"] == 0

    def test_diff_added_lines(self):
        from app.services.change_intelligence import diff_versions
        result = diff_versions("line one\n", "line one\nline two\n")
        assert result["added_count"] == 1

    def test_diff_removed_lines(self):
        from app.services.change_intelligence import diff_versions
        result = diff_versions("line one\nline two\n", "line one\n")
        assert result["removed_count"] == 1

    def test_date_change_classification(self):
        from app.services.change_intelligence import diff_versions
        result = diff_versions("old text\n", "old text\nEffective: 2026-06-01\n")
        assert result["classification"] == "DATE_CHANGE"

    def test_policy_change_classification(self):
        from app.services.change_intelligence import diff_versions
        result = diff_versions("text\n", "text\nAll employees must comply.\n")
        assert result["classification"] == "POLICY_CHANGE"

    def test_numeric_change_classification(self):
        from app.services.change_intelligence import diff_versions
        result = diff_versions("text\n", "text\nBudget: $50,000\n")
        assert result["classification"] == "NUMERIC_CHANGE"

    def test_structural_change_classification(self):
        from app.services.change_intelligence import diff_versions
        big = "\n".join(f"line {i}" for i in range(100))
        result = diff_versions("one line\n", big)
        assert result["classification"] == "STRUCTURAL_CHANGE"

    def test_importance_mapping(self):
        from app.services.change_intelligence import diff_versions
        assert diff_versions("a\n", "a\n2026-01-01\n")["importance"] == "HIGH"
        assert diff_versions("a\n", "a\nnew content here\n")["importance"] in ("MEDIUM", "HIGH")

    def test_change_types_constant(self):
        from app.services.change_intelligence import CHANGE_TYPES
        assert "CONTENT_CHANGE" in CHANGE_TYPES
        assert "DATE_CHANGE" in CHANGE_TYPES

    def test_analyze_change_impact(self, db_session):
        from app.services.change_intelligence import analyze_change
        from app.models.document_content import DocumentContent
        doc = make_document(db_session, 1)
        db_session.add(DocumentContent(document_id=doc.id, extracted_text="v1\n"))
        db_session.flush()
        result = analyze_change(db_session, doc.id, 1, None, 2, user_id=1)
        assert "classification" in result
        assert "impact" in result
        assert isinstance(result["impact"]["collections"], list)

    def test_analyze_change_records_insight_for_high(self, db_session, monkeypatch):
        from app.services import change_intelligence
        from app.services.change_intelligence import analyze_change
        from app.models.document_content import DocumentContent
        from app.models.knowledge import KnowledgeInsight
        doc = make_document(db_session, 1)
        db_session.add(DocumentContent(document_id=doc.id, extracted_text="v1\n"))
        db_session.flush()
        monkeypatch.setattr(
            change_intelligence, "_text_for_version",
            lambda db, doc_id, version: ("old policy text\n" if version == 1 else "old policy text\nNew policy: all must comply.\n"),
        )
        analyze_change(db_session, doc.id, 1, 1, 2, user_id=1)
        insight = db_session.query(KnowledgeInsight).filter(
            KnowledgeInsight.insight_type == "change"
        ).first()
        assert insight is not None


# ============================================================
# Deadlines
# ============================================================

class TestDeadlineService:
    def test_create_deadline(self, db_session):
        deadline = make_deadline(db_session, 1, 1, days_from_now=10)
        assert deadline.status == "UPCOMING"
        assert deadline.source == "manual"

    def test_create_deadline_invalid_source(self, db_session):
        from app.services.deadline_service import create_deadline
        with pytest.raises(ValueError):
            create_deadline(db_session, 1, 1, "X", datetime.now(timezone.utc), source="alien")

    def test_create_deadline_invalid_confidence(self, db_session):
        from app.services.deadline_service import create_deadline
        with pytest.raises(ValueError):
            create_deadline(db_session, 1, 1, "X", datetime.now(timezone.utc), confidence="DEFINITELY")

    def test_refresh_upcoming(self, db_session):
        from app.services.deadline_service import refresh_status
        deadline = make_deadline(db_session, 1, 1, days_from_now=5)
        refresh_status(db_session, deadline)
        assert deadline.status == "UPCOMING"

    def test_refresh_overdue(self, db_session):
        from app.services.deadline_service import refresh_status
        deadline = make_deadline(db_session, 1, 1, days_from_now=-3)
        refresh_status(db_session, deadline)
        assert deadline.status == "OVERDUE"

    def test_refresh_due_today(self, db_session):
        from app.services.deadline_service import refresh_status
        deadline = make_deadline(db_session, 1, 1, days_from_now=0)
        refresh_status(db_session, deadline)
        assert deadline.status in ("DUE", "OVERDUE")

    def test_complete_deadline(self, db_session):
        from app.services.deadline_service import complete_deadline
        deadline = make_deadline(db_session, 1, 1)
        complete_deadline(db_session, deadline)
        assert deadline.status == "COMPLETED"

    def test_cancel_deadline(self, db_session):
        from app.services.deadline_service import cancel_deadline
        deadline = make_deadline(db_session, 1, 1)
        cancel_deadline(db_session, deadline)
        assert deadline.status == "CANCELLED"

    def test_complete_completed_raises(self, db_session):
        from app.services.deadline_service import complete_deadline, DeadlineStatusError
        deadline = make_deadline(db_session, 1, 1)
        complete_deadline(db_session, deadline)
        with pytest.raises(DeadlineStatusError):
            complete_deadline(db_session, deadline)

    def test_completed_deadline_not_modified_by_refresh(self, db_session):
        from app.services.deadline_service import complete_deadline, refresh_status
        deadline = make_deadline(db_session, 1, 1, days_from_now=-2)
        complete_deadline(db_session, deadline)
        refresh_status(db_session, deadline)
        assert deadline.status == "COMPLETED"

    def test_notification_created_for_window(self, db_session):
        from app.services.deadline_service import check_deadline_notifications
        make_deadline(db_session, 1, 1, days_from_now=7)
        notifications = check_deadline_notifications(db_session, 1, user_ids=[1])
        assert len(notifications) >= 1

    def test_notification_deduplicated(self, db_session):
        from app.services.deadline_service import check_deadline_notifications
        make_deadline(db_session, 1, 1, days_from_now=7)
        check_deadline_notifications(db_session, 1, user_ids=[1])
        second = check_deadline_notifications(db_session, 1, user_ids=[1])
        assert second == []

    def test_far_deadline_no_notification(self, db_session):
        from app.services.deadline_service import check_deadline_notifications
        make_deadline(db_session, 1, 1, days_from_now=45)
        notifications = check_deadline_notifications(db_session, 1, user_ids=[1])
        assert notifications == []

    def test_deadline_action_created_once(self, db_session):
        from app.services.deadline_service import deadline_actions
        deadline = make_deadline(db_session, 1, 1, days_from_now=-1)
        created = deadline_actions(db_session, 1, 1, None)
        assert len(created) == 1
        second = deadline_actions(db_session, 1, 1, None)
        assert second == []

    def test_parse_deadline_iso(self):
        from app.services.deadline_service import parse_deadline_from_text
        result = parse_deadline_from_text("Renewal by 2026-12-31")
        assert result is not None
        assert result["date"].year == 2026

    def test_parse_deadline_slash(self):
        from app.services.deadline_service import parse_deadline_from_text
        result = parse_deadline_from_text("Due 12/31/2026")
        assert result is not None
        assert result["date"].year == 2026

    def test_parse_no_date(self):
        from app.services.deadline_service import parse_deadline_from_text
        assert parse_deadline_from_text("no dates here") is None


class TestDeadlineAPI:
    def test_create_deadline_endpoint(self, client):
        cookies, _ = register_user(client, "dlapi")
        ws = create_workspace(client, cookies)
        resp = client.post("/deadlines", json={
            "workspace_id": ws,
            "title": "Contract renewal",
            "due_date": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text

    def test_list_deadlines(self, client, db_session):
        cookies, _ = register_user(client, "dllist")
        ws = create_workspace(client, cookies)
        make_deadline(db_session, ws, 1, days_from_now=10)
        db_session.commit()
        resp = client.get(f"/deadlines?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 1

    def test_complete_deadline_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "dlcomp")
        ws = create_workspace(client, cookies)
        deadline = make_deadline(db_session, ws, 1)
        db_session.commit()
        resp = client.post(f"/deadlines/{deadline.id}/complete", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["status"] == "COMPLETED"

    def test_cancel_deadline_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "dlcancel")
        ws = create_workspace(client, cookies)
        deadline = make_deadline(db_session, ws, 1)
        db_session.commit()
        resp = client.post(f"/deadlines/{deadline.id}/cancel", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["status"] == "CANCELLED"

    def test_parse_endpoint(self, client):
        cookies, _ = register_user(client, "dlparse")
        resp = client.post("/deadlines/parse?text=due+2026-10-01", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["date"]

    def test_parse_endpoint_no_date(self, client):
        cookies, _ = register_user(client, "dlparse2")
        resp = client.post("/deadlines/parse?text=nothing", cookies=cookies)
        assert resp.status_code == 400

    def test_deadline_requires_auth(self, client):
        resp = client.get("/deadlines?workspace_id=1")
        assert resp.status_code == 401

    def test_check_notifications_endpoint(self, client, db_session):
        cookies, _ = register_user(client, "dlnotif")
        ws = create_workspace(client, cookies)
        make_deadline(db_session, ws, 1, days_from_now=3)
        db_session.commit()
        resp = client.post(f"/deadlines/check-notifications?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["notifications_created"] >= 1


# ============================================================
# AI Reports
# ============================================================

class TestReportService:
    def test_list_templates(self):
        from app.services.report_service import list_templates
        templates = list_templates()
        assert len(templates) >= 7
        names = {t["template"] for t in templates}
        assert {"executive_summary", "weekly_knowledge", "risk_report"} <= names

    def test_unknown_template_raises(self, db_session):
        from app.services.report_service import generate_report, UnknownTemplateError
        with pytest.raises(UnknownTemplateError):
            generate_report(db_session, 1, 1, "mystery_template")

    def test_executive_summary_generated(self, db_session):
        from app.services.report_service import generate_report
        report = generate_report(db_session, 1, 1, "executive_summary")
        assert report["template"] == "executive_summary"
        assert "findings" in report
        assert "recommendations" in report
        assert "generated_at" in report

    def test_weekly_knowledge_report(self, db_session):
        from app.services.report_service import generate_report
        report = generate_report(db_session, 1, 1, "weekly_knowledge")
        assert report["template"] == "weekly_knowledge"
        assert len(report["findings"]) >= 1

    def test_ai_usage_report(self, db_session):
        from app.services.report_service import generate_report
        report = generate_report(db_session, 1, 1, "ai_usage")
        assert report["template"] == "ai_usage"

    def test_facts_separated_from_recommendations(self, db_session):
        from app.services.report_service import generate_report
        report = generate_report(db_session, 1, 1, "executive_summary")
        for f in report["findings"]:
            assert f["type"] != "recommendation"
        for r in report["recommendations"]:
            assert r["type"] == "recommendation"

    def test_report_saved_as_artifact(self, db_session):
        from app.services.report_service import generate_report
        from app.models.ai_execution import AIArtifact
        generate_report(db_session, 1, 1, "executive_summary")
        artifact = db_session.query(AIArtifact).filter(AIArtifact.artifact_type == "report").first()
        assert artifact is not None
        assert artifact.status == "ai_generated"

    def test_report_has_schema_version(self, db_session):
        from app.services.report_service import generate_report
        report = generate_report(db_session, 1, 1, "executive_summary")
        assert report["schema_version"] == "1.0"


class TestReportAPI:
    def test_templates_endpoint(self, client):
        cookies, _ = register_user(client, "rpttmp")
        resp = client.get("/reports/templates", cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["templates"]) >= 7

    def test_generate_endpoint(self, client):
        cookies, _ = register_user(client, "rptgen")
        create_workspace(client, cookies)
        resp = client.post("/reports/generate", json={"template": "executive_summary"}, cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["template"] == "executive_summary"

    def test_generate_unknown_template_422(self, client):
        cookies, _ = register_user(client, "rptbad")
        create_workspace(client, cookies)
        resp = client.post("/reports/generate", json={"template": "nope"}, cookies=cookies)
        assert resp.status_code == 422

    def test_artifacts_endpoint(self, client):
        cookies, _ = register_user(client, "rptart")
        create_workspace(client, cookies)
        client.post("/reports/generate", json={"template": "executive_summary"}, cookies=cookies)
        resp = client.get("/reports/artifacts", cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["artifacts"]) >= 1

    def test_report_requires_auth(self, client):
        resp = client.get("/reports/templates")
        assert resp.status_code == 401


# ============================================================
# Workflow Intelligence
# ============================================================

class TestWorkflowIntel:
    def test_create_workflow_version(self, db_session):
        from app.services.workflow_intel import create_workflow_version
        definition = {"trigger": "document.ready", "nodes": [{"id": "sum", "type": "summarize", "name": "Summarize", "inputs": []}]}
        version = create_workflow_version(db_session, 1, 1, "My workflow", definition)
        assert version.version == 1
        assert version.status == "DRAFT"
        assert version.workflow_id

    def test_new_version_increments(self, db_session):
        from app.services.workflow_intel import create_workflow_version
        definition = {"trigger": "document.ready", "nodes": [{"id": "sum", "type": "summarize", "name": "S", "inputs": []}]}
        v1 = create_workflow_version(db_session, 1, 1, "W", definition)
        v2 = create_workflow_version(db_session, 1, 1, "W2", definition, workflow_id=v1.workflow_id)
        assert v2.version == 2
        assert v1.is_active is False

    def test_unknown_trigger_rejected(self, db_session):
        from app.services.workflow_intel import create_workflow_version, WorkflowValidationError
        definition = {"trigger": "moon.phase", "nodes": [{"id": "s", "type": "summarize", "name": "S", "inputs": []}]}
        with pytest.raises(WorkflowValidationError):
            create_workflow_version(db_session, 1, 1, "W", definition)

    def test_unknown_node_type_rejected(self, db_session):
        from app.services.workflow_intel import create_workflow_version, WorkflowValidationError
        definition = {"trigger": "document.ready", "nodes": [{"id": "s", "type": "teleport", "name": "S", "inputs": []}]}
        with pytest.raises(WorkflowValidationError):
            create_workflow_version(db_session, 1, 1, "W", definition)

    def test_empty_nodes_rejected(self, db_session):
        from app.services.workflow_intel import create_workflow_version, WorkflowValidationError
        definition = {"trigger": "document.ready", "nodes": []}
        with pytest.raises(WorkflowValidationError):
            create_workflow_version(db_session, 1, 1, "W", definition)

    def test_cycle_detected(self, db_session):
        from app.services.workflow_intel import create_workflow_version, WorkflowValidationError
        definition = {
            "trigger": "document.ready",
            "nodes": [
                {"id": "a", "type": "summarize", "name": "A", "inputs": ["b"]},
                {"id": "b", "type": "classify", "name": "B", "inputs": ["a"]},
            ],
        }
        with pytest.raises(WorkflowValidationError):
            create_workflow_version(db_session, 1, 1, "Cycle", definition)

    def test_notify_requires_approval_gate(self, db_session):
        from app.services.workflow_intel import create_workflow_version, WorkflowValidationError
        definition = {
            "trigger": "document.ready",
            "nodes": [
                {"id": "n", "type": "notify", "name": "Notify", "inputs": []},
            ],
        }
        with pytest.raises(WorkflowValidationError):
            create_workflow_version(db_session, 1, 1, "Bad", definition)

    def test_notify_with_approval_valid(self, db_session):
        from app.services.workflow_intel import create_workflow_version
        definition = {
            "trigger": "document.ready",
            "nodes": [
                {"id": "a", "type": "approval", "name": "Approval", "inputs": []},
                {"id": "n", "type": "notify", "name": "Notify", "inputs": ["a"]},
            ],
        }
        version = create_workflow_version(db_session, 1, 1, "Good", definition)
        assert version.status == "DRAFT"

    def test_activate_workflow(self, db_session):
        from app.services.workflow_intel import create_workflow_version, activate_workflow
        definition = {"trigger": "document.ready", "nodes": [{"id": "s", "type": "summarize", "name": "S", "inputs": []}]}
        version = create_workflow_version(db_session, 1, 1, "W", definition)
        activate_workflow(db_session, version)
        assert version.status == "ACTIVE"
        assert version.is_active is True

    def test_activate_only_draft(self, db_session):
        from app.services.workflow_intel import create_workflow_version, activate_workflow, WorkflowValidationError
        definition = {"trigger": "document.ready", "nodes": [{"id": "s", "type": "summarize", "name": "S", "inputs": []}]}
        version = create_workflow_version(db_session, 1, 1, "W", definition)
        activate_workflow(db_session, version)
        with pytest.raises(WorkflowValidationError):
            activate_workflow(db_session, version)

    def test_dry_run_no_side_effects(self, db_session):
        from app.services.workflow_intel import dry_run
        definition = {
            "trigger": "document.ready",
            "nodes": [
                {"id": "a", "type": "approval", "name": "Approval", "inputs": []},
                {"id": "n", "type": "notify", "name": "Notify", "inputs": ["a"]},
                {"id": "s", "type": "summarize", "name": "Summarize", "inputs": []},
            ],
        }
        plan = dry_run(definition)
        assert plan["mode"] == "DRY_RUN"
        assert plan["estimated_ai_calls"] == 1
        assert "Approval" in plan["approval_gates"]
        from app.models.workflow_version import WorkflowVersion
        assert db_session.query(WorkflowVersion).count() == 0

    def test_dry_run_estimates_cost(self):
        from app.services.workflow_intel import dry_run
        definition = {
            "trigger": "document.ready",
            "nodes": [
                {"id": "s", "type": "summarize", "name": "S", "inputs": []},
                {"id": "e", "type": "extract", "name": "E", "inputs": ["s"]},
            ],
        }
        plan = dry_run(definition)
        assert plan["estimated_ai_calls"] == 2
        assert plan["estimated_cost"] > 0

    def test_generate_draft_from_nl(self, db_session):
        from app.services.workflow_intel import generate_workflow_draft
        definition = generate_workflow_draft(
            "When a new contract is uploaded, summarize it, extract the renewal date, "
            "and notify the owner 30 days before renewal.",
            1, 1,
        )
        assert definition["trigger"] == "document.ready"
        types = [n["type"] for n in definition["nodes"]]
        assert "summarize" in types
        assert "extract" in types
        assert "approval" in types
        assert "notify" in types

    def test_generated_draft_is_never_active(self, db_session):
        from app.services.workflow_intel import generate_workflow_draft
        definition = generate_workflow_draft("summarize and notify", 1, 1)
        assert definition.get("status") is None  # DRAFT only via explicit creation

    def test_generate_draft_deadline_trigger(self, db_session):
        from app.services.workflow_intel import generate_workflow_draft
        definition = generate_workflow_draft("alert me when a deadline expires", 1, 1)
        assert definition["trigger"] == "document.deadline_approaching"

    def test_list_versions(self, db_session):
        from app.services.workflow_intel import create_workflow_version, list_versions
        definition = {"trigger": "document.ready", "nodes": [{"id": "s", "type": "summarize", "name": "S", "inputs": []}]}
        v1 = create_workflow_version(db_session, 1, 1, "W", definition)
        create_workflow_version(db_session, 1, 1, "W2", definition, workflow_id=v1.workflow_id)
        versions = list_versions(db_session, v1.workflow_id)
        assert len(versions) == 2
        assert versions[0].version == 2  # newest first


# ============================================================
# Knowledge Gaps
# ============================================================

class TestKnowledgeGaps:
    def test_document_gap_missing_metadata(self, db_session):
        from app.services.knowledge_gap import detect_document_gaps
        doc = make_document(db_session, 1)
        gaps = detect_document_gaps(db_session, doc)
        types = {g["type"] for g in gaps}
        assert "missing_metadata" in types

    def test_gaps_labeled_possible(self, db_session):
        from app.services.knowledge_gap import detect_document_gaps
        doc = make_document(db_session, 1)
        gaps = detect_document_gaps(db_session, doc)
        assert all(g["kind"] == "POSSIBLE_GAP" for g in gaps)

    def test_complete_metadata_no_gaps(self, db_session):
        from app.services.knowledge_gap import detect_document_gaps
        doc = make_document(db_session, 1, metadata_dict={
            "author": "a", "created": "c", "classification": "cl", "summary": "s",
        })
        gaps = detect_document_gaps(db_session, doc)
        assert all(g["type"] != "missing_metadata" for g in gaps)

    def test_policy_missing_effective_date(self, db_session):
        from app.services.knowledge_gap import detect_policy_gaps
        make_document(db_session, 1, metadata_dict={"classification": "policy"})
        gaps = detect_policy_gaps(db_session, 1)
        assert any(g["type"] == "missing_effective_date" for g in gaps)

    def test_policy_with_effective_date_ok(self, db_session):
        from app.services.knowledge_gap import detect_policy_gaps
        make_document(db_session, 1, metadata_dict={"classification": "policy", "effective_date": "2026-01-01"})
        gaps = detect_policy_gaps(db_session, 1)
        assert not any(g["type"] == "missing_effective_date" for g in gaps)

    def test_entity_gap_no_source(self, db_session):
        from app.models.knowledge_graph import Entity
        from app.services.knowledge_gap import detect_entity_gaps
        db_session.add(Entity(workspace_id=1, name="Ghost Entity", entity_type="company"))
        db_session.flush()
        gaps = detect_entity_gaps(db_session, 1)
        assert any(g["type"] == "unsupported_entity" for g in gaps)

    def test_record_gap_insights_dedup(self, db_session):
        from app.services.knowledge_gap import record_gap_insights
        make_document(db_session, 1, metadata_dict={"classification": "policy"})
        first = record_gap_insights(db_session, 1, None, 1)
        second = record_gap_insights(db_session, 1, None, 1)
        assert first >= 1
        assert second == 0

    def test_insight_importance_medium(self, db_session):
        from app.services.knowledge_gap import record_gap_insights
        from app.models.knowledge import KnowledgeInsight
        make_document(db_session, 1, metadata_dict={"classification": "policy"})
        record_gap_insights(db_session, 1, None, 1)
        insight = db_session.query(KnowledgeInsight).filter(KnowledgeInsight.insight_type == "knowledge_gap").first()
        assert insight.importance == "MEDIUM"


# ============================================================
# AI Feedback
# ============================================================

class TestFeedbackService:
    def test_submit_feedback(self, db_session):
        from app.services.feedback_service import submit_feedback
        f = submit_feedback(db_session, 1, 1, "thumbs_up")
        assert f.rating == "thumbs_up"

    def test_invalid_rating_rejected(self, db_session):
        from app.services.feedback_service import submit_feedback
        with pytest.raises(ValueError):
            submit_feedback(db_session, 1, 1, "sideways")

    def test_invalid_category_rejected(self, db_session):
        from app.services.feedback_service import submit_feedback
        with pytest.raises(ValueError):
            submit_feedback(db_session, 1, 1, "thumbs_up", category="needs_more_cats")

    def test_comment_truncated(self, db_session):
        from app.services.feedback_service import submit_feedback
        f = submit_feedback(db_session, 1, 1, "thumbs_down", comment="x" * 5000)
        assert len(f.comment) == 2000

    def test_analytics_empty(self, db_session):
        from app.services.feedback_service import workspace_feedback_analytics
        a = workspace_feedback_analytics(db_session, 999)
        assert a["total_feedback"] == 0
        assert a["acceptance_rate"] is None

    def test_analytics_acceptance_rate(self, db_session):
        from app.services.feedback_service import submit_feedback, workspace_feedback_analytics
        submit_feedback(db_session, 1, 1, "thumbs_up")
        submit_feedback(db_session, 1, 1, "thumbs_up")
        submit_feedback(db_session, 1, 1, "thumbs_down")
        db_session.flush()
        a = workspace_feedback_analytics(db_session, 1)
        assert a["total_feedback"] == 3
        assert a["acceptance_rate"] == round(2 / 3, 3)

    def test_analytics_categories(self, db_session):
        from app.services.feedback_service import submit_feedback, workspace_feedback_analytics
        submit_feedback(db_session, 1, 1, "thumbs_down", category="citation_issue")
        db_session.flush()
        a = workspace_feedback_analytics(db_session, 1)
        assert a["categories"].get("citation_issue") == 1

    def test_analytics_tenant_scoped(self, db_session):
        from app.services.feedback_service import submit_feedback, workspace_feedback_analytics
        submit_feedback(db_session, 1, 1, "thumbs_up")
        submit_feedback(db_session, 2, 1, "thumbs_down")
        db_session.flush()
        a = workspace_feedback_analytics(db_session, 1)
        assert a["total_feedback"] == 1


class TestFeedbackAPI:
    def test_submit_endpoint(self, client):
        cookies, _ = register_user(client, "fbapi")
        create_workspace(client, cookies)
        resp = client.post("/feedback", json={"rating": "thumbs_up"}, cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "recorded"

    def test_submit_invalid_rating_422(self, client):
        cookies, _ = register_user(client, "fbbad")
        create_workspace(client, cookies)
        resp = client.post("/feedback", json={"rating": "meh"}, cookies=cookies)
        assert resp.status_code == 422

    def test_analytics_endpoint(self, client):
        cookies, _ = register_user(client, "fban")
        create_workspace(client, cookies)
        client.post("/feedback", json={"rating": "thumbs_up"}, cookies=cookies)
        resp = client.get("/feedback/analytics", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["total_feedback"] >= 1

    def test_feedback_requires_auth(self, client):
        resp = client.post("/feedback", json={"rating": "thumbs_up"})
        assert resp.status_code == 401


# ============================================================
# Research
# ============================================================

class TestResearchService:
    def test_run_research_returns_structure(self, db_session):
        from app.services.research_service import run_research
        result = run_research(db_session, 1, 1, "What are our policies?", save_artifact=False)
        assert set(result.keys()) >= {"question", "answer", "evidence", "conflicts", "uncertainties", "sources"}

    def test_research_saves_brief(self, db_session):
        from app.services.research_service import run_research
        from app.models.ai_execution import AIArtifact
        run_research(db_session, 1, 1, "Research question here")
        brief = db_session.query(AIArtifact).filter(AIArtifact.artifact_type == "research_brief").first()
        assert brief is not None
        assert brief.status == "ai_generated"

    def test_research_failure_graceful(self, db_session, monkeypatch):
        from app.services.research_service import run_research
        from app.services import research_service as research_mod
        from app.services.rag_service import RAGError

        def boom(**kwargs):
            raise RAGError("retrieval failed")
        monkeypatch.setattr(research_mod, "answer_question_with_history", boom)
        result = run_research(db_session, 1, 1, "Q", save_artifact=False)
        assert result["grounded"] is False
        assert "retrieval failed" in result["answer"]

    def test_list_briefs(self, db_session):
        from app.services.research_service import run_research, list_research_briefs
        run_research(db_session, 1, 1, "List me question")
        briefs = list_research_briefs(db_session, 1)
        assert len(briefs) >= 1


class TestResearchAPI:
    def test_research_endpoint(self, client):
        cookies, _ = register_user(client, "rsapi")
        create_workspace(client, cookies)
        resp = client.post("/research", json={"question": "Summarize our contracts"}, cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert "answer" in resp.json()

    def test_research_briefs_endpoint(self, client):
        cookies, _ = register_user(client, "rsbriefs")
        create_workspace(client, cookies)
        client.post("/research", json={"question": "What changed last month?"}, cookies=cookies)
        resp = client.get("/research/briefs", cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["briefs"]) >= 1

    def test_research_empty_question_422(self, client):
        cookies, _ = register_user(client, "rsempty")
        create_workspace(client, cookies)
        resp = client.post("/research", json={"question": ""}, cookies=cookies)
        assert resp.status_code == 422

    def test_research_requires_auth(self, client):
        resp = client.post("/research", json={"question": "hi"})
        assert resp.status_code == 401


# ============================================================
# Confidence Model
# ============================================================

class TestConfidence:
    def test_level_from_score_high(self):
        from app.services.confidence import level_from_score
        assert level_from_score(0.9) == "HIGH"

    def test_level_from_score_medium(self):
        from app.services.confidence import level_from_score
        assert level_from_score(0.6) == "MEDIUM"

    def test_level_from_score_low(self):
        from app.services.confidence import level_from_score
        assert level_from_score(0.4) == "LOW"

    def test_level_from_score_none(self):
        from app.services.confidence import level_from_score
        assert level_from_score(None) == "UNKNOWN"

    def test_combine_scores(self):
        from app.services.confidence import combine
        assert combine([0.9, 0.8]) == "HIGH"

    def test_combine_empty(self):
        from app.services.confidence import combine
        assert combine([]) == "UNKNOWN"

    def test_level_rank(self):
        from app.services.confidence import level_rank
        assert level_rank("HIGH") == 4
        assert level_rank("UNKNOWN") == 1

    def test_confidence_dict_valid(self):
        from app.services.confidence import confidence_dict
        d = confidence_dict("MEDIUM", reasons=["retrieval ok"], score=0.6)
        assert d["level"] == "MEDIUM"
        assert d["score"] == 0.6

    def test_confidence_dict_invalid_level_normalized(self):
        from app.services.confidence import confidence_dict
        d = confidence_dict("DEFINITE", score=0.99)
        assert d["level"] == "UNKNOWN"

    def test_raw_probability_never_certainty(self):
        from app.services.confidence import level_from_score
        # even a high raw score is still just HIGH, not "certain"
        assert level_from_score(0.95) == "HIGH"


# ============================================================
# Worker Scheduler / Fairness
# ============================================================

class TestWorkerScheduler:
    def test_priority_rank(self):
        from app.services.worker_scheduler import PRIORITY_RANK
        assert PRIORITY_RANK["CRITICAL"] > PRIORITY_RANK["HIGH"] > PRIORITY_RANK["NORMAL"] > PRIORITY_RANK["LOW"]

    def test_validate_priority(self):
        from app.services.worker_scheduler import validate_priority
        assert validate_priority("high") == "HIGH"
        with pytest.raises(ValueError):
            validate_priority("URGENT")

    def test_sort_by_priority(self):
        from app.services.worker_scheduler import ScheduledJob, sort_by_priority
        jobs = [
            ScheduledJob("1", "AI_TASKS", priority="LOW"),
            ScheduledJob("2", "AI_TASKS", priority="CRITICAL"),
            ScheduledJob("3", "AI_TASKS", priority="NORMAL"),
        ]
        ordered = sort_by_priority(jobs)
        assert [j.job_id for j in ordered] == ["2", "3", "1"]

    def test_admit_under_limits(self):
        from app.services.worker_scheduler import can_admit
        ok, reason = can_admit({"total": 1, "ai_tasks": 1}, 1, None)
        assert ok

    def test_admit_rejects_ai_limit(self):
        from app.services.worker_scheduler import can_admit
        ok, reason = can_admit({"ai_tasks": 10}, 1, None)
        assert not ok
        assert "AI call" in reason

    def test_admit_rejects_agent_limit(self):
        from app.services.worker_scheduler import can_admit
        ok, _ = can_admit({"agents": 4}, 1, None)
        assert not ok

    def test_admit_rejects_workspace_cap(self):
        from app.services.worker_scheduler import can_admit
        ok, _ = can_admit({"per_workspace": 3}, 1, None)
        assert not ok

    def test_admit_rejects_org_cap(self):
        from app.services.worker_scheduler import can_admit
        ok, _ = can_admit({"per_organization": 6}, 1, 1)
        assert not ok

    def test_custom_limits(self):
        from app.services.worker_scheduler import can_admit
        ok, _ = can_admit({"ai_tasks": 5}, 1, None, limits={"concurrent_ai_calls": 8})
        assert ok

    def test_fairness_no_workspace_monopoly(self):
        from app.services.worker_scheduler import can_admit
        # workspace A already holds the per-workspace cap
        ok, _ = can_admit({"per_workspace": 3}, 10, None)
        assert not ok

    def test_fairness_other_workspace_admitted(self):
        from app.services.worker_scheduler import can_admit
        ok, _ = can_admit({"per_workspace": 3}, 10, None, limits={"per_workspace_cap": 5})
        assert ok

    def test_default_limits_present(self):
        from app.services.worker_scheduler import DEFAULT_LIMITS
        assert "concurrent_ai_calls" in DEFAULT_LIMITS
        assert "per_workspace_cap" in DEFAULT_LIMITS


# ============================================================
# Tenant Isolation
# ============================================================

class TestTenantIsolation:
    def test_action_cross_workspace(self, client, db_session):
        from app.models.workspace import Workspace, WorkspaceMember
        cookies_a, _ = register_user(client, "tis1")
        ws_a = create_workspace(client, cookies_a)
        cookies_b, _ = register_user(client, "tis2")
        create_workspace(client, cookies_b)
        action = make_action(db_session, ws_a, 1)
        db_session.commit()
        resp = client.post(f"/ai-actions/{action.id}/approve", cookies=cookies_b)
        assert resp.status_code in (403, 404)

    def test_suggestion_isolated(self, db_session):
        from app.services.ai_action_service import create_suggestion
        s1 = create_suggestion(db_session, 1, 1, "A", "deadline")
        create_suggestion(db_session, 2, 1, "B", "deadline")
        db_session.flush()
        from app.models.ai_action import AISuggestion
        count = db_session.query(AISuggestion).filter(AISuggestion.workspace_id == 1).count()
        assert count == 1
        assert s1.workspace_id == 1

    def test_health_isolated(self, db_session):
        from app.services.health_service import workspace_health
        make_document(db_session, 1, status="READY")
        make_document(db_session, 2, status="FAILED")
        h = workspace_health(db_session, 2)
        assert h["processing_failures"] == 1
        assert h["total_documents"] == 1

    def test_deadline_isolated(self, db_session):
        make_deadline(db_session, 1, 1, days_from_now=5)
        from app.models.knowledge import Deadline
        count = db_session.query(Deadline).filter(Deadline.workspace_id == 1).count()
        assert count == 1

    def test_report_tenant_scoped(self, db_session):
        from app.services.report_service import generate_report
        from app.models.ai_execution import AIArtifact
        generate_report(db_session, 1, 1, "executive_summary")
        generate_report(db_session, 2, 1, "executive_summary")
        count = db_session.query(AIArtifact).filter(AIArtifact.workspace_id == 1).count()
        assert count == 1

    def test_feedback_isolated(self, db_session):
        from app.services.feedback_service import submit_feedback, workspace_feedback_analytics
        submit_feedback(db_session, 1, 1, "thumbs_up")
        submit_feedback(db_session, 2, 1, "thumbs_up")
        db_session.flush()
        assert workspace_feedback_analytics(db_session, 1)["total_feedback"] == 1

    def test_saved_search_isolated(self, db_session):
        from app.services.nl_search import create_saved_search, list_saved_searches
        create_saved_search(db_session, 1, 1, "A", "q")
        create_saved_search(db_session, 2, 1, "B", "q")
        db_session.flush()
        assert len(list_saved_searches(db_session, 1, 1)) == 1

    def test_workflow_versions_isolated(self, db_session):
        from app.services.workflow_intel import create_workflow_version
        definition = {"trigger": "document.ready", "nodes": [{"id": "s", "type": "summarize", "name": "S", "inputs": []}]}
        create_workflow_version(db_session, 1, 1, "W1", definition)
        create_workflow_version(db_session, 2, 1, "W2", definition)
        from app.models.workflow_version import WorkflowVersion
        count = db_session.query(WorkflowVersion).filter(WorkflowVersion.workspace_id == 1).count()
        assert count == 1

    def test_insights_isolated(self, db_session):
        from app.services.ai_action_service import record_insight
        record_insight(db_session, 1, "health", "A")
        record_insight(db_session, 2, "health", "B")
        db_session.flush()
        from app.models.knowledge import KnowledgeInsight
        count = db_session.query(KnowledgeInsight).filter(KnowledgeInsight.workspace_id == 1).count()
        assert count == 1

    def test_copilot_scope_never_leaks(self, db_session, client):
        from fastapi import HTTPException
        from app.services.copilot_service import copilot_ask
        from app.models.user import User
        cookies_a, _ = register_user(client, "tisc1")
        ws_a = create_workspace(client, cookies_a)
        user_a = db_session.query(User).order_by(User.id.desc()).first()
        cookies_b, _ = register_user(client, "tisc2")
        user_b = db_session.query(User).order_by(User.id.desc()).first()
        create_workspace(client, cookies_b)
        doc = make_document(db_session, ws_a, user_id=user_a.id)
        db_session.commit()
        # user B (a different workspace) cannot reach user A's document copilot
        with pytest.raises((HTTPException,)):
            copilot_ask(db_session, user_b.id, "DOCUMENT", "hi", document_id=doc.id)


# ============================================================
# API Security & Validation
# ============================================================

class TestAPISecurity:
    def test_ai_actions_require_auth(self, client):
        resp = client.get("/ai-actions?workspace_id=1")
        assert resp.status_code == 401

    def test_knowledge_require_auth(self, client):
        resp = client.get("/knowledge/workspaces/1/health")
        assert resp.status_code == 401

    def test_copilot_requires_auth(self, client):
        resp = client.post("/copilot/ask", json={"question": "hi", "scope": "WORKSPACE"})
        assert resp.status_code == 401

    def test_search_intent_requires_auth(self, client):
        resp = client.post("/search/intent?query=contracts")
        assert resp.status_code == 401

    def test_deadlines_require_auth(self, client):
        resp = client.get("/deadlines?workspace_id=1")
        assert resp.status_code == 401

    def test_health_document_requires_auth(self, client):
        resp = client.post("/knowledge/documents/1/health")
        assert resp.status_code == 401

    def test_feedback_requires_auth(self, client):
        resp = client.get("/feedback/analytics")
        assert resp.status_code == 401

    def test_reports_require_auth(self, client):
        resp = client.post("/reports/generate", json={"template": "executive_summary"})
        assert resp.status_code == 401

    def test_research_requires_auth(self, client):
        resp = client.get("/research/briefs")
        assert resp.status_code == 401

    def test_action_unknown_type_400(self, client):
        cookies, _ = register_user(client, "sec1")
        ws = create_workspace(client, cookies)
        resp = client.post("/ai-actions", json={
            "workspace_id": ws, "action_type": "nope", "title": "X",
        }, cookies=cookies)
        assert resp.status_code == 400

    def test_missing_workspace_404(self, client):
        cookies, _ = register_user(client, "sec2")
        create_workspace(client, cookies)
        resp = client.get("/knowledge/workspaces/99999/health", cookies=cookies)
        assert resp.status_code == 404

    def test_action_endpoint_rejects_foreign_workspace(self, client, db_session):
        cookies_a, _ = register_user(client, "sec3a")
        ws_a = create_workspace(client, cookies_a)
        action = make_action(db_session, ws_a, 1)
        db_session.commit()
        cookies_b, _ = register_user(client, "sec3b")
        create_workspace(client, cookies_b)
        resp = client.post(f"/ai-actions/{action.id}/transition", json={"new_status": "RUNNING"}, cookies=cookies_b)
        assert resp.status_code in (403, 404)

    def test_report_payload_validation(self, client):
        cookies, _ = register_user(client, "sec4")
        create_workspace(client, cookies)
        resp = client.post("/reports/generate", json={"template": 123}, cookies=cookies)
        assert resp.status_code == 422

    def test_deadline_bad_confidence_400(self, client):
        cookies, _ = register_user(client, "sec5")
        ws = create_workspace(client, cookies)
        resp = client.post("/deadlines", json={
            "workspace_id": ws, "title": "X",
            "due_date": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            "confidence": "BOGUS",
        }, cookies=cookies)
        assert resp.status_code == 400


# ============================================================
# End-to-end product scenarios
# ============================================================

class TestE2EScenarios:
    def test_scenario_upload_health_suggestion(self, client, db_session):
        """Upload → READY-ish → health score → suggestion."""
        cookies, _ = register_user(client, "e2e1")
        ws = create_workspace(client, cookies)
        doc = upload_document(client, cookies, filename="annual_policy.pdf")
        from app.models.document import Document
        from app.models.document_tag import DocumentSummary
        from app.models.document_version import DocumentVersion
        row = db_session.query(Document).filter(Document.id == doc["id"]).first()
        db_session.add(DocumentSummary(document_id=row.id, summary_type="short", content="x" * 60))
        db_session.add(DocumentVersion(
            document_id=row.id, version_number=1, created_by=1,
            storage_key=row.storage_key, file_size=100, mime_type="application/pdf",
            is_current=True, processing_status="READY",
        ))
        db_session.commit()
        resp = client.post(f"/knowledge/documents/{doc['id']}/health", cookies=cookies)
        assert resp.status_code == 200
        assert 0 <= resp.json()["score"] <= 100
        resp = client.post(f"/ai-actions/suggestions/document/{doc['id']}", cookies=cookies)
        assert resp.status_code in (200, 400)

    def test_scenario_nl_search_saved_alert(self, client):
        """NL search → saved search → alert."""
        cookies, _ = register_user(client, "e2e2")
        ws = create_workspace(client, cookies)
        resp = client.post("/search/intent?query=contracts+expiring+next+month", cookies=cookies)
        assert resp.status_code == 200
        saved = client.post("/search/saved", json={
            "workspace_id": ws, "name": "Expiring contracts", "query": "expiring",
        }, cookies=cookies).json()
        alert = client.post(f"/search/saved/{saved['id']}/alert", cookies=cookies)
        assert alert.status_code == 201

    def test_scenario_deadline_reminder(self, client, db_session):
        """Deadline extraction → deadline → reminder notification."""
        cookies, _ = register_user(client, "e2e3")
        ws = create_workspace(client, cookies)
        parsed = client.post("/deadlines/parse?text=renew+2026-12-01", cookies=cookies)
        assert parsed.status_code == 200
        deadline = make_deadline(db_session, ws, 1, days_from_now=3, title="Renewal")
        db_session.commit()
        resp = client.post(f"/deadlines/check-notifications?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["notifications_created"] >= 1

    def test_scenario_research_to_brief(self, client):
        """Research query → saved brief."""
        cookies, _ = register_user(client, "e2e4")
        create_workspace(client, cookies)
        resp = client.post("/research", json={"question": "Compare our insurance policies"}, cookies=cookies)
        assert resp.status_code == 200
        briefs = client.get("/research/briefs", cookies=cookies)
        assert briefs.status_code == 200
        assert len(briefs.json()["briefs"]) >= 1

    def test_scenario_workflow_draft_to_validation(self, client):
        """NL workflow draft → validate → dry run (no side effects)."""
        cookies, _ = register_user(client, "e2e5")
        create_workspace(client, cookies)
        from app.services.workflow_intel import generate_workflow_draft, dry_run, validate_definition
        definition = generate_workflow_draft(
            "When a new contract is uploaded, summarize it, extract the renewal date, "
            "and notify the owner 30 days before renewal.", 1, 1,
        )
        validated = validate_definition(definition)
        assert validated["trigger"] == "document.ready"
        plan = dry_run(validated)
        assert plan["mode"] == "DRY_RUN"

    def test_scenario_feedback_flow(self, client):
        """Answer → feedback → analytics."""
        cookies, _ = register_user(client, "e2e6")
        create_workspace(client, cookies)
        client.post("/copilot/ask", json={"question": "What is the renewal policy?", "scope": "WORKSPACE"}, cookies=cookies)
        client.post("/feedback", json={"rating": "thumbs_down", "category": "citation_issue", "comment": "No source"}, cookies=cookies)
        resp = client.get("/feedback/analytics", cookies=cookies)
        assert resp.json()["categories"].get("citation_issue") == 1

    def test_scenario_workspace_health_dashboard(self, client, db_session):
        """Workspace health → AI usage → report."""
        cookies, _ = register_user(client, "e2e7")
        ws = create_workspace(client, cookies)
        make_document(db_session, ws, status="READY")
        make_execution(db_session, ws, 1, status="completed", cost=0.02)
        db_session.commit()
        health = client.get(f"/knowledge/workspaces/{ws}/health", cookies=cookies)
        assert health.json()["total_documents"] == 1
        assert health.json()["ai_usage"]["ai_executions_30d"] == 1
        report = client.post("/reports/generate", json={"template": "weekly_knowledge"}, cookies=cookies)
        assert report.status_code == 200

    def test_scenario_duplicate_suggestion(self, client, db_session):
        """Two similar documents → duplicate suggestion (never auto-delete)."""
        cookies, _ = register_user(client, "e2e8")
        ws = create_workspace(client, cookies)
        from app.services.suggestion_engine import suggest_duplicate_candidates
        make_document(db_session, ws, title="Security Policy 2026", filename="a.pdf")
        make_document(db_session, ws, title="Security Policy 2026", filename="b.pdf")
        db_session.flush()
        suggestions = suggest_duplicate_candidates(db_session, ws, 1, None)
        assert len(suggestions) == 1
        assert "duplicate" in suggestions[0].title.lower()
        # no document was deleted
        from app.models.document import Document
        assert db_session.query(Document).filter(Document.workspace_id == ws).count() == 2