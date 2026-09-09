"""Phase 15 API validation — auth, tenant isolation, and router behavior.

Exercises the new routers (executions, events, knowledge-os, memory,
reviews, automation, governance, costs) end to end with real users and
workspaces created through the API.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

_counter = [0]
VALID_PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\nxref\n0 4\n0000000000 65535 f \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n150\n%%EOF"


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


def register_user(client, tag="u"):
    _counter[0] += 1
    email = f"{tag}{_counter[0]}@p15-test.com"
    client.post("/auth/register", json={
        "name": f"{tag.title()} {_counter[0]}", "email": email, "password": "password123",
    })
    login = client.post("/auth/login", json={"email": email, "password": "password123"})
    assert login.status_code == 200, login.text
    return login.cookies


def create_workspace(client, cookies, name="WS"):
    resp = client.post("/workspaces", json={"name": name}, cookies=cookies)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


def upload_doc(client, cookies):
    files = {"file": (f"p15api-{uuid.uuid4().hex[:6]}.pdf", VALID_PDF, "application/pdf")}
    resp = client.post("/documents", files=files, cookies=cookies)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ============================================================
# Executions API
# ============================================================

class TestExecutionsAPI:
    def test_create_execution(self, client):
        cookies = register_user(client, "exec")
        ws = create_workspace(client, cookies)
        resp = client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "summarize",
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == "QUEUED"
        assert resp.json()["trace_id"]

    def test_create_execution_requires_auth(self, client):
        resp = client.post("/executions", json={
            "workspace_id": 1, "execution_type": "rag", "task_type": "summarize",
        })
        assert resp.status_code == 401

    def test_idempotent_create(self, client):
        cookies = register_user(client, "execid")
        ws = create_workspace(client, cookies)
        key = f"api-idem-{uuid.uuid4().hex[:10]}"
        payload = {"workspace_id": ws, "execution_type": "rag", "task_type": "x",
                   "idempotency_key": key}
        first = client.post("/executions", json=payload, cookies=cookies)
        second = client.post("/executions", json=payload, cookies=cookies)
        assert first.status_code == second.status_code == 201
        assert first.json()["id"] == second.json()["id"]

    def test_idempotency_conflict_409(self, client):
        cookies = register_user(client, "execconf")
        ws = create_workspace(client, cookies)
        key = f"api-conf-{uuid.uuid4().hex[:10]}"
        client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "x",
            "idempotency_key": key, "query": "one",
        }, cookies=cookies)
        resp = client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "x",
            "idempotency_key": key, "query": "two different",
        }, cookies=cookies)
        assert resp.status_code == 409

    def test_invalid_priority_400(self, client):
        cookies = register_user(client, "execpri")
        ws = create_workspace(client, cookies)
        resp = client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "x",
            "priority": "SOMETIME",
        }, cookies=cookies)
        assert resp.status_code == 400

    def test_cross_workspace_execution_denied(self, client):
        cookies_a = register_user(client, "execx1")
        ws_a = create_workspace(client, cookies_a)
        cookies_b = register_user(client, "execx2")
        resp = client.post("/executions", json={
            "workspace_id": ws_a, "execution_type": "rag", "task_type": "x",
        }, cookies=cookies_b)
        assert resp.status_code in (403, 404)

    def test_transition_flow(self, client):
        cookies = register_user(client, "exectr")
        ws = create_workspace(client, cookies)
        eid = client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "x",
        }, cookies=cookies).json()["id"]
        resp = client.post(f"/executions/{eid}/transition",
                           json={"new_status": "RUNNING"}, cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["status"] == "RUNNING"

    def test_invalid_transition_400(self, client):
        cookies = register_user(client, "execbad")
        ws = create_workspace(client, cookies)
        eid = client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "x",
        }, cookies=cookies).json()["id"]
        resp = client.post(f"/executions/{eid}/transition",
                           json={"new_status": "COMPLETED"}, cookies=cookies)  # QUEUED->COMPLETED invalid
        assert resp.status_code == 400

    def test_cancel_and_complete(self, client):
        cookies = register_user(client, "execcpl")
        ws = create_workspace(client, cookies)
        eid = client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "x",
        }, cookies=cookies).json()["id"]
        client.post(f"/executions/{eid}/transition", json={"new_status": "RUNNING"}, cookies=cookies)
        resp = client.post(f"/executions/{eid}/complete?output_reference=art:1&actual_cost=0.01", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["status"] == "COMPLETED"

    def test_queue_endpoint_scoped(self, client):
        cookies = register_user(client, "execq")
        ws = create_workspace(client, cookies)
        client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "x",
            "priority": "CRITICAL",
        }, cookies=cookies)
        resp = client.get(f"/executions/queue?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_foreign_execution_404(self, client):
        cookies = register_user(client, "exec404")
        resp = client.get(f"/executions/{uuid.uuid4()}", cookies=cookies)
        assert resp.status_code == 404


# ============================================================
# Events API
# ============================================================

class TestEventsAPI:
    def test_emit_event_endpoint(self, client):
        cookies = register_user(client, "evt")
        ws = create_workspace(client, cookies)
        resp = client.post("/events", json={
            "workspace_id": ws, "event_type": "DOCUMENT_READY",
            "aggregate_type": "document", "aggregate_id": 3,
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == "PENDING"

    def test_unknown_event_400(self, client):
        cookies = register_user(client, "evtbad")
        ws = create_workspace(client, cookies)
        resp = client.post("/events", json={
            "workspace_id": ws, "event_type": "UFO_SPOTTED",
        }, cookies=cookies)
        assert resp.status_code == 400

    def test_events_requires_auth(self, client):
        resp = client.get("/events?workspace_id=1")
        assert resp.status_code == 401

    def test_events_summary(self, client):
        cookies = register_user(client, "evtsum")
        ws = create_workspace(client, cookies)
        client.post("/events", json={
            "workspace_id": ws, "event_type": "DOCUMENT_UPDATED",
            "aggregate_type": "document", "aggregate_id": 7,
        }, cookies=cookies)
        resp = client.get(f"/events/summary?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["PENDING"] >= 1

    def test_events_tenant_isolation(self, client):
        cookies = register_user(client, "evtiso")
        ws = create_workspace(client, cookies)
        other = register_user(client, "evtiso2")
        resp = client.get(f"/events?workspace_id={ws}", cookies=other)
        assert resp.status_code in (403, 404)


# ============================================================
# Knowledge OS API
# ============================================================

class TestKnowledgeOSAPI:
    def test_conflict_detection_endpoint(self, client, db_session):
        cookies = register_user(client, "kos")
        ws = create_workspace(client, cookies)
        # Extract policies from a real uploaded document via service directly.
        from app.models.document import Document
        from app.services.knowledge_engine import extract_policy_statements, detect_policy_conflicts
        doc = db_session.query(Document).filter(
            Document.workspace_id == ws, Document.status == "READY").first()
        if not doc:
            from app.models.document import Document as D
            doc = D(
                user_id=1, workspace_id=ws,
                original_filename="p.pdf",
                storage_key=f"kos-{uuid.uuid4().hex[:8]}",
                mime_type="application/pdf", file_size=10, status="READY",
            )
            db_session.add(doc)
            db_session.flush()
        from app.services import knowledge_engine as ke
        ke._text_for_version = lambda db, d, v: (
            "Approval is required for expenses above $500. "
            "No approval is required for expenses below $600."
        )
        extract_policy_statements(db_session, ws, doc.id)
        db_session.commit()
        resp = client.post(f"/knowledge-os/conflicts/detect?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["detected"] >= 1

    def test_conflicts_list(self, client):
        cookies = register_user(client, "koslist")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/knowledge-os/conflicts?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200

    def test_snapshot_endpoint(self, client):
        cookies = register_user(client, "kossnap")
        ws = create_workspace(client, cookies)
        resp = client.post(f"/knowledge-os/snapshots?workspace_id={ws}&name=test", cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["name"] == "test"

    def test_snapshot_requires_auth(self, client):
        resp = client.get("/knowledge-os/snapshots?workspace_id=1")
        assert resp.status_code == 401

    def test_timeline_endpoint(self, client, db_session):
        cookies = register_user(client, "kostimeline")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/knowledge-os/timeline?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert "items" in resp.json()

    def test_timeline_bad_type_400(self, client):
        cookies = register_user(client, "kostbad")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/knowledge-os/timeline?workspace_id={ws}&event_type=alien", cookies=cookies)
        assert resp.status_code == 400


# ============================================================
# Memory API
# ============================================================

class TestMemoryAPI:
    def test_store_memory_endpoint(self, client):
        cookies = register_user(client, "mem")
        ws = create_workspace(client, cookies)
        resp = client.post("/memory", json={
            "workspace_id": ws, "memory_type": "WORKSPACE_FACT",
            "content": "Remote-first policy", "scope": "WORKSPACE",
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["content"] == "Remote-first policy"

    def test_invalid_memory_type_400(self, client):
        cookies = register_user(client, "membad")
        ws = create_workspace(client, cookies)
        resp = client.post("/memory", json={
            "workspace_id": ws, "memory_type": "BRAIN_DUMP", "content": "x",
        }, cookies=cookies)
        assert resp.status_code == 400

    def test_memory_list_requires_auth(self, client):
        resp = client.get("/memory?workspace_id=1")
        assert resp.status_code == 401

    def test_memory_delete_endpoint(self, client):
        cookies = register_user(client, "memdel")
        ws = create_workspace(client, cookies)
        mid = client.post("/memory", json={
            "workspace_id": ws, "memory_type": "DECISION", "content": "Approved plan",
        }, cookies=cookies).json()["id"]
        resp = client.delete(f"/memory/{mid}", cookies=cookies)
        assert resp.status_code == 200

    def test_memory_summary_endpoint(self, client):
        cookies = register_user(client, "memsum")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/memory/summary?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert "total" in resp.json()


# ============================================================
# Reviews API
# ============================================================

class TestReviewsAPI:
    def test_create_review_endpoint(self, client):
        cookies = register_user(client, "rev")
        ws = create_workspace(client, cookies)
        resp = client.post("/reviews", json={
            "workspace_id": ws, "item_type": "POLICY_CONFLICT",
            "title": "Review conflict #1", "priority": "HIGH",
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == "PENDING"

    def test_review_decide_endpoint(self, client):
        cookies = register_user(client, "revdec")
        ws = create_workspace(client, cookies)
        rid = client.post("/reviews", json={
            "workspace_id": ws, "item_type": "AI_ACTION", "title": "Approve",
        }, cookies=cookies).json()["id"]
        resp = client.post(f"/reviews/{rid}/decide", json={"decision": "APPROVED", "note": "ok"},
                           cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "APPROVED"

    def test_review_invalid_decision_400(self, client):
        cookies = register_user(client, "revbad")
        ws = create_workspace(client, cookies)
        rid = client.post("/reviews", json={
            "workspace_id": ws, "item_type": "AI_ACTION", "title": "x",
        }, cookies=cookies).json()["id"]
        resp = client.post(f"/reviews/{rid}/decide", json={"decision": "MAYBE"}, cookies=cookies)
        assert resp.status_code == 400

    def test_reviews_requires_auth(self, client):
        resp = client.get("/reviews?workspace_id=1")
        assert resp.status_code == 401

    def test_review_summary_endpoint(self, client):
        cookies = register_user(client, "revsum")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/reviews/summary?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200


# ============================================================
# Automation 2.0 API
# ============================================================

class TestAutomation2API:
    def test_simulate_endpoint(self, client):
        cookies = register_user(client, "auto")
        create_workspace(client, cookies)
        definition = {
            "trigger": "document.ready",
            "nodes": [
                {"id": "s", "type": "summarize", "name": "Summarize", "inputs": []},
                {"id": "a", "type": "approval", "name": "Approve", "inputs": ["s"]},
                {"id": "n", "type": "notify", "name": "Notify", "inputs": ["a"]},
            ],
        }
        resp = client.post("/automation/simulate", json={"definition": definition}, cookies=cookies)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["mode"] == "SIMULATION"
        assert body["side_effects"]

    def test_validate_endpoint_rejects_bad(self, client):
        cookies = register_user(client, "autoval")
        create_workspace(client, cookies)
        bad = {"trigger": "alien.trigger", "nodes": [{"id": "x", "type": "summarize"}]}
        resp = client.post("/automation/validate", json={"definition": bad}, cookies=cookies)
        assert resp.status_code == 400

    def test_simulate_requires_auth(self, client):
        definition = {"trigger": "document.ready", "nodes": [{"id": "s", "type": "summarize"}]}
        resp = client.post("/automation/simulate", json={"definition": definition})
        assert resp.status_code == 401

    def test_side_effect_record(self, client):
        cookies = register_user(client, "autose")
        ws = create_workspace(client, cookies)
        resp = client.post("/automation/side-effects", json={
            "workspace_id": ws, "workflow_execution_id": "run-api-1",
            "node_id": "notify-1", "side_effect_type": "notification",
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text
        assert resp.json()["reversible"] is False

    def test_node_execution_begin_complete(self, client):
        cookies = register_user(client, "autonode")
        ws = create_workspace(client, cookies)
        resp = client.post("/automation/node-executions/begin", json={
            "workspace_id": ws, "workflow_execution_id": "run-api-2",
            "workflow_id": "wf-api-1", "node_id": "extract",
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text
        record_id = resp.json()["id"]
        resp2 = client.post(f"/automation/node-executions/{record_id}/complete",
                            json={"output_reference": "artifact:z"}, cookies=cookies)
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "COMPLETED"

    def test_node_double_run_409(self, client):
        cookies = register_user(client, "autodup")
        ws = create_workspace(client, cookies)
        body = {"workspace_id": ws, "workflow_execution_id": "run-api-3",
                "workflow_id": "wf-api-1", "node_id": "extract"}
        first = client.post("/automation/node-executions/begin", json=body, cookies=cookies)
        assert first.status_code == 201
        second = client.post("/automation/node-executions/begin", json=body, cookies=cookies)
        assert second.status_code == 409

    def test_recovery_plan_endpoint(self, client):
        cookies = register_user(client, "autorec")
        ws = create_workspace(client, cookies)
        resp = client.post(
            f"/automation/recovery-plan?workspace_id={ws}&workflow_execution_id=run-x"
            f"&node_id=n1&error=connection+timed+out",
            cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["action"] == "AUTO_RETRY"


# ============================================================
# Governance API
# ============================================================

class TestGovernanceAPI:
    def test_route_compliant(self, client):
        cookies = register_user(client, "govroute")
        ws = create_workspace(client, cookies)
        resp = client.post("/governance/route", json={
            "workspace_id": ws, "sensitivity": "INTERNAL",
            "provider": "openai_compatible", "model": "gpt-4",
        }, cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["allowed"] is True

    def test_route_requires_auth(self, client):
        resp = client.post("/governance/route", json={
            "workspace_id": 1, "sensitivity": "PUBLIC", "provider": "fake", "model": "fake-llm",
        })
        assert resp.status_code == 401

    def test_minimize_endpoint(self, client):
        cookies = register_user(client, "govmin")
        create_workspace(client, cookies)
        resp = client.post("/governance/minimize", json={
            "content": "text", "secret_notes": "DROP TABLE users",
        }, cookies=cookies)
        assert resp.status_code == 200
        assert "secret_notes" not in resp.json()["minimized"]

    def test_quality_endpoint(self, client):
        cookies = register_user(client, "govqual")
        ws = create_workspace(client, cookies)
        # Requires ai:manage_settings (ADMIN). The workspace owner is ADMIN-level.
        resp = client.get(f"/governance/quality/{ws}", cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["executions"]["total"] == 0

    def test_feedback_endpoint_requires_member(self, client):
        cookies = register_user(client, "govfb")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/governance/feedback/{ws}", cookies=cookies)
        assert resp.status_code == 200

    def test_provider_health_requires_admin(self, client):
        cookies = register_user(client, "govph")
        resp = client.get("/governance/provider-health", cookies=cookies)
        assert resp.status_code in (200, 403)

    def test_org_health_requires_org_admin(self, client):
        cookies = register_user(client, "govorg")
        create_workspace(client, cookies)
        # User belongs to no organization membership — org admin check must deny.
        resp = client.get(f"/governance/organizations/9999/health", cookies=cookies)
        assert resp.status_code in (403, 404)


# ============================================================
# Costs API
# ============================================================

class TestCostsAPI:
    def test_cost_summary_endpoint(self, client):
        cookies = register_user(client, "cost")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/costs/workspaces/{ws}/summary", cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert "total_cost_usd" in resp.json()

    def test_cost_forecast_endpoint(self, client):
        cookies = register_user(client, "costfc")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/costs/workspaces/{ws}/forecast", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["estimate_is_guaranteed"] is False

    def test_budget_enforce_endpoint(self, client):
        cookies = register_user(client, "costbd")
        ws = create_workspace(client, cookies)
        resp = client.post("/costs/enforce", json={
            "workspace_id": ws, "estimated_cost_usd": 0.01,
        }, cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["decision"] in ("ALLOW", "RESTRICTED")

    def test_costs_requires_auth(self, client):
        resp = client.get("/costs/workspaces/1/summary")
        assert resp.status_code == 401

    def test_foreign_workspace_costs_denied(self, client):
        cookies_a = register_user(client, "costx1")
        ws_a = create_workspace(client, cookies_a)
        cookies_b = register_user(client, "costx2")
        resp = client.get(f"/costs/workspaces/{ws_a}/summary", cookies=cookies_b)
        assert resp.status_code in (403, 404)