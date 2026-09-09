"""Phase 16 API + security tests — worker/ops endpoints, backfill API, page
intelligence API, policy API, webhook SSRF gate, and error semantics.
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
from app.models.document import Document  # noqa: E402
from app.models.webhook import WebhookEndpoint, WebhookEvent, WebhookDelivery  # noqa: E402
from app.models.phase16 import TraceSpan  # noqa: E402
from app.models.phase15 import PolicyStatement  # noqa: E402
from app.services.trace_service import start_span, end_span  # noqa: E402
from app.services.webhook2 import validate_delivery_url, process_due_webhooks  # noqa: E402
from app.services import worker_platform as wp  # noqa: E402

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


@pytest.fixture(autouse=True)
def _clean_jobs(db_session):
    from app.models.phase16 import WorkerJob, WorkerHeartbeat
    db_session.query(WorkerJob).delete()
    db_session.query(WorkerHeartbeat).delete()
    db_session.commit()


def register_user(client, tag="u"):
    _counter[0] += 1
    email = f"{tag}{_counter[0]}@p16-test.com"
    resp = client.post("/auth/register", json={
        "name": f"{tag.title()} {_counter[0]}", "email": email,
        "password": "password123",
    })
    assert resp.status_code in (200, 201), resp.text
    login = client.post("/auth/login", json={
        "email": email, "password": "password123"})
    assert login.status_code == 200, login.text
    return login.cookies


def create_workspace(client, cookies, name="WS"):
    resp = client.post("/workspaces", json={"name": name}, cookies=cookies)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


def create_org(client, cookies, name="Org"):
    _counter[0] += 1
    resp = client.post("/organizations", json={
        "name": name, "slug": f"p16-org-{uuid.uuid4().hex[:8]}"}, cookies=cookies)
    assert resp.status_code == 201, resp.text
    return resp.json()


def user_by_email(db, email):
    return db.query(User).filter(User.email == email).first()


def latest_user(db, tag):
    return (
        db.query(User)
        .filter(User.email.like(f"{tag}%@p16-test.com"))
        .order_by(User.id.desc())
        .first()
    )


# ============================================================
# Worker jobs API
# ============================================================

class TestWorkerJobsAPI:
    def test_list_jobs(self, client, db_session):
        cookies = register_user(client, "wjob")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/worker-jobs?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert "items" in resp.json()

    def test_list_jobs_requires_auth(self, client):
        assert client.get("/worker-jobs?workspace_id=1").status_code == 401

    def test_foreign_workspace_jobs_denied(self, client):
        cookies_a = register_user(client, "wjoba")
        ws_a = create_workspace(client, cookies_a)
        cookies_b = register_user(client, "wjobb")
        resp = client.get(f"/worker-jobs?workspace_id={ws_a}", cookies=cookies_b)
        assert resp.status_code in (403, 404)

    def test_cancel_job_endpoint(self, client, db_session):
        cookies = register_user(client, "wjc")
        ws = create_workspace(client, cookies)
        owner = latest_user(db_session, "wjc")
        job = wp.enqueue_job(db_session, "AI_EXECUTIONS", "probe", ws,
                             user_id=owner.id)
        db_session.commit()
        resp = client.post(f"/worker-jobs/{job.id}/cancel", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["status"] == "CANCELLED"

    def test_cancel_missing_job_404(self, client):
        cookies = register_user(client, "wjm")
        resp = client.post("/worker-jobs/9999999/cancel", cookies=cookies)
        assert resp.status_code == 404


# ============================================================
# Ops API (admin gated)
# ============================================================

class TestOpsAPI:
    def test_queue_metrics_workspace_admin(self, client):
        cookies = register_user(client, "opsm")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/ops/queue-metrics?workspace_id={ws}",
                          cookies=cookies)
        assert resp.status_code == 200
        assert "statuses" in resp.json()

    def test_queue_metrics_requires_auth(self, client):
        assert client.get("/ops/queue-metrics?workspace_id=1").status_code == 401

    def test_queue_metrics_foreign_denied(self, client):
        cookies_a = register_user(client, "opsa")
        ws_a = create_workspace(client, cookies_a)
        cookies_b = register_user(client, "opsb")
        resp = client.get(f"/ops/queue-metrics?workspace_id={ws_a}",
                          cookies=cookies_b)
        assert resp.status_code == 403

    def test_vector_status(self, client):
        cookies = register_user(client, "opsv")
        resp = client.get("/ops/vector-status", cookies=cookies)
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_backend"] == "json_fallback"  # SQLite test env
        assert body["pgvector_available"] is False

    def test_vector_status_requires_auth(self, client):
        assert client.get("/ops/vector-status").status_code == 401

    def test_traces_scoped_listing(self, client, db_session):
        cookies = register_user(client, "opst")
        ws = create_workspace(client, cookies)
        span = start_span(db_session, ws, "retrieval", input_summary="q")
        end_span(db_session, span)
        db_session.commit()
        resp = client.get(f"/ops/traces?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_traces_foreign_workspace_denied(self, client):
        cookies_a = register_user(client, "opsta")
        ws_a = create_workspace(client, cookies_a)
        cookies_b = register_user(client, "opstb")
        resp = client.get(f"/ops/traces?workspace_id={ws_a}", cookies=cookies_b)
        assert resp.status_code in (403, 404)

    def test_trace_summary(self, client):
        cookies = register_user(client, "opssum")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/ops/trace-summary?workspace_id={ws}",
                          cookies=cookies)
        assert resp.status_code == 200
        assert "spans" in resp.json()

    def test_capabilities_list_admin(self, client):
        cookies = register_user(client, "opscap")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/ops/capabilities?workspace_id={ws}",
                          cookies=cookies)
        assert resp.status_code == 200

    def test_capability_create_admin(self, client):
        cookies = register_user(client, "opscap2")
        ws = create_workspace(client, cookies)
        resp = client.post(f"/ops/capabilities?workspace_id={ws}", json={
            "provider": "acme", "model": "m1", "supports_tools": True,
        }, cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["model"] == "m1"

    def test_capability_create_requires_admin(self, client):
        cookies_a = register_user(client, "opscap3a")
        ws = create_workspace(client, cookies_a)
        cookies_b = register_user(client, "opscap3b")
        resp = client.post(f"/ops/capabilities?workspace_id={ws}", json={
            "provider": "acme", "model": "m1"}, cookies=cookies_b)
        assert resp.status_code == 403

    def test_provider_status_admin(self, client):
        cookies = register_user(client, "opsps")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/ops/provider-status?workspace_id={ws}",
                          cookies=cookies)
        assert resp.status_code == 200

    def test_gateway_status_no_secrets(self, client):
        cookies = register_user(client, "opsgs")
        resp = client.get("/ops/gateway-status", cookies=cookies)
        assert resp.status_code == 200
        body = resp.json()
        assert "api_key" not in body
        assert "real_gateway_configured" in body

    def test_org_admin_gate(self, client):
        cookies = register_user(client, "opsorg")
        org = create_org(client, cookies)
        resp = client.get(f"/ops/queue-metrics?organization_id={org['id']}",
                          cookies=cookies)
        assert resp.status_code == 200
        other = register_user(client, "opsorg2")
        resp = client.get(f"/ops/queue-metrics?organization_id={org['id']}",
                          cookies=other)
        assert resp.status_code == 403


# ============================================================
# Backfill API
# ============================================================

class TestBackfillAPI:
    def test_audit_requires_org_admin(self, client):
        cookies = register_user(client, "bfaud")
        org = create_org(client, cookies)
        resp = client.post("/admin/backfill/audit",
                           json={"organization_id": org["id"]},
                           cookies=cookies)
        assert resp.status_code == 200
        assert "documents_without_workspace" in resp.json()

    def test_audit_denied_for_non_member(self, client):
        cookies_a = register_user(client, "bfaa")
        org = create_org(client, cookies_a)
        cookies_b = register_user(client, "bfab")
        resp = client.post("/admin/backfill/audit",
                           json={"organization_id": org["id"]},
                           cookies=cookies_b)
        assert resp.status_code == 403

    def test_audit_requires_auth(self, client):
        assert client.post("/admin/backfill/audit",
                           json={"organization_id": 1}).status_code == 401

    def test_create_run_dry_run(self, client, db_session):
        cookies = register_user(client, "bfrun")
        org = create_org(client, cookies)
        owner = latest_user(db_session, "bfrun")
        ws = create_workspace(client, cookies)
        # attach workspace to the org for realism
        ws_row = db_session.query(Workspace).filter(
            Workspace.id == ws).first()
        ws_row.organization_id = org["id"]
        db_session.commit()
        doc = Document(
            user_id=owner.id, workspace_id=None,
            original_filename="legacy.txt",
            storage_key=f"legacy-{uuid.uuid4().hex}",
            mime_type="text/plain", file_size=5, status="READY",
        )
        db_session.add(doc)
        db_session.commit()
        resp = client.post("/admin/backfill/runs", json={
            "organization_id": org["id"], "kind": "document_workspace",
            "dry_run": True, "user_id": owner.id,
        }, cookies=cookies)
        assert resp.status_code == 201
        assert resp.json()["dry_run"] is True
        run_id = resp.json()["id"]
        run_resp = client.post(f"/admin/backfill/runs/{run_id}/execute",
                               json={"organization_id": org["id"],
                                     "complete": True}, cookies=cookies)
        assert run_resp.status_code == 200
        body = run_resp.json()
        assert body["assigned"] == 1
        db_session.refresh(doc)
        assert doc.workspace_id is None  # dry run wrote nothing

    def test_execute_real_run(self, client, db_session):
        cookies = register_user(client, "bfx")
        org = create_org(client, cookies)
        owner = latest_user(db_session, "bfx")
        ws = create_workspace(client, cookies)
        ws_row = db_session.query(Workspace).filter(
            Workspace.id == ws).first()
        ws_row.organization_id = org["id"]
        db_session.commit()
        doc = Document(
            user_id=owner.id, workspace_id=None,
            original_filename="legacy2.txt",
            storage_key=f"legacy2-{uuid.uuid4().hex}",
            mime_type="text/plain", file_size=5, status="READY",
        )
        db_session.add(doc)
        db_session.commit()
        resp = client.post("/admin/backfill/runs", json={
            "organization_id": org["id"], "kind": "document_workspace",
            "dry_run": False, "user_id": owner.id,
        }, cookies=cookies)
        run_id = resp.json()["id"]
        exec_resp = client.post(f"/admin/backfill/runs/{run_id}/execute",
                                json={"organization_id": org["id"],
                                      "complete": True}, cookies=cookies)
        assert exec_resp.status_code == 200
        assert exec_resp.json()["assigned"] == 1
        db_session.refresh(doc)
        assert doc.workspace_id == ws

    def test_list_runs(self, client):
        cookies = register_user(client, "bflist")
        org = create_org(client, cookies)
        client.post("/admin/backfill/audit",
                    json={"organization_id": org["id"]}, cookies=cookies)
        resp = client.get(f"/admin/backfill/runs?organization_id={org['id']}",
                          cookies=cookies)
        assert resp.status_code == 200

    def test_invalid_kind_400(self, client):
        cookies = register_user(client, "bfbad")
        org = create_org(client, cookies)
        resp = client.post("/admin/backfill/runs", json={
            "organization_id": org["id"], "kind": "nonsense"}, cookies=cookies)
        assert resp.status_code == 400


# ============================================================
# Document pages API
# ============================================================

class TestPagesAPI:
    def _upload_and_pages(self, client, db_session):
        from tests.test_phase16_api import register_user as ru
        cookies = register_user(client, "pgapi")
        ws = create_workspace(client, cookies)
        owner = latest_user(db_session, "pgapi")
        doc = Document(
            user_id=owner.id, workspace_id=ws,
            original_filename="p.txt",
            storage_key=f"p-{uuid.uuid4().hex}",
            mime_type="text/plain", file_size=5, status="READY",
        )
        db_session.add(doc)
        db_session.commit()
        return cookies, ws, doc

    def test_ingest_and_list_pages(self, client, db_session):
        cookies, ws, doc = self._upload_and_pages(client, db_session)
        resp = client.post(f"/documents/{doc.id}/pages", json={
            "version": 1,
            "pages": [{"page_number": 1,
                       "text": "Heading\n\nBody paragraph here.\n"}],
        }, cookies=cookies)
        assert resp.status_code == 201, resp.text
        listed = client.get(f"/documents/{doc.id}/pages", cookies=cookies)
        assert listed.status_code == 200
        assert listed.json()["count"] == 1

    def test_pages_requires_auth(self, client):
        assert client.post("/documents/1/pages", json={
            "version": 1, "pages": []}).status_code == 401

    def test_foreign_document_pages_denied(self, client, db_session):
        cookies_a = register_user(client, "pgb")
        ws_a = create_workspace(client, cookies_a)
        owner_a = latest_user(db_session, "pgb")
        doc = Document(
            user_id=owner_a.id, workspace_id=ws_a,
            original_filename="x.txt", storage_key=f"x-{uuid.uuid4().hex}",
            mime_type="text/plain", file_size=5, status="READY",
        )
        db_session.add(doc)
        db_session.commit()
        cookies_b = register_user(client, "pgc")
        resp = client.get(f"/documents/{doc.id}/pages", cookies=cookies_b)
        assert resp.status_code in (403, 404)

    def test_missing_document_404(self, client):
        cookies = register_user(client, "pgd")
        assert client.get("/documents/999999/pages",
                          cookies=cookies).status_code == 404

    def test_region_search(self, client, db_session):
        cookies, ws, doc = self._upload_and_pages(client, db_session)
        client.post(f"/documents/{doc.id}/pages", json={
            "version": 1,
            "pages": [{"page_number": 1,
                       "text": "2. Overview\n\nBody paragraph here.\n"}],
        }, cookies=cookies)
        resp = client.get(f"/documents/{doc.id}/pages/regions?kind=heading",
                          cookies=cookies)
        assert resp.status_code == 200
        assert any(r["kind"] == "heading" for r in resp.json()["items"])


# ============================================================
# Policy intelligence API
# ============================================================

class TestPolicyAPI:
    def test_semanticize(self, client):
        cookies = register_user(client, "pol")
        resp = client.post("/policy-intelligence/semanticize", json={
            "statement": "Approval required for expenses above $1,000."},
            cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["threshold"] == 1000.0

    def test_semanticize_requires_auth(self, client):
        assert client.post("/policy-intelligence/semanticize",
                           json={"statement": "x"}).status_code == 401

    def test_compare_verdict(self, client):
        cookies = register_user(client, "polc")
        resp = client.post("/policy-intelligence/compare", json={
            "statement_a": "Approval required for expenses above $1,000.",
            "statement_b": "Approval is not required for expenses above $1,000.",
        }, cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["result"] == "CONTRADICTS"

    def test_scan_and_list_conflicts(self, client, db_session):
        cookies = register_user(client, "pols")
        ws = create_workspace(client, cookies)
        db_session.add(PolicyStatement(
            workspace_id=ws, statement=(
                "Approval required for expenses above $1,000.")))
        db_session.add(PolicyStatement(
            workspace_id=ws, statement=(
                "Expenses below $2,000 require no approval.")))
        db_session.commit()
        resp = client.post("/policy-intelligence/scan",
                           json={"workspace_id": ws}, cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["conflicts_recorded"] == 1
        listed = client.get(f"/policy-intelligence/conflicts?workspace_id={ws}",
                            cookies=cookies)
        assert listed.status_code == 200
        assert listed.json()["count"] >= 1

    def test_scan_foreign_workspace_denied(self, client):
        cookies_a = register_user(client, "polx1")
        ws_a = create_workspace(client, cookies_a)
        cookies_b = register_user(client, "polx2")
        resp = client.post("/policy-intelligence/scan",
                           json={"workspace_id": ws_a}, cookies=cookies_b)
        assert resp.status_code in (403, 404)


# ============================================================
# Webhook SSRF gate + delivery worker
# ============================================================

class TestWebhookSafety:
    def test_block_loopback(self):
        ok, _ = validate_delivery_url("http://localhost:9000/hook")
        assert not ok

    def test_block_metadata_ip(self):
        ok, _ = validate_delivery_url("http://169.254.169.254/latest/meta-data")
        assert not ok

    def test_block_private_range(self):
        ok, _ = validate_delivery_url("http://10.0.0.5/hook")
        assert not ok
        ok, _ = validate_delivery_url("http://192.168.1.5/hook")
        assert not ok
        ok, _ = validate_delivery_url("http://127.0.0.1/hook")
        assert not ok

    def test_block_internal_dns_names(self):
        ok, _ = validate_delivery_url("http://metadata.google.internal/hook")
        assert not ok
        ok, _ = validate_delivery_url("http://db.internal/hook")
        assert not ok

    def test_block_non_http_schemes(self):
        ok, _ = validate_delivery_url("file:///etc/passwd")
        assert not ok
        ok, _ = validate_delivery_url("ftp://example.com/x")
        assert not ok

    def test_allow_public_https(self):
        ok, reason = validate_delivery_url("https://hooks.example.com/flow")
        assert ok

    def test_worker_blocks_unsafe_endpoint(self, client, db_session):
        user = User(name="Webhook SSRF", email=f"wssrf-{uuid.uuid4().hex[:8]}@p16-test.com",
                    password_hash="x")
        db_session.add(user)
        db_session.flush()
        ws = Workspace(name="wh-ssrf", owner_id=user.id)
        db_session.add(ws)
        db_session.flush()
        endpoint = WebhookEndpoint(
            workspace_id=ws.id, user_id=user.id,
            url="http://127.0.0.1:1/hook",
            secret_hash="not-a-secret-for-test", events_json='["*"]',
        )
        db_session.add(endpoint)
        db_session.flush()
        event = WebhookEvent(
            event_id=f"evt-{uuid.uuid4().hex[:12]}",
            event_type="document.created", workspace_id=ws.id,
            payload_json="{}",
        )
        db_session.add(event)
        db_session.flush()
        delivery = WebhookDelivery(event_id=event.id, endpoint_id=endpoint.id,
                                   status="PENDING")
        db_session.add(delivery)
        db_session.commit()
        result = process_due_webhooks(db_session, max_jobs=5)
        assert result["blocked"] == 1
        db_session.commit()
        db_session.refresh(delivery)
        assert delivery.status == "FAILED"
        assert "SSRF" in (delivery.last_error or "")

    def test_delivery_worker_success(self, client, db_session, monkeypatch):
        import urllib.request

        class FakeResp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=10):
            return FakeResp()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        user = User(name="Webhook OK", email=f"wok-{uuid.uuid4().hex[:8]}@p16-test.com",
                    password_hash="x")
        db_session.add(user)
        db_session.flush()
        ws = Workspace(name="wh-ok", owner_id=user.id)
        db_session.add(ws)
        db_session.flush()
        endpoint = WebhookEndpoint(
            workspace_id=ws.id, user_id=user.id,
            url="https://hooks.example.com/flow",
            secret_hash="s3cret-value-for-signing", events_json='["*"]',
        )
        db_session.add(endpoint)
        db_session.flush()
        event = WebhookEvent(
            event_id=f"evt-{uuid.uuid4().hex[:12]}",
            event_type="document.created", workspace_id=ws.id,
            payload_json="{}",
        )
        db_session.add(event)
        db_session.flush()
        delivery = WebhookDelivery(event_id=event.id, endpoint_id=endpoint.id,
                                   status="PENDING")
        db_session.add(delivery)
        db_session.commit()
        result = process_due_webhooks(db_session, max_jobs=5)
        assert result["delivered"] == 1
        db_session.commit()
        db_session.refresh(delivery)
        assert delivery.status == "DELIVERED"
        assert delivery.attempt_count == 1


# ============================================================
# Error semantics / bounds
# ============================================================

class TestErrorSemantics:
    def test_invalid_payload_422(self, client):
        cookies = register_user(client, "err422")
        resp = client.post("/worker-jobs/abc/cancel", cookies=cookies)
        assert resp.status_code == 422

    def test_not_found_404(self, client):
        cookies = register_user(client, "err404")
        resp = client.get("/ops/traces?workspace_id=987654",
                          cookies=cookies)
        assert resp.status_code in (403, 404)

    def test_list_bound_limit(self, client):
        cookies = register_user(client, "errlim")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/worker-jobs?workspace_id={ws}&limit=10000",
                          cookies=cookies)
        assert resp.status_code == 200

    def test_idempotency_conflict_preserved_409(self, client):
        cookies = register_user(client, "err409")
        ws = create_workspace(client, cookies)
        key = f"api16-{uuid.uuid4().hex[:10]}"
        first = client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "x",
            "idempotency_key": key, "query": "one"}, cookies=cookies)
        assert first.status_code == 201
        second = client.post("/executions", json={
            "workspace_id": ws, "execution_type": "rag", "task_type": "x",
            "idempotency_key": key, "query": "two"}, cookies=cookies)
        assert second.status_code == 409
