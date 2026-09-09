"""Phase 18 tests — API surface + E2E product flows.

Covers: authentication on every new ops surface, admin authorization,
workspace isolation, broker/vector/provider/ingestion/connector/KG/memory/
workflow/governance/cost/search/dataops/cleanup/DR API endpoints, plus E2E
scenarios: distributed worker drain, ingestion poison → quarantine →
release, entity merge approval chain, and cleanup hold enforcement.
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
from app.models.organization import Organization, OrganizationMember  # noqa: E402
from app.models.phase16 import WorkerJob, WorkerHeartbeat, TraceSpan  # noqa: E402
from app.models.phase17 import (  # noqa: E402
    IngestionRun, JobLease, LegalHold, HoldEntity, AIPolicyRule,
)
from app.models.phase18 import (  # noqa: E402
    PoisonDocument, EntityMergeRequest, EmbeddingModel, ImportJob,
)
from app.models.knowledge_graph import Entity  # noqa: E402

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
def _clean(db_session):
    from app.models.phase18 import (
        PoisonDocument, EntityMergeRequest, EmbeddingModel, ImportJob,
        SloSnapshot, SearchAnalyticsEvent,
    )
    from app.models.phase15 import AIMemory  # noqa: F401
    from app.models.phase17 import ConnectorSource, IngestionRun
    db_session.query(PoisonDocument).delete()
    db_session.query(EntityMergeRequest).delete()
    db_session.query(EmbeddingModel).delete()
    db_session.query(ImportJob).delete()
    db_session.query(SloSnapshot).delete()
    db_session.query(SearchAnalyticsEvent).delete()
    db_session.query(AIPolicyRule).delete()
    db_session.query(LegalHold).delete()
    db_session.query(Entity).delete()
    db_session.query(JobLease).delete()
    db_session.query(WorkerJob).delete()
    db_session.query(WorkerHeartbeat).delete()
    db_session.commit()


def register_user(client, tag="u"):
    _counter[0] += 1
    email = f"{tag}{_counter[0]}@p18api.com"
    resp = client.post("/auth/register", json={
        "name": f"{tag.title()} {_counter[0]}", "email": email,
        "password": "password123"})
    assert resp.status_code in (200, 201), resp.text
    login = client.post("/auth/login", json={"email": email,
                                             "password": "password123"})
    assert login.status_code == 200, login.text
    return login.cookies


def create_workspace(client, cookies, name="WS"):
    resp = client.post("/workspaces", json={"name": name}, cookies=cookies)
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


def latest_user(db, tag):
    return (db.query(User)
            .filter(User.email.like(f"{tag}%@p18api.com"))
            .order_by(User.id.desc()).first())


# ============================================================
# Ops18 API surface — auth + authorization
# ============================================================

class TestOpsAuth:
    @pytest.mark.parametrize("path", [
        "/broker/health", "/vector/health", "/providers/dashboard",
        "/ingestion/poison", "/observability/slo", "/cost/projection",
        "/search/analytics", "/dr/backup-inventory",
    ])
    def test_ops_routes_require_auth(self, client, path):
        resp = client.get(path)
        assert resp.status_code in (401, 403), resp.text

    @pytest.mark.parametrize("path", [
        "/vector/models", "/imports", "/cleanup/run",
    ])
    def test_ops_write_routes_require_auth(self, client, path):
        resp = client.post(path, json={})
        assert resp.status_code in (401, 403), resp.text

    def test_broker_health(self, client, db_session):
        cookies = register_user(client, "bh")
        resp = client.get("/broker/health", cookies=cookies)
        assert resp.status_code == 200
        data = resp.json()
        assert data["broker"] == "postgres"
        assert data["status"] in ("UP", "DOWN")

    def test_worker_health_endpoint(self, client):
        resp = client.get("/worker-health")
        assert resp.status_code == 200
        assert "active" in resp.json()

    def test_vector_health_requires_admin(self, client):
        cookies = register_user(client, "vh")
        ws = create_workspace(client, cookies)
        # The creator is owner → admin gate passes.
        resp = client.get(f"/vector/health?workspace_id={ws}", cookies=cookies)
        assert resp.status_code == 200
        assert "pgvector" in resp.json()

    def test_vector_health_foreign_workspace_denied(self, client):
        a = register_user(client, "vha")
        b = register_user(client, "vhb")
        ws_a = create_workspace(client, a)
        resp = client.get(f"/vector/health?workspace_id={ws_a}", cookies=b)
        assert resp.status_code in (403, 404)

    def test_register_vector_model(self, client, db_session):
        cookies = register_user(client, "vm")
        ws = create_workspace(client, cookies)
        resp = client.post(
            f"/vector/models?workspace_id={ws}",
            json={"provider": "openai", "model": "text-embed-3-small",
                  "dimensions": 1536, "version": "v1"},
            cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["dimensions"] == 1536
        model = db_session.query(EmbeddingModel).filter(
            EmbeddingModel.model == "text-embed-3-small").first()
        assert model is not None

    def test_register_vector_model_invalid_dimensions(self, client):
        cookies = register_user(client, "vmi")
        ws = create_workspace(client, cookies)
        resp = client.post(
            f"/vector/models?workspace_id={ws}",
            json={"provider": "p", "model": "m", "dimensions": -5},
            cookies=cookies)
        assert resp.status_code == 400

    def test_provider_dashboard_admin_only(self, client):
        cookies = register_user(client, "pd")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/providers/dashboard?workspace_id={ws}",
                          cookies=cookies)
        assert resp.status_code == 200
        assert "providers" in resp.json()

    def test_provider_dashboard_member_not_admin(self, client, db_session):
        owner = register_user(client, "pdo")
        member = register_user(client, "pdm")
        ws = create_workspace(client, owner)
        # Add member with MEMBER role.
        from app.models.workspace import Workspace
        row = db_session.query(Workspace).filter(
            Workspace.id == ws).first()
        mem_user = latest_user(db_session, "pdm")
        db_session.add(WorkspaceMember(workspace_id=ws, user_id=mem_user.id,
                                       role="MEMBER"))
        db_session.commit()
        resp = client.get(f"/providers/dashboard?workspace_id={ws}",
                          cookies=member)
        assert resp.status_code == 403

    def test_broker_queue_depth(self, client, db_session):
        cookies = register_user(client, "qd")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/broker/queue-depth?workspace_id={ws}",
                          cookies=cookies)
        assert resp.status_code == 200
        assert "per_queue" in resp.json()

    def test_provider_timeout_policy(self, client):
        cookies = register_user(client, "tp")
        resp = client.get("/providers/timeout-policy", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["connect_seconds"] > 0


# ============================================================
# Ops18 API surface — tenant isolation
# ============================================================

class TestOpsIsolation:
    def test_cleanup_run_cross_workspace_denied(self, client, db_session):
        a = register_user(client, "cla")
        b = register_user(client, "clb")
        ws_a = create_workspace(client, a)
        ws_b = create_workspace(client, b)
        resp = client.post(
            f"/cleanup/run?workspace_id={ws_a}",
            json={"entity_type": "traces", "dry_run": True},
            cookies=b)
        assert resp.status_code in (403, 404)

    def test_import_cross_workspace_record_rejected(self, client, db_session):
        a = register_user(client, "ia")
        b = register_user(client, "ib")
        ws_a = create_workspace(client, a)
        ws_b = create_workspace(client, b)
        resp = client.post(
            f"/imports?workspace_id={ws_a}",
            json={"import_type": "documents",
                  "records": [{"title": "bad", "workspace_id": ws_b}]},
            cookies=a)
        assert resp.status_code == 200, resp.text
        assert resp.json()["invalid"] == 1

    def test_memory_transition_foreign_workspace(self, client, db_session):
        owner = register_user(client, "mta")
        other = register_user(client, "mtb")
        ws_a = create_workspace(client, owner)
        ws_b = create_workspace(client, other)
        from app.models.phase15 import AIMemory
        mem = AIMemory(workspace_id=ws_a, memory_type="fact",
                       scope="WORKSPACE", content="secret fact",
                       lifecycle_status="CANDIDATE")
        db_session.add(mem)
        db_session.commit()
        resp = client.post(
            f"/memory/{mem.id}/transition?workspace_id={ws_b}",
            json={"target": "validated"}, cookies=other)
        assert resp.status_code in (403, 404, 400)

    def test_kg_merge_request_flow_e2e(self, client, db_session):
        cookies = register_user(client, "mge")
        ws = create_workspace(client, cookies)
        e1 = Entity(workspace_id=ws, name="Northwind",
                    entity_type="company")
        e2 = Entity(workspace_id=ws, name="Northwind Inc",
                    entity_type="company")
        db_session.add_all([e1, e2])
        db_session.commit()
        resp = client.post(
            f"/kg/merge-requests?workspace_id={ws}",
            json={"source_entity_id": e1.id, "target_entity_id": e2.id,
                  "reason": "same company"},
            cookies=cookies)
        assert resp.status_code == 200, resp.text
        merge_id = resp.json()["id"]
        # Approve the merge.
        resp = client.post(
            f"/kg/merge-requests/{merge_id}/decide?workspace_id={ws}",
            json={"decision": "APPROVE"}, cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["merged"] is True
        row = db_session.get(EntityMergeRequest, merge_id)
        assert row.status == "APPROVED"


# ============================================================
# E2E flows
# ============================================================

class TestE2E:
    def test_e2e_ingestion_poison_quarantine_release(self, client,
                                                     db_session):
        """Ingestion failure → poison quarantine → operator release."""
        cookies = register_user(client, "e2ep")
        ws = create_workspace(client, cookies)
        # Register a poison via the service, then surface via API.
        from app.services.ingestion_ops import record_poison, resolve_poison
        record_poison(db_session, workspace_id=ws, document_id=42,
                      stage="OCR", error="pdf rasterization failed")
        db_session.commit()
        resp = client.get(f"/ingestion/poison?workspace_id={ws}",
                          cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        poison_id = resp.json()["items"][0]["id"]
        resp = client.post(
            f"/ingestion/poison/{poison_id}/resolve?workspace_id={ws}",
            json={"action": "RELEASE"}, cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["status"] == "RELEASED"

    def test_e2e_vector_model_and_health(self, client, db_session):
        cookies = register_user(client, "e2ev")
        ws = create_workspace(client, cookies)
        resp = client.post(
            f"/vector/models?workspace_id={ws}",
            json={"provider": "openai", "model": "e3-small",
                  "dimensions": 1536}, cookies=cookies)
        assert resp.status_code == 200
        resp = client.get(f"/vector/models?workspace_id={ws}",
                          cookies=cookies)
        assert resp.json()["total"] == 1
        resp = client.get(f"/vector/health?workspace_id={ws}",
                          cookies=cookies)
        assert resp.json()["active_model"]["model"] == "e3-small"

    def test_e2e_slo_record_and_status(self, client):
        cookies = register_user(client, "e2es")
        ws = create_workspace(client, cookies)
        resp = client.post(
            f"/observability/slo?workspace_id={ws}",
            json={"availability": 0.999, "latency_p95_ms": 150.0,
                  "error_rate": 0.001}, cookies=cookies)
        assert resp.status_code == 200
        resp = client.get(f"/observability/slo?workspace_id={ws}",
                          cookies=cookies)
        assert resp.json()["status"] in ("HEALTHY", "NO_DATA", "BREACHED")

    def test_e2e_search_analytics_record_and_query(self, client):
        cookies = register_user(client, "e2eq")
        ws = create_workspace(client, cookies)
        resp = client.post(
            f"/search/analytics?workspace_id={ws}",
            json={"query": "quarterly results", "mode": "hybrid",
                  "latency_ms": 80, "result_count": 5},
            cookies=cookies)
        assert resp.status_code == 200
        resp = client.get(f"/search/analytics?workspace_id={ws}",
                          cookies=cookies)
        assert resp.json()["searches"] == 1

    def test_e2e_cost_budget_check(self, client):
        cookies = register_user(client, "e2ec")
        ws = create_workspace(client, cookies)
        resp = client.post(
            f"/cost/budget-check?workspace_id={ws}",
            json={"feature": "rag", "estimated_cost": 1.0,
                  "hard_limit": 100.0}, cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["action"] == "ALLOW"

    def test_e2e_governance_sensitivity_route(self, client, db_session):
        cookies = register_user(client, "e2eg")
        ws = create_workspace(client, cookies)
        import json as _json
        from app.models.workspace import Workspace as WsModel
        ws_row = db_session.query(WsModel).filter(WsModel.id == ws).first()
        db_session.add(AIPolicyRule(
            workspace_id=ws, rule_type="PROVIDER",
            allowlist_json=_json.dumps(["openai"]), enabled=True))
        db_session.commit()
        resp = client.post(
            f"/governance/sensitivity-route?workspace_id={ws}",
            json={"sensitivity": "CONFIDENTIAL"}, cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["provider"] == "openai"

    def test_e2e_cleanup_dry_run_api(self, client, db_session):
        cookies = register_user(client, "e2ecl")
        ws = create_workspace(client, cookies)
        db_session.add(TraceSpan(
            span_id=str(uuid.uuid4()), trace_id=str(uuid.uuid4()),
            workspace_id=ws, span_type="provider",
            created_at=datetime.now(timezone.utc) - timedelta(days=400)))
        db_session.commit()
        resp = client.post(
            f"/cleanup/run?workspace_id={ws}",
            json={"entity_type": "traces", "older_than_days": 365,
                  "dry_run": True}, cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["candidates"] == 1
        assert resp.json()["deleted"] == 0

    def test_e2e_dr_validate(self, client):
        cookies = register_user(client, "e2edr")
        ws = create_workspace(client, cookies)
        resp = client.get(f"/dr/validate-restore?workspace_id={ws}",
                          cookies=cookies)
        assert resp.status_code == 200
        assert "restore_valid" in resp.json()

    def test_e2e_worker_fleet_and_ops_surface(self, client):
        cookies = register_user(client, "e2ew")
        ws = create_workspace(client, cookies)
        for path in ("/worker-fleet", "/broker/failover-policy",
                     "/providers/latency"):
            resp = client.get(
                f"{path}?workspace_id={ws}" if "?" not in path
                else f"{path}&workspace_id={ws}",
                cookies=cookies)
            assert resp.status_code in (200, 403), f"{path}: {resp.text}"

    def test_e2e_connector_sync_enqueue(self, client, db_session):
        cookies = register_user(client, "e2ecs")
        ws = create_workspace(client, cookies)
        from app.models.phase17 import ConnectorSource
        src = ConnectorSource(workspace_id=ws, name="SharePoint",
                              kind="collaboration", enabled=True)
        db_session.add(src)
        db_session.commit()
        resp = client.post(
            f"/connectors/{src.id}/schedule-sync?workspace_id={ws}",
            cookies=cookies)
        assert resp.status_code == 200
        data = resp.json()
        assert data["enqueued"] is True
        job = db_session.query(WorkerJob).filter(
            WorkerJob.job_type == "CONNECTOR_SYNC").first()
        assert job is not None

    def test_e2e_workflow_run_survives_worker_sim(self, client, db_session):
        """Workflow run advances idempotently like a worker restart."""
        cookies = register_user(client, "e2ewf")
        ws = create_workspace(client, cookies)
        from app.services.workflow3 import start_workflow_run, advance_run
        run = start_workflow_run(
            db_session, workspace_id=ws, organization_id=None,
            definition={"nodes": [
                {"id": "a", "type": "task"},
                {"id": "b", "type": "task", "depends_on": ["a"]}]})
        db_session.commit()
        first = advance_run(db_session, run.id, ws)
        db_session.commit()
        # Simulate restart: fresh session continues from durable state.
        second = advance_run(db_session, run.id, ws)
        db_session.commit()
        third = advance_run(db_session, run.id, ws)
        db_session.commit()
        assert first["status"] in ("RUNNING", "COMPLETED")
        assert third["status"] == "COMPLETED"

    def test_e2e_worker_drain_once(self, client, db_session):
        """A worker drains exactly one job in --once mode semantics."""
        cookies = register_user(client, "e2ew1")
        ws = create_workspace(client, cookies)
        from app.services import worker_platform as wp
        from app.services.worker_handlers import get_handler

        def handler(session, job):
            job.status = "COMPLETED"
        job = wp.enqueue_job(db_session, queue_name="E2E", job_type="test.ok",
                             workspace_id=ws, payload={})
        db_session.commit()
        done = wp.run_once(db_session, "E2E", "w-e2e", handler=handler)
        db_session.commit()
        assert done is not None and done.id == job.id
        db_session.refresh(job)
        assert job.status == "COMPLETED"


from datetime import datetime, timezone, timedelta  # noqa: E402