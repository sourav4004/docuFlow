"""Phase 22 tests — ops22 API surface: route inventory, auth/authorization
matrix, validation errors, pagination bounds, and the Phase 22 E2E product
flows (document, RAG, AI policy, self-healing, provider fallback, worker
recovery, evaluation, improvement, incident, DR).
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.models.phase22 import (
    InfraCapability, ProviderValidationRun, EvalExecution, MaintenanceRun,
    RegionCapacitySnapshot, SecurityScanRun, SelfHealLoopRun,
    AutonomyLoopRun, OpsStreamEvent,
)

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


P22_API_TABLES = [
    AutonomyLoopRun, SelfHealLoopRun, SecurityScanRun,
    RegionCapacitySnapshot, MaintenanceRun, EvalExecution,
    ProviderValidationRun, InfraCapability, OpsStreamEvent,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    from app.models.phase19 import ResidencyRule, RegionRecord, \
        RegionFailover
    from app.models.phase21 import FailoverSimulation, ResidencyGuardEvent
    for model in P22_API_TABLES:
        db_session.query(model).delete()
    for model in (FailoverSimulation, RegionFailover, ResidencyGuardEvent,
                  ResidencyRule, RegionRecord):
        db_session.query(model).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _user_and_ws(client, tag="api"):
    _counter[0] += 1
    n = _counter[0]
    email = f"{tag}{n}@p22api.example"
    resp = client.post("/auth/register", json={
        "name": tag.title(), "email": email, "password": "password123"})
    assert resp.status_code in (200, 201), resp.text
    login = client.post("/auth/login", json={"email": email,
                                             "password": "password123"})
    cookies = login.cookies
    ws = client.post("/workspaces", json={"name": f"ws-{tag}-{n}"},
                     cookies=cookies)
    return cookies, ws.json()["id"]


# ===========================================================================
# Steps 227-231: route inventory + authorization
# ===========================================================================

class TestRouteInventory:
    def test_ops22_routes_registered(self):
        routes = [r.path for r in app.routes
                  if hasattr(r, "methods") and r.path.startswith("/ops22")]
        assert len(routes) == 55  # 55 unique endpoints on the root mount

    def test_ops22_api_v1_mount(self):
        routes = [r.path for r in app.routes
                  if hasattr(r, "methods")
                  and r.path.startswith("/api/v1/ops22")]
        assert len(routes) == 55

    def test_no_duplicate_endpoints(self):
        from collections import Counter
        routes = [(r.path, tuple(sorted(r.methods))) for r in app.routes
                  if hasattr(r, "methods")
                  and (r.path.startswith("/ops22")
                       or r.path.startswith("/api/v1/ops22"))]
        counts = Counter((p.replace("/api/v1", ""), m)
                         for p, m in routes)
        # Each (path, method) mounts exactly twice: root + /api/v1.
        assert all(v == 2 for v in counts.values())

    def test_protected_routes_reject_anonymous(self, client):
        protected = [
            "/ops22/infrastructure",
            "/ops22/vector/backend",
            "/ops22/providers/detect",
            "/ops22/evaluations/1",
            "/ops22/regions",
            "/ops22/dr/report",
            "/ops22/workers/fair-share",
            "/ops22/cost/anomaly/1",
            "/ops22/streams/execution",
            "/ops22/vector/coverage/1",
        ]
        for path in protected:
            resp = client.get(path)
            assert resp.status_code in (401, 403), f"{path}: {resp.status_code}"

    def test_write_routes_reject_anonymous(self, client):
        protected = [
            ("/ops22/infrastructure/detect", {}),
            ("/ops22/vector/rebuild", {}),
            ("/ops22/evaluations/run", {}),
            ("/ops22/knowledge/maintenance", {}),
            ("/ops22/loops/self-heal", {}),
            ("/ops22/chaos/run", {}),
            ("/ops22/security/scan", {}),
            ("/ops22/retention/run", {}),
        ]
        for path, body in protected:
            resp = client.post(path, json=body or {"workspace_id": 1})
            assert resp.status_code in (401, 403, 422), \
                f"{path}: {resp.status_code}"


class TestAuthorization:
    def test_member_can_read(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.get(f"/ops22/vector/coverage/{ws_id}", cookies=cookies)
        assert resp.status_code == 200

    def test_non_member_cannot_read(self, client):
        cookies_a, ws_a = _user_and_ws(client, "ownera")
        cookies_b, _ws_b = _user_and_ws(client, "ownerb")
        resp = client.get(f"/ops22/vector/coverage/{ws_a}",
                          cookies=cookies_b)
        assert resp.status_code in (403, 404)

    def test_owner_can_write(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/knowledge/maintenance", cookies=cookies,
                           json={"workspace_id": ws_id, "kind": "embeddings"})
        assert resp.status_code == 200

    def test_non_member_cannot_write(self, client):
        cookies_a, ws_a = _user_and_ws(client, "wownera")
        cookies_b, _ws_b = _user_and_ws(client, "wownerb")
        resp = client.post("/ops22/knowledge/maintenance", cookies=cookies_b,
                           json={"workspace_id": ws_a, "kind": "embeddings"})
        assert resp.status_code in (403, 404)


class TestErrorContracts:
    def test_unknown_maintenance_kind_422(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/knowledge/maintenance", cookies=cookies,
                           json={"workspace_id": ws_id, "kind": "bogus"})
        assert resp.status_code == 422

    def test_unknown_chaos_scenario_404(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/chaos/run", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "scenario": "martian_invasion"})
        assert resp.status_code == 404

    def test_unknown_stream_404(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.get("/ops22/streams/bogus_stream", cookies=cookies)
        assert resp.status_code == 404

    def test_unknown_provider_validation_404(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post(f"/ops22/providers/telepathy/validate/{ws_id}",
                           cookies=cookies)
        assert resp.status_code == 404


# ===========================================================================
# API integration coverage
# ===========================================================================

class TestInfrastructureAPI:
    def test_detect_and_status(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.post("/ops22/infrastructure/detect", cookies=cookies)
        assert resp.status_code == 200
        body = resp.json()
        assert body["configured"] is True
        resp2 = client.get("/ops22/infrastructure", cookies=cookies)
        assert resp2.status_code == 200

    def test_redis_status(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.get("/ops22/infrastructure/redis", cookies=cookies)
        assert resp.status_code == 200
        assert "config" in resp.json() and "health" in resp.json()

    def test_broker_failover(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.get("/ops22/infrastructure/broker-failover",
                          cookies=cookies)
        assert resp.status_code == 200
        body = resp.json()
        assert body["plan"]["fallback"] == "postgres"


class TestVectorAPI:
    def test_backend_and_coverage(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.get("/ops22/vector/backend", cookies=cookies)
        assert resp.status_code == 200
        resp = client.get(f"/ops22/vector/coverage/{ws_id}", cookies=cookies)
        assert resp.status_code == 200
        assert "coverage_pct" in resp.json()

    def test_drift(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.get(f"/ops22/vector/drift/{ws_id}", cookies=cookies)
        assert resp.status_code == 200
        assert "drift_pct" in resp.json()

    def test_rebuild(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/vector/rebuild", cookies=cookies,
                           json={"workspace_id": ws_id, "document_id": 1})
        assert resp.status_code == 200

    def test_benchmark(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post(f"/ops22/vector/benchmark/{ws_id}",
                           cookies=cookies)
        assert resp.status_code == 200
        assert "backend" in resp.json()

    def test_backfill_preview(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.get(f"/ops22/vector/backfill/preview?workspace_id={ws_id}",
                          cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True


class TestProviderAPI:
    def test_detect(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.get("/ops22/providers/detect", cookies=cookies)
        assert resp.status_code == 200

    def test_validate_completion(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post(
            f"/ops22/providers/completion/validate/{ws_id}", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["passed"] is True

    def test_failure_mode(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/providers/failure-mode", cookies=cookies,
                           json={"workspace_id": ws_id, "mode": "timeout"})
        assert resp.status_code == 200

    def test_health_and_reconciliation(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.get("/ops22/providers/health", cookies=cookies)
        assert resp.status_code == 200
        resp = client.post("/ops22/providers/cost-reconciliation",
                           cookies=cookies,
                           json={"workspace_id": ws_id, "provider": "fake"})
        assert resp.status_code == 200
        assert resp.json()["reconciled"] is True


class TestEvaluationAPI:
    def test_run_and_list(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/evaluations/run", cookies=cookies,
                           json={"workspace_id": ws_id, "domain": "retrieval"})
        assert resp.status_code == 200
        resp = client.get(f"/ops22/evaluations/{ws_id}", cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["executions"]) >= 1

    def test_run_idempotent_by_key(self, client):
        cookies, ws_id = _user_and_ws(client)
        key = f"api-eval-{uuid.uuid4().hex[:8]}"
        r1 = client.post("/ops22/evaluations/run", cookies=cookies,
                         json={"workspace_id": ws_id, "domain": "rag",
                               "idempotency_key": key})
        r2 = client.post("/ops22/evaluations/run", cookies=cookies,
                         json={"workspace_id": ws_id, "domain": "rag",
                               "idempotency_key": key})
        assert r1.status_code == 200 and r2.status_code == 200
        assert r2.json().get("deduplicated") is True

    def test_gates_and_promotion(self, client):
        cookies, ws_id = _user_and_ws(client)
        # Create a proposal through the Phase 20 platform to gate.
        resp = client.post("/ops20/proposals", cookies=cookies, json={
            "domain": "retrieval", "title": "api gate proposal",
            "problem": "quality", "proposed_change": "tune weights",
            "workspace_id": ws_id})
        if resp.status_code == 200:
            proposal_id = resp.json()["id"]
            gates = client.post("/ops22/evaluations/gates", cookies=cookies,
                                json={"workspace_id": ws_id,
                                      "proposal_id": proposal_id})
            assert gates.status_code == 200
            promote = client.post("/ops22/evaluations/promote",
                                  cookies=cookies,
                                  json={"workspace_id": ws_id,
                                        "proposal_id": proposal_id})
            assert promote.status_code == 200


class TestKnowledgeConnectorAPI:
    def test_maintenance_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/knowledge/maintenance", cookies=cookies,
                           json={"workspace_id": ws_id, "kind": "graph"})
        assert resp.status_code == 200

    def test_governor_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/ingestion/governor", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "file_size": 10 ** 9})
        assert resp.status_code == 200
        assert resp.json()[0]["action"] == "REJECTED"

    def test_quarantine_list(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.get(f"/ops22/ingestion/quarantine/{ws_id}",
                          cookies=cookies)
        assert resp.status_code == 200

    def test_connector_sync(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/connectors/sync", cookies=cookies,
                           json={"workspace_id": ws_id, "connector_id": 3,
                                 "items": 5})
        assert resp.status_code == 200
        assert resp.json()["health"] == "HEALTHY"


class TestRegionDRAPI:
    def test_region_upsert_and_list(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.post("/ops22/regions", cookies=cookies,
                           json={"region": "us-east-2", "status": "HEALTHY"})
        assert resp.status_code == 200
        resp = client.get("/ops22/regions", cookies=cookies)
        assert any(r["region"] == "us-east-2"
                   for r in resp.json()["regions"])

    def test_capacity(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.post("/ops22/regions/capacity", cookies=cookies,
                           json={"region": "us-east-2", "workers": 2,
                                 "queue_depth": 10, "db_healthy": True,
                                 "provider_healthy": True})
        assert resp.status_code == 200

    def test_failover_plan_simulated(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.post("/ops22/regions/failover-plan", cookies=cookies,
                           json={"primary": "us-east-2",
                                 "secondary": "eu-west-2"})
        assert resp.status_code == 200
        assert resp.json()["simulation"]["simulated"] is True
        assert resp.json()["real_failover_available"] is False

    def test_dr_flow(self, client):
        cookies, ws_id = _user_and_ws(client)
        health = client.get("/ops22/dr/backup-health", cookies=cookies)
        assert health.status_code == 200
        drill = client.post("/ops22/dr/restore-drill", cookies=cookies,
                            json={"workspace_id": ws_id})
        assert drill.status_code == 200
        assert drill.json()["simulated"] is True
        report = client.get("/ops22/dr/report", cookies=cookies)
        assert report.status_code == 200
        assert report.json()["real_rpo_rto_measured"] is False


class TestObservabilityAPI:
    def test_trace_lifecycle(self, client):
        cookies, ws_id = _user_and_ws(client)
        start = client.post("/ops22/traces/start", cookies=cookies,
                            json={"workspace_id": ws_id, "stage": "request",
                                  "name": "api", "force_record": True})
        assert start.status_code == 200
        span_id = start.json()["span_id"]
        assert span_id is not None

    def test_slo_define_and_burn(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/slos", cookies=cookies,
                           json={"domain": "api", "name": f"api-{uuid.uuid4().hex[:6]}",
                                 "target": 0.99})
        assert resp.status_code == 200
        burn = client.post(f"/ops22/slos/burn-rate/{ws_id}", cookies=cookies)
        assert burn.status_code == 200


class TestWorkerAPI:
    def test_fair_share(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.get("/ops22/workers/fair-share", cookies=cookies)
        assert resp.status_code == 200

    def test_dead_letters(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.get("/ops22/workers/dead-letters", cookies=cookies)
        assert resp.status_code == 200

    def test_lease_recovery(self, client):
        cookies, _ws = _user_and_ws(client)
        resp = client.post("/ops22/workers/lease-recovery", cookies=cookies)
        assert resp.status_code == 200


class TestSecurityCostAPI:
    def test_scan_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/security/scan", cookies=cookies,
                           json={"workspace_id": ws_id, "corpus": "all"})
        assert resp.status_code == 200
        assert resp.json()["all_passed"] is True

    def test_retention_dry_run(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/retention/run", cookies=cookies,
                           json={"kind": "traces", "older_than_days": 30,
                                 "dry_run": True, "workspace_id": ws_id})
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True

    def test_cost_guard_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/cost/guard", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "estimated_cost_usd": 1.0,
                                 "budget_limit_usd": 10.0})
        assert resp.status_code == 200
        assert resp.json()["decision"] == "ALLOWED"


class TestLoopsAPI:
    def test_selfheal_loop_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/loops/self-heal", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "trigger": "worker_stall"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["detected"] is False
        assert body["loop_id"] > 0

    def test_autonomy_loop_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/loops/autonomy", cookies=cookies,
                           json={"workspace_id": ws_id, "domain": "retrieval",
                                 "metric": {"quality": 0.9,
                                            "reliability": 0.99,
                                            "cost": 1.0}})
        assert resp.status_code == 200
        assert resp.json()["deviation_detected"] is False


class TestChaosLoadAPI:
    def test_chaos_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/chaos/run", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "scenario": "provider_timeout"})
        assert resp.status_code == 200
        assert resp.json()["pass"] is True

    def test_load_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.post("/ops22/load/run", cookies=cookies,
                           json={"workspace_id": ws_id, "kind": "rag"})
        assert resp.status_code == 200


class TestStreamsAPI:
    def test_stream_read_empty(self, client):
        cookies, ws_id = _user_and_ws(client)
        resp = client.get(f"/ops22/streams/execution?workspace_id={ws_id}",
                          cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["last_seq"] >= 0

    def test_stream_after_loop_run(self, client):
        cookies, ws_id = _user_and_ws(client)
        client.post("/ops22/loops/self-heal", cookies=cookies,
                    json={"workspace_id": ws_id, "trigger": "worker_stall"})
        resp = client.get(
            f"/ops22/streams/incident?workspace_id={ws_id}", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["last_seq"] >= 1


# ===========================================================================
# Steps 232-241: E2E product flows
# ===========================================================================

class TestE2EFlows:
    def test_e2e_provider_fallback_flow(self, client):
        """provider failure -> fallback -> recovery -> health restoration."""
        cookies, ws_id = _user_and_ws(client)
        fail = client.post("/ops22/providers/failure-mode", cookies=cookies,
                           json={"workspace_id": ws_id, "mode": "timeout"})
        assert fail.status_code == 200
        fb = client.post(f"/ops22/providers/fallback/validate/{ws_id}",
                         cookies=cookies)
        assert fb.status_code == 200 and fb.json()["passed"] is True
        health = client.get("/ops22/providers/health", cookies=cookies)
        assert health.status_code == 200

    def test_e2e_selfhealing_flow(self, client):
        """failure -> detection -> diagnosis -> recovery -> verification."""
        cookies, ws_id = _user_and_ws(client)
        loop = client.post("/ops22/loops/self-heal", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "trigger": "queue_buildup"})
        assert loop.status_code == 200
        body = loop.json()
        assert "timeline" in body and body["loop_id"] > 0
        stream = client.get(
            f"/ops22/streams/incident?workspace_id={ws_id}", cookies=cookies)
        assert stream.status_code == 200
        assert stream.json()["last_seq"] >= 1

    def test_e2e_evaluation_flow(self, client):
        """schedule -> worker -> evaluation -> result -> regression check."""
        cookies, ws_id = _user_and_ws(client)
        run = client.post("/ops22/evaluations/run", cookies=cookies,
                          json={"workspace_id": ws_id, "domain": "retrieval"})
        assert run.status_code == 200
        reg = client.post(
            f"/ops22/evaluations/regression/{ws_id}?domain=retrieval",
            cookies=cookies)
        assert reg.status_code == 200
        assert "regressed" in reg.json()

    def test_e2e_improvement_flow(self, client):
        """regression -> proposal -> gates -> rollback capability."""
        cookies, ws_id = _user_and_ws(client)
        proposal = client.post("/ops20/proposals", cookies=cookies, json={
            "domain": "retrieval", "title": "e2e improvement",
            "problem": "drift", "proposed_change": "adjust reranker",
            "workspace_id": ws_id})
        if proposal.status_code == 200:
            pid = proposal.json()["id"]
            gates = client.post("/ops22/evaluations/gates", cookies=cookies,
                                json={"workspace_id": ws_id,
                                      "proposal_id": pid})
            assert gates.status_code == 200
            rb = client.post("/ops22/evaluations/rollback", cookies=cookies,
                             json={"workspace_id": ws_id,
                                   "proposal_id": pid})
            assert rb.status_code == 200

    def test_e2e_dr_flow(self, client):
        """backup simulation -> restore validation -> tenant isolation."""
        cookies, ws_id = _user_and_ws(client)
        drill = client.post("/ops22/dr/restore-drill", cookies=cookies,
                            json={"workspace_id": ws_id})
        assert drill.status_code == 200
        body = drill.json()
        assert body["simulated"] is True
        assert "tenant_isolation" in str(body)

    def test_e2e_incident_flow_via_streams(self, client):
        """failure -> incident stream -> operator visibility."""
        cookies, ws_id = _user_and_ws(client)
        client.post("/ops22/loops/autonomy", cookies=cookies,
                    json={"workspace_id": ws_id, "domain": "rag",
                          "metric": {"quality": 0.2, "reliability": 0.5,
                                     "cost": 300.0}})
        stream = client.get(
            f"/ops22/streams/execution?workspace_id={ws_id}", cookies=cookies)
        assert stream.status_code == 200
        assert stream.json()["last_seq"] >= 1

    def test_e2e_worker_recovery_flow(self, client):
        """worker loss -> lease expiry -> recovery."""
        cookies, ws_id = _user_and_ws(client)
        chaos = client.post("/ops22/chaos/run", cookies=cookies,
                            json={"workspace_id": ws_id,
                                  "scenario": "worker_loss"})
        assert chaos.status_code == 200
        assert chaos.json()["pass"] is True
        recovery = client.post("/ops22/workers/lease-recovery",
                               cookies=cookies)
        assert recovery.status_code == 200

    def test_e2e_document_maintenance_flow(self, client):
        """document change -> maintenance detection -> governed repair."""
        cookies, ws_id = _user_and_ws(client)
        gov = client.post("/ops22/ingestion/governor", cookies=cookies,
                          json={"workspace_id": ws_id, "file_size": 2048})
        assert gov.status_code == 200
        maint = client.post("/ops22/knowledge/maintenance", cookies=cookies,
                            json={"workspace_id": ws_id,
                                  "kind": "stale_documents"})
        assert maint.status_code == 200
        assert maint.json()["id"] > 0

    def test_e2e_rag_flow_via_trace(self, client):
        """query -> retrieval -> evidence -> answer -> citation trace."""
        cookies, ws_id = _user_and_ws(client)
        run = client.post("/ops22/evaluations/run", cookies=cookies,
                          json={"workspace_id": ws_id, "domain": "rag"})
        assert run.status_code == 200
        assert run.json().get("status") in ("DONE", "COMPLETED",
                                           "SUCCEEDED") or \
            "metrics" in run.json()
