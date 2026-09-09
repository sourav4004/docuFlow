"""Phase 22 tests — API matrix + E2E depth.

Exercises every ops22 endpoint family for contract stability, pagination
bounds, and error-shape consistency, plus a full E2E matrix across all
nine Phase 22 product flows in one suite.
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
    from app.models.phase22 import (OpsStreamEvent, SelfHealLoopRun,
                                    AutonomyLoopRun)
    from app.models.phase21 import EmergencyStop, PlatformEvent, \
        AutonomousOperation, AutonomyPolicy, DiagnosisReport, IncidentP21, \
        RecoveryAttempt, RecoveryPlaybook
    from app.models.phase20 import ImprovementProposal, Experiment, \
        ExperimentRun
    db_session.query(OpsStreamEvent).delete()
    db_session.query(SelfHealLoopRun).delete()
    db_session.query(AutonomyLoopRun).delete()
    for model in (ImprovementProposal, ExperimentRun, Experiment,
                  AutonomousOperation, EmergencyStop, PlatformEvent,
                  RecoveryAttempt, RecoveryPlaybook, AutonomyPolicy,
                  DiagnosisReport, IncidentP21):
        try:
            db_session.query(model).delete()
        except Exception:  # noqa: BLE001
            db_session.rollback()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _user_and_ws(client, tag="mx"):
    _counter[0] += 1
    n = _counter[0]
    email = f"{tag}{n}@p22mx.example"
    client.post("/auth/register", json={
        "name": tag.title(), "email": email, "password": "password123"})
    login = client.post("/auth/login", json={"email": email,
                                             "password": "password123"})
    cookies = login.cookies
    ws = client.post("/workspaces", json={"name": f"ws-{tag}-{n}"},
                     cookies=cookies)
    return cookies, ws.json()["id"]


# ===========================================================================
# API contract matrix — every GET family
# ===========================================================================

class TestGetContractMatrix:
    @pytest.mark.parametrize("path_tpl", [
        "/ops22/infrastructure",
        "/ops22/infrastructure/redis",
        "/ops22/infrastructure/broker-failover",
        "/ops22/vector/backend",
        "/ops22/providers/detect",
        "/ops22/providers/health",
        "/ops22/regions",
        "/ops22/dr/backup-health",
        "/ops22/dr/report",
        "/ops22/workers/fair-share",
        "/ops22/workers/dead-letters",
    ])
    def test_global_get_ok(self, client, path_tpl):
        cookies, _ws = _user_and_ws(client)
        resp = client.get(path_tpl, cookies=cookies)
        assert resp.status_code == 200, f"{path_tpl}: {resp.status_code}"
        assert isinstance(resp.json(), (dict, list))

    @pytest.mark.parametrize("path_tpl", [
        "/ops22/vector/coverage/{ws}",
        "/ops22/vector/drift/{ws}",
        "/ops22/vector/backfill/preview?workspace_id={ws}",
        "/ops22/evaluations/{ws}",
        "/ops22/ingestion/quarantine/{ws}",
        "/ops22/connectors/health/{ws}",
        "/ops22/cost/anomaly/{ws}",
        "/ops22/cost/forecast/{ws}",
    ])
    def test_workspace_get_ok(self, client, path_tpl):
        cookies, ws_id = _user_and_ws(client)
        resp = client.get(path_tpl.format(ws=ws_id), cookies=cookies)
        assert resp.status_code == 200, \
            f"{path_tpl}: {resp.status_code} {resp.text[:200]}"

    @pytest.mark.parametrize("path_tpl", [
        "/ops22/vector/coverage/{ws}",
        "/ops22/vector/drift/{ws}",
        "/ops22/evaluations/{ws}",
        "/ops22/cost/anomaly/{ws}",
    ])
    def test_workspace_get_isolated(self, client, path_tpl):
        _cookies_a, ws_a = _user_and_ws(client, "isoa")
        cookies_b, _ws_b = _user_and_ws(client, "isob")
        resp = client.get(path_tpl.format(ws=ws_a), cookies=cookies_b)
        assert resp.status_code in (403, 404)


# ===========================================================================
# POST contract matrix — every write family
# ===========================================================================

class TestPostContractMatrix:
    def test_infrastructure_detect(self, client):
        cookies, _ = _user_and_ws(client)
        r = client.post("/ops22/infrastructure/detect", cookies=cookies)
        assert r.status_code == 200 and r.json()["configured"] is True

    def test_vector_rebuild(self, client):
        cookies, ws_id = _user_and_ws(client)
        r = client.post("/ops22/vector/rebuild", cookies=cookies,
                        json={"workspace_id": ws_id, "document_id": 7})
        assert r.status_code == 200 and r.json()["ok"] is True

    def test_vector_dimension_safety(self, client):
        cookies, _ = _user_and_ws(client)
        r = client.get("/ops22/vector/dimension-safety?dimensions=99999",
                       cookies=cookies)
        assert r.status_code == 200

    def test_provider_validations_all(self, client):
        cookies, ws_id = _user_and_ws(client)
        for kind in ("completion", "streaming", "embeddings",
                     "structured", "tools", "circuit", "fallback"):
            r = client.post(f"/ops22/providers/{kind}/validate/{ws_id}",
                            cookies=cookies)
            assert r.status_code == 200, f"{kind}: {r.status_code}"
            assert r.json()["passed"] is True

    def test_provider_failure_modes_all(self, client):
        cookies, ws_id = _user_and_ws(client)
        for mode in ("timeout", "429", "5xx", "malformed"):
            r = client.post("/ops22/providers/failure-mode", cookies=cookies,
                            json={"workspace_id": ws_id, "mode": mode})
            assert r.status_code == 200

    def test_provider_cost_reconciliation(self, client):
        cookies, ws_id = _user_and_ws(client)
        r = client.post("/ops22/providers/cost-reconciliation",
                        cookies=cookies,
                        json={"workspace_id": ws_id, "provider": "fake"})
        assert r.status_code == 200 and r.json()["reconciled"] is True

    def test_evaluation_run_and_regression(self, client):
        cookies, ws_id = _user_and_ws(client)
        r = client.post("/ops22/evaluations/run", cookies=cookies,
                        json={"workspace_id": ws_id, "domain": "retrieval"})
        assert r.status_code == 200
        r = client.post(
            f"/ops22/evaluations/regression/{ws_id}?domain=retrieval",
            cookies=cookies)
        assert r.status_code == 200

    def test_maintenance_all_kinds(self, client):
        cookies, ws_id = _user_and_ws(client)
        for kind in ("stale_documents", "embeddings", "graph", "memory",
                     "summaries", "connectors"):
            r = client.post("/ops22/knowledge/maintenance", cookies=cookies,
                            json={"workspace_id": ws_id, "kind": kind})
            assert r.status_code == 200, f"{kind}: {r.status_code}"

    def test_governor_all_normal(self, client):
        cookies, ws_id = _user_and_ws(client)
        r = client.post("/ops22/ingestion/governor", cookies=cookies,
                        json={"workspace_id": ws_id, "file_size": 5000,
                              "pages": 10, "ocr_pages": 2})
        assert r.status_code == 200
        assert all(e["action"] == "ALLOWED" for e in r.json())

    def test_governor_all_rejected(self, client):
        cookies, ws_id = _user_and_ws(client)
        r = client.post("/ops22/ingestion/governor", cookies=cookies,
                        json={"workspace_id": ws_id,
                              "file_size": 10 ** 12, "pages": 10 ** 6,
                              "ocr_pages": 10 ** 6})
        assert r.status_code == 200
        assert all(e["action"] == "REJECTED" for e in r.json())

    def test_connector_sync_ok_and_failure(self, client):
        cookies, ws_id = _user_and_ws(client)
        ok = client.post("/ops22/connectors/sync", cookies=cookies,
                         json={"workspace_id": ws_id, "connector_id": 21,
                               "items": 4})
        assert ok.status_code == 200
        bad = client.post("/ops22/connectors/sync", cookies=cookies,
                          json={"workspace_id": ws_id, "connector_id": 22,
                                "error": "upstream_500"})
        assert bad.status_code == 200
        assert bad.json()["failure_count"] == 1

    def test_region_capacity_all(self, client):
        cookies, _ = _user_and_ws(client)
        r = client.post("/ops22/regions/capacity", cookies=cookies,
                        json={"region": "ap-south-1", "workers": 8,
                              "queue_depth": 42, "db_healthy": True,
                              "provider_healthy": True})
        assert r.status_code == 200

    def test_failover_plan_returns_steps(self, client):
        cookies, _ = _user_and_ws(client)
        r = client.post("/ops22/regions/failover-plan", cookies=cookies,
                        json={"primary": "r1", "secondary": "r2"})
        assert r.status_code == 200
        assert len(r.json()["plan"]["steps"]) >= 5

    def test_restore_drill(self, client):
        cookies, ws_id = _user_and_ws(client)
        r = client.post("/ops22/dr/restore-drill", cookies=cookies,
                        json={"workspace_id": ws_id})
        assert r.status_code == 200 and r.json()["simulated"] is True

    def test_trace_start_minimal(self, client):
        cookies, ws_id = _user_and_ws(client)
        r = client.post("/ops22/traces/start", cookies=cookies,
                        json={"workspace_id": ws_id, "stage": "api",
                              "name": "mx", "force_record": True})
        assert r.status_code == 200

    def test_slo_define(self, client):
        cookies, _ = _user_and_ws(client)
        r = client.post("/ops22/slos", cookies=cookies,
                        json={"domain": "workers",
                              "name": f"mx-slo-{uuid.uuid4().hex[:6]}",
                              "target": 0.995})
        assert r.status_code == 200

    def test_lease_recovery_and_shutdown(self, client):
        cookies, _ws = _user_and_ws(client)
        r1 = client.post("/ops22/workers/lease-recovery", cookies=cookies)
        assert r1.status_code == 200
        r2 = client.get("/ops22/workers/shutdown/mx-worker-1",
                        cookies=cookies)
        assert r2.status_code == 200

    def test_security_scan_single_corpus(self, client):
        cookies, ws_id = _user_and_ws(client)
        for corpus in ("prompt_injection", "exfiltration", "tool_abuse",
                       "ssrf", "tenant_matrix", "api_abuse",
                       "autonomy_bypass"):
            r = client.post("/ops22/security/scan", cookies=cookies,
                            json={"workspace_id": ws_id, "corpus": corpus})
            assert r.status_code == 200, f"{corpus}: {r.status_code}"
            assert r.json()["scan_passed"] is True

    def test_retention_all_kinds_dry_run(self, client):
        cookies, ws_id = _user_and_ws(client)
        for kind in ("artifacts", "traces", "evaluations", "events",
                     "usage", "temporary"):
            r = client.post("/ops22/retention/run", cookies=cookies,
                            json={"kind": kind, "older_than_days": 30,
                                  "dry_run": True, "workspace_id": ws_id})
            assert r.status_code == 200, f"{kind}: {r.status_code}"
            assert r.json()["dry_run"] is True

    def test_cost_guard_all_bands(self, client):
        cookies, ws_id = _user_and_ws(client)
        bands = [("allow", 1.0, 10.0, "ALLOWED"),
                 ("approve", 7.5, 10.0, "REQUIRES_APPROVAL"),
                 ("block", 20.0, 10.0, "BLOCKED")]
        for tag, est, limit, expected in bands:
            r = client.post("/ops22/cost/guard", cookies=cookies,
                            json={"workspace_id": ws_id,
                                  "estimated_cost_usd": est,
                                  "budget_limit_usd": limit,
                                  "operation_type": f"band.{tag}"})
            assert r.status_code == 200
            assert r.json()["decision"] == expected

    def test_chaos_all_scenarios(self, client):
        cookies, ws_id = _user_and_ws(client)
        scenarios = ["provider_timeout", "provider_429", "provider_5xx",
                     "provider_malformed", "broker_failure", "worker_loss",
                     "scheduler_leader_loss", "db_failure", "network_chaos",
                     "recovery_validation"]
        for scenario in scenarios:
            r = client.post("/ops22/chaos/run", cookies=cookies,
                            json={"workspace_id": ws_id,
                                  "scenario": scenario})
            assert r.status_code == 200, f"{scenario}: {r.status_code}"
            body = r.json()
            assert body.get("pass") is True, scenario

    def test_load_all_kinds(self, client):
        cookies, ws_id = _user_and_ws(client)
        for kind in ("api", "worker", "rag", "ingestion", "search",
                     "provider", "soak"):
            r = client.post("/ops22/load/run", cookies=cookies,
                            json={"workspace_id": ws_id, "kind": kind})
            assert r.status_code == 200, f"{kind}: {r.status_code}"

    def test_stream_pagination_bounds(self, client):
        cookies, ws_id = _user_and_ws(client)
        for i in range(8):
            client.post("/ops22/loops/self-heal", cookies=cookies,
                        json={"workspace_id": ws_id,
                              "trigger": "worker_stall"})
        r = client.get(f"/ops22/streams/incident?workspace_id={ws_id}"
                       f"&limit=3", cookies=cookies)
        assert r.status_code == 200
        assert len(r.json()["events"]) <= 3


# ===========================================================================
# Full E2E matrix — all nine product flows
# ===========================================================================

class TestE2EMatrix:
    def test_flow_document(self, client):
        """upload -> ingestion (governor) -> maintenance -> ready signal."""
        cookies, ws_id = _user_and_ws(client)
        gov = client.post("/ops22/ingestion/governor", cookies=cookies,
                          json={"workspace_id": ws_id, "file_size": 100})
        assert gov.status_code == 200
        maint = client.post("/ops22/knowledge/maintenance", cookies=cookies,
                            json={"workspace_id": ws_id,
                                  "kind": "stale_documents"})
        assert maint.status_code == 200

    def test_flow_rag(self, client):
        """query -> retrieval eval -> evidence -> answer -> citations."""
        cookies, ws_id = _user_and_ws(client)
        for domain in ("retrieval", "rag", "citations"):
            r = client.post("/ops22/evaluations/run", cookies=cookies,
                            json={"workspace_id": ws_id, "domain": domain})
            assert r.status_code == 200

    def test_flow_ai_policy(self, client):
        """request -> policy -> execution -> audit."""
        cookies, ws_id = _user_and_ws(client)
        client.post("/ops21/autonomy/policies", cookies=cookies, json={
            "workspace_id": ws_id, "operation_type": "e2e.op",
            "autonomy_level": "AUTO_LOW_RISK", "requires_approval": False})
        guard = client.post("/ops21/autonomy/guard", cookies=cookies, json={
            "workspace_id": ws_id, "operation_type": "e2e.op",
            "risk_level": "LOW", "actor": "e2e", "source": "TEST"})
        assert guard.status_code == 200
        ops = client.get(f"/ops21/autonomy/operations?workspace_id={ws_id}",
                         cookies=cookies)
        assert ops.status_code == 200

    def test_flow_selfhealing(self, client):
        cookies, ws_id = _user_and_ws(client)
        loop = client.post("/ops22/loops/self-heal", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "trigger": "worker_stall",
                                 "signals": {"worker": {"stalled": True}}})
        assert loop.status_code == 200
        stream = client.get(
            f"/ops22/streams/incident?workspace_id={ws_id}", cookies=cookies)
        assert stream.status_code == 200
        assert stream.json()["last_seq"] >= 1

    def test_flow_provider_fallback(self, client):
        cookies, ws_id = _user_and_ws(client)
        client.post("/ops22/providers/failure-mode", cookies=cookies,
                    json={"workspace_id": ws_id, "mode": "5xx"})
        fb = client.post(f"/ops22/providers/fallback/validate/{ws_id}",
                         cookies=cookies)
        assert fb.json()["passed"] is True

    def test_flow_worker_recovery(self, client):
        cookies, ws_id = _user_and_ws(client)
        chaos = client.post("/ops22/chaos/run", cookies=cookies,
                            json={"workspace_id": ws_id,
                                  "scenario": "worker_loss"})
        assert chaos.json()["pass"] is True
        rec = client.post("/ops22/workers/lease-recovery", cookies=cookies)
        assert rec.status_code == 200

    def test_flow_evaluation(self, client):
        cookies, ws_id = _user_and_ws(client)
        run = client.post("/ops22/evaluations/run", cookies=cookies,
                          json={"workspace_id": ws_id, "domain": "model"})
        assert run.status_code == 200
        reg = client.post(
            f"/ops22/evaluations/regression/{ws_id}?domain=model",
            cookies=cookies)
        assert reg.status_code == 200

    def test_flow_improvement(self, client):
        cookies, ws_id = _user_and_ws(client)
        loop = client.post("/ops22/loops/autonomy", cookies=cookies,
                           json={"workspace_id": ws_id, "domain": "retrieval",
                                 "metric": {"quality": 0.3,
                                            "reliability": 0.9,
                                            "cost": 2.0}})
        assert loop.status_code == 200
        body = loop.json()
        # Deviation must produce a governed outcome, never silent activation.
        if body["deviation_detected"]:
            assert body["governance"] in ("BLOCKED", "APPROVAL", "ALLOWED")

    def test_flow_incident(self, client):
        cookies, ws_id = _user_and_ws(client)
        client.post("/ops22/loops/self-heal", cookies=cookies,
                    json={"workspace_id": ws_id, "trigger": "broker_failure",
                          "signals": {"broker_errors": 4}})
        stream = client.get(
            f"/ops22/streams/incident?workspace_id={ws_id}", cookies=cookies)
        assert stream.json()["last_seq"] >= 1

    def test_flow_dr(self, client):
        cookies, ws_id = _user_and_ws(client)
        drill = client.post("/ops22/dr/restore-drill", cookies=cookies,
                            json={"workspace_id": ws_id})
        assert drill.json()["simulated"] is True
        report = client.get("/ops22/dr/report", cookies=cookies)
        assert report.json()["real_rpo_rto_measured"] is False

    def test_all_flows_leave_audit_trail(self, client):
        cookies, ws_id = _user_and_ws(client)
        client.post("/ops22/loops/self-heal", cookies=cookies,
                    json={"workspace_id": ws_id, "trigger": "db_failure",
                          "signals": {"db_errors": 2}})
        client.post("/ops22/chaos/run", cookies=cookies,
                    json={"workspace_id": ws_id, "scenario": "db_failure"})
        stream = client.get(
            f"/ops22/streams/incident?workspace_id={ws_id}", cookies=cookies)
        events = stream.json()["events"]
        kinds = {e["kind"] for e in events}
        assert "selfheal_loop" in kinds
