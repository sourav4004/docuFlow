"""Phase 23 tests — ops23 API surface (Step 89 route audit + contract cases).

Covers: route registration (root + /api/v1 mounts, no duplicates), anonymous
rejection on every endpoint, workspace isolation, validation failure modes,
and happy-path contracts for every ops23 route group.
"""

import uuid

import pytest
from collections import Counter
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db
from tests.shared_db import TestingSessionLocal, override_get_db

# Single canonical test DB: fixtures and API requests must share one
# SQLite connection (same convention as test_phase22_api.py).
app.dependency_overrides[get_db] = override_get_db

from app.models.phase21 import SecurityIncidentP21, AutonomyAbuseAttempt
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember
from app.models.phase22 import (
    EvalExecution, OpsStreamEvent,
)
from app.models.phase23 import (
    StorageMigrationPlan, StorageMigrationObject, ProviderReadinessScore,
    ProviderRoutingDecision, ResidencyDecisionLog, EmbeddingModelVersion,
    DocumentEmbeddingStatus, SearchShadowComparison, RegionDrainOperation,
    DependencyEdge, KnowledgeFreshnessState, ConsistencyCheckRun,
    RequestDedupRecord, SchedulerTaskRun, WebhookDeliveryAttempt,
    WebhookEndpointHealth, ReviewDecision,
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


P23_API_TABLES = [
    StorageMigrationPlan, StorageMigrationObject, ProviderReadinessScore,
    ProviderRoutingDecision, ResidencyDecisionLog, EmbeddingModelVersion,
    DocumentEmbeddingStatus, SearchShadowComparison, RegionDrainOperation,
    DependencyEdge, KnowledgeFreshnessState, ConsistencyCheckRun,
    RequestDedupRecord, SchedulerTaskRun, WebhookDeliveryAttempt,
    WebhookEndpointHealth, ReviewDecision,
    # shared tables touched via ops23 flows
    EvalExecution, OpsStreamEvent, SecurityIncidentP21, AutonomyAbuseAttempt,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    from app.models.phase19 import ResidencyRule, RegionRecord, \
        RegionFailover
    from app.models.phase21 import FailoverSimulation
    for model in P23_API_TABLES:
        db_session.query(model).delete()
    for model in (FailoverSimulation, RegionFailover, ResidencyRule,
                  RegionRecord):
        db_session.query(model).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _user_and_ws(client, tag="api"):
    _counter[0] += 1
    n = _counter[0]
    email = f"{tag}{n}@p23api.example"
    resp = client.post("/auth/register", json={
        "name": tag.title(), "email": email, "password": "password123"})
    assert resp.status_code in (200, 201), resp.text
    login = client.post("/auth/login", json={"email": email,
                                             "password": "password123"})
    cookies = login.cookies
    ws = client.post("/workspaces", json={"name": f"ws-{tag}-{n}"},
                     cookies=cookies)
    return cookies, ws.json()["id"]


ALL_OPS23 = sorted({
    (r.path, tuple(sorted(r.methods)))
    for r in app.routes
    if hasattr(r, "methods") and r.path.startswith("/ops23")
})


# ===========================================================================
# Steps 89 + 38: route inventory, duplicates, auth
# ===========================================================================

class TestRouteInventory:
    def test_ops23_route_count(self):
        assert len(ALL_OPS23) == 81

    def test_ops23_api_v1_mount_matches(self):
        v1 = {(r.path.replace("/api/v1", ""), tuple(sorted(r.methods)))
              for r in app.routes
              if hasattr(r, "methods") and r.path.startswith("/api/v1/ops23")}
        assert set(ALL_OPS23) == v1

    def test_no_duplicate_endpoints(self):
        counts = Counter(ALL_OPS23)
        assert all(v == 1 for v in counts.values()), \
            [k for k, v in counts.items() if v > 1]

    def test_anonymous_rejected_on_every_route(self, client):
        # Every route must reject anonymous access (or reject the empty
        # body at validation before anything runs).
        for path, methods in ALL_OPS23:
            for m in methods:
                if m == "GET":
                    resp = client.get(path)
                    assert resp.status_code in (401, 403, 422), \
                        f"{m} {path}: {resp.status_code}"
                elif m in ("POST", "DELETE"):
                    resp = client.request(m, path, json={})
                    assert resp.status_code in (401, 403, 422), \
                        f"{m} {path}: {resp.status_code}"
                else:
                    raise AssertionError(f"unexpected method {m} {path}")

    def test_compat_metadata_reports_v1_v2(self, client):
        cookies, _ = _user_and_ws(client, "compat")
        resp = client.get("/ops23/api/compatibility", cookies=cookies)
        assert resp.status_code == 200
        body = resp.json()
        assert body["supported_versions"] == ["v1", "v2"]
        assert body["current_version"] == "v1"
        assert body["v2_status"] == "READY"


# ===========================================================================
# Global control plane endpoints (Steps 1, 18-19)
# ===========================================================================

class TestGlobalControlPlaneAPI:
    def test_capabilities_list(self, client):
        cookies, _ = _user_and_ws(client, "cap")
        resp = client.get("/ops23/capabilities", cookies=cookies)
        assert resp.status_code == 200
        caps = resp.json()
        assert isinstance(caps, list) and len(caps) > 0
        for c in caps:
            assert {"component", "state"} <= set(c)

    def test_capability_detail_unknown(self, client):
        cookies, _ = _user_and_ws(client, "capd")
        resp = client.get("/ops23/capabilities/definitely-not-real",
                          cookies=cookies)
        assert resp.status_code == 404

    def test_capability_check_and_refresh(self, client):
        cookies, _ = _user_and_ws(client, "capr")
        check = client.post("/ops23/capabilities/check", cookies=cookies,
                            json={})
        assert check.status_code == 200
        refresh = client.post("/ops23/capabilities/refresh", cookies=cookies,
                              json={})
        assert refresh.status_code == 200

    def test_global_health_shape(self, client):
        cookies, _ = _user_and_ws(client, "gh")
        resp = client.get("/ops23/global-health", cookies=cookies)
        assert resp.status_code == 200
        body = resp.json()
        for key in ("status", "capabilities", "regions",
                    "degraded_components", "database", "broker"):
            assert key in body
        assert body["status"] in ("HEALTHY", "DEGRADED", "UNHEALTHY")

    def test_readiness_and_degraded(self, client):
        cookies, _ = _user_and_ws(client, "rd")
        ready = client.get("/ops23/readiness", cookies=cookies)
        assert ready.status_code == 200
        assert "ready" in ready.json()
        deg = client.get("/ops23/degraded-components", cookies=cookies)
        assert deg.status_code == 200

    def test_dependency_graph_and_impact(self, client):
        cookies, _ = _user_and_ws(client, "dep")
        deps = client.get("/ops23/dependencies", cookies=cookies)
        assert deps.status_code == 200
        impact = client.get("/ops23/dependencies/database/impact",
                            cookies=cookies)
        assert impact.status_code == 200
        body = impact.json()
        assert {"component", "upstream_hard", "downstream_hard",
                "blast_radius"} <= set(body)

    def test_dependency_impact_unknown_component(self, client):
        cookies, _ = _user_and_ws(client, "dep2")
        resp = client.get("/ops23/dependencies/nope/impact", cookies=cookies)
        assert resp.status_code == 404


# ===========================================================================
# Broker + storage migration endpoints (Steps 3, 4, 5)
# ===========================================================================

class TestMigrationAPIs:
    def test_broker_migration_plan_and_get(self, client):
        cookies, ws = _user_and_ws(client, "bm")
        plan = client.post("/ops23/broker/migration/plan", cookies=cookies,
                           json={"workspace_id": ws})
        assert plan.status_code == 200
        mid = plan.json()["migration_id"]
        got = client.get(f"/ops23/broker/migration/{mid}", cookies=cookies)
        assert got.status_code == 200
        assert got.json()["migration_id"] == mid

    def test_broker_drain_check(self, client):
        cookies, ws = _user_and_ws(client, "bdc")
        resp = client.get("/ops23/broker/drain-check",
                          params={"workspace_id": ws}, cookies=cookies)
        assert resp.status_code == 200

    def test_storage_plan_requires_workspace(self, client):
        cookies, _ = _user_and_ws(client, "sp")
        resp = client.post("/ops23/storage-migration/plans", cookies=cookies,
                           json={})
        assert resp.status_code == 422

    def test_storage_plan_dry_run(self, client, db_session):
        from app.models.document import Document
        cookies, ws = _user_and_ws(client, "spd")
        user = db_session.query(User).filter(
            User.email.like("%@p23api.example")).order_by(
            User.id.desc()).first()
        doc = Document(workspace_id=ws, user_id=user.id,
                       original_filename="a.txt",
                       storage_key="local://a.txt", mime_type="text/plain",
                       file_size=10, status="UPLOADED")
        db_session.add(doc)
        db_session.commit()
        plan = client.post("/ops23/storage-migration/plans", cookies=cookies,
                           json={"workspace_id": ws, "batch_size": 10,
                                 "dry_run": True})
        assert plan.status_code == 200
        pid = plan.json()["plan_id"]
        run = client.post(f"/ops23/storage-migration/plans/{pid}/run",
                          cookies=cookies)
        assert run.status_code == 200
        got = client.get(f"/ops23/storage-migration/plans/{pid}",
                         cookies=cookies)
        assert got.status_code == 200
        assert got.json()["mode"] == "DRY_RUN"

    def test_storage_capability(self, client):
        cookies, _ = _user_and_ws(client, "scap")
        resp = client.get("/ops23/storage/capability", cookies=cookies)
        assert resp.status_code == 200

    def test_storage_integrity_unknown_document(self, client):
        cookies, ws = _user_and_ws(client, "si")
        resp = client.get("/ops23/storage/integrity/999999",
                          params={"workspace_id": ws}, cookies=cookies)
        assert resp.status_code == 404


# ===========================================================================
# Provider gate + routing + cost endpoints (Steps 6-9)
# ===========================================================================

class TestProviderAPIs:
    def test_provider_validation_and_matrix(self, client):
        cookies, _ = _user_and_ws(client, "pv")
        val = client.post("/ops23/providers/fake/readiness", cookies=cookies)
        assert val.status_code == 200
        matrix = client.get("/ops23/providers/readiness", cookies=cookies)
        assert matrix.status_code == 200

    def test_routing_decision_member_only(self, client):
        cookies, ws = _user_and_ws(client, "rt")
        resp = client.post("/ops23/routing/decide", cookies=cookies,
                           json={"workspace_id": ws,
                                 "operation": "completion"})
        assert resp.status_code == 200
        body = resp.json()
        assert {"decision", "provider"} <= set(body)
        if body["decision"] == "ROUTED":
            assert body["provider"]

    def test_admission_check_budget_block(self, client):
        cookies, ws = _user_and_ws(client, "adm")
        resp = client.post("/ops23/routing/admission", cookies=cookies,
                           json={"workspace_id": ws,
                                 "estimated_cost": 10_000.0,
                                 "budget_remaining": 1.0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["admitted"] is False
        assert any(c["check"] == "budget" and not c["ok"]
                   for c in body["checks"])

    def test_reconcile_roundtrip(self, client):
        cookies, ws = _user_and_ws(client, "rc")
        rec = client.post("/ops23/cost/reconcile", cookies=cookies,
                          json={"workspace_id": ws, "provider": "fake",
                                "model": "fake-model",
                                "estimated_tokens": 100,
                                "actual_input_tokens": 80,
                                "actual_output_tokens": 20,
                                "estimated_cost": 0.01,
                                "actual_cost": "0.012"})
        assert rec.status_code == 200
        listing = client.get(f"/ops23/cost/reconcile/{ws}", cookies=cookies)
        assert listing.status_code == 200
        assert listing.json()["count"] >= 1


# ===========================================================================
# Vector + region + residency endpoints (Steps 10-17)
# ===========================================================================

class TestVectorRegionAPIs:
    def test_vector_activation_and_readiness(self, client):
        cookies, _ = _user_and_ws(client, "va")
        act = client.get("/ops23/vector/activation", cookies=cookies)
        assert act.status_code == 200
        rd = client.get("/ops23/vector/migration-readiness", cookies=cookies)
        assert rd.status_code == 200

    def test_vector_model_registration(self, client):
        cookies, _ = _user_and_ws(client, "vm")
        resp = client.post("/ops23/vector/models", cookies=cookies,
                           json={"model_name": "fake-embed-1",
                                 "version": 1, "dimension": 384})
        assert resp.status_code == 200

    def test_region_crud_and_detail(self, client):
        cookies, _ = _user_and_ws(client, "rg")
        reg = client.post("/ops23/regions", cookies=cookies,
                          json={"region": "us-east-23", "status": "ACTIVE",
                                "residency": ["US"]})
        assert reg.status_code == 200
        listed = client.get("/ops23/regions", cookies=cookies)
        assert listed.status_code == 200
        detail = client.get("/ops23/regions/us-east-23", cookies=cookies)
        assert detail.status_code == 200

    def test_residency_evaluate_and_log(self, client):
        cookies, ws = _user_and_ws(client, "res")
        ev = client.post("/ops23/residency/evaluate", cookies=cookies,
                         json={"workspace_id": ws, "source_region": "us",
                               "dest_region": "eu",
                               "sensitivity": "RESTRICTED"})
        assert ev.status_code == 200
        log = client.get(f"/ops23/residency/log/{ws}", cookies=cookies)
        assert log.status_code == 200
        assert log.json()["count"] >= 1

    def test_failback_and_history(self, client):
        cookies, ws = _user_and_ws(client, "fb")
        fb = client.get("/ops23/regions/failback",
                        params={"workspace_id": ws}, cookies=cookies)
        assert fb.status_code == 200
        hist = client.get("/ops23/regions/failover-history",
                          params={"workspace_id": ws}, cookies=cookies)
        assert hist.status_code == 200


# ===========================================================================
# RAG8 + knowledge endpoints (Steps 23-28)
# ===========================================================================

class TestRag8KnowledgeAPIs:
    def test_rag8_answer_requires_evidence_discipline(self, client):
        cookies, ws = _user_and_ws(client, "rag")
        resp = client.post("/ops23/rag8/answer", cookies=cookies,
                           json={"workspace_id": ws, "query": "what?",
                                 "evidence": [], "record_quality": True})
        assert resp.status_code == 200
        body = resp.json()
        # Insufficient evidence → explicit refusal with reduced confidence.
        assert (body["refusal"] is True
                or body["stages"]["sufficiency"]["sufficient"] is False)
        assert body["confidence"] < 0.5

    def test_rag8_change_impact_unknown_doc(self, client):
        cookies, ws = _user_and_ws(client, "ci")
        resp = client.get("/ops23/rag8/change-impact/999999",
                          params={"workspace_id": ws,
                                  "change_class": "CONTENT_UPDATE"},
                          cookies=cookies)
        assert resp.status_code == 404

    def test_freshness_compute_and_get(self, client):
        cookies, ws = _user_and_ws(client, "fr")
        comp = client.post(f"/ops23/knowledge/freshness/{ws}",
                           cookies=cookies)
        assert comp.status_code == 200
        got = client.get(f"/ops23/knowledge/freshness/{ws}", cookies=cookies)
        assert got.status_code == 200

    def test_drift_endpoint(self, client):
        cookies, ws = _user_and_ws(client, "dr")
        resp = client.get(f"/ops23/knowledge/drift/{ws}", cookies=cookies)
        assert resp.status_code == 200

    def test_maintenance_run_and_get(self, client):
        cookies, ws = _user_and_ws(client, "mt")
        run = client.post(f"/ops23/knowledge/maintenance/{ws}",
                          cookies=cookies, json={"dry_run": True})
        assert run.status_code == 200
        got = client.get(f"/ops23/knowledge/maintenance/{ws}",
                         cookies=cookies)
        assert got.status_code == 200


# ===========================================================================
# API platform + cache endpoints (Steps 39-45)
# ===========================================================================

class TestApiPlatformAPIs:
    def test_contract_matrix(self, client):
        cookies, _ = _user_and_ws(client, "cm")
        resp = client.get("/ops23/api/contract-matrix", cookies=cookies)
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 8
        assert "tenant_isolation" in body["cases"]

    def test_cache_stats(self, client):
        cookies, _ = _user_and_ws(client, "cs")
        resp = client.get("/ops23/cache/stats", cookies=cookies)
        assert resp.status_code == 200


# ===========================================================================
# Webhook + scheduler + review endpoints (Steps 35, 48, 51)
# ===========================================================================

class TestWebhookSchedulerReviewAPIs:
    def test_sign_preview_never_uses_real_secret(self, client):
        cookies, _ = _user_and_ws(client, "wsp")
        resp = client.post("/ops23/webhooks/sign-preview", cookies=cookies,
                           json={"payload": {"a": 1}})
        assert resp.status_code == 200
        assert "preview-secret-not-real" not in resp.text
        assert resp.json()["signature_shape"].startswith("t=")

    def test_delivery_deadletter_reenable_cycle(self, client):
        cookies, ws = _user_and_ws(client, "wdl")
        ep = wr_endpoints = client.post("/ops23/webhooks/delivery",
                                        cookies=cookies,
                                        json={"workspace_id": ws,
                                              "endpoint_id": 5001,
                                              "event_type": "doc.updated",
                                              "ok": False,
                                              "response_status": 500,
                                              "delivery_id": "d-1"})
        assert ep.status_code == 200
        # repeat failures disable the endpoint eventually
        for i in range(6):
            client.post("/ops23/webhooks/delivery", cookies=cookies,
                        json={"workspace_id": ws, "endpoint_id": 5001,
                              "event_type": "doc.updated", "ok": False,
                              "response_status": 500,
                              "delivery_id": f"d-{i + 2}"})
        dl = client.get(f"/ops23/webhooks/dead-letters/{ws}", cookies=cookies)
        assert dl.status_code == 200
        re = client.post("/ops23/webhooks/endpoints/5001/reenable",
                         cookies=cookies)
        assert re.status_code == 200

    def test_scheduler_claim_finish(self, client):
        cookies, _ = _user_and_ws(client, "sch")
        claim = client.post("/ops23/scheduler/tasks/claim", cookies=cookies,
                            json={"task_key": "nightly-eval",
                                  "leader_id": "t-leader"})
        assert claim.status_code == 200
        assert claim.json()["claimed"] is True
        dup = client.post("/ops23/scheduler/tasks/claim", cookies=cookies,
                          json={"task_key": "nightly-eval",
                                "leader_id": "other"})
        assert dup.json()["claimed"] is False
        finish = client.post(
            f"/ops23/scheduler/tasks/{claim.json()['run_id']}/finish",
            cookies=cookies, json={"ok": True})
        assert finish.status_code == 200

    def test_review_create_and_decide(self, client):
        cookies, ws = _user_and_ws(client, "rv")
        item = client.post("/ops23/reviews", cookies=cookies,
                           json={"workspace_id": ws,
                                 "item_type": "agent_handoff",
                                 "title": "risky plan",
                                 "payload": {"risk": "HIGH"}})
        assert item.status_code == 200, item.text
        item_id = item.json()["id"]
        dec = client.post(f"/ops23/reviews/{item_id}/decide", cookies=cookies,
                          json={"decision": "APPROVE",
                                "workspace_id": ws,
                                "reason": "reviewed"})
        assert dec.status_code == 200, dec.text


# ===========================================================================
# Security + evaluation + search + streams endpoints (Steps 36, 56, 52-53, 62)
# ===========================================================================

class TestSecurityEvalSearchStreamAPIs:
    def test_finding_lifecycle(self, client):
        cookies, ws = _user_and_ws(client, "sf")
        created = client.post("/ops23/security/findings", cookies=cookies,
                              json={"workspace_id": ws,
                                    "category": "prompt_injection",
                                    "severity": "HIGH", "title": "probe hit",
                                    "evidence": {"src": "corpus"}})
        assert created.status_code == 200, created.text
        fid = created.json()["id"]
        upd = client.post(f"/ops23/security/findings/{fid}/status",
                          cookies=cookies, json={"new_status": "MITIGATED",
                                                 "workspace_id": ws})
        assert upd.status_code == 200, upd.text
        listing = client.get(f"/ops23/security/findings/{ws}", cookies=cookies)
        assert listing.status_code == 200
        assert listing.json()["count"] >= 1

    def test_severity_matrix(self, client):
        cookies, _ = _user_and_ws(client, "sm")
        resp = client.get("/ops23/security/severity-matrix", cookies=cookies)
        assert resp.status_code == 200
        assert "prompt_injection" in resp.json()

    def test_scan_creates_findings(self, client):
        cookies, ws = _user_and_ws(client, "scan")
        resp = client.post(f"/ops23/security/scan/{ws}", cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["created"] >= 1

    def test_evaluation_record_and_gates(self, client):
        cookies, ws = _user_and_ws(client, "ev")
        rec = client.post("/ops23/evaluations", cookies=cookies,
                          json={"workspace_id": ws, "domain": "retrieval",
                                "metrics": {"precision": 0.9}})
        assert rec.status_code == 200
        gates = client.post("/ops23/evaluations/promotion-gates",
                            cookies=cookies,
                            json={"workspace_id": ws, "quality": 0.95,
                                  "security_findings": 0,
                                  "latency_p95_ms": 200, "cost": 0.01,
                                  "regression": 0.0})
        assert gates.status_code == 200
        assert gates.json()["promote"] is True

    def test_evaluation_rejects_unknown_domain(self, client):
        cookies, ws = _user_and_ws(client, "ev2")
        resp = client.post("/ops23/evaluations", cookies=cookies,
                           json={"workspace_id": ws, "domain": "nope"})
        assert resp.status_code == 422

    def test_search_explain_and_diversity(self, client):
        cookies, _ = _user_and_ws(client, "se")
        ex = client.post("/ops23/search/explain", cookies=cookies,
                         json={"candidates": [{"id": "a", "score": 0.9}]})
        assert ex.status_code == 200
        dv = client.post("/ops23/search/diversity", cookies=cookies,
                         json={"ranking": [{"id": "r1", "document_id": 1}]})
        assert dv.status_code == 200

    def test_self_evaluation(self, client):
        cookies, ws = _user_and_ws(client, "sse")
        resp = client.post(f"/ops23/search/self-evaluation/{ws}",
                           cookies=cookies,
                           json={"zero_result_queries": 30,
                                 "total_queries": 100,
                                 "reformulations": 60,
                                 "precision_estimate": 0.4})
        assert resp.status_code == 200
        assert resp.json()["proposal_created"] is True

    def test_stream_emit_and_read_with_seq(self, client):
        cookies, ws = _user_and_ws(client, "str")
        e1 = client.post("/ops23/streams/worker", cookies=cookies,
                         json={"workspace_id": ws, "kind": "worker.event",
                               "payload": {"n": 1}})
        assert e1.status_code == 200
        client.post("/ops23/streams/worker", cookies=cookies,
                    json={"workspace_id": ws, "kind": "worker.event",
                          "payload": {"n": 2}})
        got = client.get(f"/ops23/streams/worker/{ws}", cookies=cookies)
        assert got.status_code == 200
        events = got.json()["events"]
        assert len(events) == 2
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs) and len(set(seqs)) == 2
        after = client.get(f"/ops23/streams/worker/{ws}",
                           params={"after_seq": seqs[0]}, cookies=cookies)
        assert all(e["seq"] > seqs[0] for e in after.json()["events"])

    def test_stream_unknown_kind_rejected(self, client):
        cookies, ws = _user_and_ws(client, "str2")
        resp = client.post("/ops23/streams/not-a-stream", cookies=cookies,
                           json={"workspace_id": ws})
        assert resp.status_code == 422

    def test_stream_cross_workspace_hidden(self, client):
        cookies_a, ws_a = _user_and_ws(client, "swa")
        cookies_b, ws_b = _user_and_ws(client, "swb")
        client.post("/ops23/streams/worker", cookies=cookies_a,
                    json={"workspace_id": ws_a, "payload": {"x": 1}})
        got = client.get(f"/ops23/streams/worker/{ws_b}", cookies=cookies_b)
        assert got.status_code == 200
        assert got.json()["events"] == []
        # and B cannot read A's stream
        resp = client.get(f"/ops23/streams/worker/{ws_a}", cookies=cookies_b)
        assert resp.status_code in (403, 404)


# ===========================================================================
# Contract cases (Step 40) — parametrized failure modes on representative
# POST endpoints
# ===========================================================================

CONTRACT_CASES = [
    # (path-template, payload-modifier, expected)
    ("/ops23/routing/decide", {"workspace_id": 99999999}, (403, 404)),
    ("/ops23/routing/admission", {"workspace_id": 99999999}, (403, 404)),
    ("/ops23/cost/reconcile", {"workspace_id": 99999999}, (403, 404)),
    ("/ops23/evaluations", {"workspace_id": 99999999}, (403, 404)),
]


class TestContractFailureCases:
    @pytest.mark.parametrize("path,extra,expected", CONTRACT_CASES)
    def test_cross_workspace_rejected(self, client, path, extra, expected):
        cookies, ws = _user_and_ws(client, "cc")
        payload = {"workspace_id": ws, "domain": "retrieval"}
        payload.update(extra)
        resp = client.post(path, cookies=cookies, json=payload)
        assert resp.status_code in expected

    def test_missing_required_fields_422(self, client):
        cookies, _ = _user_and_ws(client, "mf")
        resp = client.post("/ops23/evaluations", cookies=cookies,
                           json={"workspace_id": 1})
        assert resp.status_code == 422

    def test_error_contract_is_structured(self, client):
        resp = client.get("/ops23/capabilities")
        assert resp.status_code == 401
        assert resp.json()["detail"]
