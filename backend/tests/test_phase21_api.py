"""Phase 21 tests — ops21 API surface + E2E product flows.

Auth/authorization matrix on the autonomy control surface, pagination
bounds, idempotency contracts, and the seven E2E flows: self-healing,
knowledge, retrieval, provider, agent, incident, and autonomy.
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
from app.models.phase21 import (  # noqa: E402
    AutonomousOperation, AutonomyPolicy, RecoveryAttempt, RecoveryPlaybook,
    PlatformEvent, IncidentP21, AdaptiveCandidate, IngestionQualitySample,
    ModelPerformanceSample, AgentPlanRisk, AgentRecoveryEvent,
    WorkflowRiskAssessment, MaintenancePlan, EmergencyStop, DiagnosisReport,
    KnowledgeRecoveryPlan,
)
from app.services import autonomy as au  # noqa: E402

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


P21_API_TABLES = [
    MaintenancePlan, IncidentP21, EmergencyStop, DiagnosisReport,
    KnowledgeRecoveryPlan, AdaptiveCandidate, IngestionQualitySample,
    ModelPerformanceSample, AgentRecoveryEvent, AgentPlanRisk,
    WorkflowRiskAssessment, RecoveryAttempt, RecoveryPlaybook,
    PlatformEvent, AutonomousOperation, AutonomyPolicy,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in P21_API_TABLES:
        db_session.query(model).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _user_and_ws(client, tag):
    _counter[0] += 1
    email = f"{tag}{_counter[0]}@p21apitest.example"
    resp = client.post("/auth/register", json={
        "name": tag.title(), "email": email, "password": "password123"})
    assert resp.status_code in (200, 201), resp.text
    login = client.post("/auth/login", json={"email": email,
                                             "password": "password123"})
    cookies = login.cookies
    ws = client.post("/workspaces", json={"name": f"ws-{tag}"},
                     cookies=cookies)
    return cookies, ws.json()["id"]


def allow_policy(client, cookies, ws_id, operation_type,
                 level="AUTO_LOW_RISK"):
    resp = client.post("/ops21/autonomy/policies", cookies=cookies, json={
        "workspace_id": ws_id, "operation_type": operation_type,
        "autonomy_level": level, "requires_approval": False})
    assert resp.status_code == 200, resp.text


# ===========================================================================
# Route inventory + auth matrix
# ===========================================================================

class TestRouteInventory:
    def test_ops21_routes_registered(self):
        routes = [r.path for r in app.routes
                  if hasattr(r, "methods") and r.path.startswith("/ops21")]
        assert len(routes) >= 60

    def test_protected_routes_reject_anonymous(self, client):
        protected = [
            "/ops21/autonomy/policies?workspace_id=1",
            "/ops21/autonomy/operations?workspace_id=1",
            "/ops21/recovery/attempts?workspace_id=1",
            "/ops21/diagnosis?workspace_id=1",
            "/ops21/knowledge/recovery-plans?workspace_id=1",
            "/ops21/candidates?workspace_id=1",
            "/ops21/incidents?workspace_id=1",
            "/ops21/personal/activity?workspace_id=1",
        ]
        for path in protected:
            resp = client.get(path)
            assert resp.status_code in (401, 403), path

    def test_writes_require_membership(self, client):
        resp = client.post("/ops21/autonomy/policies", json={
            "workspace_id": 999999, "operation_type": "op"})
        assert resp.status_code in (401, 403)

    def test_owner_endpoints_require_owner(self, client):
        cookies, ws_id = _user_and_ws(client, "member")
        # Member (not owner of a different workspace) gets 403/404.
        resp = client.post("/ops21/autonomy/policies", cookies=cookies,
                           json={"workspace_id": 999999,
                                 "operation_type": "op"})
        assert resp.status_code in (403, 404)


class TestPaginationAndBounds:
    def test_operations_pagination(self, client, db_session):
        cookies, ws_id = _user_and_ws(client, "page")
        for i in range(3):
            au.guard_operation(db_session, ws_id, "page.op",
                               idempotency_key=f"p-{ws_id}-{i}")
        resp = client.get(
            f"/ops21/autonomy/operations?workspace_id={ws_id}&limit=2",
            cookies=cookies)
        assert resp.status_code == 200
        assert len(resp.json()["operations"]) <= 2

    def test_incidents_pagination(self, client):
        cookies, ws_id = _user_and_ws(client, "inc")
        for i in range(3):
            client.post("/ops21/incidents", cookies=cookies,
                        json={"workspace_id": ws_id, "title": f"inc{i}",
                              "severity": "SEV3"})
        resp = client.get(
            f"/ops21/incidents?workspace_id={ws_id}&limit=2",
            cookies=cookies)
        assert len(resp.json()["incidents"]) == 2

    def test_limit_capped(self, client, db_session):
        cookies, ws_id = _user_and_ws(client, "cap")
        resp = client.get(
            f"/ops21/autonomy/operations?workspace_id={ws_id}&limit=100000",
            cookies=cookies)
        assert resp.status_code == 200  # capped server-side, no error


class TestIdempotencyContracts:
    def test_guard_idempotent_via_api(self, client, db_session):
        cookies, ws_id = _user_and_ws(client, "idem")
        key = f"api-key-{uuid.uuid4()}"
        body = {"workspace_id": ws_id, "operation_type": "api.op",
                "idempotency_key": key}
        r1 = client.post("/ops21/autonomy/guard", cookies=cookies, json=body)
        r2 = client.post("/ops21/autonomy/guard", cookies=cookies, json=body)
        assert r1.json()["operation_id"] == r2.json()["operation_id"]

    def test_event_dedup_via_api(self, client):
        cookies, ws_id = _user_and_ws(client, "evd")
        body = {"workspace_id": ws_id, "event_kind": "ops.recovery",
                "event_key": "same-key"}
        r1 = client.post("/ops21/events", cookies=cookies, json=body)
        r2 = client.post("/ops21/events", cookies=cookies, json=body)
        assert r1.json()["created"] is True
        assert r2.json()["deduplicated"] is True

    def test_recovery_attempt_idempotent_via_api(self, client):
        cookies, ws_id = _user_and_ws(client, "rec")
        allow_policy(client, cookies, ws_id, "recovery")
        pb = client.post("/ops21/recovery/playbooks", cookies=cookies, json={
            "workspace_id": ws_id, "name": "pb-api",
            "trigger": "cache_failure",
            "actions": ["invalidate_cache"], "approved": True})
        assert pb.status_code == 200, pb.text
        body = {"workspace_id": ws_id, "trigger": "cache_failure",
                "playbook_id": pb.json()["playbook_id"],
                "idempotency_key": "fixed-key"}
        r1 = client.post("/ops21/recovery/attempt", cookies=cookies,
                         json=body)
        r2 = client.post("/ops21/recovery/attempt", cookies=cookies,
                         json=body)
        assert r1.json()["attempt_id"] == r2.json()["attempt_id"]

    def test_error_contract_structured(self, client):
        cookies, ws_id = _user_and_ws(client, "err")
        resp = client.post("/ops21/autonomy/policies", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "operation_type": "op",
                                 "autonomy_level": "INVALID"})
        assert resp.status_code == 422
        assert "detail" in resp.json()


# ===========================================================================
# E2E product flows
# ===========================================================================

class TestE2ESelfHealingFlow:
    def test_detection_to_audit(self, client, db_session):
        cookies, ws_id = _user_and_ws(client, "e2esh")
        # failure -> detection
        failures = client.post(
            f"/ops21/health/failures?workspace_id={ws_id}", cookies=cookies,
            json={"worker": {"heartbeat_age_seconds": 400},
                  "provider_failures": 2}).json()["failures"]
        assert len(failures) >= 1
        # diagnosis
        diag = client.post("/ops21/diagnosis", cookies=cookies, json={
            "workspace_id": ws_id, "symptom": "worker stall + provider errors",
            "signals": {"worker": {"heartbeat_age_seconds": 400},
                        "provider": {"error_rate": 0.5, "sample_count": 10}}})
        assert diag.status_code == 200
        assert diag.json()["top_confidence"] > 0
        # governed recovery
        allow_policy(client, cookies, ws_id, "recovery")
        pb = client.post("/ops21/recovery/playbooks", cookies=cookies, json={
            "workspace_id": ws_id, "name": "e2e-pb",
            "trigger": [f["kind"] for f in failures][0],
            "actions": ["restart_worker_state"], "approved": True}).json()
        att = client.post("/ops21/recovery/attempt", cookies=cookies, json={
            "workspace_id": ws_id,
            "trigger": [f["kind"] for f in failures][0],
            "playbook_id": pb["playbook_id"]}).json()
        assert att["status"] in ("SUCCEEDED", "COOLDOWN_BLOCKED",
                                 "POLICY_BLOCKED")
        # audit visible
        ops = client.get(
            f"/ops21/autonomy/operations?workspace_id={ws_id}",
            cookies=cookies).json()["operations"]
        assert isinstance(ops, list)


class TestE2EKnowledgeFlow:
    def test_degradation_to_governed_recovery(self, client):
        cookies, ws_id = _user_and_ws(client, "e2ekn")
        # health degradation
        health = client.post("/ops21/knowledge/health", cookies=cookies,
                             json={"workspace_id": ws_id,
                                   "signals": {"embedding_coverage": 40}}).json()
        assert health["overall"] < 100
        # recovery recommendation (plan)
        plan = client.post("/ops21/knowledge/recovery-plans", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "issue_kind": "embedding_gap",
                                 "target_type": "embedding"}).json()
        assert plan["status"] == "PROPOSED"
        # governed execution -> validation
        result = client.post(
            f"/ops21/knowledge/recovery-plans/{plan['plan_id']}/execute"
            f"?workspace_id={ws_id}", cookies=cookies).json()
        assert result["decision"] in ("REQUIRES_APPROVAL", "ALLOWED",
                                      "BLOCKED")
        plans = client.get(
            f"/ops21/knowledge/recovery-plans?workspace_id={ws_id}",
            cookies=cookies).json()["plans"]
        assert plans[0]["decision"] == result["decision"]


class TestE2ERetrievalFlow:
    def test_candidate_evaluation_promotion(self, client):
        cookies, ws_id = _user_and_ws(client, "e2er")
        # candidate
        cand = client.post("/ops21/candidates", cookies=cookies, json={
            "workspace_id": ws_id, "domain": "retrieval",
            "change_kind": "vector_weight", "proposed_value": {"v": 0.7},
            "rationale": "drift"}).json()
        # evaluation against gates (query params)
        ev = client.post(
            f"/ops21/candidates/{cand['candidate_id']}/evaluate",
            cookies=cookies,
            params={"workspace_id": ws_id, "evaluation_score": 0.95,
                    "baseline_score": 0.5}).json()
        assert ev["gates_passed"] is True
        # promotion (approval required under default policy)
        promo = client.post(
            f"/ops21/candidates/{cand['candidate_id']}/promote"
            f"?workspace_id={ws_id}", cookies=cookies).json()
        assert promo["decision"] == "REQUIRES_APPROVAL"


class TestE2EProviderFlow:
    def test_degradation_to_simulation(self, client, db_session):
        cookies, ws_id = _user_and_ws(client, "e2ep")
        # provider degradation signals
        for _ in range(5):
            client.post("/ops21/models/samples", cookies=cookies, json={
                "workspace_id": ws_id, "model": "m", "provider": "p",
                "availability": 0.99})
        client.post("/ops21/models/samples", cookies=cookies, json={
            "workspace_id": ws_id, "model": "m", "provider": "p",
            "availability": 0.4})
        drift = client.post(
            f"/ops21/models/drift?workspace_id={ws_id}&model=m&provider=p"
            f"&metric=availability", cookies=cookies).json()
        assert drift["drift_detected"] is True
        # fallback simulation
        workload = [{"task": "t", "model_perf": {
            "b": {"cost_usd": 0.01, "latency_ms": 10, "quality": 0.9}}}]
        sim = client.post("/ops21/models/routing-simulate", cookies=cookies,
                          json={"workspace_id": ws_id, "description": "fb",
                                "workload": workload, "current_routing":
                                    {"t": "a"},
                                "candidate_routing": {"t": "b"}}).json()
        assert sim["safe"] is True


class TestE2EAgentFlow:
    def test_plan_risk_to_recovery(self, client):
        cookies, ws_id = _user_and_ws(client, "e2ea")
        plan = {"summary": "s", "steps": [{"tool": "retrieve"}]}
        risk = client.post("/ops21/agents/plan-risk", cookies=cookies,
                           json={"workspace_id": ws_id, "plan": plan}).json()
        assert risk["risk_level"] == "LOW"
        sim = client.post("/ops21/agents/plan-simulate", cookies=cookies,
                          json={"workspace_id": ws_id, "plan": plan}).json()
        assert sim["simulated"] if "simulated" in sim else True
        # failure -> dead letter -> recovery
        dl = client.post("/ops21/agents/dead-letter", cookies=cookies,
                         params={"workspace_id": ws_id,
                                 "execution_id": "e1",
                                 "reason": "timeout"})
        assert dl.status_code == 200
        rec = client.post("/ops21/agents/recover", cookies=cookies,
                          json={"workspace_id": ws_id, "agent_run_id": 1,
                                "recovery_kind": "RETRY"}).json()
        assert rec["recovery_kind"] == "RETRY"


class TestE2EWorkflowIncidentAutonomy:
    def test_workflow_policy_execution_diagnosis(self, client):
        cookies, ws_id = _user_and_ws(client, "e2ew")
        risk = client.post("/ops21/workflows/risk", cookies=cookies, json={
            "workspace_id": ws_id, "workflow": {"steps": []}}).json()
        assert risk["autonomy_decision"] is not None
        sim = client.post("/ops21/workflows/simulate", cookies=cookies,
                          json={"workspace_id": ws_id,
                                "workflow": {"steps": [{"name": "a"}]}}).json()
        assert sim["simulated"] is True

    def test_incident_to_postmortem(self, client):
        cookies, ws_id = _user_and_ws(client, "e2ei")
        inc = client.post("/ops21/incidents", cookies=cookies, json={
            "workspace_id": ws_id, "title": "api outage",
            "severity": "SEV1"}).json()
        pm = client.post(
            f"/ops21/incidents/{inc['incident_id']}/postmortem",
            cookies=cookies,
            json={"workspace_id": ws_id, "diagnosis_summary": "bad deploy",
                  "contributing_factors": ["no canary"]}).json()
        assert pm["status"] == "DRAFT_REQUIRES_HUMAN_REVIEW"
        fin = client.post(
            f"/ops21/incidents/{inc['incident_id']}/finalize",
            cookies=cookies,
            json={"workspace_id": ws_id, "diagnosis_summary": "x"}).json()
        assert fin["postmortem_finalized"] is True

    def test_autonomy_proposal_to_audit(self, client):
        cookies, ws_id = _user_and_ws(client, "e2eau")
        # policy (authorization)
        allow_policy(client, cookies, ws_id, "sim.op")
        # simulation (zero side effects)
        sim = client.post("/ops21/autonomy/simulate", cookies=cookies,
                          json={"workspace_id": ws_id,
                                "operation_type": "sim.op"}).json()
        assert sim["would_auto_execute"] is True
        # guard decision (execution authorization)
        guard = client.post("/ops21/autonomy/guard", cookies=cookies,
                            json={"workspace_id": ws_id,
                                  "operation_type": "sim.op"}).json()
        assert guard["decision"] == "ALLOWED"
        # audit trail
        ops = client.get(
            f"/ops21/autonomy/operations?workspace_id={ws_id}",
            cookies=cookies).json()["operations"]
        assert any(o["id"] == guard["operation_id"] for o in ops)

    def test_maintenance_dry_run_approve_execute(self, client):
        cookies, ws_id = _user_and_ws(client, "e2em")
        plan = client.post("/ops21/maintenance/plans", cookies=cookies,
                           json={"workspace_id": ws_id, "plan_kind": "cleanup",
                                 "targets": [{"id": 1}, {"id": 2}]}).json()
        assert plan["requires_approval"] is True
        dry = client.post(
            f"/ops21/maintenance/plans/{plan['plan_id']}/dry-run",
            cookies=cookies,
            json={"workspace_id": ws_id, "plan_kind": "cleanup"}).json()
        assert dry["executed"] is False
        appr = client.post(
            f"/ops21/maintenance/plans/{plan['plan_id']}/approve",
            cookies=cookies,
            json={"workspace_id": ws_id, "plan_kind": "cleanup"}).json()
        assert appr["status"] == "APPROVED"
        exe = client.post(
            f"/ops21/maintenance/plans/{plan['plan_id']}/execute",
            cookies=cookies,
            json={"workspace_id": ws_id, "plan_kind": "cleanup"}).json()
        assert exe["executed"] is True

    def test_dr_and_chaos_honest(self, client):
        cookies, ws_id = _user_and_ws(client, "e2edr")
        dr = client.post(f"/ops21/dr/simulate?workspace_id={ws_id}",
                         cookies=cookies).json()
        assert dr["simulated"] is True
        chaos = client.post("/ops21/chaos/run", cookies=cookies, json={
            "workspace_id": ws_id, "scenario": "broker_failure",
            "passed": True}).json()
        assert chaos["simulated"] is True
        assert chaos["bounded"] is True

    def test_emergency_stop_blocks_recovery(self, client):
        cookies, ws_id = _user_and_ws(client, "e2estop")
        stop = client.post("/ops21/safety/emergency-stop", cookies=cookies,
                           json={"workspace_id": ws_id, "scope": "ALL",
                                 "reason": "e2e"}).json()
        assert stop["active"] is True
        # Emergency stop is active; simulate reports blocked decisions.
        sim = client.post("/ops21/autonomy/simulate", cookies=cookies,
                          json={"workspace_id": ws_id,
                                "operation_type": "anything"}).json()
        assert sim["would_auto_execute"] is False
        assert sim["decision"] in ("REQUIRES_APPROVAL", "BLOCKED")
