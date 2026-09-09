"""Phase 21 tests — remaining API surfaces + extended coverage.

Cost autopilot APIs (forecast, guard, optimize, anomalies, scheduling),
infrastructure healing APIs (workers, broker, scheduler, db, cache),
graph repair APIs, personal AI APIs, artifact intelligence + search
autopilot services, event replay API, evaluation schedule API, residency
and failover APIs, emergency stop interplay, and tenant isolation
spot-checks across every Phase 21 subsystem.
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
from app.models.phase20 import SearchQualityEvent  # noqa: E402
from app.models.phase21 import (  # noqa: E402
    AutonomousOperation, AutonomyPolicy, CostForecast, CostGuardDecision,
    CostOptimizationEvent, CostAwareQueueDecision, CostAnomaly,
    WorkerHealthScore, BrokerHealthSnapshot, SchedulerHealthSnapshot,
    SlowQueryRecord, CacheHealthSnapshot, GraphRepairProposal,
    PersonalAutonomySetting, AIActivityItem, MemoryAutonomyEvent,
    ArtifactQualityScore, AdaptiveCandidate, PlatformEvent,
    EvaluationSchedule, EvaluationRun, EmergencyStop, RegionHealthP21,
    FailoverSimulation, ResidencyGuardEvent, RecoveryPlaybook,
)
from app.services import personal_ai as pa  # noqa: E402
from app.services import autonomy as au  # noqa: E402
from app.services import infra_heal as ih  # noqa: E402

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


P21_SURFACE_TABLES = [
    ResidencyGuardEvent, FailoverSimulation, RegionHealthP21,
    EmergencyStop, EvaluationRun, EvaluationSchedule, PlatformEvent,
    AdaptiveCandidate, ArtifactQualityScore, MemoryAutonomyEvent,
    AIActivityItem, PersonalAutonomySetting, CacheHealthSnapshot,
    SlowQueryRecord, SchedulerHealthSnapshot, BrokerHealthSnapshot,
    WorkerHealthScore, CostAnomaly, CostAwareQueueDecision,
    CostOptimizationEvent, CostGuardDecision, CostForecast,
    AutonomousOperation, AutonomyPolicy, SearchQualityEvent,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in P21_SURFACE_TABLES:
        db_session.query(model).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _user_and_ws(client, tag):
    _counter[0] += 1
    email = f"{tag}{_counter[0]}@p21surf.example"
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
# Cost APIs
# ===========================================================================

class TestCostAPIs:
    def test_forecast_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "cost")
        resp = client.post("/ops21/cost/forecast", cookies=cookies,
                           params={"workspace_id": ws_id,
                                   "current_spend_usd": 100.0,
                                   "daily_run_rate_usd": 5.0,
                                   "budget_usd": 1000.0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["over_budget"] is False
        assert body["forecast_usd"] == 250.0

    def test_guard_endpoint_allows_and_blocks(self, client):
        cookies, ws_id = _user_and_ws(client, "guard")
        ok = client.post("/ops21/cost/guard", cookies=cookies, json={
            "workspace_id": ws_id, "operation_type": "small.op",
            "estimated_cost_usd": 1.0, "remaining_budget_usd": 10.0})
        assert ok.json()["decision"] == "ALLOWED"
        blocked = client.post("/ops21/cost/guard", cookies=cookies, json={
            "workspace_id": ws_id, "operation_type": "big.op",
            "estimated_cost_usd": 100.0, "remaining_budget_usd": 10.0})
        assert blocked.json()["decision"] == "BLOCKED"

    def test_optimize_endpoint_governed(self, client):
        cookies, ws_id = _user_and_ws(client, "opt")
        resp = client.post("/ops21/cost/optimize", cookies=cookies, json={
            "workspace_id": ws_id, "optimization_kind": "cache_reuse"})
        assert resp.status_code == 200
        assert resp.json()["applied"] is False  # default policy

    def test_optimize_endpoint_with_policy(self, client):
        cookies, ws_id = _user_and_ws(client, "opt2")
        allow_policy(client, cookies, ws_id, "cost.optimize.batching")
        resp = client.post("/ops21/cost/optimize", cookies=cookies, json={
            "workspace_id": ws_id, "optimization_kind": "batching"})
        assert resp.json()["applied"] is True

    def test_anomaly_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "anom")
        resp = client.post("/ops21/cost/anomalies", cookies=cookies,
                           params={"workspace_id": ws_id,
                                   "baseline_value": 100.0,
                                   "observed_value": 400.0})
        assert resp.json()["anomaly_detected"] is True
        assert resp.json()["severity"] == "HIGH"


# ===========================================================================
# Infrastructure healing APIs
# ===========================================================================

class TestInfraAPIs:
    def test_worker_health_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "worker")
        resp = client.post("/ops21/workers/health", cookies=cookies, json={
            "workspace_id": ws_id, "worker_id": "w-1",
            "heartbeat_age_seconds": 5, "throughput": 9.0})
        assert resp.json()["state"] == "HEALTHY"

    def test_worker_recover_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "wkrec")
        health = client.post("/ops21/workers/health", cookies=cookies, json={
            "workspace_id": ws_id, "worker_id": "w-2",
            "heartbeat_age_seconds": 900, "failure_count": 8}).json()
        assert health["quarantined"] is True
        rec = client.post(
            f"/ops21/workers/{health['score_id']}/recover?workspace_id={ws_id}",
            cookies=cookies).json()
        assert rec["state"] == "RECOVERED"

    def test_capacity_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "cap")
        resp = client.post("/ops21/workers/capacity", cookies=cookies,
                           params={"workspace_id": ws_id,
                                   "current_workers": 4,
                                   "queue_depth": 20})
        body = resp.json()
        assert body["fairness_preserved"] is True

    def test_broker_health_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "brk")
        resp = client.post("/ops21/broker/health", cookies=cookies, json={
            "workspace_id": ws_id, "depth": 900, "reconnects": 6})
        body = resp.json()
        assert body["state"] == "DEGRADED"
        assert "reconnect_broker" in body["plan"]["actions"]

    def test_scheduler_health_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "sched")
        resp = client.post("/ops21/scheduler/health", cookies=cookies,
                           params={"workspace_id": ws_id, "has_leader": True,
                                   "missed_schedules": 100})
        body = resp.json()
        assert body["recovered_schedules"] == 50  # bounded

    def test_slow_query_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "slow")
        resp = client.post("/ops21/db/slow-queries", cookies=cookies,
                           params={"workspace_id": ws_id,
                                   "statement": "SELECT * FROM t WHERE a=1",
                                   "duration_ms": 3000.0})
        body = resp.json()
        assert body["recorded"] is True
        assert body["auto_applied"] is False

    def test_cache_health_and_invalidate(self, client):
        cookies, ws_id = _user_and_ws(client, "cache")
        health = client.post("/ops21/cache/health", cookies=cookies, json={
            "workspace_id": ws_id, "cache": "embedding_cache",
            "hits": 2, "misses": 98}).json()
        assert health["anomaly"] is True
        inv = client.post("/ops21/cache/invalidate", cookies=cookies, json={
            "workspace_id": ws_id,
            "cache_keys": [f"ws:{ws_id}:doc:1", "ws:999:doc:1"]}).json()
        assert inv["invalidated"] == [f"ws:{ws_id}:doc:1"]
        assert inv["tenant_safe"] is True


# ===========================================================================
# Graph repair + personal AI APIs
# ===========================================================================

class TestGraphPersonalAPIs:
    def test_graph_repair_flow(self, client):
        cookies, ws_id = _user_and_ws(client, "graph")
        prop = client.post("/ops21/graph/repairs", cookies=cookies, json={
            "workspace_id": ws_id, "repair_kind": "stale_edge_refresh",
            "target_entity_id": 3}).json()
        assert prop["status"] == "PROPOSED"
        result = client.post(
            f"/ops21/graph/repairs/{prop['proposal_id']}/execute"
            f"?workspace_id={ws_id}", cookies=cookies).json()
        assert result["decision"] == "REQUIRES_APPROVAL"

    def test_personal_settings_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "pers")
        resp = client.post("/ops21/personal/settings", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "autonomy_level": "OBSERVE",
                                 "workspace_policy_level": "RECOMMEND"})
        assert resp.json()["effective_level"] == "OBSERVE"

    def test_personal_memory_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "mem")
        resp = client.post("/ops21/personal/memory", cookies=cookies, json={
            "workspace_id": ws_id, "action": "inspect"})
        assert resp.json()["performed"] is True
        denied = client.post("/ops21/personal/memory", cookies=cookies,
                             json={"workspace_id": ws_id, "action": "delete",
                                   "memory_id": 3})
        assert denied.json()["performed"] is False

    def test_activity_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "act")
        client.post("/ops21/personal/activity", cookies=cookies, json={
            "workspace_id": ws_id, "item_kind": "SUGGESTION",
            "title": "refine query",
            "explanation": "evidence: 2 zero-result searches"})
        feed = client.get(
            f"/ops21/personal/activity?workspace_id={ws_id}",
            cookies=cookies).json()["items"]
        assert len(feed) == 1
        assert feed[0]["explanation"].startswith("evidence:")


# ===========================================================================
# Event replay + evaluation scheduling APIs
# ===========================================================================

class TestEventEvalAPIs:
    def test_replay_requires_owner(self, client):
        cookies, ws_id = _user_and_ws(client, "rep")
        client.post("/ops21/events", cookies=cookies, json={
            "workspace_id": ws_id, "event_kind": "ops.recovery",
            "event_key": "k1"})
        resp = client.post("/ops21/events/replay", cookies=cookies,
                           params={"workspace_id": ws_id,
                                   "event_kind": "ops.recovery"})
        assert resp.status_code == 200
        assert resp.json()["replayed"] == 1

    def test_eval_schedule_and_run(self, client):
        cookies, ws_id = _user_and_ws(client, "eval")
        sched = client.post("/ops21/evaluation/schedules", cookies=cookies,
                            json={"workspace_id": ws_id, "domain": "retrieval",
                                  "dataset_version": "v7"}).json()
        assert sched["dataset_version"] == "v7"
        run = client.post(
            f"/ops21/evaluation/schedules/{sched['schedule_id']}/run",
            cookies=cookies,
            params={"workspace_id": ws_id},
            json={"metrics": {"hit_rate": 0.4},
                  "baseline_metrics": {"hit_rate": 0.9}}).json()
        assert run["regression_detected"] is True
        assert run["gate_passed"] is False


# ===========================================================================
# Region / DR APIs
# ===========================================================================

def _make_org_with_member(client, db, ws_id, tag):
    """Create an org owned by the caller's user and attach the workspace."""
    from app.models.organization import Organization, OrganizationMember
    ws = db.query(Workspace).filter_by(id=ws_id).first()
    org = Organization(name=f"org-{tag}-{uuid.uuid4().hex[:6]}",
                       slug=f"org-{tag}-{uuid.uuid4().hex[:6]}",
                       owner_id=ws.owner_id)
    db.add(org)
    db.flush()
    db.add(OrganizationMember(organization_id=org.id, user_id=ws.owner_id,
                              role="OWNER"))
    ws.organization_id = org.id
    db.commit()
    return org.id


class TestRegionAPIs:
    def test_region_health_endpoint(self, client, db_session):
        cookies, ws_id = _user_and_ws(client, "reg")
        org_id = _make_org_with_member(client, db_session, ws_id, "reg")
        resp = client.post("/ops21/regions/health", cookies=cookies, json={
            "organization_id": org_id, "region_id": "us-east",
            "workers": 4})
        assert resp.status_code == 200, resp.text
        assert resp.json()["failover_ready"] is True

    def test_failover_simulate_endpoint(self, client, db_session):
        cookies, ws_id = _user_and_ws(client, "fail")
        org_id = _make_org_with_member(client, db_session, ws_id, "fail")
        ih.record_region_health(db_session, org_id, "us-east", workers=4)
        resp = client.post("/ops21/regions/failover-simulate",
                           cookies=cookies,
                           json={"organization_id": org_id,
                                 "from_region": "us-west",
                                 "to_region": "us-east"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ready"] is True
        assert body["executed"] is False

    def test_residency_check_endpoint(self, client, db_session):
        cookies, ws_id = _user_and_ws(client, "res")
        org_id = _make_org_with_member(client, db_session, ws_id, "res")
        resp = client.post("/ops21/regions/residency-check", cookies=cookies,
                           params={"organization_id": org_id,
                                   "requested_region": "eu-west"},
                           json={"__allowed__": ["us-east"]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["blocked"] is True


# ===========================================================================
# Artifact intelligence + search autopilot (service level)
# ===========================================================================

class TestArtifactsAndSearch:
    def test_artifact_scoring(self, db_session):
        ws = _ws(db_session)
        row = pa.score_artifact(db_session, ws, artifact_id=1,
                                groundedness=0.9, citation_coverage=0.8,
                                completeness=0.7, provenance_ok=True)
        assert row.quality_score == pytest.approx(81.0, abs=0.5)

    def test_artifact_provenance_caps_score(self, db_session):
        ws = _ws(db_session)
        row = pa.score_artifact(db_session, ws, artifact_id=2,
                                groundedness=1.0, citation_coverage=1.0,
                                completeness=1.0, provenance_ok=False)
        assert row.quality_score <= 50.0

    def test_artifact_version_regression(self, db_session):
        ws = _ws(db_session)
        pa.score_artifact(db_session, ws, artifact_id=3, version=1,
                          groundedness=0.9, citation_coverage=0.9,
                          completeness=0.9)
        pa.score_artifact(db_session, ws, artifact_id=3, version=2,
                          groundedness=0.4, citation_coverage=0.4,
                          completeness=0.4)
        result = pa.compare_artifact_versions(db_session, ws, 3)
        assert result["comparable"] is True
        assert result["regression"] is True

    def test_search_quality_from_events(self, db_session):
        ws = _ws(db_session)
        events = [{"zero_results": False, "feedback": "up"}] * 8 + \
                 [{"zero_results": True}] * 2
        result = pa.search_quality_snapshot(db_session, ws, events=events)
        assert result["zero_result_rate"] == pytest.approx(0.2, abs=0.01)
        assert 0 < result["quality"] <= 100

    def test_search_drift(self, db_session):
        result = pa.detect_search_drift(db_session, 0, baseline=0.8,
                                        current=0.6)
        assert result["degraded"] is True

    def test_search_candidate_allowed_kinds(self, db_session):
        ws = _ws(db_session)
        cand = pa.propose_search_candidate(db_session, ws, "ranking_weights",
                                           {"fresh": 0.1}, "drift")
        assert cand.domain == "search"
        with pytest.raises(ValueError):
            pa.propose_search_candidate(db_session, ws, "randomness", {},
                                        "x")

    def _ws(self, db_session):
        return _fresh_ws(db_session)


# ===========================================================================
# Cross-cutting isolation + misc
# ===========================================================================

class TestIsolation:
    def test_operations_tenant_scoped(self, client, db_session):
        cookies1, ws1 = _user_and_ws(client, "iso1")
        cookies2, ws2 = _user_and_ws(client, "iso2")
        au.guard_operation(db_session, ws1, "iso.op",
                           idempotency_key=f"iso-{ws1}")
        rows2 = au.list_operations(db_session, ws2)
        assert all(r.workspace_id == ws2 for r in rows2)

    def test_playbooks_tenant_scoped(self, client, db_session):
        from app.services import selfheal as sh
        ws1, ws2 = _fresh_ws(db_session), _fresh_ws(db_session)
        sh.create_playbook(db_session, ws1, "iso-pb", "cache_failure",
                           ["invalidate_cache"])
        assert db_session.query(RecoveryPlaybook).filter_by(
            workspace_id=ws2).count() == 0

    def test_emergency_stop_scoped(self, client, db_session):
        cookies1, ws1 = _user_and_ws(client, "estop1")
        cookies2, ws2 = _user_and_ws(client, "estop2")
        au.create_policy(db_session, ws2, "op",
                         autonomy_level="AUTO_LOW_RISK")
        client.post("/ops21/safety/emergency-stop", cookies=cookies1,
                    json={"workspace_id": ws1, "scope": "ALL",
                          "reason": "iso"})
        # ws2 has no stop — auto still allowed there.
        sim2 = au.simulate_operation(db_session, ws2, "op")
        assert sim2["would_auto_execute"] is True

    def test_activity_feed_user_scoped(self, db_session):
        ws = _fresh_ws(db_session)
        pa.record_activity(db_session, ws, user_id=1, item_kind="SUGGESTION",
                           title="a")
        pa.record_activity(db_session, ws, user_id=2, item_kind="ACTION",
                           title="b")
        assert len(pa.activity_feed(db_session, ws, user_id=1)) == 1
        assert len(pa.activity_feed(db_session, ws, user_id=2)) == 1


class TestMisc:
    def test_simulation_is_zero_side_effect(self, db_session):
        ws = _fresh_ws(db_session)
        before = db_session.query(AutonomousOperation).count()
        au.simulate_operation(db_session, ws, "any.op")
        assert db_session.query(AutonomousOperation).count() == before

    def test_policies_listing_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "plist")
        allow_policy(client, cookies, ws_id, "listed.op")
        rows = client.get(
            f"/ops21/autonomy/policies?workspace_id={ws_id}",
            cookies=cookies).json()["policies"]
        assert any(p["operation_type"] == "listed.op" for p in rows)

    def test_transitions_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "trans")
        resp = client.get(
            f"/ops21/autonomy/transitions?workspace_id={ws_id}",
            cookies=cookies)
        assert resp.status_code == 200

    def test_health_latest_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "hlate")
        client.post("/ops21/health/aggregate", cookies=cookies, json={
            "workspace_id": ws_id,
            "component_states": {"db": "UNHEALTHY", "api": "HEALTHY"}})
        latest = client.get(
            f"/ops21/health/latest?workspace_id={ws_id}",
            cookies=cookies).json()["snapshot"]
        assert latest["overall_state"] == "UNHEALTHY"

    def test_agent_risk_endpoint(self, client):
        cookies, ws_id = _user_and_ws(client, "arisk")
        resp = client.post("/ops21/agents/plan-risk", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "plan": {"summary": "s",
                                          "steps": [{"tool": "search"}]}})
        assert resp.json()["risk_level"] == "LOW"


def _fresh_ws(db_session):
    _counter[0] += 1
    user = User(name=f"own{_counter[0]}",
                email=f"own{_counter[0]}@p21surf.example",
                password_hash="x")
    db_session.add(user)
    db_session.flush()
    ws = Workspace(name=f"wss{_counter[0]}", owner_id=user.id)
    db_session.add(ws)
    db_session.commit()
    return ws.id


def _ws(db_session):
    return _fresh_ws(db_session)
