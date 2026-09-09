"""Phase 19 tests — ops19 API surface + E2E flows.

Auth/authorization matrix on the new control-plane surface, scheduler
leadership, control-plane config snapshots + rollback, broker/health/SLO/DR
readouts, consistency checks, quality gates, and E2E flows: operator
control-plane snapshot cycle, scheduler leadership election + takeover, DR
readiness, and guarded write authorization (member cannot pause another
member's workflow).
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
from app.models.phase19 import (  # noqa: E402
    SchedulerLeader, ControlPlaneSnapshot, SloBudgetWindow, SloDefinition,
    BackupRecord, QualityGate, ApiAbuseEvent, ConsistencyReport,
)
from app.models.phase17 import JobLease, WorkflowRun  # noqa: E402
from app.models.phase16 import WorkerJob  # noqa: E402

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
    db_session.query(SloBudgetWindow).delete()
    db_session.query(SloDefinition).delete()
    db_session.query(QualityGate).delete()
    db_session.query(ApiAbuseEvent).delete()
    db_session.query(ConsistencyReport).delete()
    db_session.query(BackupRecord).delete()
    db_session.query(ControlPlaneSnapshot).delete()
    db_session.query(SchedulerLeader).delete()
    db_session.query(WorkflowRun).delete()
    db_session.query(JobLease).delete()
    db_session.query(WorkerJob).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.commit()
    yield


def register_user(client, tag="u"):
    _counter[0] += 1
    email = f"{tag}{_counter[0]}@p19api.com"
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


# ============================================================
# Auth matrix
# ============================================================

class TestOps19Auth:
    @pytest.mark.parametrize("path", [
        "/ops19/health", "/ops19/broker", "/ops19/scheduler/leader",
        "/ops19/config", "/ops19/slo/health", "/ops19/dr/readiness",
        "/ops19/providers/capabilities", "/ops19/workers/autoscale",
        "/ops19/backpressure", "/ops19/vector/drift",
    ])
    def test_ops19_get_routes_require_auth(self, client, path):
        resp = client.get(path)
        assert resp.status_code in (401, 403), resp.text

    @pytest.mark.parametrize("path", [
        "/ops19/config/snapshots", "/ops19/scheduler/acquire",
        "/ops19/cost/anomalies/detect", "/ops19/consistency/run",
        "/ops19/dr/backups", "/ops19/providers/capabilities",
    ])
    def test_ops19_write_routes_require_auth(self, client, path):
        resp = client.post(path, json={})
        assert resp.status_code in (401, 403), resp.text

    def test_quarantine_requires_auth(self, client):
        resp = client.post("/ops19/workers/w1/quarantine")
        assert resp.status_code in (401, 403)


# ============================================================
# Control plane API
# ============================================================

class TestControlPlaneApi:
    def test_global_health(self, client):
        cookies = register_user(client, "gh")
        resp = client.get("/ops19/health", cookies=cookies)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data or "components" in data or "healthy" in data

    def test_snapshot_requires_scope_id_for_tenant(self, client):
        cookies = register_user(client, "sn")
        resp = client.post("/ops19/config/snapshots",
                           json={"scope_type": "WORKSPACE", "config": {}},
                           cookies=cookies)
        assert resp.status_code == 400

    def test_snapshot_lifecycle(self, client, db_session):
        cookies = register_user(client, "snp")
        ws = create_workspace(client, cookies)
        resp = client.post("/ops19/config/snapshots",
                           json={"scope_type": "WORKSPACE", "scope_id": ws,
                                 "config": {"models_allowed": ["gpt-4o"]},
                                 "reason": "policy change"},
                           cookies=cookies)
        assert resp.status_code == 200, resp.text
        snapshot_id = resp.json()["snapshot_id"]
        activated = client.post(f"/ops19/config/activate?snapshot_id="
                                f"{snapshot_id}", cookies=cookies)
        assert activated.status_code == 200, activated.text
        assert activated.json()["status"] == "ACTIVATED"

    def test_snapshot_rollback_flow(self, client, db_session):
        cookies = register_user(client, "sbr")
        ws = create_workspace(client, cookies)
        first = client.post("/ops19/config/snapshots",
                            json={"scope_type": "WORKSPACE",
                                  "scope_id": ws,
                                  "config": {"budget_max_usd": 10},
                                  "reason": "v1"},
                            cookies=cookies).json()
        second = client.post("/ops19/config/snapshots",
                             json={"scope_type": "WORKSPACE",
                                   "scope_id": ws,
                                   "config": {"budget_max_usd": 100},
                                   "reason": "v2"},
                             cookies=cookies).json()
        rolled = client.post("/ops19/config/rollback",
                             json={"scope_type": "WORKSPACE",
                                   "scope_id": ws,
                                   "target_version": first["version"]},
                             cookies=cookies)
        assert rolled.status_code == 200, rolled.text
        assert rolled.json()["rollback"] is True
        assert rolled.json()["restores_version"] == first["version"]

    def test_region_upsert_and_list(self, client):
        cookies = register_user(client, "reg")
        resp = client.put("/ops19/regions", json={
            "region_id": "eu-west-1", "health_score": 0.9}, cookies=cookies)
        assert resp.status_code == 200, resp.text
        listed = client.get("/ops19/regions", cookies=cookies)
        assert listed.status_code == 200


class TestSchedulerLeadership:
    def test_acquire_and_leader_readout(self, client):
        cookies = register_user(client, "sched")
        resp = client.post(f"/ops19/scheduler/acquire?leader_id="
                           f"scheduler-{uuid.uuid4().hex[:8]}",
                           cookies=cookies)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "LEADER"
        leader = client.get("/ops19/scheduler/leader", cookies=cookies)
        assert leader.status_code == 200
        assert leader.json()["leader"] is not None

    def test_no_leader_initially(self, client):
        cookies = register_user(client, "sl0")
        leader = client.get("/ops19/scheduler/leader", cookies=cookies)
        assert leader.json()["leader"] is None

    def test_double_acquire_renews_same_leader(self, client):
        cookies = register_user(client, "dl")
        lid = f"s-{uuid.uuid4().hex[:8]}"
        assert client.post(f"/ops19/scheduler/acquire?leader_id={lid}",
                           cookies=cookies).json()["status"] == "LEADER"
        again = client.post(f"/ops19/scheduler/acquire?leader_id={lid}",
                            cookies=cookies)
        assert again.status_code == 200
        assert again.json()["renewed"] is True


# ============================================================
# Operations readouts
# ============================================================

class TestOpsReadouts:
    def test_broker_info(self, client):
        cookies = register_user(client, "brk")
        resp = client.get("/ops19/broker", cookies=cookies)
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["redis_optional"] is True
        assert "guarantees" in data and "recovery" in data

    def test_slo_health_endpoint(self, client):
        cookies = register_user(client, "slh")
        resp = client.get("/ops19/slo/health", cookies=cookies)
        assert resp.status_code == 200
        assert "definitions" in resp.json()

    def test_dr_readiness(self, client):
        cookies = register_user(client, "drr")
        resp = client.get("/ops19/dr/readiness", cookies=cookies)
        assert resp.status_code == 200
        score = resp.json()
        assert 0.0 <= score["score"] <= 1.0

    def test_dr_backup_and_invalid_scope(self, client):
        cookies = register_user(client, "drb")
        ok = client.post(f"/ops19/dr/backups?scope=DATABASE&backup_ref="
                         f"bkp-{uuid.uuid4().hex[:8]}", cookies=cookies)
        assert ok.status_code == 200, ok.text
        bad = client.post(f"/ops19/dr/backups?scope=CASSETTE&backup_ref=x",
                          cookies=cookies)
        assert bad.status_code == 400

    def test_consistency_run(self, client):
        cookies = register_user(client, "crn")
        resp = client.post("/ops19/consistency/run?dry_run=true",
                           cookies=cookies)
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True

    def test_provider_capabilities_register_and_route(self, client):
        cookies = register_user(client, "cap")
        registered = client.post("/ops19/providers/capabilities", json={
            "provider": "openai", "model": "gpt-4o",
            "supports_tools": True, "supports_structured": True,
            "context_window": 128000, "cost_per_1k_input": 0.005,
        }, cookies=cookies)
        assert registered.status_code == 200, registered.text
        routed = client.get("/ops19/providers/route?task=summarize",
                            cookies=cookies)
        assert routed.status_code == 200


class TestAuthorization:
    def test_member_cannot_pause_workflow_in_foreign_workspace(
            self, client, db_session):
        owner = register_user(client, "pa")
        cookies_b = register_user(client, "pb")
        ws_a = create_workspace(client, owner)
        # B is not even a member of ws_a.
        resp = client.post(f"/ops19/workflows/1/pause?workspace_id={ws_a}",
                           cookies=cookies_b)
        assert resp.status_code in (403, 404)

    def test_member_cannot_read_other_workspace_analytics(
            self, client, db_session):
        owner = register_user(client, "aa")
        ws = create_workspace(client, owner)
        outsider = register_user(client, "ab")
        resp = client.get(f"/ops19/analytics/workspace?workspace_id={ws}",
                          cookies=outsider)
        assert resp.status_code in (403, 404)

    def test_owner_can_run_workspace_analytics(self, client):
        owner = register_user(client, "ac")
        ws = create_workspace(client, owner)
        resp = client.get(f"/ops19/analytics/workspace?workspace_id={ws}",
                          cookies=owner)
        assert resp.status_code == 200
        assert "document_count" in resp.json()


# ============================================================
# E2E flows
# ============================================================

class TestE2E:
    def test_control_plane_snapshot_e2e(self, client, db_session):
        """Operator: snapshot → activate → verify effective config."""
        cookies = register_user(client, "e2e-cp")
        ws = create_workspace(client, cookies)
        snap = client.post("/ops19/config/snapshots", json={
            "scope_type": "WORKSPACE", "scope_id": ws,
            "config": {"budget_max_usd": 50}, "reason": "e2e"},
            cookies=cookies)
        assert snap.status_code == 200, snap.text
        sid = snap.json()["snapshot_id"]
        version = snap.json()["version"]
        act = client.post(f"/ops19/config/activate?snapshot_id={sid}",
                          cookies=cookies)
        assert act.json()["status"] == "ACTIVATED"
        eff = client.get(f"/ops19/config?scope_type=WORKSPACE"
                         f"&scope_id={ws}", cookies=cookies)
        assert eff.status_code == 200
        # the activated snapshot is now the effective active version
        assert eff.json()["active_version"] == version

    def test_scheduler_leadership_e2e(self, client, db_session):
        """Leader election — acquire visible via /leader readout."""
        cookies = register_user(client, "e2e-lead")
        lid = f"sched-{uuid.uuid4().hex[:6]}"
        assert client.post(f"/ops19/scheduler/acquire?leader_id={lid}",
                           cookies=cookies).json()["status"] == "LEADER"
        readout = client.get("/ops19/scheduler/leader", cookies=cookies)
        assert readout.json()["leader"]["leader_id"] == lid

    def test_dr_ops_e2e(self, client, db_session):
        """Backup recorded → readiness reports deterministic score."""
        cookies = register_user(client, "e2e-dr")
        client.post(f"/ops19/dr/backups?scope=DATABASE&backup_ref="
                    f"bkp-{uuid.uuid4().hex[:8]}", cookies=cookies)
        score = client.get("/ops19/dr/readiness", cookies=cookies).json()
        assert score["components"]["backup_present"] > 0

    def test_quality_gate_e2e(self, client, db_session):
        """Register a quality gate through the ops service surface."""
        from app.services import quality_platform as qp
        db = db_session
        gate = qp.add_gate(db, name="e2e-recall", metric="recall",
                           operator=">=", threshold=0.4)
        result = qp.check_gate(db, metric="recall", value=0.9)
        assert result["all_passed"] is True
        assert any(r["gate"] == "e2e-recall" for r in result["results"])