"""Phase 21 tests — autonomy control plane, self-healing, diagnosis.

Autonomy policies/levels/transitions, zero-side-effect simulation, the
central guard (allow/approval/block + idempotency), operation history;
system health aggregation, failure detection, playbooks, governed recovery
with idempotency + cooldowns + escalation; and the diagnosis engine
(correlation, ranked hypotheses with evidence + deterministic confidence).
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
    AutonomyPolicy, AutonomyTransition, AutonomousOperation,
    SystemHealthSnapshot, RecoveryPlaybook, RecoveryAttempt, DiagnosisReport,
)

from app.services import autonomy as au  # noqa: E402
from app.services import selfheal as sh  # noqa: E402
from app.services import diagnosis as dg  # noqa: E402

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


PH21_CORE_TABLES = [
    AutonomousOperation, AutonomyTransition, AutonomyPolicy,
    SystemHealthSnapshot, RecoveryAttempt, RecoveryPlaybook, DiagnosisReport,
]

# Stops/tables from earlier suites must never leak into this suite's
# AUTO-mode assertions (suites share one in-memory database).
from app.models.phase21 import EmergencyStop as _EmergencyStop  # noqa: E402
from app.models.phase21 import PlatformEvent as _PlatformEvent  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in PH21_CORE_TABLES:
        db_session.query(model).delete()
    db_session.query(_EmergencyStop).delete()
    db_session.query(_PlatformEvent).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def register_user(client, tag="u"):
    _counter[0] += 1
    email = f"{tag}{_counter[0]}@p21autonomy.example"
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


def fresh_workspace(db_session):
    _counter[0] += 1
    user = User(name=f"owner{_counter[0]}",
                email=f"owner{_counter[0]}@p21autonomy.example",
                password_hash="x")
    db_session.add(user)
    db_session.flush()
    ws = Workspace(name=f"ws{_counter[0]}", owner_id=user.id)
    db_session.add(ws)
    db_session.commit()
    return ws.id


# ===========================================================================
# Autonomy policy model + levels
# ===========================================================================

class TestAutonomyPolicy:
    def test_create_policy_defaults(self, db_session):
        ws = fresh_workspace(db_session)
        policy = au.create_policy(db_session, ws, "knowledge.reprocess")
        assert policy.autonomy_level == "RECOMMEND"
        assert policy.risk_level == "LOW"
        assert policy.requires_approval is True
        assert policy.requires_audit is True

    def test_invalid_level_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            au.create_policy(db_session, ws, "op", autonomy_level="GOD_MODE")

    def test_invalid_risk_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            au.create_policy(db_session, ws, "op", risk_level="EXTREME")

    def test_policy_upsert_updates_with_transition(self, db_session):
        ws = fresh_workspace(db_session)
        p1 = au.create_policy(db_session, ws, "op", autonomy_level="OBSERVE")
        p2 = au.create_policy(db_session, ws, "op", autonomy_level="RECOMMEND")
        assert p1.id == p2.id
        transitions = db_session.query(AutonomyTransition).filter_by(
            policy_id=p1.id).all()
        assert len(transitions) == 2
        assert transitions[0].previous_level is None
        assert transitions[1].previous_level == "OBSERVE"

    def test_get_policy_fallback_to_any_risk(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "op.risk", risk_level="LOW")
        resolved = au.get_policy(db_session, ws, "op.risk", risk_level="HIGH")
        assert resolved is not None
        assert resolved.operation_type == "op.risk"

    def test_missing_policy_returns_none(self, db_session):
        ws = fresh_workspace(db_session)
        assert au.get_policy(db_session, ws, "never-defined") is None


class TestAutonomyLevels:
    def test_valid_transition_observe_to_recommend(self, db_session):
        ws = fresh_workspace(db_session)
        policy = au.create_policy(db_session, ws, "op",
                                  autonomy_level="OBSERVE")
        au.set_autonomy_level(db_session, policy, "RECOMMEND",
                              actor="alice", reason="steady")
        assert policy.autonomy_level == "RECOMMEND"

    def test_invalid_transition_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        policy = au.create_policy(db_session, ws, "op",
                                  autonomy_level="OBSERVE")
        with pytest.raises(ValueError):
            au.set_autonomy_level(db_session, policy, "AUTO_APPROVAL")

    def test_all_five_levels_exist(self):
        assert au.AUTONOMY_LEVELS == ["OBSERVE", "RECOMMEND", "AUTO_LOW_RISK",
                                      "AUTO_APPROVAL", "MANUAL_ONLY"]

    def test_unknown_level_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        policy = au.create_policy(db_session, ws, "op")
        with pytest.raises(ValueError):
            au.set_autonomy_level(db_session, policy, "SKYNET")

    def test_transition_records_actor(self, db_session):
        ws = fresh_workspace(db_session)
        policy = au.create_policy(db_session, ws, "op")
        au.set_autonomy_level(db_session, policy, "AUTO_LOW_RISK",
                              actor="bob")
        row = (db_session.query(AutonomyTransition)
               .filter_by(policy_id=policy.id)
               .order_by(AutonomyTransition.id.desc()).first())
        assert row.actor == "bob"
        assert row.previous_level == "RECOMMEND"
        assert row.new_level == "AUTO_LOW_RISK"

    def test_same_level_is_noop(self, db_session):
        ws = fresh_workspace(db_session)
        policy = au.create_policy(db_session, ws, "op")
        before = db_session.query(AutonomyTransition).count()
        au.set_autonomy_level(db_session, policy, "RECOMMEND")
        assert db_session.query(AutonomyTransition).count() == before


# ===========================================================================
# Simulation + guard
# ===========================================================================

class TestSimulation:
    def test_simulate_zero_side_effects(self, db_session):
        ws = fresh_workspace(db_session)
        before = db_session.query(AutonomousOperation).count()
        result = au.simulate_operation(db_session, ws, "some.op")
        assert db_session.query(AutonomousOperation).count() == before
        assert result["simulated"] is True

    def test_recommend_level_never_auto(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "op", autonomy_level="RECOMMEND")
        result = au.simulate_operation(db_session, ws, "op")
        assert result["would_auto_execute"] is False
        assert result["decision"] == "REQUIRES_APPROVAL"

    def test_auto_low_risk_allows_low_only(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "op", autonomy_level="AUTO_LOW_RISK")
        low = au.simulate_operation(db_session, ws, "op", risk_level="LOW")
        high = au.simulate_operation(db_session, ws, "op", risk_level="HIGH")
        assert low["would_auto_execute"] is True
        assert high["would_auto_execute"] is False
        assert high["decision"] == "REQUIRES_APPROVAL"

    def test_auto_approval_allows_medium(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "op", autonomy_level="AUTO_APPROVAL")
        medium = au.simulate_operation(db_session, ws, "op",
                                       risk_level="MEDIUM")
        assert medium["would_auto_execute"] is True

    def test_manual_only_blocks(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "op", autonomy_level="MANUAL_ONLY")
        result = au.simulate_operation(db_session, ws, "op")
        assert result["decision"] == "BLOCKED"

    def test_budget_exceeded_blocks(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "op",
                         autonomy_level="AUTO_LOW_RISK", budget_limit_usd=1.0)
        result = au.simulate_operation(db_session, ws, "op",
                                       estimated_cost_usd=5.0)
        assert result["would_auto_execute"] is False
        assert "budget" in result["reason"].lower()


class TestGuard:
    def test_guard_records_decision(self, db_session):
        ws = fresh_workspace(db_session)
        op = au.guard_operation(db_session, ws, "op.recorded",
                                risk_level="LOW", actor="agent",
                                source="AI")
        assert op.decision in ("ALLOWED", "REQUIRES_APPROVAL", "BLOCKED")
        assert op.status == "DECIDED"
        assert op.policy_level == "RECOMMEND"

    def test_guard_idempotency(self, db_session):
        ws = fresh_workspace(db_session)
        key = f"op-key-{uuid.uuid4()}"
        first = au.guard_operation(db_session, ws, "op", idempotency_key=key)
        second = au.guard_operation(db_session, ws, "op", idempotency_key=key)
        assert first.id == second.id

    def test_guard_default_idem_key_repeats(self, db_session):
        ws = fresh_workspace(db_session)
        a = au.guard_operation(db_session, ws, "op.default", actor="agent")
        b = au.guard_operation(db_session, ws, "op.default", actor="agent")
        assert a.id == b.id

    def test_guard_blocks_high_risk_at_recommend(self, db_session):
        ws = fresh_workspace(db_session)
        op = au.guard_operation(db_session, ws, "op", risk_level="HIGH")
        assert op.decision == "REQUIRES_APPROVAL"

    def test_execute_allowed_only(self, db_session):
        ws = fresh_workspace(db_session)
        op = au.guard_operation(db_session, ws, "op.exec", risk_level="LOW")
        op.decision = "ALLOWED"  # force for unit test
        db_session.commit()
        result = au.execute_allowed(db_session, op, result={"done": True},
                                    rollback_info={"undo": "recreate"})
        assert result.status == "SUCCEEDED"
        assert json.loads(result.result) == {"done": True}

    def test_execute_rejected_on_non_allowed(self, db_session):
        ws = fresh_workspace(db_session)
        op = au.guard_operation(db_session, ws, "op.exec2")
        with pytest.raises(ValueError):
            au.execute_allowed(db_session, op, result={})

    def test_rollback_contract(self, db_session):
        ws = fresh_workspace(db_session)
        op = au.guard_operation(db_session, ws, "op.rb", risk_level="LOW")
        op.decision = "ALLOWED"
        db_session.commit()
        au.execute_allowed(db_session, op, result={})
        au.mark_rolled_back(db_session, op, reason="bad outcome")
        assert op.status == "ROLLED_BACK"

    def test_operation_history_listing(self, db_session):
        ws = fresh_workspace(db_session)
        au.guard_operation(db_session, ws, "op.list",
                           idempotency_key=f"k-{uuid.uuid4()}")
        rows = au.list_operations(db_session, ws)
        assert len(rows) >= 1


# ===========================================================================
# Self-healing platform
# ===========================================================================

class TestHealthAggregation:
    def test_all_healthy(self, db_session):
        ws = fresh_workspace(db_session)
        result = sh.aggregate_health(db_session, ws,
                                     {c: "HEALTHY" for c in sh.COMPONENTS})
        assert result["overall_state"] == "HEALTHY"
        assert result["unhealthy_count"] == 0

    def test_any_unhealthy_rolls_up(self, db_session):
        ws = fresh_workspace(db_session)
        states = {c: "HEALTHY" for c in sh.COMPONENTS}
        states["db"] = "UNHEALTHY"
        result = sh.aggregate_health(db_session, ws, states)
        assert result["overall_state"] == "UNHEALTHY"
        assert result["unhealthy_count"] == 1

    def test_degraded_without_unhealthy(self, db_session):
        ws = fresh_workspace(db_session)
        states = {c: "HEALTHY" for c in sh.COMPONENTS}
        states["broker"] = "DEGRADED"
        result = sh.aggregate_health(db_session, ws, states)
        assert result["overall_state"] == "DEGRADED"

    def test_unknown_when_no_data(self, db_session):
        ws = fresh_workspace(db_session)
        result = sh.aggregate_health(db_session, ws, {})
        assert result["overall_state"] == "UNKNOWN"

    def test_snapshot_persisted(self, db_session):
        ws = fresh_workspace(db_session)
        sh.aggregate_health(db_session, ws, {"api": "HEALTHY"})
        snaps = db_session.query(SystemHealthSnapshot).filter_by(
            workspace_id=ws).all()
        assert len(snaps) == 1
        components = json.loads(snaps[0].components)
        assert components["api"] == "HEALTHY"

    def test_invalid_component_ignored(self, db_session):
        ws = fresh_workspace(db_session)
        result = sh.aggregate_health(db_session, ws, {"warp_drive": "HEALTHY"})
        assert "warp_drive" not in result["components"]


class TestFailureDetection:
    def test_worker_stall_detected(self, db_session):
        ws = fresh_workspace(db_session)
        failures = sh.detect_failures(db_session, ws, {
            "worker": {"heartbeat_age_seconds": 300, "throughput": 0.0}})
        kinds = [f["kind"] for f in failures]
        assert "worker_stall" in kinds

    def test_queue_buildup_detected(self, db_session):
        ws = fresh_workspace(db_session)
        failures = sh.detect_failures(db_session, ws, {"queue_depth": 1000})
        assert any(f["kind"] == "queue_buildup" for f in failures)

    def test_provider_failure_detected(self, db_session):
        ws = fresh_workspace(db_session)
        failures = sh.detect_failures(db_session, ws,
                                      {"provider_failures": 2})
        assert any(f["kind"] == "provider_failure" for f in failures)

    def test_healthy_system_no_failures(self, db_session):
        ws = fresh_workspace(db_session)
        failures = sh.detect_failures(db_session, ws, {})
        assert failures == []

    def test_all_nine_failure_kinds_detectable(self, db_session):
        ws = fresh_workspace(db_session)
        failures = sh.detect_failures(db_session, ws, {
            "worker": {"heartbeat_age_seconds": 500},
            "queue_depth": 900, "provider_failures": 1, "db_errors": 1,
            "broker_errors": 1, "connector_errors": 1,
            "ingestion_failures": 1, "vector_errors": 1,
            "cache_errors": 1})
        kinds = {f["kind"] for f in failures}
        assert kinds == set(sh.FAILURE_KINDS)


class TestRecoveryPlaybooks:
    def test_create_playbook(self, db_session):
        ws = fresh_workspace(db_session)
        pb = sh.create_playbook(db_session, ws, "pb1", "worker_stall",
                                ["restart_worker_state"])
        assert pb.risk_level == "LOW"
        assert pb.approved is False

    def test_destructive_action_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            sh.create_playbook(db_session, ws, "pb2", "queue_buildup",
                               ["drop_data"])

    def test_invalid_risk_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            sh.create_playbook(db_session, ws, "pb3", "trigger",
                               ["reconnect_broker"], risk_level="EXTREME")

    def test_approve_playbook(self, db_session):
        ws = fresh_workspace(db_session)
        pb = sh.create_playbook(db_session, ws, "pb4", "broker_failure",
                                ["reconnect_broker"])
        sh.approve_playbook(db_session, pb)
        assert pb.approved is True


class TestRecoveryExecution:
    def test_policy_blocked_without_approval(self, db_session):
        ws = fresh_workspace(db_session)
        pb = sh.create_playbook(db_session, ws, "pbA", "worker_stall",
                                ["restart_worker_state"], approved=False)
        att = sh.attempt_recovery(db_session, ws, "worker_stall", pb)
        assert att.status == "POLICY_BLOCKED"

    def test_auto_recovery_succeeds_with_policy(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "recovery", risk_level="LOW",
                         autonomy_level="AUTO_LOW_RISK",
                         requires_approval=False)
        pb = sh.create_playbook(db_session, ws, "pbB", "broker_failure",
                                ["reconnect_broker"], approved=True)
        att = sh.attempt_recovery(db_session, ws, "broker_failure", pb)
        assert att.status == "SUCCEEDED"
        assert att.auto_applied is True
        results = json.loads(att.action_results)
        assert results == {"reconnect_broker": "ok"}

    def test_recovery_idempotency(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "recovery", risk_level="LOW",
                         autonomy_level="AUTO_LOW_RISK")
        pb = sh.create_playbook(db_session, ws, "pbC", "worker_stall",
                                ["restart_worker_state"], approved=True)
        key = f"rec-{uuid.uuid4()}"
        a = sh.attempt_recovery(db_session, ws, "worker_stall", pb, key)
        b = sh.attempt_recovery(db_session, ws, "worker_stall", pb, key)
        assert a.id == b.id

    def test_recovery_cooldown_blocks(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "recovery", risk_level="LOW",
                         autonomy_level="AUTO_LOW_RISK")
        pb = sh.create_playbook(db_session, ws, "pbD", "cache_failure",
                                ["invalidate_cache"], approved=True,
                                cooldown_seconds=3600)
        first = sh.attempt_recovery(db_session, ws, "cache_failure", pb,
                                    f"cd-{uuid.uuid4()}")
        assert first.status == "SUCCEEDED"
        second = sh.attempt_recovery(db_session, ws, "cache_failure", pb,
                                     f"cd2-{uuid.uuid4()}")
        assert second.status == "COOLDOWN_BLOCKED"

    def test_escalation_after_max_attempts(self, db_session):
        ws = fresh_workspace(db_session)
        au.create_policy(db_session, ws, "recovery", risk_level="LOW",
                         autonomy_level="AUTO_LOW_RISK")
        pb = sh.create_playbook(db_session, ws, "pbE", "vector_failure",
                                ["rebuild_embeddings"], approved=True,
                                cooldown_seconds=0, max_attempts=1)
        first = sh.attempt_recovery(db_session, ws, "vector_failure", pb,
                                    f"esc1-{uuid.uuid4()}")
        assert first.status == "SUCCEEDED"
        second = sh.attempt_recovery(db_session, ws, "vector_failure", pb,
                                     f"esc2-{uuid.uuid4()}")
        assert second.status == "ESCALATED"
        assert second.attempts_so_far == 2

    def test_playbook_trigger_mismatch_rejected(self, db_session):
        ws = fresh_workspace(db_session)
        pb = sh.create_playbook(db_session, ws, "pbF", "worker_stall",
                                ["restart_worker_state"])
        with pytest.raises(ValueError):
            sh.attempt_recovery(db_session, ws, "broker_failure", pb)

    def test_destructive_never_auto(self, db_session):
        ws = fresh_workspace(db_session)
        with pytest.raises(ValueError):
            sh.create_playbook(db_session, ws, "pbG", "db_failure",
                               ["delete_documents"], approved=True)


# ===========================================================================
# Diagnosis engine
# ===========================================================================

class TestDiagnosis:
    def test_correlation_groups_by_component(self):
        failures = [
            {"component": "db", "kind": "database_errors", "at": 1},
            {"component": "db", "kind": "database_errors", "at": 2},
            {"component": "provider", "kind": "provider_failure", "at": 3},
        ]
        clusters = dg.correlate_failures(failures)
        assert len(clusters) == 2
        assert clusters[0]["count"] == 2  # db cluster first (largest)

    def test_hypotheses_ranked_with_confidence(self):
        report = dg.diagnose("api errors", {
            "provider": {"error_rate": 0.9, "sample_count": 10},
            "queue_depth": 900,
        })
        assert report["hypotheses"] == sorted(
            report["hypotheses"], key=lambda h: -h["confidence"])
        assert 0 < report["top_confidence"] <= 95

    def test_confidence_capped_small_samples(self):
        report = dg.diagnose("flaky", {
            "worker": {"heartbeat_age_seconds": 999}})
        top = report["hypotheses"][0]
        assert top["confidence"] <= 40.0  # small sample never overclaims

    def test_no_signal_honest_report(self):
        report = dg.diagnose("mystery", {})
        assert "insufficient" in report["top_cause"]

    def test_recent_change_hypothesis(self):
        report = dg.diagnose("regression", {
            "recent_change": "deploy 2026-09-01"})
        assert any("recent deployment" in h["cause"]
                   for h in report["hypotheses"])

    def test_persist_report(self, db_session):
        ws = fresh_workspace(db_session)
        report = dg.diagnose("errors", {"db_errors": 3})
        row = dg.persist_diagnosis(db_session, ws, "errors", report)
        assert row.top_confidence == report["top_confidence"]
        assert row.correlated_failures == 0

    def test_report_contains_no_hidden_reasoning(self, db_session):
        ws = fresh_workspace(db_session)
        report = dg.diagnose("errors", {"db_errors": 3})
        row = dg.persist_diagnosis(db_session, ws, "errors", report)
        # Evidence + confidence + concise reason only.
        assert "confidence" in row.report
        assert row.report.endswith("cluster(s).")

    def test_list_reports_tenant_scoped(self, db_session):
        ws1, ws2 = fresh_workspace(db_session), fresh_workspace(db_session)
        report = dg.diagnose("s", {"db_errors": 3})
        dg.persist_diagnosis(db_session, ws1, "s", report)
        assert dg.list_reports(db_session, ws2) == []
        assert len(dg.list_reports(db_session, ws1)) == 1


# ===========================================================================
# Autonomy API surface
# ===========================================================================

class TestAutonomyAPI:
    def test_policy_requires_owner(self, client):
        cookies = register_user(client, "own")
        ws_id = create_workspace(client, cookies)
        resp = client.post("/ops21/autonomy/policies", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "operation_type": "op",
                                 "autonomy_level": "OBSERVE"})
        assert resp.status_code in (200, 201), resp.text

    def test_simulate_endpoint(self, client):
        cookies = register_user(client, "sim")
        ws_id = create_workspace(client, cookies)
        resp = client.post("/ops21/autonomy/simulate", cookies=cookies,
                           json={"workspace_id": ws_id,
                                 "operation_type": "test.op"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["simulated"] is True
        assert "decision" in body

    def test_unauthenticated_rejected(self, client):
        resp = client.get("/ops21/autonomy/policies?workspace_id=1")
        assert resp.status_code in (401, 403)
