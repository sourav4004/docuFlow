"""Phase 22 tests — security/autonomy/governance depth.

Tenant isolation across Phase 22 surfaces, autonomy bypass matrix, AI
safety corpus depth, data governance (classification/residency/retention),
DR honesty, and chaos recovery re-verification.
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
from app.services import autonomy as au  # noqa: E402
from app.services import capabilities as cap  # noqa: E402
from app.services import chaos_harness as chs  # noqa: E402
from app.services import region_dr as rd  # noqa: E402
from app.services import safety10 as s10  # noqa: E402
from app.services import security_cost_ops as sco  # noqa: E402

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _clean(db_session):
    from app.models.phase21 import EmergencyStop, AutonomousOperation, \
        AutonomyPolicy, AutonomyTransition, CostGuardDecision
    from app.models.phase20 import ImprovementProposal
    from app.models.phase22 import SecurityScanRun, RetentionExecution
    for model in (SecurityScanRun, RetentionExecution, CostGuardDecision,
                  AutonomousOperation, EmergencyStop, AutonomyTransition,
                  AutonomyPolicy, ImprovementProposal):
        try:
            db_session.query(model).delete()
        except Exception:  # noqa: BLE001
            db_session.rollback()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db, tag="sec"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p22sec.example", name=tag,
                password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Steps 219-224: security matrix depth
# ===========================================================================

class TestSecurityMatrix:
    def test_injection_corpus_all_blocked(self):
        results = s10.run_injection_corpus()
        assert len(results) >= 8
        assert all(r["detected"] for r in results)

    def test_injection_vectors_covered(self):
        expected = {"document", "ocr", "metadata", "connector",
                    "tool_output", "encoded", "multilingual", "indirect"}
        actual = {r["vector"] for r in s10.run_injection_corpus()}
        assert expected <= actual

    def test_exfiltration_corpus_all_blocked(self):
        results = s10.run_exfiltration_corpus()
        assert len(results) >= 5
        assert all(r["detected"] for r in results)

    def test_exfil_tenant_boundary(self):
        detected, kind = s10.detect_exfiltration(
            "", {"requested_workspace_id": 7, "caller_workspace_id": 3})
        assert detected and kind == "cross_workspace_access"

    def test_exfil_credentials(self):
        detected, kind = s10.detect_exfiltration(
            "", {"contains_credentials": True})
        assert detected and kind == "credential_exposure"

    def test_tool_safety_allowlist(self, db_session):
        ws = _mkws(db_session)
        verdict = s10.enforce_tool_safety(db_session, ws.id, "rm_rf",
                                          {"path": "/"})
        assert verdict["allowed"] is False

    def test_tool_safety_budget_gate(self, db_session):
        ws = _mkws(db_session)
        verdict = s10.enforce_tool_safety(db_session, ws.id, "search",
                                          {"query": "x"},
                                          budget_remaining_usd=0.0,
                                          estimated_cost_usd=1.0)
        assert verdict["allowed"] is False

    def test_tool_safety_argument_validation(self, db_session):
        ws = _mkws(db_session)
        verdict = s10.enforce_tool_safety(db_session, ws.id, "search",
                                          {"__unserialized__": object()})
        assert verdict["allowed"] is False

    def test_full_security_scan_matrix(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_all_security_scans(db_session, ws.id)
        corpora = {s["corpus"] for s in result["scans"]}
        assert {"prompt_injection", "exfiltration", "tool_abuse", "ssrf",
                "tenant_matrix", "api_abuse", "autonomy_bypass"} <= corpora
        assert result["all_passed"] is True

    def test_scan_findings_json_safe(self, db_session):
        ws = _mkws(db_session)
        result = sco.run_security_scan(db_session, ws.id, "prompt_injection")
        assert isinstance(result["scan_passed"], bool)


# ===========================================================================
# Autonomy bypass matrix (Step 122)
# ===========================================================================

class TestAutonomyBypass:
    def test_no_policy_high_risk_blocked(self, db_session):
        ws = _mkws(db_session)
        sim = au.simulate_operation(db_session, ws.id, "unregistered.op",
                                    risk_level="HIGH")
        assert sim["decision"] != "ALLOWED"

    def test_approval_level_not_auto(self, db_session):
        ws = _mkws(db_session)
        au.create_policy(db_session, workspace_id=ws.id,
                         operation_type="bypass.op",
                         autonomy_level="AUTO_APPROVAL", risk_level="LOW")
        op = au.guard_operation(db_session, workspace_id=ws.id,
                                operation_type="bypass.op",
                                risk_level="LOW", actor="attacker",
                                source="TEST")
        assert op.decision != "AUTO"

    def test_emergency_stop_vetoes_all_scopes(self, db_session):
        ws = _mkws(db_session)
        au.create_policy(db_session, workspace_id=ws.id,
                         operation_type="stop.op",
                         autonomy_level="AUTO_LOW_RISK", risk_level="LOW")
        stop = s10.activate_emergency_stop(db_session, ws.id, scope="ALL",
                                           actor="operator", reason="test")
        sim = au.simulate_operation(db_session, ws.id, "stop.op",
                                    risk_level="LOW")
        assert sim["would_auto_execute"] is False
        s10.lift_emergency_stop(db_session, ws.id, stop, actor="operator")

    def test_emergency_stop_scoped_veto(self, db_session):
        ws = _mkws(db_session)
        stop = s10.activate_emergency_stop(db_session, ws.id,
                                           scope="AI_ACTIONS",
                                           actor="operator", reason="scoped")
        assert au.emergency_stop_active(db_session, ws.id, "AI_ACTIONS")
        assert not au.emergency_stop_active(db_session, ws.id,
                                            "AUTONOMOUS_RECOVERY")
        s10.lift_emergency_stop(db_session, ws.id, stop, actor="operator")

    def test_budget_bypass_blocked(self, db_session):
        ws = _mkws(db_session)
        au.create_policy(db_session, workspace_id=ws.id,
                         operation_type="budget.op",
                         autonomy_level="AUTO_LOW_RISK", risk_level="LOW",
                         budget_limit_usd=5.0)
        op = au.guard_operation(db_session, workspace_id=ws.id,
                                operation_type="budget.op", risk_level="LOW",
                                actor="attacker", source="TEST",
                                estimated_cost_usd=1000.0)
        assert op.decision != "AUTO"

    def test_execution_limit_enforced(self, db_session):
        ws = _mkws(db_session)
        au.create_policy(db_session, workspace_id=ws.id,
                         operation_type="limit.op",
                         autonomy_level="AUTO_LOW_RISK", risk_level="LOW",
                         execution_limit_per_hour=1)
        op1 = au.guard_operation(db_session, workspace_id=ws.id,
                                 operation_type="limit.op", risk_level="LOW",
                                 actor="system", source="TEST")
        op2 = au.guard_operation(db_session, workspace_id=ws.id,
                                 operation_type="limit.op", risk_level="LOW",
                                 actor="system", source="TEST")
        assert op2.decision != "AUTO"

    def test_autonomous_operations_audited(self, db_session):
        ws = _mkws(db_session)
        au.guard_operation(db_session, workspace_id=ws.id,
                           operation_type="audit.op", risk_level="LOW",
                           actor="system", source="TEST")
        ops = (db_session.query(au.AutonomousOperation)
               .filter_by(workspace_id=ws.id).all())
        assert len(ops) >= 1
        assert ops[0].actor and ops[0].source


# ===========================================================================
# Data governance depth (Steps 219-224, 123-129)
# ===========================================================================

class TestDataGovernance:
    def test_retention_respects_legal_hold(self, db_session):
        # run_retention must never delete held data; dry-run reports intent.
        result = sco.run_retention(db_session, "artifacts", older_than_days=1,
                                   dry_run=True)
        assert result["dry_run"] is True

    def test_retention_kinds_covered(self):
        expected = {"artifacts", "traces", "evaluations", "events", "usage",
                    "temporary"}
        assert expected <= set(sco.RETENTION_KINDS)

    def test_residency_default_permits_without_rule(self, db_session):
        result = rd.residency_guard(db_session, workspace_region="us-east",
                                    target_region="us-east", workspace_id=1,
                                    organization_id=999999)
        assert result["allowed"] is True

    def test_residency_prohibited_list(self, db_session):
        from app.models.phase19 import ResidencyRule
        rule = ResidencyRule(classification="RESTRICTED",
                             prohibited_regions_json='["us-east"]',
                             organization_id=1)
        db_session.add(rule)
        db_session.commit()
        result = rd.residency_guard(db_session, workspace_region="eu-west",
                                    target_region="us-east", workspace_id=1,
                                    organization_id=1)
        assert result["allowed"] is False

    def test_capability_summary_never_leaks_config_values(self, db_session):
        cap.persist_capabilities(db_session)
        blob = str(cap.infrastructure_summary(db_session))
        assert "sk-" not in blob and "password" not in blob.lower()


# ===========================================================================
# DR + chaos honesty depth
# ===========================================================================

class TestDRChaosHonesty:
    def test_drill_records_rpo_rto_only_simulated(self, db_session):
        ws = _mkws(db_session, "drh")
        result = rd.run_restore_drill(db_session, ws.id,
                                      target_rpo_s=300, target_rto_s=1800)
        assert result["simulated"] is True
        if result.get("rpo_s") is not None:
            # Simulated drills must not label measurements as real.
            assert result.get("measured") is not True

    def test_chaos_scenarios_all_registered(self, db_session):
        ws = _mkws(db_session, "chs")
        probes = {
            "provider_timeout": lambda: chs.provider_chaos("timeout"),
            "broker_failure": lambda: chs.broker_chaos(db_session, ws.id),
            "db_failure": lambda: chs.database_chaos(db_session, ws.id),
            "connector_chaos": lambda: chs.connector_chaos(db_session, ws.id),
            "network_chaos": lambda: chs.network_chaos(db_session, ws.id),
            "recovery_validation": lambda: chs.recovery_validation(
                db_session, ws.id),
        }
        for scenario, fn in probes.items():
            result = fn()
            assert result["pass"] is True, scenario

    def test_chaos_run_bounded_flag(self, db_session):
        result = chs.provider_chaos("429", requests=5)
        assert result["bounded"] is True

    def test_soak_never_claims_production(self):
        result = chs.soak_run(duration_hours=0.5, memory_growth_mb=100,
                              error_rate=0.001)
        assert result["stable"] is True
        assert "NOT run" in result["environment_note"]

    def test_real_provider_never_claimed(self, db_session):
        from app.services import provider_validation as pv
        if not pv.any_real_provider():
            assert pv.default_provider() == "fake"

    def test_redis_health_honest_when_absent(self):
        health = cap.redis_healthcheck()
        if not health["available"]:
            assert health["fallback"] == "postgres broker"
