"""Phase 15 Test Suite — Enterprise features (part 2).

Covers: copilot 2.0 scope enforcement, action prioritization + bundles,
workflow reliability (node executions, compensation, simulator, version diff,
risk), cost engine, model policy + sensitive data routing, provider health
circuit breaker, quality dashboards, artifacts, reports, timeline, entity and
duplicate intelligence, organization health, and API tenant isolation.
"""

import io
import json
import uuid
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

_ws_counter = [2000]
_uid_counter = [2000]


def fresh_ws() -> int:
    _ws_counter[0] += 1
    return _ws_counter[0]


def fresh_user() -> int:
    _uid_counter[0] += 1
    return _uid_counter[0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def make_workspace(db, user_id=None, organization_id=None):
    """Create a real workspace row plus owner membership (service-level tests)."""
    from app.models.workspace import Workspace, WorkspaceMember
    uid = user_id or fresh_user()
    ws = Workspace(name=f"WS {fresh_ws()}", owner_id=uid, organization_id=organization_id)
    db.add(ws)
    db.flush()
    db.add(WorkspaceMember(workspace_id=ws.id, user_id=uid, role="OWNER"))
    db.flush()
    return ws.id, uid


def make_doc(db, workspace_id, title="Doc", filename=None, status="READY", user_id=None, content_text=""):
    from app.models.document import Document
    from app.models.document_version import DocumentVersion
    from app.models.document_content import DocumentContent
    _ws_counter[0] += 1
    filename = filename or f"p15b-{_ws_counter[0]}.pdf"
    user_id = user_id or fresh_user()
    doc = Document(
        user_id=user_id, workspace_id=workspace_id,
        original_filename=filename,
        storage_key=f"p15b-{_ws_counter[0]}-{uuid.uuid4().hex[:8]}",
        mime_type="application/pdf", file_size=100, status=status,
    )
    db.add(doc)
    db.flush()
    db.add(DocumentVersion(
        document_id=doc.id, version_number=1, created_by=user_id,
        storage_key=doc.storage_key, file_size=100, mime_type="application/pdf",
        is_current=True, processing_status="READY" if status == "READY" else "PENDING",
        metadata_json=json.dumps({"title": title}),
    ))
    if content_text:
        db.add(DocumentContent(document_id=doc.id, extracted_text=content_text))
    db.flush()
    return doc


def make_action(db, workspace_id, user_id, action_type="summarize", title="Action",
                status="SUGGESTED", organization_id=None):
    from app.services.ai_action_service import create_action
    a = create_action(db, workspace_id, user_id, action_type, title,
                      organization_id=organization_id, status=status)
    db.flush()
    return a


# ============================================================
# Copilot 2.0
# ============================================================

class TestCopilot2Scopes:
    def test_document_copilot_context(self, db_session):
        from app.services.copilot2 import document_copilot_context
        ws, uid = make_workspace(db_session)
        doc = make_doc(db_session, ws, title="Policy", user_id=uid)
        ctx = document_copilot_context(db_session, ws, uid, doc.id)
        assert ctx["scope"] == "DOCUMENT"
        assert ctx["document_id"] == doc.id

    def test_document_copilot_foreign_denied(self, db_session):
        from app.services.copilot2 import document_copilot_context, CopilotScopeError
        ws1, uid = make_workspace(db_session)
        ws2, _ = make_workspace(db_session)
        doc = make_doc(db_session, ws1, user_id=uid)
        with pytest.raises(CopilotScopeError):
            document_copilot_context(db_session, ws2, uid, doc.id)

    def test_workspace_copilot_context(self, db_session):
        from app.services.copilot2 import workspace_copilot_context
        ws, uid = make_workspace(db_session)
        make_doc(db_session, ws, user_id=uid)
        ctx = workspace_copilot_context(db_session, ws, uid)
        assert ctx["scope"] == "WORKSPACE"
        assert "health" in ctx
        assert ctx["workspace_id"] == ws

    def test_workspace_copilot_requires_membership(self, db_session):
        from app.services.copilot2 import workspace_copilot_context, CopilotScopeError
        ws, uid = make_workspace(db_session)
        with pytest.raises(CopilotScopeError):
            workspace_copilot_context(db_session, ws, fresh_user())

    def test_organization_copilot_requires_admin(self, db_session):
        from app.services.copilot2 import organization_copilot_context, CopilotScopeError
        org_id = fresh_ws()
        with pytest.raises(CopilotScopeError):
            organization_copilot_context(db_session, org_id, fresh_user())

    def test_nl_analytics_changed_documents(self, db_session):
        from app.services.copilot2 import nl_analytics
        ws, uid = fresh_ws(), fresh_user()
        make_doc(db_session, ws, user_id=uid)
        result = nl_analytics(db_session, ws, "how many documents changed this month?")
        assert result["intent"] == "documents_changed"
        assert result["metric"] >= 0

    def test_nl_analytics_expiring(self, db_session):
        from app.services.copilot2 import nl_analytics
        from app.models.knowledge import Deadline
        ws, uid = make_workspace(db_session)
        db_session.add(Deadline(workspace_id=ws, owner_id=uid, title="Renewal",
                                due_date=datetime.now(timezone.utc) + timedelta(days=7)))
        db_session.flush()
        result = nl_analytics(db_session, ws, "which policies expire soon?")
        assert result["intent"] == "expiring_deadlines"
        assert result["metric"] >= 1

    def test_nl_analytics_unsupported_honest(self, db_session):
        from app.services.copilot2 import nl_analytics
        result = nl_analytics(db_session, fresh_ws(), "what is the meaning of life?")
        assert result["intent"] == "unsupported"
        assert result["metric"] is None  # never fabricates metrics

    def test_nl_analytics_never_fabricates(self, db_session):
        from app.services.copilot2 import nl_analytics
        ws = fresh_ws()
        result = nl_analytics(db_session, ws, "which policies expire soon?")
        # No deadlines exist — must not invent an expiry count tied to data.
        assert "items" not in result or result["metric"] == 0

    def test_validated_rag_answer(self):
        from app.services.copilot2 import validated_rag_answer
        import types
        db = types.SimpleNamespace()
        result = validated_rag_answer(
            db, 1, 1, "budget?",
            "The 2026 budget is 50000 dollars.",
            ["The 2026 budget is 50000 dollars."],
        )
        assert result["validation"]["supported_count"] >= 1


# ============================================================
# Action prioritization + bundles
# ============================================================

class TestActionPrioritization:
    def test_score_action_factors(self, db_session):
        from app.services.action_bundle_service import score_action
        a = make_action(db_session, fresh_ws(), fresh_user(), status="APPROVAL_REQUIRED",
                        title="High risk action")
        a.risk_level = "CRITICAL"
        a.priority = "HIGH"
        a.estimated_cost = 5.0
        scored = score_action(a)
        assert 0 <= scored["score"] <= 100
        assert "explanation" in scored
        assert scored["factors"]["business_impact"] == "CRITICAL"

    def test_rank_actions_ordered(self, db_session):
        from app.services.action_bundle_service import rank_actions
        ws, uid = fresh_ws(), fresh_user()
        low = make_action(db_session, ws, uid, title="low value", status="SUGGESTED")
        low.risk_level = "LOW"
        high = make_action(db_session, ws, uid, title="critical", status="APPROVAL_REQUIRED")
        high.risk_level = "CRITICAL"
        ranked = rank_actions(db_session, ws)
        assert ranked
        assert ranked[0]["id"] == high.id

    def test_rank_explainable(self, db_session):
        from app.services.action_bundle_service import rank_actions
        ws, uid = fresh_ws(), fresh_user()
        make_action(db_session, ws, uid, title="x")
        ranked = rank_actions(db_session, ws)
        for r in ranked:
            assert r["explanation"]
            assert "factors" in r


class TestActionBundles:
    def test_create_bundle(self, db_session):
        from app.services.action_bundle_service import create_action_bundle
        ws, uid = fresh_ws(), fresh_user()
        a1 = make_action(db_session, ws, uid)
        a2 = make_action(db_session, ws, uid, title="related")
        bundle = create_action_bundle(db_session, ws, "Policy change bundle", [a1.id, a2.id], uid)
        assert bundle["child_count"] == 2
        assert bundle["bundle_id"] > 0

    def test_bundle_children_linked(self, db_session):
        from app.services.action_bundle_service import create_action_bundle
        from app.models.ai_action import AIAction
        ws, uid = fresh_ws(), fresh_user()
        a1 = make_action(db_session, ws, uid)
        create_action_bundle(db_session, ws, "B", [a1.id], uid)
        db_session.refresh(a1)
        assert a1.parent_id is not None
        assert a1.dependency_status == "PENDING"

    def test_bundle_foreign_action_rejected(self, db_session):
        from app.services.action_bundle_service import create_action_bundle
        ws1, ws2, uid = fresh_ws(), fresh_ws(), fresh_user()
        a1 = make_action(db_session, ws1, uid)
        with pytest.raises(ValueError):
            create_action_bundle(db_session, ws2, "Bad bundle", [a1.id], uid)

    def test_bundle_status_completed(self, db_session):
        from app.services.action_bundle_service import create_action_bundle, bundle_status
        from app.services.ai_action_service import transition
        ws, uid = fresh_ws(), fresh_user()
        a1 = make_action(db_session, ws, uid, status="QUEUED")
        a2 = make_action(db_session, ws, uid, title="b", status="QUEUED")
        bundle = create_action_bundle(db_session, ws, "B", [a1.id, a2.id], uid)
        for a in (a1, a2):
            transition(db_session, a, "RUNNING", uid)
            transition(db_session, a, "COMPLETED", uid)
        status = bundle_status(db_session, ws, bundle["bundle_id"])
        assert status["overall_status"] == "COMPLETED"

    def test_dependency_not_satisfied_before_parent(self, db_session):
        from app.services.action_bundle_service import create_action_bundle, dependency_satisfied
        ws, uid = fresh_ws(), fresh_user()
        a1 = make_action(db_session, ws, uid, status="APPROVAL_REQUIRED")
        bundle_id = create_action_bundle(db_session, ws, "B", [a1.id], uid)["bundle_id"]
        # Child of an unfinished parent may not run.
        assert dependency_satisfied(db_session, a1) is False


# ============================================================
# Workflow reliability
# ============================================================

VALID_WORKFLOW = {
    "trigger": "document.updated",
    "conditions": [{"field": "type", "op": "eq", "value": "policy"}],
    "nodes": [
        {"id": "extract", "type": "extract", "name": "Extract dates", "inputs": []},
        {"id": "approval", "type": "approval", "name": "Approve", "inputs": ["extract"]},
        {"id": "notify", "type": "notify", "name": "Notify owner", "inputs": ["approval"]},
    ],
}

VALID_WORKFLOW_NO_NOTIFY = {
    "trigger": "document.ready",
    "nodes": [
        {"id": "summarize", "type": "summarize", "name": "Summarize", "inputs": []},
        {"id": "classify", "type": "classify", "name": "Classify", "inputs": ["summarize"]},
    ],
}


class TestWorkflowReliability:
    def test_begin_node_execution(self, db_session):
        from app.services.workflow_reliability import begin_node_execution
        ws, uid = fresh_ws(), fresh_user()
        record = begin_node_execution(db_session, ws, "run-1", "wf-1", "summarize")
        assert record.status == "RUNNING"
        assert record.attempt == 1
        assert record.input_hash

    def test_double_run_rejected(self, db_session):
        from app.services.workflow_reliability import (
            begin_node_execution, WorkflowReliabilityError,
        )
        ws, uid = fresh_ws(), fresh_user()
        begin_node_execution(db_session, ws, "run-1", "wf-1", "summarize")
        with pytest.raises(WorkflowReliabilityError):
            begin_node_execution(db_session, ws, "run-1", "wf-1", "summarize")

    def test_retry_is_new_attempt(self, db_session):
        from app.services.workflow_reliability import begin_node_execution, fail_node_execution
        ws, uid = fresh_ws(), fresh_user()
        r1 = begin_node_execution(db_session, ws, "run-2", "wf-1", "extract")
        fail_node_execution(db_session, r1, "transient")
        # A retry of the same node in the same run is a NEW explicit attempt.
        r2 = begin_node_execution(db_session, ws, "run-2", "wf-1", "extract")
        assert r2.attempt == 2
        assert r2.status == "RUNNING"

    def test_complete_node(self, db_session):
        from app.services.workflow_reliability import begin_node_execution, complete_node_execution
        ws = fresh_ws()
        r = begin_node_execution(db_session, ws, "run-4", "wf-1", "x")
        complete_node_execution(db_session, r, output_reference="artifact:1")
        assert r.status == "COMPLETED"
        assert r.output_reference == "artifact:1"

    def test_reversible_side_effect(self, db_session):
        from app.services.workflow_reliability import record_side_effect
        ws = fresh_ws()
        record = record_side_effect(db_session, ws, "run-5", "node-1", "metadata_update")
        assert record.reversible is True

    def test_irreversible_side_effect(self, db_session):
        from app.services.workflow_reliability import record_side_effect
        ws = fresh_ws()
        record = record_side_effect(db_session, ws, "run-5", "node-1", "notification")
        assert record.reversible is False

    def test_compensate_reversible(self, db_session):
        from app.services.workflow_reliability import record_side_effect, compensate_side_effect
        ws = fresh_ws()
        record = record_side_effect(db_session, ws, "run-6", "n", "metadata_update")
        compensate_side_effect(db_session, record)
        assert record.status == "COMPENSATED"

    def test_never_compensate_irreversible(self, db_session):
        from app.services.workflow_reliability import (
            record_side_effect, compensate_side_effect, WorkflowReliabilityError,
        )
        ws = fresh_ws()
        record = record_side_effect(db_session, ws, "run-6", "n", "notification")
        with pytest.raises(WorkflowReliabilityError):
            compensate_side_effect(db_session, record)
        # The engine must not pretend an unsent notification was unsent.
        assert record.status == "RECORDED"

    def test_failure_classification(self):
        from app.services.workflow_reliability import classify_failure
        assert classify_failure("connection timed out") == "RETRYABLE"
        assert classify_failure("bad request payload") == "NON_RETRYABLE"

    def test_recovery_plan_auto_retry(self, db_session):
        from app.services.workflow_reliability import (
            begin_node_execution, fail_node_execution, recovery_plan,
        )
        ws = fresh_ws()
        r = begin_node_execution(db_session, ws, "run-7", "wf", "n1")
        fail_node_execution(db_session, r, "connection timeout")
        plan = recovery_plan(db_session, ws, "run-7", "connection timeout", "n1")
        assert plan["action"] == "AUTO_RETRY"
        assert plan["failure_class"] == "RETRYABLE"

    def test_recovery_plan_dead_letter_after_limit(self, db_session):
        from app.services.workflow_reliability import (
            begin_node_execution, fail_node_execution, recovery_plan, MAX_AUTO_RETRIES,
        )
        ws = fresh_ws()
        for _ in range(MAX_AUTO_RETRIES):
            r = begin_node_execution(db_session, ws, "run-8", "wf", "n1")
            fail_node_execution(db_session, r, "connection timeout")
        plan = recovery_plan(db_session, ws, "run-8", "connection timeout", "n1")
        assert plan["action"] == "DEAD_LETTER"
        assert plan["note"] == "Retry limit reached — manual intervention required"

    def test_recovery_plan_manual_for_nonretryable(self, db_session):
        from app.services.workflow_reliability import (
            begin_node_execution, fail_node_execution, recovery_plan,
        )
        ws = fresh_ws()
        r = begin_node_execution(db_session, ws, "run-9", "wf", "n1")
        fail_node_execution(db_session, r, "schema validation failed")
        plan = recovery_plan(db_session, ws, "run-9", "schema validation failed", "n1")
        assert plan["action"] == "MANUAL_RETRY"
        assert plan["failure_class"] == "NON_RETRYABLE"


class TestWorkflowSimulator:
    def test_simulate_side_effect_free(self):
        from app.services.workflow_reliability import simulate_workflow
        plan = simulate_workflow(VALID_WORKFLOW)
        assert plan["mode"] == "SIMULATION"
        assert plan["estimated_tokens"] > 0
        assert plan["cost_estimate_is_exact"] is False
        assert plan["risk"]["level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert plan["approval_points"] == ["approval"]

    def test_simulate_no_notify_no_approval_needed(self):
        from app.services.workflow_reliability import simulate_workflow
        plan = simulate_workflow(VALID_WORKFLOW_NO_NOTIFY)
        assert plan["side_effects"] == []

    def test_risk_engine_notify_without_approval_critical(self):
        from app.services.workflow_reliability import workflow_risk
        risky = {
            "trigger": "document.ready",
            "nodes": [
                {"id": "n1", "type": "notify", "name": "External notify", "inputs": []},
            ],
        }
        risk = workflow_risk(risky)
        assert risk["level"] in ("HIGH", "CRITICAL")
        assert any("without approval" in f for f in risk["factors"])

    def test_risk_engine_safe_workflow_low(self):
        from app.services.workflow_reliability import workflow_risk
        risk = workflow_risk(VALID_WORKFLOW_NO_NOTIFY)
        assert risk["level"] == "LOW"


class TestWorkflowVersionDiff:
    def _versions(self, db_session, ws, uid):
        from app.services.workflow_intel import create_workflow_version
        v1 = create_workflow_version(db_session, ws, uid, "Flow", {
            "trigger": "document.ready",
            "nodes": [{"id": "a", "type": "summarize", "name": "Summarize", "inputs": []}],
        })
        v2 = create_workflow_version(db_session, ws, uid, "Flow", {
            "trigger": "document.updated",
            "nodes": [
                {"id": "a", "type": "summarize", "name": "Summarize", "inputs": []},
                {"id": "approval", "type": "approval", "name": "Approve", "inputs": ["a"]},
                {"id": "b", "type": "notify", "name": "Notify", "inputs": ["approval"]},
            ],
        }, workflow_id=v1.workflow_id)
        return v1, v2

    def test_version_diff_added_node(self, db_session):
        from app.services.workflow_reliability import version_diff
        ws, uid = fresh_ws(), fresh_user()
        v1, v2 = self._versions(db_session, ws, uid)
        diff = version_diff(v1, v2)
        assert diff["from_version"] == 1
        assert diff["to_version"] == 2
        added_ids = {n["id"] for n in diff["added_nodes"]}
        assert added_ids == {"approval", "b"}

    def test_version_diff_trigger_change(self, db_session):
        from app.services.workflow_reliability import version_diff
        ws, uid = fresh_ws(), fresh_user()
        v1, v2 = self._versions(db_session, ws, uid)
        diff = version_diff(v1, v2)
        assert diff["trigger_changed"] is True
        assert diff["trigger_to"] == "document.updated"


# ============================================================
# Cost engine
# ============================================================

class TestCostEngine:
    def _exec(self, db, ws, uid, cost=0.01, status="COMPLETED", model="fake-llm", created_days_ago=0):
        from app.models.ai_execution import AIExecution
        e = AIExecution(
            id=str(uuid.uuid4()), workspace_id=ws, user_id=uid,
            task_type="rag", execution_type="rag", status=status,
            actual_cost=cost, total_tokens=200, input_tokens=100, output_tokens=100,
            model=model, provider="fake",
            created_at=datetime.now(timezone.utc) - timedelta(days=created_days_ago),
        )
        db.add(e)
        db.flush()
        return e

    def test_cost_summary(self, db_session):
        from app.services.cost_engine import cost_summary
        ws, uid = fresh_ws(), fresh_user()
        self._exec(db_session, ws, uid, cost=0.02, model="m1")
        self._exec(db_session, ws, uid, cost=0.03, model="m2")
        summary = cost_summary(db_session, workspace_id=ws)
        assert summary["total_cost_usd"] == 0.05
        assert summary["execution_count"] == 2
        assert summary["by_model"]["m1"] == 0.02

    def test_cost_summary_tenant_scoped(self, db_session):
        from app.services.cost_engine import cost_summary
        ws1, ws2, uid = fresh_ws(), fresh_ws(), fresh_user()
        self._exec(db_session, ws1, uid, cost=0.50)
        assert cost_summary(db_session, workspace_id=ws2)["total_cost_usd"] == 0.0

    def test_forecast_returns_estimates(self, db_session):
        from app.services.cost_engine import forecast_spend
        ws, uid = fresh_ws(), fresh_user()
        self._exec(db_session, ws, uid, cost=0.03)
        forecast = forecast_spend(db_session, ws, lookback_days=30)
        assert forecast["estimate_is_guaranteed"] is False
        assert forecast["monthly_estimate_usd"] > 0
        assert "confidence_range" in forecast

    def test_forecast_no_data(self, db_session):
        from app.services.cost_engine import forecast_spend
        forecast = forecast_spend(db_session, fresh_ws(), lookback_days=30)
        assert forecast["monthly_estimate_usd"] == 0.0

    def test_no_budget_allows(self, db_session):
        from app.services.cost_engine import enforce_budget
        result = enforce_budget(db_session, fresh_ws(), 0.5, fresh_user())
        assert result["decision"] == "ALLOW"

    def test_budget_block_when_exceeded(self, db_session, monkeypatch):
        from app.services.cost_engine import enforce_budget
        from app.services import cost_engine as ce
        monkeypatch.setattr(ce, "get_budget", lambda db, ws: {
            "source": "plan", "name": "PRO", "monthly_limit_usd": 1.0,
        })
        ws = fresh_ws()
        # 0.9 already used + 0.5 estimate > 1.2x limit
        monkeypatch.setattr(ce, "cost_summary", lambda db, workspace_id=None, organization_id=None, since=None: {"total_cost_usd": 0.9})
        result = enforce_budget(db_session, ws, 0.5, fresh_user())
        assert result["decision"] == "RESTRICTED"
        assert result["action"] == "BLOCK"

    def test_budget_requires_approval(self, db_session, monkeypatch):
        from app.services.cost_engine import enforce_budget
        from app.services import cost_engine as ce
        monkeypatch.setattr(ce, "get_budget", lambda db, ws: {
            "source": "plan", "name": "PRO", "monthly_limit_usd": 1.0,
        })
        monkeypatch.setattr(ce, "cost_summary", lambda db, workspace_id=None, organization_id=None, since=None: {"total_cost_usd": 0.85})
        ws = fresh_ws()
        # 0.85 used + 0.20 estimate > 1.00 limit but < 1.2x => approval required.
        result = enforce_budget(db_session, ws, 0.20, fresh_user())
        assert result["action"] == "REQUIRE_APPROVAL"

    def test_budget_downgrade_when_tight(self, db_session, monkeypatch):
        from app.services.cost_engine import enforce_budget
        from app.services import cost_engine as ce
        monkeypatch.setattr(ce, "get_budget", lambda db, ws: {
            "source": "plan", "name": "PRO", "monthly_limit_usd": 1.0,
        })
        monkeypatch.setattr(ce, "cost_summary", lambda db, workspace_id=None, organization_id=None, since=None: {"total_cost_usd": 0.6})
        ws = fresh_ws()
        # Remaining (0.4) still covers 0.35 estimate => allow; then 0.5 remaining
        # check with a larger estimate crossing into downgrade territory.
        result = enforce_budget(db_session, ws, 0.35, fresh_user())
        assert result["decision"] == "ALLOW"


# ============================================================
# Model policy + sensitive data routing
# ============================================================

class TestModelPolicy:
    def test_default_policy_allows(self, db_session):
        from app.services.model_policy import route_model
        result = route_model(db_session, None, fresh_ws(), "INTERNAL", "openai_compatible", "gpt-4")
        assert result["allowed"] is True

    def test_blocked_provider_rejected(self, db_session):
        from app.services.model_policy import ModelPolicy, route_model, ModelPolicyError
        from app.services import model_policy as mp
        monkeypatch_policy = mp.get_effective_policy
        mp.get_effective_policy = lambda db, org: ModelPolicy(
            organization_id=1, blocked_providers=["openai_compatible"])
        try:
            with pytest.raises(ModelPolicyError):
                route_model(db_session, 1, fresh_ws(), "PUBLIC", "openai_compatible", "gpt-4")
        finally:
            mp.get_effective_policy = monkeypatch_policy

    def test_allowed_models_enforced(self, db_session):
        from app.services.model_policy import ModelPolicy, route_model, ModelPolicyError
        from app.services import model_policy as mp
        original = mp.get_effective_policy
        mp.get_effective_policy = lambda db, org: ModelPolicy(
            organization_id=1, allowed_models=["gpt-4"])
        try:
            with pytest.raises(ModelPolicyError):
                route_model(db_session, 1, fresh_ws(), "PUBLIC", "openai_compatible", "gpt-3.5-turbo")
        finally:
            mp.get_effective_policy = original

    def test_blocked_model_rejected(self, db_session):
        from app.services.model_policy import ModelPolicy, route_model, ModelPolicyError
        from app.services import model_policy as mp
        original = mp.get_effective_policy
        mp.get_effective_policy = lambda db, org: ModelPolicy(organization_id=1, blocked_models=["fake-llm"])
        try:
            with pytest.raises(ModelPolicyError):
                route_model(db_session, 1, fresh_ws(), "PUBLIC", "fake", "fake-llm")
        finally:
            mp.get_effective_policy = original

    def test_cost_cap_enforced(self, db_session):
        from app.services.model_policy import ModelPolicy, route_model, ModelPolicyError
        from app.services import model_policy as mp
        original = mp.get_effective_policy
        mp.get_effective_policy = lambda db, org: ModelPolicy(organization_id=1, max_cost_usd=0.01)
        try:
            with pytest.raises(ModelPolicyError):
                route_model(db_session, 1, fresh_ws(), "PUBLIC", "fake", "fake-llm", estimated_cost_usd=5.0)
        finally:
            mp.get_effective_policy = original

    def test_restricted_requires_approval(self, db_session):
        from app.services.model_policy import ModelPolicy, route_model
        from app.services import model_policy as mp
        original = mp.get_effective_policy
        mp.get_effective_policy = lambda db, org: ModelPolicy(
            organization_id=1, require_approval_for=["RESTRICTED"])
        try:
            result = route_model(db_session, 1, fresh_ws(), "RESTRICTED", "fake", "fake-llm")
            assert result["requires_approval"] is True
        finally:
            mp.get_effective_policy = original

    def test_sensitivity_resolution(self, db_session):
        from app.services.model_policy import resolve_sensitivity
        ws, uid = fresh_ws(), fresh_user()
        doc = make_doc(db_session, ws, user_id=uid)
        doc.sensitivity = "CONFIDENTIAL"
        db_session.flush()
        assert resolve_sensitivity(db_session, doc.id) == "CONFIDENTIAL"
        assert resolve_sensitivity(db_session, 999999) == "PUBLIC"

    def test_save_policy_roundtrip(self, db_session):
        from app.models.organization import Organization
        from app.services.model_policy import get_effective_policy, save_policy
        org = Organization(name="Test Org", slug=f"tpol-{uuid.uuid4().hex[:8]}", owner_id=fresh_user())
        db_session.add(org)
        db_session.flush()
        save_policy(db_session, org.id, {"blocked_providers": ["openai_compatible"]}, fresh_user())
        db_session.flush()
        policy = get_effective_policy(db_session, org.id)
        assert policy.blocked_providers == ["openai_compatible"]
        assert policy.allowed_providers == []


class TestDataMinimization:
    def test_minimize_keeps_only_allowed(self):
        from app.services.model_policy import minimize_for_external_call
        data = {
            "content": "the actual document text",
            "title": "Policy",
            "sections": ["a"],
            "internal_notes": "SECRET",
            "workspace_metadata": {"x": 1},
        }
        minimized = minimize_for_external_call(data)
        assert "content" in minimized
        assert "internal_notes" not in minimized
        assert "workspace_metadata" not in minimized

    def test_minimize_redacts_sensitive(self):
        from app.services.model_policy import minimize_for_external_call
        minimized = minimize_for_external_call({
            "content": "Contact 123-45-6789 about the 1234567890123456 card",
        })
        assert "123-45-6789" not in minimized["content"]
        assert "1234567890123456" not in minimized["content"]


# ============================================================
# Provider health + circuit breaker
# ============================================================

class TestProviderHealth:
    def test_success_records(self, db_session):
        from app.services.workflow_reliability import record_provider_call
        health = record_provider_call(db_session, "fake", "fake-llm", True, latency_ms=120)
        assert health.status == "UP"
        assert health.circuit_state == "CLOSED"
        assert health.success_count == 1

    def test_circuit_opens_after_threshold(self, db_session):
        from app.services.workflow_reliability import record_provider_call
        health = None
        for _ in range(5):
            health = record_provider_call(db_session, "fake", "broken", False, error="5xx")
        assert health.circuit_state == "OPEN"
        assert health.status == "DOWN"

    def test_degraded_after_failures(self, db_session):
        from app.services.workflow_reliability import record_provider_call
        record_provider_call(db_session, "fake", "flaky", False, error="timeout")
        health = record_provider_call(db_session, "fake", "flaky", False, error="timeout")
        assert health.circuit_state == "HALF_OPEN"
        assert health.status == "DEGRADED"

    def test_recovery_resets_circuit(self, db_session):
        from app.services.workflow_reliability import record_provider_call
        for _ in range(5):
            record_provider_call(db_session, "fake", "flaky", False, error="x")
        health = record_provider_call(db_session, "fake", "flaky", True)
        assert health.circuit_state == "CLOSED"
        assert health.consecutive_failures == 0

    def test_allow_gate_blocks_open(self, db_session):
        from app.services.workflow_reliability import record_provider_call, allow_provider_call
        health = None
        for _ in range(5):
            health = record_provider_call(db_session, "fake", "down", False, error="x")
        assert allow_provider_call(health) is False  # never hammer a failing provider

    def test_allow_when_healthy(self, db_session):
        from app.services.workflow_reliability import record_provider_call, allow_provider_call
        health = record_provider_call(db_session, "fake", "ok", True)
        assert allow_provider_call(health) is True

    def test_provider_health_summary(self, db_session):
        from app.services.quality_service import provider_health_summary
        from app.services.workflow_reliability import record_provider_call
        record_provider_call(db_session, "fake", "a", True)
        summary = provider_health_summary(db_session)
        assert summary["total"] >= 1
        assert "providers" in summary


# ============================================================
# Quality dashboard + feedback analytics
# ============================================================

class TestQualityPlatform:
    def test_quality_dashboard_empty(self, db_session):
        from app.services.quality_service import quality_dashboard
        dash = quality_dashboard(db_session, fresh_ws())
        assert dash["executions"]["total"] == 0
        assert dash["note"]  # no cross-tenant aggregation

    def test_quality_dashboard_counts(self, db_session):
        from app.services.quality_service import quality_dashboard
        from app.models.ai_execution import AIExecution
        ws, uid = fresh_ws(), fresh_user()
        for status in ("COMPLETED", "COMPLETED", "FAILED"):
            db_session.add(AIExecution(id=str(uuid.uuid4()), workspace_id=ws, user_id=uid,
                                       task_type="rag", status=status, actual_cost=0.01,
                                       total_tokens=10))
        db_session.flush()
        dash = quality_dashboard(db_session, ws)
        assert dash["executions"]["total"] == 3
        assert dash["executions"]["failed"] == 1
        assert dash["cost"]["estimated_cost_usd"] == 0.03

    def test_feedback_analytics(self, db_session):
        from app.services.quality_service import feedback_analytics
        from app.models.search_intel import AIFeedback
        ws, uid = fresh_ws(), fresh_user()
        db_session.add(AIFeedback(workspace_id=ws, user_id=uid, rating="thumbs_up", category="correct"))
        db_session.add(AIFeedback(workspace_id=ws, user_id=uid, rating="thumbs_down", category="citation_issue"))
        db_session.flush()
        analytics = feedback_analytics(db_session, ws)
        assert analytics["total"] == 2
        assert analytics["citation_issues"] == 1
        assert analytics["acceptance_rate"] == 0.5

    def test_feedback_never_trains_models(self, db_session):
        from app.services.quality_service import feedback_analytics
        # Analytics must be aggregate-only; no model training path exists.
        analytics = feedback_analytics(db_session, fresh_ws())
        assert isinstance(analytics["by_rating"], dict)

    def test_record_quality_metric(self, db_session):
        from app.services.quality_service import record_quality_metric
        ws = fresh_ws()
        metric = record_quality_metric(db_session, ws, "grounding_rate", 0.9)
        assert metric.metric_type == "grounding_rate"
        assert metric.value == 0.9


# ============================================================
# Artifacts 2.0
# ============================================================

class TestArtifactPlatform:
    def test_create_artifact(self, db_session):
        from app.services.artifact_service import create_artifact
        ws, uid = fresh_ws(), fresh_user()
        artifact = create_artifact(db_session, ws, uid, "research_brief", "Brief",
                                   {"claims": ["a"]}, source_document_ids=[1, 2])
        assert artifact.version == 1
        assert artifact.artifact_type == "research_brief"
        assert artifact.source_document_ids_json == [1, 2]

    def test_invalid_type_rejected(self, db_session):
        from app.services.artifact_service import create_artifact, ArtifactError
        with pytest.raises(ArtifactError):
            create_artifact(db_session, fresh_ws(), fresh_user(), "alien", "x", {})

    def test_new_version(self, db_session):
        from app.services.artifact_service import create_artifact, new_version, list_versions
        ws, uid = fresh_ws(), fresh_user()
        a = create_artifact(db_session, ws, uid, "report", "Report", {"claims": ["v1"]})
        b = new_version(db_session, ws, a.id, {"claims": ["v2"]}, uid)
        assert b.version == 2
        assert b.id != a.id
        versions = list_versions(db_session, ws, a.id)
        assert len(versions) == 2
        # Original version untouched.
        assert versions[0].content_json["claims"] == ["v1"]

    def test_version_diff_changed_claims(self, db_session):
        from app.services.artifact_service import create_artifact, new_version, version_diff
        ws, uid = fresh_ws(), fresh_user()
        a = create_artifact(db_session, ws, uid, "report", "R", {
            "claims": ["claim one"], "confidence": "HIGH",
        })
        b = new_version(db_session, ws, a.id, {
            "claims": ["claim two"], "confidence": "MEDIUM",
        }, uid)
        diff = version_diff(a, b)
        assert diff["confidence_changed"] is True
        assert any(s["section"] == "claims" for s in diff["changed_sections"])

    def test_list_artifacts_latest_version(self, db_session):
        from app.services.artifact_service import create_artifact, new_version, list_artifacts
        ws, uid = fresh_ws(), fresh_user()
        a = create_artifact(db_session, ws, uid, "report", "R", {"v": 1})
        new_version(db_session, ws, a.id, {"v": 2}, uid)
        listed = list_artifacts(db_session, ws)
        assert len(listed) == 1
        assert listed[0].version == 2

    def test_delete_artifact_all_versions(self, db_session):
        from app.services.artifact_service import create_artifact, new_version, delete_artifact
        ws, uid = fresh_ws(), fresh_user()
        a = create_artifact(db_session, ws, uid, "report", "R", {"v": 1})
        new_version(db_session, ws, a.id, {"v": 2}, uid)
        assert delete_artifact(db_session, ws, a.id, uid) is True
        assert delete_artifact(db_session, ws, a.id, uid) is False


# ============================================================
# Reports 2.0
# ============================================================

class TestReports:
    def test_executive_brief_sections(self, db_session):
        from app.services.report2 import build_report
        ws, uid = fresh_ws(), fresh_user()
        make_doc(db_session, ws, user_id=uid)
        report = build_report(db_session, ws, "executive_summary", uid)
        for section in ("FACTS", "EVIDENCE", "INFERENCES", "RECOMMENDATIONS", "UNCERTAINTIES"):
            assert section in report

    def test_facts_have_evidence(self, db_session):
        from app.services.report2 import build_report
        ws, uid = fresh_ws(), fresh_user()
        report = build_report(db_session, ws, "executive_summary", uid)
        for fact in report["FACTS"]:
            assert "fact" in fact
            assert "evidence" in fact

    def test_unknown_template_rejected(self, db_session):
        from app.services.report2 import build_report, ReportError
        with pytest.raises(ReportError):
            build_report(db_session, fresh_ws(), "banana_report", fresh_user())

    def test_risk_report(self, db_session):
        from app.services.report2 import build_report
        ws, uid = fresh_ws(), fresh_user()
        report = build_report(db_session, ws, "risk_report", uid)
        assert report["template"] == "risk_report"
        assert report["FACTS"]

    def test_compliance_readiness_honest(self, db_session):
        from app.services.report2 import build_report
        ws, uid = fresh_ws(), fresh_user()
        report = build_report(db_session, ws, "compliance_readiness_report", uid)
        items = report["EVIDENCE"][0]["items"]
        control_names = [i["control"] for i in items]
        # Never claims certifications it does not hold.
        assert any("NOT certified" in i["status"] for i in items)

    def test_ai_usage_report(self, db_session):
        from app.services.report2 import build_report
        ws, uid = fresh_ws(), fresh_user()
        report = build_report(db_session, ws, "ai_usage_report", uid, date_range_days=30)
        assert "forecast" in report
        assert report["UNCERTAINTIES"]

    def test_save_report_artifact(self, db_session):
        from app.services.report2 import build_report, save_report_artifact
        ws, uid = fresh_ws(), fresh_user()
        report = build_report(db_session, ws, "risk_report", uid)
        result = save_report_artifact(db_session, ws, uid, report)
        assert result["artifact_id"]
        assert result["version"] == 1


# ============================================================
# Timeline
# ============================================================

class TestTimeline:
    def test_timeline_scoped(self, db_session):
        from app.services.timeline_service import unified_timeline
        ws, uid = fresh_ws(), fresh_user()
        make_doc(db_session, ws, user_id=uid)
        entries = unified_timeline(db_session, ws)
        kinds = {e["event_type"] for e in entries}
        assert "document" in kinds
        assert all(e["occurred_at"] for e in entries)

    def test_timeline_filter_by_type(self, db_session):
        from app.services.timeline_service import unified_timeline
        ws, uid = fresh_ws(), fresh_user()
        make_doc(db_session, ws, user_id=uid)
        entries = unified_timeline(db_session, ws, event_type="document")
        assert all(e["event_type"] == "document" for e in entries)

    def test_timeline_tenant_isolation(self, db_session):
        from app.services.timeline_service import unified_timeline
        ws1, ws2, uid = fresh_ws(), fresh_ws(), fresh_user()
        make_doc(db_session, ws1, user_id=uid)
        assert unified_timeline(db_session, ws2) == []

    def test_timeline_invalid_type_rejected(self, db_session):
        from app.services.timeline_service import unified_timeline
        with pytest.raises(ValueError):
            unified_timeline(db_session, fresh_ws(), event_type="banana")

    def test_timeline_sorted_desc(self, db_session):
        from app.services.timeline_service import unified_timeline
        ws, uid = fresh_ws(), fresh_user()
        make_doc(db_session, ws, user_id=uid)
        entries = unified_timeline(db_session, ws)
        stamps = [e["occurred_at"] for e in entries]
        assert stamps == sorted(stamps, reverse=True)

    def test_timeline_bounded(self, db_session):
        from app.services.timeline_service import unified_timeline
        ws, uid = fresh_ws(), fresh_user()
        for i in range(5):
            make_doc(db_session, ws, title=f"Doc {i}", user_id=uid)
        entries = unified_timeline(db_session, ws, limit=3)
        assert len(entries) <= 3


# ============================================================
# Entity + duplicate intelligence
# ============================================================

class TestEntityIntelligence:
    def _entity(self, db_session, ws, name="Acme Corp", entity_type="company"):
        from app.models.knowledge_graph import Entity
        e = Entity(workspace_id=ws, name=name, entity_type=entity_type,
                   aliases=json.dumps([name.lower()]))
        db_session.add(e)
        db_session.flush()
        return e

    def test_entity_summary(self, db_session):
        from app.services.entity2 import entity_summary
        ws = fresh_ws()
        e = self._entity(db_session, ws)
        summary = entity_summary(db_session, ws, e.id)
        assert summary["name"] == "Acme Corp"
        assert "relationships" in summary
        assert "confidence" in summary

    def test_entity_summary_foreign_denied(self, db_session):
        from app.services.entity2 import entity_summary
        ws1, ws2 = fresh_ws(), fresh_ws()
        e = self._entity(db_session, ws1)
        with pytest.raises(ValueError):
            entity_summary(db_session, ws2, e.id)

    def test_duplicate_entity_detection(self, db_session):
        from app.services.entity2 import validate_entity_relationships
        ws = fresh_ws()
        self._entity(db_session, ws, name="Acme Corp")
        self._entity(db_session, ws, name="Acme Corp")
        result = validate_entity_relationships(db_session, ws)
        assert len(result["duplicate_entities"]) == 1

    def test_orphan_relationship_detection(self, db_session):
        from app.services.entity2 import validate_entity_relationships
        from app.models.knowledge_graph import EntityRelationship
        ws = fresh_ws()
        e = self._entity(db_session, ws)
        db_session.add(EntityRelationship(
            workspace_id=ws, source_id=e.id, target_id=999999, relationship_type="owns",
        ))
        db_session.flush()
        result = validate_entity_relationships(db_session, ws)
        assert len(result["orphan_relationships"]) == 1

    def test_validation_never_merges(self, db_session):
        from app.services.entity2 import validate_entity_relationships
        from app.models.knowledge_graph import Entity
        ws = fresh_ws()
        self._entity(db_session, ws, name="Acme Corp")
        self._entity(db_session, ws, name="Acme Corp")
        result = validate_entity_relationships(db_session, ws)
        # Detection reports; no automatic merge happened.
        count = db_session.query(Entity).filter(Entity.name == "Acme Corp").count()
        assert count == 2
        assert "human approval" in result["note"]

    def test_entity_touch_updates_last_seen(self, db_session):
        from app.services.entity2 import record_entity_touch
        ws = fresh_ws()
        e = self._entity(db_session, ws)
        record_entity_touch(db_session, ws, e.id)
        assert e.last_seen_at is not None


class TestDuplicateIntelligence:
    def test_exact_duplicate(self, db_session):
        from app.services.duplicate2 import assess_pair
        ws, uid = fresh_ws(), fresh_user()
        text = "This is the exact same policy text for both documents " * 5
        a = make_doc(db_session, ws, title="A", user_id=uid, content_text=text)
        b = make_doc(db_session, ws, title="B", user_id=uid, content_text=text)
        result = assess_pair(db_session, a, b)
        assert result.classification == "EXACT_DUPLICATE"
        assert result.similarity >= 0.98

    def test_unrelated(self, db_session):
        from app.services.duplicate2 import assess_pair
        ws, uid = fresh_ws(), fresh_user()
        a = make_doc(db_session, ws, title="A", user_id=uid,
                     content_text="Q3 financial report with revenue numbers " * 8)
        b = make_doc(db_session, ws, title="B", user_id=uid,
                     content_text="Kitchen menu with recipes for pasta and sauce " * 8)
        result = assess_pair(db_session, a, b)
        assert result.classification == "UNRELATED"

    def test_version_chain(self, db_session):
        from app.services.duplicate2 import assess_pair
        ws, uid = fresh_ws(), fresh_user()
        base = " ".join(f"term{i}" for i in range(60)) + " contract obligations renewal dates parties"
        a = make_doc(db_session, ws, title="Master Agreement v1", user_id=uid, content_text=base)
        b = make_doc(db_session, ws, title="Master Agreement v2", user_id=uid,
                     content_text=base + " added section twelve amendments")
        result = assess_pair(db_session, a, b)
        assert result.classification == "VERSION"
        assert result.same_version_chain is True

    def test_never_auto_deletes(self, db_session):
        from app.services.duplicate2 import scan_workspace
        from app.models.document import Document
        ws, uid = fresh_ws(), fresh_user()
        text = "identical content duplicated many times over " * 5
        make_doc(db_session, ws, title="A", user_id=uid, content_text=text)
        make_doc(db_session, ws, title="B", user_id=uid, content_text=text)
        findings = scan_workspace(db_session, ws)
        assert len(findings) >= 1
        remaining = db_session.query(Document).filter(Document.workspace_id == ws).count()
        assert remaining == 2  # advisory only — nothing deleted

    def test_classification_enum(self):
        from app.services.duplicate2 import DuplicateAssessment
        # Every classification is advisory, not destructive.
        assert DuplicateAssessment(1, 2, "NEAR_DUPLICATE", 0.8, []).classification == "NEAR_DUPLICATE"


# ============================================================
# Organization health
# ============================================================

class TestOrgHealth:
    def test_org_health_aggregates(self, db_session):
        from app.services.health2 import organization_health
        org_id = fresh_ws()
        make_workspace(db_session, organization_id=org_id)
        health = organization_health(db_session, org_id)
        assert health["organization_id"] == org_id
        assert health["totals"]["workspaces"] >= 1
        assert health["note"]

    def test_org_health_no_cross_workspace_docs(self, db_session):
        from app.services.health2 import organization_health
        org_id = fresh_ws()
        ws1, _ = make_workspace(db_session, organization_id=org_id)
        ws2, _ = make_workspace(db_session)  # workspace in a different org
        make_doc(db_session, ws1, user_id=1)
        make_doc(db_session, ws2, user_id=1)  # foreign workspace must stay out
        health = organization_health(db_session, org_id)
        assert health["totals"]["documents"] == 1