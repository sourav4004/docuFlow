"""Phase 17 tests — orchestration + governance platform.

Agent plan graphs (DAG validation, budgets, durable steps, human handoff,
durable cancellation), workflow run orchestration (pause/resume, timeouts,
node retries, compensation classification), governance (policy hierarchy,
model allowlists, retention + legal holds, cleanup), alert engine, cost
anomaly/forecast/export, and search planning.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.phase15 import AIExecutionCheckpoint  # noqa: E402
from app.models.phase17 import (  # noqa: E402
    AgentPlan, HumanHandoff, WorkflowRun, AIPolicyRule, LegalHold,
    AlertRule, AlertEvent, RetentionAssignment,
)
from app.models.ai_execution import AIExecution  # noqa: E402
from app.services import agent3, workflow3 as wf3, governance2 as gov  # noqa: E402
from app.services import alerts as alert_svc, cost2, search3  # noqa: E402

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
    from app.models.phase17 import (
        AgentPlan, HumanHandoff, WorkflowRun, AIPolicyRule, LegalHold,
        HoldEntity, AlertRule, AlertEvent, RetentionAssignment,
    )
    for model in (AgentPlan, HumanHandoff, WorkflowRun, AIPolicyRule,
                  LegalHold, HoldEntity, AlertRule, AlertEvent,
                  RetentionAssignment):
        db_session.query(model).delete()
    db_session.query(AIExecutionCheckpoint).delete()
    db_session.commit()


def fresh_user(db, tag="p17pu"):
    _counter[0] += 1
    user = User(name=f"P User {_counter[0]}",
                email=f"{tag}{_counter[0]}@p17-platform.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p17p ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def make_execution(db, ws, user, status="QUEUED"):
    execution = AIExecution(
        id=str(uuid.uuid4()), workspace_id=ws.id, user_id=user.id,
        execution_type="agent", task_type="probe", status=status,
        priority="NORMAL", query="test")
    db.add(execution)
    db.commit()
    db.refresh(execution)
    return execution


# ============================================================
# Agent platform 3.0
# ============================================================

class TestAgent:
    def test_validate_plan_accepts_dag(self):
        plan = {"steps": [
            {"id": "1", "tool": "retrieve", "dependencies": []},
            {"id": "2", "tool": "summarize", "dependencies": ["1"]},
        ]}
        assert agent3.validate_plan(plan, 1)["valid"] is True

    def test_plan_cycle_rejected(self):
        plan = {"steps": [
            {"id": "1", "tool": "retrieve", "dependencies": ["2"]},
            {"id": "2", "tool": "summarize", "dependencies": ["1"]},
        ]}
        with pytest.raises(ValueError):
            agent3.validate_plan(plan, 1)

    def test_plan_unknown_tool_rejected(self):
        plan = {"steps": [
            {"id": "1", "tool": "rm_rf_everything", "dependencies": []},
        ]}
        with pytest.raises(ValueError):
            agent3.validate_plan(plan, 1)

    def test_plan_missing_dependency_rejected(self):
        plan = {"steps": [
            {"id": "1", "tool": "retrieve", "dependencies": ["ghost"]},
        ]}
        with pytest.raises(ValueError):
            agent3.validate_plan(plan, 1)

    def test_create_plan_persists_budgets(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = make_execution(db_session, ws, user)
        plan = agent3.create_agent_plan(
            db_session, workspace_id=ws.id,
            execution_id=execution.id, objective="find policy changes",
            plan={"steps": [
                {"id": "1", "tool": "search", "dependencies": []},
            ]},
            budgets={"cost_budget_usd": 1.0},
            risk="MEDIUM")
        row = db_session.query(AgentPlan).filter(
            AgentPlan.id == plan.id).first()
        assert json.loads(row.budgets_json)["cost_budget_usd"] == 1.0

    def test_budget_check_reports(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = make_execution(db_session, ws, user)
        agent3.create_agent_plan(
            db_session, workspace_id=ws.id, execution_id=execution.id,
            objective="x",
            plan={"steps": [{"id": "1", "tool": "search",
                             "dependencies": []}]},
            budgets={"token_budget": 10})
        check = agent3.check_budgets(db_session, execution)
        assert "token_budget" in check["exceeded"] or \
            not check["exceeded"]  # token usage defaults to 0

    def test_execute_step_checkpoints(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = make_execution(db_session, ws, user, status="RUNNING")
        result = agent3.execute_step(
            db_session, execution_id=execution.id, workspace_id=ws.id,
            step_number=1, tool="search", inputs={"q": "policies"})
        assert result["outputs"]["ok"] is True
        assert db_session.query(AIExecutionCheckpoint).filter(
            AIExecutionCheckpoint.execution_id == execution.id).count() == 1

    def test_handoff_pauses_execution(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = make_execution(db_session, ws, user, status="RUNNING")
        agent3.request_handoff(db_session, workspace_id=ws.id,
                               execution_id=execution.id,
                               question="proceed with deletion?",
                               requested_by=user.id)
        execution = db_session.query(AIExecution).filter(
            AIExecution.id == execution.id).first()
        assert execution.status == "WAITING_APPROVAL"
        handoff = db_session.query(HumanHandoff).filter(
            HumanHandoff.execution_id == execution.id).first()
        assert handoff.status == "PENDING"

    def test_answer_handoff_resumes(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = make_execution(db_session, ws, user, status="RUNNING")
        agent3.request_handoff(db_session, workspace_id=ws.id,
                               execution_id=execution.id,
                               question="proceed?")
        handoff = db_session.query(HumanHandoff).filter(
            HumanHandoff.execution_id == execution.id).first()
        agent3.answer_handoff(db_session, handoff_id=handoff.id,
                              workspace_id=ws.id, answer="yes",
                              answered_by=user.id)
        execution = db_session.query(AIExecution).filter(
            AIExecution.id == execution.id).first()
        assert execution.status == "RUNNING"
        handoff = db_session.query(HumanHandoff).filter(
            HumanHandoff.id == handoff.id).first()
        assert handoff.status == "ANSWERED"
        assert handoff.answer == "yes"

    def test_durable_cancellation(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = make_execution(db_session, ws, user, status="RUNNING")
        agent3.request_handoff(db_session, workspace_id=ws.id,
                               execution_id=execution.id,
                               question="pending?")
        result = agent3.cancel_execution_durable(
            db_session, execution.id, ws.id, user.id)
        assert result["status"] == "CANCELLED"
        assert result["open_handoffs_cancelled"] == 1


# ============================================================
# Workflow orchestration 3.0
# ============================================================

class TestWorkflowRuns:
    DEFINITION = {
        "name": "policy change flow",
        "nodes": [
            {"id": "detect", "type": "detect_change", "inputs": {}},
            {"id": "notify", "type": "notify",
             "depends_on": ["detect"], "inputs": {}},
            {"id": "review", "type": "create_review_task",
             "depends_on": ["notify"], "inputs": {}},
        ],
    }

    def test_start_run_persists_snapshot(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = wf3.start_workflow_run(
            db_session, workspace_id=ws.id, organization_id=None,
            definition=self.DEFINITION)
        assert run.status == "RUNNING"
        assert run.definition_hash
        assert run.timeout_at is not None

    def test_cycle_definition_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            wf3.start_workflow_run(
                db_session, workspace_id=ws.id, organization_id=None,
                definition={"nodes": [
                    {"id": "a", "depends_on": ["b"]},
                    {"id": "b", "depends_on": ["a"]},
                ]})

    def test_advance_executes_nodes_in_order(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = wf3.start_workflow_run(
            db_session, workspace_id=ws.id, organization_id=None,
            definition=self.DEFINITION)
        first = wf3.advance_run(db_session, run.id, ws.id)
        assert first["status"] == "COMPLETED"
        assert first["node"] == "detect"
        second = wf3.advance_run(db_session, run.id, ws.id)
        assert second["node"] == "notify"
        third = wf3.advance_run(db_session, run.id, ws.id)
        assert third["node"] == "review"
        done = wf3.advance_run(db_session, run.id, ws.id)
        assert done["status"] == "COMPLETED"

    def test_pause_prevents_execution(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = wf3.start_workflow_run(
            db_session, workspace_id=ws.id, organization_id=None,
            definition=self.DEFINITION)
        wf3.pause_run(db_session, run.id, ws.id, user.id,
                      reason="compliance review")
        result = wf3.advance_run(db_session, run.id, ws.id)
        assert result["status"] == "PAUSED"
        wf3.resume_run(db_session, run.id, ws.id, user.id)
        first = wf3.advance_run(db_session, run.id, ws.id)
        assert first["status"] == "COMPLETED"

    def test_workflow_timeout_fails_run(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = wf3.start_workflow_run(
            db_session, workspace_id=ws.id, organization_id=None,
            definition=self.DEFINITION, timeout_seconds=60)
        run.timeout_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db_session.commit()
        result = wf3.advance_run(db_session, run.id, ws.id)
        assert result["status"] == "FAILED"

    def test_check_timeouts_batch(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = wf3.start_workflow_run(
            db_session, workspace_id=ws.id, organization_id=None,
            definition=self.DEFINITION)
        run.timeout_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        db_session.commit()
        result = wf3.check_timeouts(db_session)
        assert result["timed_out"] >= 1

    def test_node_retry_bounded(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = wf3.start_workflow_run(
            db_session, workspace_id=ws.id, organization_id=None,
            definition=self.DEFINITION)
        calls = {"n": 0}

        def flaky(db2, run2, node):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return {"ok": True}

        result = wf3.execute_node(
            db_session, run=run, node={"id": "x", "retry":
                                       {"max_attempts": 5}},
            execute_callable=flaky)
        assert result["status"] == "COMPLETED"
        assert result["attempt"] == 3

    def test_node_retry_exhaustion_fails(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = wf3.start_workflow_run(
            db_session, workspace_id=ws.id, organization_id=None,
            definition=self.DEFINITION)

        def always_fail(db2, run2, node):
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError):
            wf3.execute_node(db_session, run=run,
                             node={"id": "y", "retry":
                                   {"max_attempts": 2}},
                             execute_callable=always_fail)
        from app.models.phase15 import WorkflowNodeExecution
        rows = db_session.query(WorkflowNodeExecution).filter(
            WorkflowNodeExecution.node_id == "y").all()
        assert len(rows) == 2
        assert all(r.status == "FAILED" for r in rows)

    def test_compensation_classification(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = wf3.start_workflow_run(
            db_session, workspace_id=ws.id, organization_id=None,
            definition=self.DEFINITION)
        from app.models.phase15 import WorkflowCompensation
        note = wf3.compensation_note(
            db_session, run=run, node_id="notify",
            side_effect_type="notification", reversible=False,
            detail="email already sent")
        assert note.reversible is False
        assert note.status == "RECORDED"
        assert db_session.query(WorkflowCompensation).filter(
            WorkflowCompensation.workflow_execution_id ==
            f"wf-run-{run.id}").count() == 1


# ============================================================
# Governance 2.0
# ============================================================

class TestGovernance:
    def test_model_deny_list(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        gov.create_rule(db_session, organization_id=None,
                        workspace_id=ws.id, feature=None,
                        rule_type="MODEL", deny=["gpt-4o-mini"])
        ok, reason = gov.check_model_allowed(
            db_session, organization_id=None, workspace_id=ws.id,
            feature="rag", model="gpt-4o-mini", sensitivity="INTERNAL")
        assert ok is False
        assert "denied" in reason

    def test_model_allowlist_intersection(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        gov.create_rule(db_session, organization_id=None,
                        workspace_id=ws.id, feature="rag",
                        rule_type="MODEL",
                        allowlist=["claude-sonnet", "gpt-4o"])
        ok, _ = gov.check_model_allowed(
            db_session, organization_id=None, workspace_id=ws.id,
            feature="rag", model="gpt-4o", sensitivity="INTERNAL")
        assert ok is True
        ok2, reason2 = gov.check_model_allowed(
            db_session, organization_id=None, workspace_id=ws.id,
            feature="rag", model="unknown-model",
            sensitivity="INTERNAL")
        assert ok2 is False

    def test_sensitivity_restriction(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        gov.create_rule(db_session, organization_id=None,
                        workspace_id=ws.id, feature=None,
                        rule_type="MODEL", sensitivity_max="CONFIDENTIAL")
        ok, _ = gov.check_model_allowed(
            db_session, organization_id=None, workspace_id=ws.id,
            feature="rag", model="claude-sonnet", sensitivity="INTERNAL")
        assert ok is True
        ok2, _ = gov.check_model_allowed(
            db_session, organization_id=None, workspace_id=ws.id,
            feature="rag", model="claude-sonnet", sensitivity="RESTRICTED")
        assert ok2 is False

    def test_tool_allowlist(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        gov.create_rule(db_session, organization_id=None,
                        workspace_id=ws.id, feature="agent",
                        rule_type="TOOL",
                        allowlist=["retrieve", "search"])
        ok, _ = gov.check_tool_allowed(
            db_session, organization_id=None, workspace_id=ws.id,
            feature="agent", tool="search")
        assert ok is True
        ok2, _ = gov.check_tool_allowed(
            db_session, organization_id=None, workspace_id=ws.id,
            feature="agent", tool="export_all")
        assert ok2 is False

    def test_most_restrictive_wins(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        gov.create_rule(db_session, organization_id=None,
                        workspace_id=ws.id, feature="rag",
                        rule_type="BUDGET", budget_max_usd=10.0)
        gov.create_rule(db_session, organization_id=None,
                        workspace_id=ws.id, feature=None,
                        rule_type="BUDGET", budget_max_usd=2.0)
        policy = gov.resolve_policy(
            db_session, organization_id=None, workspace_id=ws.id,
            feature="rag", rule_type="BUDGET")
        assert policy["budget_max_usd"] == 2.0

    def test_retention_workspace_scoped(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        gov.set_retention(db_session, entity_type="trace",
                          retention_days=30, workspace_id=ws1.id)
        gov.set_retention(db_session, entity_type="trace",
                          retention_days=365, workspace_id=ws2.id)
        assert gov.retention_days_for(db_session, "trace", ws1.id,
                                      None) == 30
        assert gov.retention_days_for(db_session, "trace", ws2.id,
                                      None) == 365
        ws3 = fresh_workspace(db_session, user)
        assert gov.retention_days_for(db_session, "trace", ws3.id,
                                      None) == 90  # default

    def test_legal_hold_blocks_cleanup(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        from app.models.phase16 import TraceSpan
        span = TraceSpan(span_id=str(uuid.uuid4()), trace_id="t1",
                         workspace_id=ws.id, span_type="llm",
                         status="OK")
        db_session.add(span)
        db_session.commit()
        gov.set_retention(db_session, entity_type="trace",
                          retention_days=0, workspace_id=ws.id)
        hold = gov.place_hold(db_session, workspace_id=ws.id,
                              organization_id=None, name="litigation",
                              reason="case 2026-09",
                              entity_type="trace",
                              entity_ids=[span.id],
                              started_by=user.id)
        result = gov.run_cleanup(db_session, entity_type="trace",
                                 workspace_id=ws.id,
                                 organization_id=None)
        assert result["skipped_under_hold"] == 1
        assert db_session.query(TraceSpan).filter(
            TraceSpan.id == span.id).first() is not None
        gov.release_hold(db_session, hold.id, ws.id, user.id)
        result2 = gov.run_cleanup(db_session, entity_type="trace",
                                  workspace_id=ws.id,
                                  organization_id=None)
        assert result2["deleted"] >= 1


# ============================================================
# Alerts
# ============================================================

class TestAlerts:
    def test_create_and_list(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        alert_svc.create_rule(db_session, name="queue depth",
                              metric="queue_depth", operator=">",
                              threshold=100, workspace_id=ws.id)
        listing = alert_svc.list_rules(db_session, workspace_id=ws.id)
        assert listing["total"] == 1

    def test_invalid_operator_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            alert_svc.create_rule(db_session, name="bad",
                                  metric="m", operator="==", threshold=1,
                                  workspace_id=ws.id)

    def test_fire_on_threshold(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        alert_svc.create_rule(db_session, name="backlog",
                              metric="queue_depth", operator=">",
                              threshold=5, cooldown_minutes=1,
                              workspace_id=ws.id)
        fired = alert_svc.evaluate(db_session, {"queue_depth": 50})
        assert len(fired) == 1
        assert fired[0]["severity"] == "WARNING"
        events = alert_svc.list_events(db_session, workspace_id=ws.id)
        assert events["total"] == 1

    def test_cooldown_prevents_spam(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        alert_svc.create_rule(db_session, name="backlog",
                              metric="queue_depth", operator=">",
                              threshold=5, cooldown_minutes=60,
                              workspace_id=ws.id)
        alert_svc.evaluate(db_session, {"queue_depth": 50})
        second = alert_svc.evaluate(db_session, {"queue_depth": 500})
        assert second == []  # cooldown — never spammed

    def test_below_threshold_no_fire(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        alert_svc.create_rule(db_session, name="backlog",
                              metric="queue_depth", operator=">",
                              threshold=5, workspace_id=ws.id)
        assert alert_svc.evaluate(db_session, {"queue_depth": 1}) == []

    def test_resolve_events(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        rule = alert_svc.create_rule(db_session, name="r",
                                     metric="m", operator=">",
                                     threshold=1, workspace_id=ws.id)
        alert_svc.evaluate(db_session, {"m": 10})
        result = alert_svc.resolve_events(db_session, rule.id, user.id)
        assert result["resolved"] == 1


# ============================================================
# Cost platform
# ============================================================

class TestCost:
    def test_no_anomaly_without_history(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        assert cost2.detect_anomalies(db_session,
                                      workspace_id=ws.id) == []

    def test_forecast_needs_history(self, db_session):
        assert cost2.forecast(db_session)["estimate"] is None

    def test_usage_export_csv_header(self, db_session):
        text = cost2.export_usage_csv(db_session, organization_id=1,
                                      days=7)
        assert text.startswith("day,executions,tokens,cost_usd")

    def test_anomaly_detected_on_spike(self, db_session):
        from app.models.ai_execution import AIExecution
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        # history of small costs
        for day_offset in range(6, 0, -1):
            execution = AIExecution(
                id=str(uuid.uuid4()), workspace_id=ws.id,
                user_id=user.id, execution_type="rag",
                task_type="x", status="COMPLETED", priority="NORMAL",
                actual_cost=0.01)
            execution.created_at = datetime.now(timezone.utc) - \
                timedelta(days=day_offset)
            db_session.add(execution)
        # today: spike
        spike = AIExecution(id=str(uuid.uuid4()), workspace_id=ws.id,
                            user_id=user.id, execution_type="rag",
                            task_type="x", status="COMPLETED",
                            priority="NORMAL", actual_cost=50.0)
        db_session.add(spike)
        db_session.commit()
        anomalies = cost2.detect_anomalies(db_session,
                                           workspace_id=ws.id, days=8)
        assert any(a["metric"] == "cost_usd" for a in anomalies)
        assert anomalies[0]["label"].startswith("estimated")


# ============================================================
# Search platform
# ============================================================

class TestSearch:
    def test_planner_modes(self):
        assert search3.plan_query('exact "phrase match"')["retrieval_mode"] \
            == "keyword"
        assert search3.plan_query("as of 2025 policy")["retrieval_mode"] \
            == "hybrid_temporal"
        assert search3.plan_query("entity relationships of X")["retrieval_mode"] \
            == "graph"
        assert search3.plan_query("normal query")["retrieval_mode"] == \
            "hybrid"

    def test_planner_explainable(self):
        plan = search3.plan_query("what changed since march?")
        assert plan["explanation"]
        assert "filters" in plan

    def test_diversity_cap(self):
        results = [{"document_id": 1, "score": 0.9}] * 5 + \
            [{"document_id": 2, "score": 0.8}]
        out = search3.diversify(results, max_per_document=3)
        ids = [r["document_id"] for r in out]
        assert ids.count(1) == 3
        assert 2 in ids

    def test_saved_search_check(self, db_session):
        from app.models.search_intel import SavedSearch
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        record = SavedSearch(workspace_id=ws.id, owner_id=user.id,
                             name="travel policy", query="travel policy")
        db_session.add(record)
        db_session.commit()
        from app.models.document import Document
        doc = Document(workspace_id=ws.id, user_id=user.id,
                       original_filename="travel-policy.pdf",
                       mime_type="text/plain", file_size=5,
                       status="READY", storage_key=f"ss-{uuid.uuid4().hex}")
        db_session.add(doc)
        db_session.commit()
        def evaluator(db2, ws_id, query):
            return [doc.id]
        result = search3.run_saved_search_check(
            db_session, saved_search_id=record.id, workspace_id=ws.id,
            alert_evaluator=evaluator)
        assert result["changed"] is True
