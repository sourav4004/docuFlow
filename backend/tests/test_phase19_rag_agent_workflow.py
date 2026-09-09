"""Phase 19 tests — RAG 7.0 + agent platform 4.0 + workflow platform 4.0 +
AI action platform 3.0.

RAG: intent planning, retrieval plans, evidence diversity/authority,
sufficiency gating, claim matrices, citation correctness/coverage,
conflict-aware answers, temporal reasoning, refusal, bounded repair, and
quality scores. Agents: plan validation/simulation, budgets, checkpoints,
resume, cancellation, handoffs, dead letters. Workflows: DAG validation,
simulation, node checkpoints, joins, timeouts, retries, compensation,
replay safety, pause/resume. Actions: risk engine, approval policy,
previews, expiry, audit.
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
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.models.ai_execution import AIExecution, AIApproval  # noqa: E402
from app.models.phase17 import WorkflowRun, HumanHandoff  # noqa: E402
from app.models.phase19 import AgentDeadLetter  # noqa: E402
from app.services import rag7  # noqa: E402
from app.services import agent4  # noqa: E402
from app.services import workflow4 as wf4  # noqa: E402
from app.services import action3  # noqa: E402

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
    db_session.query(AgentDeadLetter).delete()
    db_session.query(HumanHandoff).delete()
    db_session.query(AIApproval).delete()
    db_session.query(AIExecution).delete()
    db_session.query(WorkflowRun).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p19raw"):
    _counter[0] += 1
    user = User(name=f"P19 RAW {_counter[0]}",
                email=f"{tag}{_counter[0]}@p19-raw.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p19 raw ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def execution(db, ws, user, status="QUEUED", exec_type="agent"):
    _counter[0] += 1
    row = AIExecution(
        id=f"exec-{uuid.uuid4().hex[:32]}", workspace_id=ws.id,
        user_id=user.id, execution_type=exec_type, task_type="test",
        status=status)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def workflow_run(db, ws, definition=None):
    _counter[0] += 1
    row = WorkflowRun(workspace_id=ws.id,
                      definition_hash="h" * 64,
                      definition_json=json.dumps(definition or {}),
                      status="RUNNING", control_state="ACTIVE")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ===========================================================================
# RAG 7.0
# ===========================================================================

class TestIntentPlanner:
    def test_temporal_intent(self):
        result = rag7.intent_planner_v2("What was the policy as of 2023?")
        assert result["intent"] == "temporal"

    def test_comparative_intent(self):
        result = rag7.intent_planner_v2("Compare plan A vs plan B")
        assert result["intent"] == "comparative"

    def test_procedural_intent(self):
        result = rag7.intent_planner_v2("How do I onboard a new vendor?")
        assert result["intent"] == "procedural"

    def test_investigative_intent(self):
        result = rag7.intent_planner_v2("What happened during the outage?")
        assert result["intent"] == "investigative"

    def test_analytical_intent(self):
        result = rag7.intent_planner_v2("Analyze the revenue trend")
        assert result["intent"] == "analytical"

    def test_multi_document_intent(self):
        result = rag7.intent_planner_v2(
            "Summarize the common findings across all documents")
        assert result["intent"] == "multi_document"

    def test_entity_intent(self):
        result = rag7.intent_planner_v2("Tell me about the Acme Corp")
        assert result["intent"] == "entity"

    def test_default_factual(self):
        result = rag7.intent_planner_v2("What color is the sky?")
        assert result["intent"] == "factual"

    def test_reasons_are_explainable(self):
        result = rag7.intent_planner_v2("Why did the build fail?")
        assert result["reasons"] and all(
            isinstance(r, str) for r in result["reasons"])


class TestRetrievalPlan:
    def test_sources_for_temporal(self):
        plan = rag7.retrieval_plan_v2("What was the rule as of 2022?")
        assert "versions" in plan["sources"]

    def test_sources_for_entity(self):
        plan = rag7.retrieval_plan_v2("Who runs the Acme Corp?")
        assert "graph" in plan["sources"]

    def test_plan_explainable(self):
        plan = rag7.retrieval_plan_v2("Summarize our security policy")
        assert plan["intent"] == "policy"
        assert plan["note"].startswith("retrieval plan is deterministic")

    def test_plan_bounded_sources(self):
        for query in ("x", "What happened in Q3?", "How do I file a claim?"):
            plan = rag7.retrieval_plan_v2(query)
            assert 1 <= len(plan["sources"]) <= 4


class TestEvidenceDiversity:
    def test_removes_near_duplicates(self):
        evidence = [
            {"id": "a", "content": "the quick brown fox jumps"},
            {"id": "b", "content": "the quick brown fox jumps high"},
            {"id": "c", "content": "completely unrelated topic here"},
        ]
        kept = rag7.evidence_diversity(evidence)
        assert len(kept) == 2
        assert {e["id"] for e in kept} == {"a", "c"}

    def test_distinct_evidence_kept(self):
        evidence = [{"id": "a", "content": "alpha beta gamma"},
                    {"id": "b", "content": "delta epsilon zeta"}]
        assert len(rag7.evidence_diversity(evidence)) == 2

    def test_empty_evidence(self):
        assert rag7.evidence_diversity([]) == []


class TestAuthorityRanking:
    def test_orders_by_score(self):
        evidence = [
            {"id": "low", "content": "x y", "relevance": 0.2,
             "source": "memory:1"},
            {"id": "high", "content": "x y z", "relevance": 0.95,
             "source": "policy:official"},
        ]
        ranked = rag7.evidence_authority_rank(evidence)
        assert ranked[0]["id"] == "high"

    def test_authority_inference(self):
        assert rag7._default_authority("policy:security") >= 0.8
        assert rag7._default_authority("document:3") >= 0.7
        assert rag7._default_authority("memory:1") < 0.6

    def test_missing_fields_degrade_deterministically(self):
        evidence = [{"id": "a", "content": "q", "relevance": 0.9},
                    {"id": "b", "content": "q", "relevance": 0.9}]
        ranked = rag7.evidence_authority_rank(evidence)
        assert ranked[0]["_authority_score"] == \
            ranked[1]["_authority_score"]


class TestSufficiencyRefusal:
    def test_sufficient_evidence(self):
        check = rag7.evidence_sufficiency2(
            [{"id": "a", "content": "the sky is blue today",
              "relevance": 0.9}], "what color is the sky")
        assert check["sufficient"] is True

    def test_insufficient_evidence(self):
        check = rag7.evidence_sufficiency2(
            [{"id": "a", "content": "unrelated content here",
              "relevance": 0.1}], "what color is the sky")
        assert check["sufficient"] is False

    def test_refusal_on_insufficient(self):
        refusal = rag7.refuse_unsupported(
            "who won the 2019 award?",
            [{"id": "a", "content": "irrelevant text"}])
        assert refusal["refuse"] is True

    def test_no_refusal_when_supported(self):
        refusal = rag7.refuse_unsupported(
            "what is our refund policy?",
            [{"id": "a", "content": "our refund policy allows 30 days",
              "relevance": 0.95}])
        assert refusal["refuse"] is False


class TestClaimMatrix:
    def test_supporting_evidence_found(self):
        matrix = rag7.claim_matrix2(
            ["refunds take 30 days"],
            [{"id": "d1", "content": "refunds take 30 days per policy"}])
        assert matrix["supported"] == 1
        assert matrix["claims"][0]["supporting_evidence"] == ["d1"]

    def test_negated_claim_downgrades_confidence(self):
        matrix = rag7.claim_matrix2(
            ["refunds are not offered"],
            [{"id": "d1", "content": "refunds are not offered at all"}])
        row = matrix["claims"][0]
        assert row["supporting_evidence"]
        assert row["confidence"] == "MEDIUM"

    def test_contradicting_evidence_listed(self):
        matrix = rag7.claim_matrix2(
            ["refunds are allowed"],
            [{"id": "d2", "content": "refunds are not allowed",
              "contradicts": ["refunds are allowed"]}])
        row = matrix["claims"][0]
        assert "d2" in row["contradicting_evidence"]

    def test_unsupported_claim_low_confidence(self):
        matrix = rag7.claim_matrix2(
            ["unicorns run payroll"],
            [{"id": "d1", "content": "refunds take 30 days"}])
        assert matrix["claims"][0]["confidence"] == "LOW"


class TestCitations:
    def test_correct_citation(self):
        result = rag7.citation_correctness2(
            "the sky is blue",
            [{"id": "c1", "content": "the sky is blue in daylight"}])
        assert result["correct"] is True
        assert result["citation_ok"] is True

    def test_incorrect_citation(self):
        result = rag7.citation_correctness2(
            "unicorns run payroll",
            [{"id": "c1", "content": "refunds take 30 days"}])
        assert result["correct"] is False

    def test_coverage_partial(self):
        result = rag7.citation_coverage2(
            ["refunds take 30 days", "unicorns run payroll"],
            [{"id": "c1", "content": "refunds take 30 days"}])
        assert result["supported_claims"] == 1
        assert result["coverage"] == 0.5

    def test_coverage_full(self):
        result = rag7.citation_coverage2(
            ["sky is blue"],
            [{"id": "c1", "content": "the sky is blue today"}])
        assert result["coverage"] == 1.0


class TestConflictsTemporal:
    def test_contradiction_surfaced(self):
        conflicts = rag7.detect_conflicts2([
            {"id": "a", "content": "remote work is allowed"},
            {"id": "b", "content": "remote work is not allowed"}])
        assert len(conflicts) == 1
        assert conflicts[0]["type"] == "CONTRADICTION"

    def test_no_false_conflict(self):
        conflicts = rag7.detect_conflicts2([
            {"id": "a", "content": "meetings are on tuesdays"},
            {"id": "b", "content": "lunches are on fridays"}])
        assert conflicts == []

    def test_temporal_as_of_excludes_future(self):
        today = datetime.now(timezone.utc).isoformat()
        evidence = [
            {"id": "a", "valid_from": "2030-01-01T00:00:00+00:00",
             "content": "future rule"},
            {"id": "b", "valid_from": "2020-01-01T00:00:00+00:00",
             "valid_to": "2025-01-01T00:00:00+00:00", "content": "old rule"},
            {"id": "c", "content": "no date — currently valid"},
        ]
        kept = rag7.temporal_filter(evidence, as_of=today)
        ids = {e["id"] for e in kept}
        assert "a" not in ids and "b" not in ids and "c" in ids

    def test_temporal_between(self):
        kept = rag7.temporal_filter(
            [{"id": "a", "valid_from": "2021-06-01T00:00:00+00:00"},
             {"id": "b", "valid_from": "2024-06-01T00:00:00+00:00"}],
            between_from="2021-01-01T00:00:00+00:00",
            between_to="2023-01-01T00:00:00+00:00")
        assert [e["id"] for e in kept] == ["a"]

    def test_invalid_date_ignored_safely(self):
        kept = rag7.temporal_filter(
            [{"id": "a", "valid_from": "not-a-date"}], as_of="2022-01-01")
        assert [e["id"] for e in kept] == ["a"]


class TestRepairQuality:
    def test_repair_removes_unsupported(self):
        result = rag7.repair_answer2(
            "answer text", ["supported claim", "fabricated claim"],
            [{"id": "d1", "content": "supported claim evidence"}])
        assert result["removed_unsupported"] == 1
        assert "fabricated claim" not in result["repaired_claims"]

    def test_repair_bounded_rounds(self):
        result = rag7.repair_answer2(
            "a", ["c1", "c2", "c3"], [], max_rounds=2)
        assert result["rounds"] <= 2
        assert result["loop_terminated"] is True

    def test_quality_score_components(self):
        score = rag7.rag_quality_score(
            query="refund policy", claims=["refunds take 30 days"],
            evidence=[{"id": "d1", "content": "refunds take 30 days",
                       "relevance": 0.95}])
        assert score["evidence_sufficiency"] is True
        assert score["citation_coverage"] == 1.0
        assert "grade" in score and 0.0 <= score["score"] <= 1.0

    def test_quality_score_poor_without_evidence(self):
        score = rag7.rag_quality_score(
            query="refund policy", claims=["refunds take 30 days"],
            evidence=[{"id": "d1", "content": "unrelated"}])
        assert score["evidence_sufficiency"] is False


# ===========================================================================
# Agent platform 4.0
# ===========================================================================

class TestPlanValidation:
    def test_valid_plan(self):
        plan = {"objective": "summarize",
                "budgets": {"tokens": 1000, "cost_usd": 1.0, "steps": 3,
                            "time_s": 300, "tool_calls": 5},
                "steps": [{"id": "s1", "tool": "search_documents",
                           "scope": "workspace"}]}
        result = agent4.validate_plan_v2(None, plan, workspace_id=1)
        assert result["valid"] is True

    def test_missing_objective(self):
        result = agent4.validate_plan_v2(
            None, {"steps": [{"id": "s1", "tool": "search_documents",
                              "scope": "workspace"}]}, workspace_id=1)
        assert result["valid"] is False
        assert any("objective" in e for e in result["errors"])

    def test_unavailable_tool_rejected(self):
        plan = {"objective": "x", "budgets": {"tokens": 1, "cost_usd": 1,
                                              "steps": 1, "time_s": 1,
                                              "tool_calls": 1},
                "steps": [{"id": "s1", "tool": "drop_database",
                           "scope": "workspace"}]}
        result = agent4.validate_plan_v2(None, plan, workspace_id=1)
        assert result["valid"] is False
        assert any("not available" in e for e in result["errors"])

    def test_cycle_detected(self):
        plan = {"objective": "x",
                "budgets": {"tokens": 1, "cost_usd": 1, "steps": 2,
                            "time_s": 1, "tool_calls": 1},
                "steps": [{"id": "a", "tool": "search_documents",
                           "scope": "workspace", "deps": ["b"]},
                          {"id": "b", "tool": "search_documents",
                           "scope": "workspace", "deps": ["a"]}]}
        result = agent4.validate_plan_v2(None, plan, workspace_id=1)
        assert result["valid"] is False
        assert any("cycle" in e for e in result["errors"])

    def test_destructive_step_needs_high_risk(self):
        plan = {"objective": "x",
                "budgets": {"tokens": 1, "cost_usd": 1, "steps": 1,
                            "time_s": 1, "tool_calls": 1},
                "steps": [{"id": "s1", "tool": "search_documents",
                           "destructive": True, "risk": "LOW",
                           "scope": "workspace"}]}
        result = agent4.validate_plan_v2(None, plan, workspace_id=1)
        assert any("destructive" in e for e in result["errors"])

    def test_missing_scope_rejected(self):
        plan = {"objective": "x",
                "budgets": {"tokens": 1, "cost_usd": 1, "steps": 1,
                            "time_s": 1, "tool_calls": 1},
                "steps": [{"id": "s1", "tool": "search_documents"}]}
        result = agent4.validate_plan_v2(None, plan, workspace_id=1)
        assert any("scope" in e for e in result["errors"])


class TestSimulationBudgets:
    def test_simulation_has_no_side_effects(self):
        plan = {"objective": "x",
                "steps": [{"id": "s1", "tool": "delete_document",
                           "input": "DELETE EVERYTHING"}]}
        result = agent4.simulate_plan(None, plan)
        assert result["simulated"] is True
        assert all(s["side_effects"] == "NONE (simulation)"
                   for s in result["steps"])

    def test_budget_validation_caps(self):
        result = agent4.validate_budgets(
            {"tokens": 500000, "cost_usd": 5.0})
        assert result["valid"] is False
        assert any(v["budget"] == "tokens" for v in result["violations"])

    def test_budget_consumption_bounded(self):
        budgets = {"tokens": 100.0}
        result = agent4.consume_budget(budgets, kind="tokens", amount=250.0)
        assert result["remaining"] == 0.0
        assert result["exhausted"] is True

    def test_dangerous_tool_requires_approval(self):
        result = agent4.authorize_tool_call(
            None, workspace_id=1, tool="delete_document", risk="HIGH")
        assert result["authorized"] is False


class TestCheckpointsResume:
    def test_checkpoint_before_external(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        result = agent4.checkpoint_external(
            db_session, execution_id=exec_row.id, workspace_id=ws.id,
            step_number=2, phase="before_external", state={"step": 2})
        assert result["checkpointed"] is True
        assert result["phase"] == "before_external"

    def test_checkpoint_invalid_phase(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        with pytest.raises(ValueError):
            agent4.checkpoint_external(
                db_session, execution_id=exec_row.id, workspace_id=ws.id,
                step_number=1, phase="mid_side_effect", state={})

    def test_resume_returns_next_step(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        agent4.checkpoint_external(
            db_session, execution_id=exec_row.id, workspace_id=ws.id,
            step_number=3, phase="after_external", state={"ok": True})
        result = agent4.resume_agent(db_session, execution_id=exec_row.id,
                                     workspace_id=ws.id)
        assert result["resumable"] is True
        assert result["next_step"] == 4


class TestCancellationHandoff:
    def test_cancel_running_execution(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user, status="RUNNING")
        result = agent4.cancel_agent(db_session, execution_id=exec_row.id,
                                     workspace_id=ws.id, user_id=user.id)
        assert result["status"] == "CANCELLED"

    def test_cancel_is_idempotent(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user, status="CANCELLED")
        result = agent4.cancel_agent(db_session, execution_id=exec_row.id,
                                     workspace_id=ws.id)
        assert result["status"] == "ALREADY_CANCELLED"
        assert result["idempotent"] is True

    def test_terminal_execution_not_cancellable(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user, status="COMPLETED")
        result = agent4.cancel_agent(db_session, execution_id=exec_row.id,
                                     workspace_id=ws.id)
        assert result["status"] == "NOT_CANCELLABLE"

    def test_handoff_modes_validated(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        with pytest.raises(ValueError):
            agent4.request_handoff(db_session, workspace_id=ws.id,
                                   execution_id=exec_row.id,
                                   question="q", mode="WAITING_MAGIC")

    def test_handoff_answered(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        handoff = agent4.request_handoff(
            db_session, workspace_id=ws.id, execution_id=exec_row.id,
            question="approve?", mode="WAITING_APPROVAL")
        result = agent4.answer_handoff(
            db_session, handoff_id=handoff["handoff_id"],
            workspace_id=ws.id, answer="yes", answered_by=user.id)
        assert result["status"] == "ANSWERED"

    def test_handoff_answered_once(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        handoff = agent4.request_handoff(
            db_session, workspace_id=ws.id, execution_id=exec_row.id,
            question="q", mode="WAITING_INPUT")
        agent4.answer_handoff(db_session, handoff_id=handoff["handoff_id"],
                              workspace_id=ws.id, answer="a",
                              answered_by=user.id)
        second = agent4.answer_handoff(
            db_session, handoff_id=handoff["handoff_id"],
            workspace_id=ws.id, answer="b", answered_by=user.id)
        assert second["status"] == "ALREADY_ANSWERED"


class TestDeadLetters:
    def test_pre_side_effect_checkpoint_replayable(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user, status="FAILED")
        result = agent4.dead_letter_agent(
            db_session, execution_id=exec_row.id, workspace_id=ws.id,
            reason="provider outage",
            last_checkpoint={"phase": "before_side_effect"})
        assert result["safe_replay"] is True

    def test_post_side_effect_checkpoint_not_replayable(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user, status="FAILED")
        result = agent4.dead_letter_agent(
            db_session, execution_id=exec_row.id, workspace_id=ws.id,
            reason="crash after side effect",
            last_checkpoint={"phase": "after_side_effect"})
        assert result["safe_replay"] is False

    def test_dead_letter_listed(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user, status="FAILED")
        agent4.dead_letter_agent(db_session, execution_id=exec_row.id,
                                 workspace_id=ws.id, reason="boom")
        letters = agent4.list_dead_letters(db_session, workspace_id=ws.id)
        assert len(letters) == 1
        assert letters[0].status == "OPEN"


# ===========================================================================
# Workflow platform 4.0
# ===========================================================================

class TestWorkflowValidation:
    def test_valid_workflow(self):
        definition = {"nodes": [
            {"id": "a", "kind": "task", "scope": "workspace"},
            {"id": "b", "kind": "task", "scope": "workspace",
             "deps": ["a"]},
        ]}
        result = wf4.validate_workflow(definition)
        assert result["valid"] is True
        assert result["acyclic"] is True

    def test_cycle_rejected(self):
        definition = {"nodes": [
            {"id": "a", "kind": "task", "scope": "workspace",
             "deps": ["b"]},
            {"id": "b", "kind": "task", "scope": "workspace",
             "deps": ["a"]},
        ]}
        result = wf4.validate_workflow(definition)
        assert result["valid"] is False
        assert any("cycle" in e for e in result["errors"])

    def test_timeout_cap_enforced(self):
        definition = {"timeout_s": 999999,
                      "nodes": [{"id": "a", "kind": "task",
                                 "scope": "workspace"}]}
        result = wf4.validate_workflow(definition)
        assert any("timeout" in e for e in result["errors"])

    def test_destructive_without_approval_rejected(self):
        definition = {"nodes": [{"id": "a", "kind": "task",
                                 "destructive": True,
                                 "scope": "workspace"}]}
        result = wf4.validate_workflow(definition)
        assert any("destructive" in e for e in result["errors"])

    def test_unknown_capability_rejected(self):
        definition = {"nodes": [{"id": "a", "kind": "quantum_teleport",
                                 "scope": "workspace"}]}
        result = wf4.validate_workflow(definition)
        assert any("capability" in e for e in result["errors"])

    def test_duplicate_node_rejected(self):
        definition = {"nodes": [
            {"id": "a", "kind": "task", "scope": "workspace"},
            {"id": "a", "kind": "task", "scope": "workspace"},
        ]}
        result = wf4.validate_workflow(definition)
        assert any("duplicate" in e for e in result["errors"])


class TestWorkflowSimulationJoins:
    def test_simulation_dependency_order(self):
        result = wf4.simulate_workflow({"nodes": [
            {"id": "b", "kind": "task", "deps": ["a"]},
            {"id": "a", "kind": "task"},
        ]})
        assert result["valid"] is True
        assert result["order"] == ["a", "b"]

    def test_simulation_detects_cycle(self):
        result = wf4.simulate_workflow({"nodes": [
            {"id": "a", "deps": ["b"]}, {"id": "b", "deps": ["a"]},
        ]})
        assert result["valid"] is False

    def test_join_requires_all_branches(self):
        definition = {"nodes": [
            {"id": "j", "kind": "join", "deps": ["a", "b"]},
            {"id": "a", "kind": "task"}, {"id": "b", "kind": "task"},
        ]}
        result = wf4.join_ready(
            definition=definition,
            branch_statuses={"a": "COMPLETED", "b": "RUNNING"})
        assert result["joins"][0]["ready"] is False
        assert result["joins"][0]["pending_dependencies"] == ["b"]

    def test_join_ready_when_all_complete(self):
        definition = {"nodes": [
            {"id": "j", "kind": "join", "deps": ["a", "b"]},
        ]}
        result = wf4.join_ready(definition=definition,
                                branch_statuses={"a": "COMPLETED",
                                                 "b": "COMPLETED"})
        assert result["joins"][0]["ready"] is True
        assert result["all_ready"] is True


class TestWorkflowCheckpointsTimeouts:
    def test_node_checkpoint_persisted(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = workflow_run(db_session, ws)
        result = wf4.node_checkpoint(db_session, run_id=run.id,
                                     workspace_id=ws.id, node_id="a",
                                     status="COMPLETED", detail="ok")
        assert result["status"] == "COMPLETED"
        db_session.refresh(run)
        assert '"a"' in run.node_state_json

    def test_node_checkpoint_unknown_run(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(KeyError):
            wf4.node_checkpoint(db_session, run_id=99999,
                                workspace_id=ws.id, node_id="a",
                                status="RUNNING")

    def test_timed_out_nodes_detected(self):
        started = (datetime.now(timezone.utc) - timedelta(minutes=30)
                   ).isoformat()
        definition = {"nodes": [
            {"id": "slow", "kind": "task", "timeout_s": 60},
            {"id": "done", "kind": "task", "timeout_s": 60},
        ]}
        node_state = {"slow": {"status": "RUNNING",
                               "started_at": started},
                      "done": {"status": "COMPLETED",
                               "started_at": started}}
        result = wf4.detect_timed_out_nodes(definition=definition,
                                            node_state=node_state)
        assert [t["node_id"] for t in result["timed_out"]] == ["slow"]

    def test_no_timeout_within_budget(self):
        definition = {"nodes": [{"id": "a", "timeout_s": 3600}]}
        result = wf4.detect_timed_out_nodes(
            definition=definition,
            node_state={"a": {"status": "RUNNING",
                              "started_at": datetime.now(
                                  timezone.utc).isoformat()}})
        assert result["timed_out"] == []

    def test_retry_delay_bounded_exponential(self):
        assert wf4.retry_delay(0) == 2.0
        assert wf4.retry_delay(3) == 16.0
        assert wf4.retry_delay(20) == 300.0  # capped

    def test_retry_policy_exhausted(self):
        result = wf4.retry_policy({"retry": {"max_attempts": 2}}, 2)
        assert result["retry"] is False

    def test_timeout_recovery_retries(self):
        result = wf4.timeout_recovery({"retry": {"max_attempts": 3}}, 0)
        assert result["action"] == "RETRY"

    def test_timeout_recovery_fails_when_exhausted(self):
        result = wf4.timeout_recovery({"retry": {"max_attempts": 1}}, 1)
        assert result["action"] == "FAIL"


class TestCompensationReplay:
    def test_destructive_non_compensatable(self):
        result = wf4.classify_compensation({"kind": "task",
                                            "destructive": True})
        assert result["class"] == "non_compensatable"

    def test_read_only_reversible(self):
        result = wf4.classify_compensation({"kind": "decision",
                                            "read_only": True})
        assert result["class"] == "reversible"

    def test_explicit_class_respected(self):
        result = wf4.classify_compensation(
            {"kind": "external", "compensation": {"class": "compensatable"}})
        assert result["class"] == "compensatable"

    def test_replay_safe_idempotent(self):
        result = wf4.replay_safe({"kind": "task", "idempotent": True})
        assert result["safe"] is True

    def test_replay_unsafe_destructive(self):
        result = wf4.replay_safe({"kind": "task", "destructive": True})
        assert result["safe"] is False

    def test_replay_unsafe_side_effect_without_idempotency(self):
        result = wf4.replay_safe({"kind": "external", "side_effect": True})
        assert result["safe"] is False


class TestWorkflowPauseResume:
    def test_pause_and_resume(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = workflow_run(db_session, ws)
        paused = wf4.pause_workflow(db_session, run_id=run.id,
                                    workspace_id=ws.id,
                                    reason="operator review")
        assert paused["status"] == "PAUSED"
        resumed = wf4.resume_workflow(db_session, run_id=run.id,
                                      workspace_id=ws.id)
        assert resumed["status"] == "ACTIVE"

    def test_pause_idempotent(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = workflow_run(db_session, ws)
        wf4.pause_workflow(db_session, run_id=run.id, workspace_id=ws.id)
        second = wf4.pause_workflow(db_session, run_id=run.id,
                                    workspace_id=ws.id)
        assert second["status"] == "ALREADY_PAUSED"

    def test_pause_cross_workspace_blocked(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        run = workflow_run(db_session, ws1)
        with pytest.raises(KeyError):
            wf4.pause_workflow(db_session, run_id=run.id,
                               workspace_id=ws2.id)


# ===========================================================================
# AI action platform 3.0
# ===========================================================================

class TestRiskEngine:
    def test_low_risk_public_read(self):
        risk = action3.risk_score(sensitivity="PUBLIC")
        assert risk["level"] == "LOW"

    def test_medium_risk_external_side_effect(self):
        risk = action3.risk_score(external_side_effect=True,
                                  sensitivity="INTERNAL")
        assert risk["level"] in ("MEDIUM", "HIGH")

    def test_critical_risk_destructive_irreversible(self):
        risk = action3.risk_score(sensitivity="RESTRICTED",
                                  destructive=True, reversible=False,
                                  external_side_effect=True)
        assert risk["level"] == "CRITICAL"

    def test_high_risk_many_resources(self):
        risk = action3.risk_score(affected_resources=500,
                                  destructive=True)
        assert risk["level"] in ("HIGH", "CRITICAL")

    def test_cross_tenant_increases_risk(self):
        base = action3.risk_score(sensitivity="INTERNAL")
        cross = action3.risk_score(sensitivity="INTERNAL",
                                   cross_tenant=True)
        assert cross["score"] > base["score"]


class TestApprovalPolicy:
    def test_low_auto_allowed(self):
        policy = action3.approval_policy(
            action3.risk_score(sensitivity="PUBLIC"))
        assert policy["action"] == "ALLOW"

    def test_medium_requires_approval(self):
        policy = action3.approval_policy(
            action3.risk_score(external_side_effect=True))
        assert policy["action"] == "REQUIRE_APPROVAL"

    def test_critical_always_blocked(self):
        policy = action3.approval_policy(
            action3.risk_score(destructive=True, reversible=False,
                               external_side_effect=True,
                               sensitivity="RESTRICTED"))
        assert policy["action"] == "BLOCK"

    def test_preview_shows_required_approval(self):
        preview = action3.action_preview(
            action="send_email", target="vendor@example.com",
            external_side_effect=True, sensitivity="CONFIDENTIAL")
        assert preview["required_approval"] is True
        assert preview["target"] == "vendor@example.com"

    def test_preview_never_reveals_reasoning(self):
        preview = action3.action_preview(
            action="delete", target="doc", destructive=True,
            external_side_effect=True, sensitivity="RESTRICTED",
            reversible=False)
        assert preview["blocked"] is True
        assert "score" in preview["risk"]


class TestApprovalLifecycle:
    def test_approve_pending(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        requested = action3.request_approval(
            db_session, execution_id=exec_row.id, workspace_id=ws.id,
            user_id=user.id, action="send_email", risk_level="HIGH")
        result = action3.approve_action(
            db_session, requested["approval_id"], approved_by=user.id)
        assert result["status"] == "approved"

    def test_reject_action(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        requested = action3.request_approval(
            db_session, execution_id=exec_row.id, workspace_id=ws.id,
            user_id=user.id, action="send_email", risk_level="MEDIUM")
        result = action3.approve_action(
            db_session, requested["approval_id"], approved_by=user.id,
            rejection_reason="not authorized")
        assert result["status"] == "rejected"

    def test_approval_valid_after_approve(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        requested = action3.request_approval(
            db_session, execution_id=exec_row.id, workspace_id=ws.id,
            user_id=user.id, action="send_email", risk_level="MEDIUM")
        action3.approve_action(db_session, requested["approval_id"],
                               approved_by=user.id)
        check = action3.approval_valid(db_session,
                                       requested["approval_id"])
        assert check["valid"] is True

    def test_expired_approval_cannot_execute(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        requested = action3.request_approval(
            db_session, execution_id=exec_row.id, workspace_id=ws.id,
            user_id=user.id, action="send_email", risk_level="HIGH")
        row = db_session.query(AIApproval).get(requested["approval_id"])
        row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        db_session.commit()
        check = action3.approval_valid(db_session,
                                       requested["approval_id"])
        assert check["valid"] is False
        assert check["reason"] == "approval expired"

    def test_double_decision_idempotent(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_row = execution(db_session, ws, user)
        requested = action3.request_approval(
            db_session, execution_id=exec_row.id, workspace_id=ws.id,
            user_id=user.id, action="send_email", risk_level="MEDIUM")
        action3.approve_action(db_session, requested["approval_id"],
                               approved_by=user.id)
        second = action3.approve_action(db_session,
                                        requested["approval_id"],
                                        approved_by=user.id)
        assert second["status"] == "ALREADY_DECIDED"

    def test_unknown_approval_invalid(self, db_session):
        check = action3.approval_valid(db_session, "nope-nope-nope")
        assert check["valid"] is False

    def test_audit_record_shape(self):
        audit = action3.action_audit(
            requester=1, approver=2, action="send_email", target="x@y.z",
            policy="REQUIRE_APPROVAL", outcome="approved",
            request_id="req-1")
        assert audit["requester"] == 1 and audit["approver"] == 2
        assert "timestamp" in audit