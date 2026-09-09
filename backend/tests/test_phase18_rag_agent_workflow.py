"""Phase 18 tests — RAG 6.0, durable agent runtime, workflow runtime.

RAG: retrieval planning, evidence ranking/sufficiency, claim matrix,
citation coverage/correctness, refusal behavior, bounded repair, conflict
detection. Agents: plan persistence, per-step authorization, checkpoints,
resume, dead letters, budget enforcement. Workflows: parallel branches,
join unblocking, safe replay.
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
from app.models.ai_execution import AIExecution  # noqa: E402
from app.models.phase15 import AIExecutionCheckpoint  # noqa: E402
from app.models.phase17 import WorkflowRun, AgentPlan  # noqa: E402
from app.services import rag6  # noqa: E402
from app.services import agent_runtime as ar  # noqa: E402
from app.services import workflow_runtime as wr  # noqa: E402

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
    db_session.query(AIExecutionCheckpoint).delete()
    db_session.query(AgentPlan).delete()
    db_session.query(AIExecution).delete()
    db_session.query(WorkflowRun).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p18rw"):
    _counter[0] += 1
    user = User(name=f"P18 RW {_counter[0]}",
                email=f"{tag}{_counter[0]}@p18-rw.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p18 rw ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


def fresh_execution(db, ws, user, status="RUNNING"):
    execution = AIExecution(
        id=str(uuid.uuid4()), workspace_id=ws.id, user_id=user.id,
        execution_type="agent", task_type="test", status=status,
        priority="NORMAL")
    db.add(execution)
    db.commit()
    db.refresh(execution)
    return execution


# ============================================================
# RAG 6.0
# ============================================================

class TestRag6:
    def test_retrieval_plan_default_hybrid(self):
        plan = rag6.retrieval_plan("who manages the budget?")
        assert plan["strategy"] == "hybrid"
        assert "keyword" in plan["sources"]
        assert "vector" in plan["sources"]

    def test_retrieval_plan_entity_graph(self):
        plan = rag6.retrieval_plan("acme", {"entity_id": 5})
        assert plan["strategy"] == "graph"

    def test_retrieval_plan_memory(self):
        plan = rag6.retrieval_plan("q", {"memory": True})
        assert plan["strategy"] == "memory"

    def test_retrieval_plan_explainable(self):
        plan = rag6.retrieval_plan("policy renewal deadline", {})
        assert plan["explainable"] is True
        assert plan["max_evidence"] == rag6.MAX_EVIDENCE

    def test_rank_evidence_deterministic_order(self):
        chunks = [
            {"chunk_id": 1, "text": "a", "score": 0.9},
            {"chunk_id": 2, "text": "b", "score": 0.5},
            {"chunk_id": 3, "text": "c", "score": 0.7},
        ]
        ranked = rag6.rank_evidence(chunks)
        assert [e["chunk_id"] for e in ranked] == [1, 3, 2]

    def test_rank_evidence_tiebreak_by_id(self):
        chunks = [
            {"chunk_id": "b", "text": "x", "score": 0.5},
            {"chunk_id": "a", "text": "y", "score": 0.5},
        ]
        ranked = rag6.rank_evidence(chunks)
        assert [e["chunk_id"] for e in ranked] == ["a", "b"]

    def test_rank_evidence_conflict_penalty(self):
        base = {"chunk_id": 1, "text": "x", "score": 0.9}
        clean = rag6.rank_evidence([dict(base)])[0]
        conflicted = rag6.rank_evidence(
            [dict(base, has_conflict=True)])[0]
        assert conflicted["score"] < clean["score"]

    def test_evidence_sufficiency_ok(self):
        evidence = [{"chunk_id": 1, "text": "revenue grew 20%",
                     "relevance": 0.8},
                    {"chunk_id": 2, "text": "revenue grew 20%",
                     "relevance": 0.7}]
        result = rag6.evidence_sufficiency(evidence, "revenue?")
        assert result["sufficient"] is True
        assert result["refusal"] is False

    def test_evidence_sufficiency_insufficient_refuses(self):
        evidence = [{"chunk_id": 1, "text": "unrelated",
                     "relevance": 0.1}]
        result = rag6.evidence_sufficiency(evidence, "revenue?")
        assert result["sufficient"] is False
        assert result["refusal"] is True
        assert result["suggestion"] is not None

    def test_claim_matrix_supported(self):
        claims = ["Revenue grew by 20%"]
        evidence = [{"chunk_id": 1,
                     "text": "Revenue grew by 20 percent last year"}]
        matrix = rag6.claim_matrix(claims, evidence)
        assert matrix["claims"][0]["support_status"] == "SUPPORTED"
        assert matrix["coverage"] == 1.0

    def test_claim_matrix_unsupported(self):
        claims = ["The moon is made of cheese"]
        evidence = [{"chunk_id": 1, "text": "Revenue grew 20%"}]
        matrix = rag6.claim_matrix(claims, evidence)
        assert matrix["claims"][0]["support_status"] == "UNSUPPORTED"
        assert matrix["coverage"] == 0.0
        assert len(matrix["unsupported"]) == 1

    def test_citation_correctness(self):
        claims = ["Revenue grew", "Aliens exist"]
        evidence = [{"chunk_id": 1, "text": "Revenue grew 20 percent"}]
        result = rag6.citation_correctness(claims, evidence)
        assert result["total_claims"] == 2
        assert result["correct_citations"] == 1
        assert result["coverage"] == 0.5
        assert result["unsupported_claims"] == ["Aliens exist"]

    def test_repair_answer_marks_unsupported(self):
        answer = "Revenue grew 20%. Aliens exist."
        claims = ["Revenue grew 20%", "Aliens exist"]
        evidence = [{"chunk_id": 1, "text": "Revenue grew 20 percent"}]
        result = rag6.repair_answer(answer, claims, evidence)
        assert "UNSUPPORTED" in result["repaired"]
        assert result["rounds"] <= rag6.MAX_REPAIR_ROUNDS
        # The unsupported claim was flagged — it no longer appears bare.
        assert "Aliens exist." not in result["repaired"].replace(
            "[UNSUPPORTED: evidence insufficient for: Aliens exist]", "")

    def test_repair_bounded_rounds(self):
        answer = "A. B. C. D."
        claims = ["A", "B", "C", "D"]
        result = rag6.repair_answer(answer, claims, [], max_rounds=2)
        assert result["rounds"] <= 2

    def test_repair_no_invented_evidence(self):
        answer = "Claim one. Claim two."
        result = rag6.repair_answer(answer, ["Claim one", "Claim two"], [],
                                    max_rounds=1)
        # Only markers are inserted — never fabricated citations.
        assert "evidence" not in result["repaired"].lower() \
            or "UNSUPPORTED" in result["repaired"]

    def test_assemble_answer_refusal(self):
        result = rag6.assemble_answer("question?", [])
        assert result["refused"] is True
        assert result["answer"] is None

    def test_assemble_answer_with_conflict_note(self):
        evidence = [
            {"chunk_id": 1, "text": "budget is 100 USD", "relevance": 0.9},
            {"chunk_id": 2, "text": "budget is 200 USD", "relevance": 0.9},
        ]
        result = rag6.assemble_answer("budget?", evidence)
        assert result["refused"] is False
        assert result["conflicts"]
        assert "USD" in result["conflicts"][0]["field"]

    def test_detect_conflicts_numeric(self):
        evidence = [
            {"chunk_id": 1, "text": "growth was 10 percent"},
            {"chunk_id": 2, "text": "growth was 25 percent"},
        ]
        conflicts = rag6.detect_conflicts(evidence)
        assert len(conflicts) == 1
        assert conflicts[0]["severity"] == "CONFLICT"
        assert conflicts[0]["values"] == [10.0, 25.0]

    def test_detect_conflicts_none(self):
        evidence = [{"chunk_id": 1, "text": "no numbers here"}]
        assert rag6.detect_conflicts(evidence) == []


# ============================================================
# Durable agent runtime
# ============================================================

class TestAgentRuntime:
    def test_persist_plan(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        result = ar.persist_plan(
            db_session, execution_id=execution.id, workspace_id=ws.id,
            plan={"objective": "summarize", "steps": [
                {"step": 1, "tool": "retrieve", "inputs": {"q": "x"}}]})
        db_session.commit()
        assert result["steps"] == 1
        assert result["plan_hash"]
        plan = db_session.query(AgentPlan).filter(
            AgentPlan.execution_id == execution.id).first()
        assert plan is not None

    def test_authorize_step_ok(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = ar.authorize_step(
            db_session, workspace_id=ws.id,
            step={"tool": "retrieve", "scope": "workspace"},
            allowed_tools=["retrieve"], allowed_scopes=["workspace"])
        assert result["authorized"] is True

    def test_authorize_step_denied_tool(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            ar.authorize_step(
                db_session, workspace_id=ws.id,
                step={"tool": "shell"}, allowed_tools=["retrieve"])

    def test_authorize_step_denied_scope(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            ar.authorize_step(
                db_session, workspace_id=ws.id,
                step={"tool": "retrieve", "scope": "organization"},
                allowed_scopes=["workspace"])

    def test_authorize_step_destructive_requires_approval(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            ar.authorize_step(
                db_session, workspace_id=ws.id,
                step={"tool": "delete", "destructive": True})

    def test_authorize_step_sensitive_needs_provider(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            ar.authorize_step(
                db_session, workspace_id=ws.id,
                step={"tool": "retrieve", "sensitivity": "RESTRICTED"})

    def test_checkpoint_before_side_effect(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        cp = ar.checkpoint_before_side_effect(
            db_session, execution_id=execution.id, workspace_id=ws.id,
            step_number=2, state={"tool": "send"})
        db_session.commit()
        assert cp.step_number == 2
        assert "before_side_effect" in cp.state_json

    def test_checkpoint_after_side_effect(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        cp = ar.checkpoint_after_side_effect(
            db_session, execution_id=execution.id, workspace_id=ws.id,
            step_number=2, state={"ok": True},
            output_reference="artifact://42")
        db_session.commit()
        assert cp.tool_output_reference == "artifact://42"

    def test_resume_from_checkpoint_first_step(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        resume = ar.resume_from_checkpoint(
            db_session, execution_id=execution.id, workspace_id=ws.id)
        assert resume["resumable"] is True
        assert resume["next_step"] == 1

    def test_resume_after_checkpoints(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        ar.checkpoint_after_side_effect(
            db_session, execution_id=execution.id, workspace_id=ws.id,
            step_number=3, state={"ok": True})
        db_session.commit()
        resume = ar.resume_from_checkpoint(
            db_session, execution_id=execution.id, workspace_id=ws.id)
        assert resume["next_step"] == 4

    def test_resume_cancelled_not_resumable(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user, status="CANCELLED")
        resume = ar.resume_from_checkpoint(
            db_session, execution_id=execution.id, workspace_id=ws.id)
        assert resume["resumable"] is False

    def test_run_agent_steps_checkpoints(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        calls = []

        def run_step(step):
            calls.append(step["tool"])
            return {"artifact_reference": f"art-{len(calls)}"}

        result = ar.run_agent_steps(
            db_session, execution_id=execution.id, workspace_id=ws.id,
            steps=[
                {"step": 1, "tool": "retrieve", "inputs": {"q": "x"}},
                {"step": 2, "tool": "classify", "inputs": {}},
            ],
            run_step=run_step,
            allowed_tools=["retrieve", "classify"],
            allowed_scopes=["workspace"])
        db_session.commit()
        assert len(result["steps"]) == 2
        assert calls == ["retrieve", "classify"]
        cps = db_session.query(AIExecutionCheckpoint).filter(
            AIExecutionCheckpoint.execution_id == execution.id).count()
        assert cps == 4  # before+after for each step

    def test_run_agent_steps_skips_completed(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        ar.checkpoint_after_side_effect(
            db_session, execution_id=execution.id, workspace_id=ws.id,
            step_number=1, state={"ok": True})
        db_session.commit()
        calls = []

        def run_step(step):
            calls.append(step["tool"])
            return {}
        result = ar.run_agent_steps(
            db_session, execution_id=execution.id, workspace_id=ws.id,
            steps=[
                {"step": 1, "tool": "retrieve", "inputs": {}},
                {"step": 2, "tool": "classify", "inputs": {}},
            ],
            run_step=run_step)
        db_session.commit()
        assert calls == ["classify"]

    def test_run_agent_steps_unauthorized_stops(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)

        def run_step(step):
            return {}
        with pytest.raises(ValueError):
            ar.run_agent_steps(
                db_session, execution_id=execution.id, workspace_id=ws.id,
                steps=[{"step": 1, "tool": "shell"}],
                run_step=run_step, allowed_tools=["retrieve"])

    def test_agent_dead_letter(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws, user)
        result = ar.agent_dead_letter(
            db_session, execution_id=execution.id, workspace_id=ws.id,
            error="unrecoverable tool failure", actor_id=user.id)
        db_session.commit()
        assert result["dead_lettered"] is True
        db_session.refresh(execution)
        assert execution.status == "FAILED"
        assert "unrecoverable" in (execution.failure_reason or "")

    def test_cross_workspace_execution_denied(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        execution = fresh_execution(db_session, ws1, user)
        with pytest.raises(KeyError):
            ar.resume_from_checkpoint(
                db_session, execution_id=execution.id, workspace_id=ws2.id)


# ============================================================
# Workflow runtime
# ============================================================

class TestWorkflowRuntime:
    def fresh_run(self, db, ws, nodes):
        from app.services.workflow3 import start_workflow_run
        run = start_workflow_run(
            db, workspace_id=ws.id, organization_id=None,
            definition={"nodes": nodes}, timeout_seconds=600)
        db.commit()
        db.refresh(run)
        return run

    def test_parallel_branches(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self.fresh_run(db_session, ws, [
            {"id": "a", "type": "task", "branch": "b1"},
            {"id": "b", "type": "task", "branch": "b2"},
            {"id": "c", "type": "join", "branch": "main",
             "depends_on": ["a", "b"]},
        ])
        branches = wr.parallel_branches(db_session, run_id=run.id,
                                        workspace_id=ws.id)
        assert "b1" in branches["ready_branches"]
        assert "b2" in branches["ready_branches"]

    def test_mark_branch_running_persists(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self.fresh_run(db_session, ws, [
            {"id": "a", "type": "task"},
            {"id": "b", "type": "task", "depends_on": ["a"]},
        ])
        result = wr.mark_branch_running(db_session, run_id=run.id,
                                        workspace_id=ws.id,
                                        node_ids=["a"])
        db_session.commit()
        assert "a" in result["running"]

    def test_complete_branch_nodes_unblocks_join(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self.fresh_run(db_session, ws, [
            {"id": "a", "type": "task", "branch": "b1"},
            {"id": "b", "type": "task", "branch": "b2"},
            {"id": "join", "type": "join", "depends_on": ["a", "b"]},
        ])
        wr.mark_branch_running(db_session, run_id=run.id,
                               workspace_id=ws.id,
                               node_ids=["a", "b"])
        db_session.commit()
        result = wr.complete_branch_nodes(
            db_session, run_id=run.id, workspace_id=ws.id,
            node_ids=["a"])
        db_session.commit()
        assert result["unblocked_joins"] == []  # b not done yet
        result = wr.complete_branch_nodes(
            db_session, run_id=run.id, workspace_id=ws.id,
            node_ids=["b"])
        db_session.commit()
        assert result["unblocked_joins"] == ["join"]

    def test_replay_safe_nodes_only(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self.fresh_run(db_session, ws, [
            {"id": "classify", "type": "task", "side_effect": "classify"},
            {"id": "send", "type": "task", "side_effect": "send"},
        ])
        from app.services.workflow3 import advance_run
        advance_run(db_session, run.id, ws.id)
        advance_run(db_session, run.id, ws.id)
        db_session.commit()
        result = wr.replay_safe_nodes(db_session, run_id=run.id,
                                      workspace_id=ws.id,
                                      operator_user_id=user.id)
        db_session.commit()
        assert len(result["replayed"]) == 1  # classify only
        assert result["skipped"][0]["node_id"] == "send"
        assert "non-replayable" in result["skipped"][0]["reason"]

    def test_run_state_view(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        run = self.fresh_run(db_session, ws, [
            {"id": "a", "type": "task"},
        ])
        state = wr.run_state(db_session, run.id, ws.id)
        assert state["run_id"] == run.id
        assert state["status"] == "RUNNING"

    def test_run_state_cross_workspace_denied(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        run = self.fresh_run(db_session, ws1, [
            {"id": "a", "type": "task"},
        ])
        with pytest.raises(KeyError):
            wr.run_state(db_session, run.id, ws2.id)