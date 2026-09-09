"""Phase 18 workflow runtime — join nodes, parallel branch scheduling,
replay of idempotent steps only.

Builds on the Phase 17 durable WorkflowRun state machine (DAG validation,
pause/resume, timeouts, per-node retry) without rewriting it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from ..models.phase17 import WorkflowRun
from ..models.phase15 import WorkflowNodeExecution

logger = logging.getLogger(__name__)

REPLAYABLE_SIDE_EFFECTS = {"notify", "classify", "extract", "validate",
                           "transform", "merge", "report"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _owned_run(db: Session, run_id: int, workspace_id: int) -> WorkflowRun:
    run = (db.query(WorkflowRun)
           .filter(WorkflowRun.id == run_id,
                   WorkflowRun.workspace_id == workspace_id)
           .first())
    if run is None:
        raise KeyError(f"workflow run {run_id} not found in workspace")
    return run


def _definition(run: WorkflowRun) -> dict:
    try:
        return json.loads(run.definition_json)
    except (ValueError, TypeError) as exc:
        raise ValueError("corrupt workflow definition") from exc


def _nodes(definition: dict) -> list[dict]:
    return definition.get("nodes", [])


def parallel_branches(db: Session, *, run_id: int, workspace_id: int) -> dict:
    """Compute ready-to-run nodes grouped by branch (deterministic)."""
    run = _owned_run(db, run_id, workspace_id)
    definition = _definition(run)
    nodes = _nodes(definition)
    state = json.loads(run.node_state_json or "{}")
    completed = set(state.get("completed", []))
    running = set(state.get("running", []))
    by_branch: dict[str, list[dict]] = {}
    for node in nodes:
        node_id = str(node.get("id"))
        if node_id in completed or node_id in running:
            continue
        deps = [str(d) for d in (node.get("depends_on")
                                 or node.get("dependencies") or [])]
        if any(dep not in completed for dep in deps if dep):
            continue
        branch = str(node.get("branch", "main"))
        by_branch.setdefault(branch, []).append(node)
    return {
        "run_id": run_id,
        "ready_branches": {b: [n.get("id") for n in ns]
                           for b, ns in by_branch.items()},
        "completed": sorted(completed),
        "running": sorted(running),
    }


def mark_branch_running(db: Session, *, run_id: int, workspace_id: int,
                        node_ids: list[str]) -> dict:
    """Persist in-flight branch nodes so joins wait for them (idempotent)."""
    run = _owned_run(db, run_id, workspace_id)
    state = json.loads(run.node_state_json or "{}")
    running = set(state.get("running", []))
    running.update(str(n) for n in node_ids)
    state["running"] = sorted(running)
    run.node_state_json = json.dumps(state)
    db.flush()
    return {"run_id": run_id, "running": sorted(running)}


def complete_branch_nodes(db: Session, *, run_id: int, workspace_id: int,
                          node_ids: list[str]) -> dict:
    """Persist branch completion; join nodes unblock when all deps done."""
    run = _owned_run(db, run_id, workspace_id)
    state = json.loads(run.node_state_json or "{}")
    completed = set(state.get("completed", []))
    running = set(state.get("running", []))
    completed.update(str(n) for n in node_ids)
    running.difference_update(str(n) for n in node_ids)
    state["completed"] = sorted(completed)
    state["running"] = sorted(running)
    run.node_state_json = json.dumps(state)
    db.flush()
    # Return whether any join node is now unblocked.
    definition = _definition(run)
    unblocked_joins = []
    for node in _nodes(definition):
        node_id = str(node.get("id"))
        if node_id in completed:
            continue
        if node.get("type") == "join":
            deps = [str(d) for d in (node.get("depends_on")
                                     or node.get("dependencies") or [])]
            if deps and all(d in completed for d in deps):
                unblocked_joins.append(node_id)
    return {"run_id": run_id, "completed": sorted(completed),
            "unblocked_joins": unblocked_joins}


def replay_safe_nodes(db: Session, *, run_id: int, workspace_id: int,
                      operator_user_id: int,
                      execute_callable: Optional[Callable] = None) -> dict:
    """Replay ONLY idempotent/safe completed steps.

    Nodes with irreversible side effects (send, delete, pay, external) are
    never replayed. Returns the replay log.
    """
    from .workflow3 import execute_node
    run = _owned_run(db, run_id, workspace_id)
    definition = _definition(run)
    state = json.loads(run.node_state_json or "{}")
    completed = set(state.get("completed", []))
    replayed = []
    skipped = []
    from ..models.phase15 import WorkflowNodeExecution as WNE
    for node in _nodes(definition):
        node_id = str(node.get("id"))
        if node_id not in completed:
            continue
        side_effect = str(node.get("side_effect", "internal")).lower()
        if side_effect in REPLAYABLE_SIDE_EFFECTS:
            # Append a fresh attempt so the per-node unique constraint is
            # never violated while preserving the original audit trail.
            max_attempt = (db.query(WNE)
                           .filter(WNE.workflow_execution_id
                                   == f"wf-run-{run.id}",
                                   WNE.node_id == node_id)
                           .order_by(WNE.attempt.desc())
                           .first())
            attempt_start = (max_attempt.attempt + 1
                             if max_attempt is not None else 1)
            result = execute_node(db, run=run, node=node,
                                  execute_callable=execute_callable,
                                  attempt_start=attempt_start)
            replayed.append({"node_id": node_id, "status": result["status"]})
        else:
            skipped.append({"node_id": node_id,
                            "reason": f"non-replayable side effect "
                                      f"{side_effect!r}"})
    run.control_state = "ACTIVE"
    db.flush()
    _audit(db, "workflow.replay", workspace_id, operator_user_id,
           f"run {run_id}: replayed={len(replayed)} skipped={len(skipped)}")
    return {"run_id": run_id, "replayed": replayed, "skipped": skipped}


def _audit(db: Session, action: str, workspace_id: int, user_id: int,
           detail: str) -> None:
    try:
        from .audit_service import log_action
        log_action(db, workspace_id=workspace_id, user_id=user_id,
                   action=action, detail=detail[:2000])
    except Exception:  # noqa: BLE001 — audit is best-effort
        logger.warning("workflow audit write failed", exc_info=True)


def run_state(db: Session, run_id: int, workspace_id: int) -> dict:
    """Full durable state view for the ops console."""
    run = _owned_run(db, run_id, workspace_id)
    state = json.loads(run.node_state_json or "{}")
    executions = (db.query(WorkflowNodeExecution)
                  .filter(WorkflowNodeExecution.workflow_execution_id
                          == f"wf-run-{run.id}")
                  .order_by(WorkflowNodeExecution.id)
                  .limit(500).all())
    return {
        "run_id": run.id,
        "status": run.status,
        "control_state": run.control_state,
        "node_state": state,
        "timeout_at": run.timeout_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "error": (run.error or "")[:500],
        "node_executions": [
            {"node_id": e.node_id, "attempt": e.attempt, "status": e.status,
             "error": (e.error_message or "")[:200],
             "started_at": e.started_at, "completed_at": e.completed_at}
            for e in executions],
    }