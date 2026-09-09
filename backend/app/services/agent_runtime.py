"""Phase 18 durable agent runtime — plan persistence, per-step
authorization, checkpoints before/after side effects, and dead-letter
handling.

Execution survives worker restart by resuming from the last persisted
checkpoint. Every step is authorized independently before it runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from ..models.ai_execution import AIExecution
from ..models.phase15 import AIExecutionCheckpoint

logger = logging.getLogger(__name__)

CHECKPOINT_BEFORE_SIDE_EFFECT = True


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _owned_execution(db: Session, execution_id: str,
                     workspace_id: int) -> AIExecution:
    row = (db.query(AIExecution)
           .filter(AIExecution.id == execution_id,
                   AIExecution.workspace_id == workspace_id)
           .first())
    if row is None:
        raise KeyError(f"execution {execution_id} not found in workspace")
    return row


# ---------------------------------------------------------------------------
# Plan persistence + step authorization
# ---------------------------------------------------------------------------

def persist_plan(db: Session, *, execution_id: str, workspace_id: int,
                 plan: dict, risk: str = "LOW",
                 allowed_tools: Optional[set] = None) -> dict:
    """Persist the full plan DAG as a checkpoint artifact (idempotent)."""
    from ..services.agent3 import create_agent_plan
    _owned_execution(db, execution_id, workspace_id)
    steps = plan.get("steps") or []
    create_agent_plan(
        db, workspace_id=workspace_id,
        execution_id=execution_id,
        objective=plan.get("objective", ""),
        plan=plan,
        risk=risk,
        allowed_tools=allowed_tools)
    db.flush()
    return {"execution_id": execution_id, "steps": len(steps),
            "plan_hash": _plan_hash(plan)}


def _plan_hash(plan: dict) -> str:
    import hashlib
    return hashlib.sha256(
        json.dumps(plan, default=str, sort_keys=True).encode()).hexdigest()[:32]


def authorize_step(db: Session, *, workspace_id: int,
                   step: dict, allowed_tools: Optional[list[str]] = None,
                   allowed_scopes: Optional[list[str]] = None) -> dict:
    """Per-step authorization: tool + scope + sensitivity validation.

    Raises ValueError on any failure — an AI plan can never bypass this.
    """
    tool = str(step.get("tool", ""))
    if not tool:
        raise ValueError("step declares no tool")
    if allowed_tools and tool not in allowed_tools:
        raise ValueError(f"tool {tool!r} not permitted for this execution")
    scope = str(step.get("scope", "workspace"))
    if allowed_scopes and scope not in allowed_scopes:
        raise ValueError(f"scope {scope!r} not permitted")
    sensitivity = str(step.get("sensitivity", "INTERNAL"))
    if sensitivity in ("RESTRICTED", "CONFIDENTIAL") and not step.get(
            "approved_provider"):
        raise ValueError(
            f"sensitivity {sensitivity} requires an approved provider")
    if step.get("destructive"):
        raise ValueError("destructive steps require explicit human approval")
    return {"authorized": True, "tool": tool, "scope": scope}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_before_side_effect(db: Session, *, execution_id: str,
                                  workspace_id: int, step_number: int,
                                  state: dict) -> AIExecutionCheckpoint:
    """Persist a checkpoint BEFORE a side effect (durable-state-first)."""
    execution = _owned_execution(db, execution_id, workspace_id)
    row = AIExecutionCheckpoint(
        execution_id=execution_id,
        step_number=step_number,
        state_json=json.dumps({"phase": "before_side_effect",
                               **state}, default=str)[:4000],
        checksum=None,
    )
    db.add(row)
    db.flush()
    return row


def checkpoint_after_side_effect(db: Session, *, execution_id: str,
                                 workspace_id: int, step_number: int,
                                 state: dict,
                                 output_reference: Optional[str] = None,
                                 tool_reference: Optional[str] = None,
                                 ) -> AIExecutionCheckpoint:
    """Persist a checkpoint AFTER a side effect with its outcome."""
    _owned_execution(db, execution_id, workspace_id)
    row = AIExecutionCheckpoint(
        execution_id=execution_id,
        step_number=step_number,
        state_json=json.dumps({"phase": "after_side_effect",
                               **state}, default=str)[:4000],
        tool_output_reference=(output_reference or tool_reference),
        checksum=None,
    )
    db.add(row)
    db.flush()
    return row


def resume_from_checkpoint(db: Session, *, execution_id: str,
                           workspace_id: int) -> dict:
    """Next step number after the last durable checkpoint (survives restart)."""
    execution = _owned_execution(db, execution_id, workspace_id)
    last = (db.query(AIExecutionCheckpoint)
            .filter(AIExecutionCheckpoint.execution_id == execution_id)
            .order_by(AIExecutionCheckpoint.step_number.desc())
            .first())
    next_step = (last.step_number + 1) if last is not None else 1
    if execution.status == "CANCELLED":
        return {"resumable": False, "reason": "execution cancelled",
                "next_step": None}
    if execution.status in ("COMPLETED", "FAILED", "TIMED_OUT"):
        return {"resumable": False, "reason": execution.status,
                "next_step": None}
    return {"resumable": True, "next_step": next_step,
            "last_checkpoint": last.step_number if last else None}


# ---------------------------------------------------------------------------
# Durable execution driver
# ---------------------------------------------------------------------------

def run_agent_steps(db: Session, *, execution_id: str, workspace_id: int,
                    steps: list[dict],
                    run_step: Callable[[dict], dict],
                    allowed_tools: Optional[list[str]] = None,
                    allowed_scopes: Optional[list[str]] = None,
                    stop_step: Optional[int] = None) -> dict:
    """Drive steps with per-step authorization + before/after checkpoints.

    Raises on the first unauthorized/budget/failure so the caller (worker)
    can fail the job; the execution remains resumable from the last
    checkpoint.
    """
    from .agent3 import check_budgets
    from . import ai_execution_service as aes

    execution = _owned_execution(db, execution_id, workspace_id)
    resume = resume_from_checkpoint(db, execution_id=execution_id,
                                    workspace_id=workspace_id)
    if not resume["resumable"]:
        return {"execution_id": execution_id,
                "status": "NOT_RESUMABLE", "reason": resume["reason"]}
    start = resume["next_step"]
    results = []
    for step in steps:
        step_number = int(step.get("step", 1))
        if step_number < start:
            continue
        if stop_step is not None and step_number > stop_step:
            break
        authz = authorize_step(db, workspace_id=workspace_id, step=step,
                               allowed_tools=allowed_tools,
                               allowed_scopes=allowed_scopes)
        budget = check_budgets(db, execution)
        if budget["exceeded"]:
            raise RuntimeError(
                f"agent budget exceeded: {sorted(budget['exceeded'])}")
        if execution.status == "CANCELLED":
            raise RuntimeError("execution cancelled")
        checkpoint_before_side_effect(
            db, execution_id=execution_id, workspace_id=workspace_id,
            step_number=step_number,
            state={"tool": step.get("tool"), "inputs_keys":
                   sorted((step.get("inputs") or {}).keys())})
        db.flush()
        try:
            output = run_step(step)
        except Exception as exc:  # noqa: BLE001 — step boundary
            checkpoint_after_side_effect(
                db, execution_id=execution_id, workspace_id=workspace_id,
                step_number=step_number,
                state={"failed": str(exc)[:500]})
            db.flush()
            raise
        checkpoint_after_side_effect(
            db, execution_id=execution_id, workspace_id=workspace_id,
            step_number=step_number,
            state={"ok": True, "tool": step.get("tool")},
            output_reference=(output or {}).get("artifact_reference"))
        db.flush()
        results.append({"step": step_number, "tool": step.get("tool"),
                        "ok": True, "authorized": authz["authorized"]})
    return {"execution_id": execution_id, "steps": results,
            "next_step": len(results) + start}


# ---------------------------------------------------------------------------
# Dead letters
# ---------------------------------------------------------------------------

def agent_dead_letter(db: Session, *, execution_id: str, workspace_id: int,
                      error: str, actor_id: int = 0) -> dict:
    """Terminal state for an unrecoverable agent execution."""
    execution = _owned_execution(db, execution_id, workspace_id)
    from . import ai_execution_service as aes
    try:
        aes.transition(db, execution, "FAILED", actor_id=actor_id)
    except Exception:  # noqa: BLE001 — terminal state is forced
        execution.status = "FAILED"
    execution.failure_reason = error[:2000]
    execution.completed_at = _utcnow()
    db.flush()
    return {"execution_id": execution_id, "status": "FAILED",
            "dead_lettered": True}