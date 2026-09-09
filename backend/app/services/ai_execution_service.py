"""AI Execution Orchestrator 2.0.

Production-grade execution lifecycle:
- Idempotent creation (same key + tenant => one execution)
- Explicit state machine with invalid-transition rejection
- Resumable checkpoints
- Priority-aware queue ordering with tenant fairness
- Stale execution recovery
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.ai_execution import AIExecution
from ..models.phase15 import (
    AIExecutionIdempotency, AIExecutionCheckpoint, EXECUTION_STATUSES,
)
from ..services.audit_service import log_audit_event
from ..services.worker_scheduler import validate_priority

# Explicit transition table — invalid transitions fail server-side.
VALID_TRANSITIONS = {
    "QUEUED": {"PLANNING", "RUNNING", "WAITING_APPROVAL", "WAITING_TOOL", "RETRYING", "FAILED", "CANCELLED", "TIMED_OUT"},
    "PLANNING": {"RUNNING", "WAITING_APPROVAL", "WAITING_TOOL", "RETRYING", "FAILED", "CANCELLED", "TIMED_OUT"},
    "RUNNING": {"WAITING_APPROVAL", "WAITING_TOOL", "RETRYING", "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"},
    "WAITING_APPROVAL": {"RUNNING", "RETRYING", "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"},
    "WAITING_TOOL": {"RUNNING", "RETRYING", "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"},
    "RETRYING": {"PLANNING", "RUNNING", "WAITING_TOOL", "FAILED", "CANCELLED", "TIMED_OUT"},
    "COMPLETED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
    "TIMED_OUT": set(),
}

TERMINAL_STATUSES = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")

# After this age without progress, an execution is considered stale.
STALE_AFTER_SECONDS = 60 * 60  # 1 hour
# Hard timeout for an execution that never completes.
HARD_TIMEOUT_SECONDS = 24 * 60 * 60  # 24h


class ExecutionStateError(Exception):
    """Raised on invalid execution state transitions."""


class IdempotencyConflict(Exception):
    """Raised when the same idempotency key is reused with a different payload."""


class StaleExecutionError(Exception):
    """Raised when an execution cannot be recovered."""


def _as_utc(value) -> Optional[datetime]:
    """Normalize naive datetimes (e.g. SQLite storage) to aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None


def _request_hash(payload: Optional[dict], query: Optional[str] = None) -> str:
    # The full user-visible request (payload AND query) determines identity —
    # reusing a key with different content must be a conflict, not a reuse.
    content = dict(payload or {})
    if query is not None:
        content["__query__"] = query
    canonical = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def create_execution(
    db: Session,
    workspace_id: int,
    user_id: int,
    execution_type: str,
    task_type: str,
    organization_id: Optional[int] = None,
    action_id: Optional[int] = None,
    priority: str = "NORMAL",
    query: Optional[str] = None,
    payload: Optional[dict] = None,
    idempotency_key: Optional[str] = None,
    parent_execution_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> AIExecution:
    """Create an execution (idempotent per tenant+key)."""
    priority = validate_priority(priority)
    request_hash = _request_hash(payload, query)

    if idempotency_key:
        existing = (
            db.query(AIExecutionIdempotency)
            .filter(
                AIExecutionIdempotency.workspace_id == workspace_id,
                AIExecutionIdempotency.idempotency_key == idempotency_key,
            )
            .first()
        )
        if existing:
            if existing.request_hash != request_hash:
                raise IdempotencyConflict(
                    "Idempotency key was already used with a different request"
                )
            execution = db.query(AIExecution).filter(AIExecution.id == existing.execution_id).first()
            if execution:
                return execution
            raise StaleExecutionError("Idempotency record points to a missing execution")

    execution = AIExecution(
        id=str(uuid.uuid4()),
        organization_id=organization_id,
        workspace_id=workspace_id,
        user_id=user_id,
        action_id=action_id,
        execution_type=execution_type,
        parent_execution_id=parent_execution_id,
        task_type=task_type,
        status="QUEUED",
        priority=priority,
        query=query,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
        input_reference=(payload or {}).get("input_reference"),
        trace_id=trace_id or str(uuid.uuid4())[:32],
        estimated_cost=(payload or {}).get("estimated_cost", 0.0),
    )
    db.add(execution)
    db.flush()

    if idempotency_key:
        db.add(AIExecutionIdempotency(
            workspace_id=workspace_id,
            organization_id=organization_id,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            execution_id=execution.id,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        ))

    log_audit_event(
        db, event_type="ai_execution", event_action="create",
        user_id=user_id, resource_type="ai_execution", resource_id=execution.id,
        details=f"Execution '{execution_type}/{task_type}' queued (priority={priority})",
    )
    return execution


def transition(
    db: Session,
    execution: AIExecution,
    new_status: str,
    actor_id: int,
    failure_reason: Optional[str] = None,
) -> AIExecution:
    """Transition an execution, enforcing the explicit state machine."""
    if new_status not in EXECUTION_STATUSES:
        raise ExecutionStateError(f"Unknown execution status: {new_status}")
    if execution.status in TERMINAL_STATUSES:
        raise ExecutionStateError(
            f"Cannot transition terminal status {execution.status}"
        )
    allowed = VALID_TRANSITIONS.get(execution.status, set())
    if new_status not in allowed:
        raise ExecutionStateError(
            f"Invalid transition: {execution.status} -> {new_status}"
        )

    now = datetime.now(timezone.utc)
    execution.status = new_status
    if new_status in ("RUNNING", "PLANNING", "RETRYING") and execution.started_at is None:
        execution.started_at = now
    if new_status == "COMPLETED":
        execution.completed_at = now
        started = _as_utc(execution.started_at)
        if started:
            execution.duration_ms = (now - started).total_seconds() * 1000
            execution.latency_ms = execution.duration_ms
    if new_status == "FAILED":
        execution.failure_reason = failure_reason
    db.flush()

    log_audit_event(
        db, event_type="ai_execution", event_action=f"transition:{new_status.lower()}",
        user_id=actor_id, resource_type="ai_execution", resource_id=execution.id,
        details=f"Execution {execution.id} -> {new_status}",
    )
    return execution


def complete_execution(
    db: Session,
    execution: AIExecution,
    actor_id: int,
    output_reference: Optional[str] = None,
    actual_cost: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> AIExecution:
    execution = transition(db, execution, "COMPLETED", actor_id)
    execution.output_reference = output_reference
    execution.actual_cost = actual_cost
    execution.input_tokens = input_tokens
    execution.output_tokens = output_tokens
    execution.total_tokens = input_tokens + output_tokens
    db.flush()
    return execution


def fail_execution(
    db: Session,
    execution: AIExecution,
    actor_id: int,
    reason: str,
    retryable: bool = True,
) -> AIExecution:
    if retryable and execution.retry_count < 3:
        execution.retry_count += 1
        return transition(db, execution, "RETRYING", actor_id, failure_reason=reason)
    return transition(db, execution, "FAILED", actor_id, failure_reason=reason)


def cancel_execution(db: Session, execution: AIExecution, actor_id: int) -> AIExecution:
    return transition(db, execution, "CANCELLED", actor_id)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    db: Session,
    execution: AIExecution,
    step_number: int,
    state: Optional[dict] = None,
    tool_output_reference: Optional[str] = None,
    artifact_reference: Optional[str] = None,
) -> AIExecutionCheckpoint:
    checkpoint = AIExecutionCheckpoint(
        execution_id=execution.id,
        step_number=step_number,
        state_json=json.dumps(state, default=str) if state else None,
        tool_output_reference=tool_output_reference,
        artifact_reference=artifact_reference,
        checksum=hashlib.sha256(json.dumps(state or {}, sort_keys=True, default=str).encode()).hexdigest(),
    )
    db.add(checkpoint)
    db.flush()
    return checkpoint


def latest_checkpoint(db: Session, execution: AIExecution) -> Optional[AIExecutionCheckpoint]:
    return (
        db.query(AIExecutionCheckpoint)
        .filter(AIExecutionCheckpoint.execution_id == execution.id)
        .order_by(AIExecutionCheckpoint.step_number.desc())
        .first()
    )


def resume_from_checkpoint(
    db: Session,
    execution: AIExecution,
    actor_id: int,
) -> tuple[AIExecution, Optional[AIExecutionCheckpoint]]:
    """Resume an execution in RETRYING/PLANNING state from its latest checkpoint."""
    if execution.status not in ("RETRYING", "PLANNING", "QUEUED"):
        raise ExecutionStateError(
            f"Cannot resume execution in status {execution.status}"
        )
    execution = transition(db, execution, "RUNNING", actor_id)
    checkpoint = latest_checkpoint(db, execution)
    return execution, checkpoint


# ---------------------------------------------------------------------------
# Queue + recovery
# ---------------------------------------------------------------------------

def queue_order(db: Session, workspace_id: Optional[int] = None, limit: int = 100) -> list[AIExecution]:
    """Return queued executions ordered by priority (CRITICAL first) then age."""
    query = db.query(AIExecution).filter(AIExecution.status == "QUEUED")
    if workspace_id is not None:
        query = query.filter(AIExecution.workspace_id == workspace_id)
    return query.order_by(
        _priority_rank_sql(),
        AIExecution.created_at.asc(),
    ).limit(limit).all()


def _priority_rank_sql():
    from sqlalchemy import case
    return case(
        {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3, "BACKGROUND": 4},
        value=AIExecution.priority,
        else_=3,
    )


def recover_stale_executions(db: Session, actor_id: int, stale_after: int = STALE_AFTER_SECONDS) -> list[AIExecution]:
    """Mark stale non-terminal executions as FAILED (recoverable => RETRYING)."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after)
    stale = (
        db.query(AIExecution)
        .filter(
            AIExecution.status.in_(("QUEUED", "PLANNING", "RUNNING", "WAITING_TOOL")),
            AIExecution.started_at.isnot(None),
            AIExecution.started_at < cutoff,
        )
        .limit(200)
        .all()
    )
    recovered = []
    for execution in stale:
        try:
            recovered.append(fail_execution(
                db, execution, actor_id, reason="Execution became stale", retryable=True
            ))
        except ExecutionStateError:
            recovered.append(transition(db, execution, "TIMED_OUT", actor_id))
    if recovered:
        db.flush()
    return recovered


def timeout_hard(db: Session, actor_id: int, timeout: int = HARD_TIMEOUT_SECONDS) -> list[AIExecution]:
    """Hard-timeout executions that exceed the absolute limit."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=timeout)
    overdue = (
        db.query(AIExecution)
        .filter(
            AIExecution.status.notin_(TERMINAL_STATUSES),
            AIExecution.started_at.isnot(None),
            AIExecution.started_at < cutoff,
        )
        .limit(200)
        .all()
    )
    for execution in overdue:
        transition(db, execution, "TIMED_OUT", actor_id)
    if overdue:
        db.flush()
    return overdue