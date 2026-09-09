"""AI Execution Orchestrator 2.0 API — lifecycle, idempotency, checkpoints, queue."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_principal, AuthPrincipal
from ..models.ai_execution import AIExecution
from ..models.phase15 import AIExecutionCheckpoint
from ..services.permission_service import require_permission, require_workspace_membership
from ..services.ai_execution_service import (
    create_execution,
    transition,
    complete_execution,
    fail_execution,
    cancel_execution,
    save_checkpoint,
    latest_checkpoint,
    resume_from_checkpoint,
    queue_order,
    recover_stale_executions,
    timeout_hard,
    IdempotencyConflict,
    ExecutionStateError,
    StaleExecutionError,
)

router = APIRouter(prefix="/executions", tags=["executions"])


class ExecutionCreate(BaseModel):
    workspace_id: int
    execution_type: str
    task_type: str
    priority: str = "NORMAL"
    query: Optional[str] = None
    payload: Optional[dict] = None
    idempotency_key: Optional[str] = None
    parent_execution_id: Optional[str] = None


class ExecutionTransition(BaseModel):
    new_status: str
    failure_reason: Optional[str] = None


def _execution_dict(e: AIExecution) -> dict:
    return {
        "id": e.id,
        "workspace_id": e.workspace_id,
        "organization_id": e.organization_id,
        "user_id": e.user_id,
        "action_id": e.action_id,
        "execution_type": e.execution_type,
        "task_type": e.task_type,
        "status": e.status,
        "priority": e.priority,
        "query": e.query,
        "model": e.model,
        "provider": e.provider,
        "input_reference": e.input_reference,
        "output_reference": e.output_reference,
        "trace_id": e.trace_id,
        "parent_execution_id": e.parent_execution_id,
        "idempotency_key": e.idempotency_key,
        "input_tokens": e.input_tokens,
        "output_tokens": e.output_tokens,
        "total_tokens": e.total_tokens,
        "estimated_cost": e.estimated_cost,
        "actual_cost": e.actual_cost,
        "started_at": e.started_at,
        "completed_at": e.completed_at,
        "latency_ms": e.latency_ms,
        "failure_reason": e.failure_reason,
        "retry_count": e.retry_count,
        "created_at": e.created_at,
    }


def _require_workspace(db, workspace_id, principal):
    workspace = require_workspace_membership(db, workspace_id, principal.user.id)
    return workspace


@router.post("", status_code=201)
def create_execution_endpoint(
    data: ExecutionCreate,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, data.workspace_id, principal)
    try:
        execution = create_execution(
            db,
            workspace_id=data.workspace_id,
            user_id=principal.user.id,
            execution_type=data.execution_type,
            task_type=data.task_type,
            priority=data.priority,
            query=data.query,
            payload=data.payload,
            idempotency_key=data.idempotency_key,
            parent_execution_id=data.parent_execution_id,
        )
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except StaleExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    db.commit()
    db.refresh(execution)
    return _execution_dict(execution)


@router.get("")
def list_executions(
    workspace_id: int,
    status: Optional[str] = None,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    _require_workspace(db, workspace_id, principal)
    query = db.query(AIExecution).filter(AIExecution.workspace_id == workspace_id)
    if status:
        query = query.filter(AIExecution.status == status)
    items = query.order_by(AIExecution.created_at.desc()).limit(min(limit, 200)).all()
    return {"items": [_execution_dict(e) for e in items]}


@router.get("/queue")
def execution_queue(
    workspace_id: int,
    limit: int = 100,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Queued executions ordered by priority then age (fairness)."""
    _require_workspace(db, workspace_id, principal)
    items = queue_order(db, workspace_id=workspace_id, limit=min(limit, 200))
    return {"items": [_execution_dict(e) for e in items], "count": len(items)}


@router.get("/{execution_id}")
def get_execution(
    execution_id: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    execution = db.query(AIExecution).filter(AIExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    _require_workspace(db, execution.workspace_id, principal)
    return _execution_dict(execution)


@router.post("/{execution_id}/transition")
def transition_execution(
    execution_id: str,
    data: ExecutionTransition,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    execution = db.query(AIExecution).filter(AIExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    _require_workspace(db, execution.workspace_id, principal)
    try:
        execution = transition(db, execution, data.new_status, principal.user.id, failure_reason=data.failure_reason)
    except ExecutionStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(execution)
    return _execution_dict(execution)


@router.post("/{execution_id}/cancel")
def cancel_execution_endpoint(
    execution_id: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    execution = db.query(AIExecution).filter(AIExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    _require_workspace(db, execution.workspace_id, principal)
    try:
        execution = cancel_execution(db, execution, principal.user.id)
    except ExecutionStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(execution)
    return _execution_dict(execution)


@router.post("/{execution_id}/resume")
def resume_execution_endpoint(
    execution_id: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    execution = db.query(AIExecution).filter(AIExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    _require_workspace(db, execution.workspace_id, principal)
    try:
        execution, checkpoint = resume_from_checkpoint(db, execution, principal.user.id)
    except ExecutionStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    return {
        "execution": _execution_dict(execution),
        "checkpoint": {
            "step_number": checkpoint.step_number,
            "state": checkpoint.state_json,
            "created_at": checkpoint.created_at,
        } if checkpoint else None,
    }


@router.get("/{execution_id}/checkpoints")
def list_checkpoints(
    execution_id: str,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    execution = db.query(AIExecution).filter(AIExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    _require_workspace(db, execution.workspace_id, principal)
    checkpoints = (
        db.query(AIExecutionCheckpoint)
        .filter(AIExecutionCheckpoint.execution_id == execution_id)
        .order_by(AIExecutionCheckpoint.step_number.asc())
        .all()
    )
    return {
        "items": [
            {
                "step_number": c.step_number,
                "tool_output_reference": c.tool_output_reference,
                "artifact_reference": c.artifact_reference,
                "checksum": c.checksum,
                "created_at": c.created_at,
            }
            for c in checkpoints
        ]
    }


@router.post("/{execution_id}/complete")
def complete_execution_endpoint(
    execution_id: str,
    output_reference: Optional[str] = None,
    actual_cost: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    execution = db.query(AIExecution).filter(AIExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    _require_workspace(db, execution.workspace_id, principal)
    try:
        execution = complete_execution(
            db, execution, principal.user.id,
            output_reference=output_reference,
            actual_cost=actual_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except ExecutionStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(execution)
    return _execution_dict(execution)


@router.post("/{execution_id}/fail")
def fail_execution_endpoint(
    execution_id: str,
    reason: str,
    retryable: bool = True,
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    execution = db.query(AIExecution).filter(AIExecution.id == execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    _require_workspace(db, execution.workspace_id, principal)
    try:
        execution = fail_execution(db, execution, principal.user.id, reason, retryable=retryable)
    except ExecutionStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    db.commit()
    db.refresh(execution)
    return _execution_dict(execution)


@router.post("/maintenance/recover-stale")
def recover_stale(
    principal: AuthPrincipal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    """Worker maintenance hook: recover stale executions (admin)."""
    require_permission(db, principal.user.id, "ai:manage_settings")
    recovered = recover_stale_executions(db, principal.user.id)
    timed_out = timeout_hard(db, principal.user.id)
    db.commit()
    return {"recovered": len(recovered), "hard_timed_out": len(timed_out)}