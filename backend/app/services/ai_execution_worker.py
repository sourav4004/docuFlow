"""Durable AI execution worker.

Connects Phase 15 ``AIExecution`` records to the Phase 16 worker platform:

    AIExecution(QUEUED)
        └─ enqueue_execution ─▶ WorkerJob(AI_EXECUTIONS)
              └─ process_execution:
                     QUEUED → RUNNING (explicit transition)
                     handler executes the task
                     ▶ COMPLETED        (execution + job)
                     ▶ RETRYING → job   (bounded backoff)
                     ▶ FAILED / TIMED_OUT / DEAD_LETTERED

Claiming is atomic (worker platform), retries are bounded, and handlers are
registered per (execution_type, task_type) so application logic stays
decoupled from the queue.
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

from sqlalchemy.orm import Session

from ..models.ai_execution import AIExecution
from ..services.ai_execution_service import (
    transition, complete_execution, fail_execution, cancel_execution,
    ExecutionStateError, save_checkpoint, latest_checkpoint,
)
from ..services.worker_platform import (
    enqueue_job, run_once, complete_job, fail_job, cancel_job, touch_heartbeat,
    mark_worker_stopped, register_worker, graceful_shutdown,
    recover_stale_workers, WorkerRetryError, JobStateError,
)

logger = logging.getLogger(__name__)

QUEUE_NAME = "AI_EXECUTIONS"

# Execution state machine additions for the worker path.
EXECUTION_HANDLERS: dict = {}
_EXECUTION_RETRYABLE_ERRORS = ("provider", "timeout", "rate_limit",
                               "dependency", "storage", "database")


def register_execution_handler(execution_type: str, task_type: str,
                               fn: Callable) -> None:
    """Register a deterministic handler for (execution_type, task_type).

    ``task_type`` may be "*" to catch all tasks of an execution type.
    """
    key = f"{execution_type}/{task_type}"
    EXECUTION_HANDLERS[key] = fn


def resolve_handler(execution_type: str, task_type: str) -> Optional[Callable]:
    fn = EXECUTION_HANDLERS.get(f"{execution_type}/{task_type}")
    if fn is None:
        fn = EXECUTION_HANDLERS.get(f"{execution_type}/*")
    return fn


def enqueue_execution(
    db: Session,
    execution: AIExecution,
    priority: Optional[str] = None,
) -> object:
    """Enqueue an AIExecution for durable worker processing (idempotent)."""
    return enqueue_job(
        db,
        queue_name=QUEUE_NAME,
        job_type=execution.task_type or execution.execution_type,
        workspace_id=execution.workspace_id,
        payload={"execution_id": execution.id},
        user_id=execution.user_id,
        organization_id=execution.organization_id,
        priority=priority or execution.priority or "NORMAL",
        dedupe_key=execution.id,
        trace_id=execution.trace_id,
        correlation_id=execution.id,
        max_attempts=3,
    )


def _retryable_failure(error_class: str) -> bool:
    return error_class in _EXECUTION_RETRYABLE_ERRORS


def process_execution(db: Session, job) -> None:
    """Run one claimed AI execution job (called by the worker loop)."""
    execution_id = job.dedupe_key or (
        (job.payload_json or {}).get("execution_id"))
    if not execution_id:
        raise WorkerRetryError("Execution job missing execution id",
                               retryable=False)
    execution = db.query(AIExecution).filter(
        AIExecution.id == execution_id).first()
    if execution is None:
        raise WorkerRetryError(
            f"Execution {execution_id} not found", retryable=False)

    if execution.status == "CANCELLED":
        cancel_job(db, job.id)
        db.flush()
        return

    handler = resolve_handler(execution.execution_type, execution.task_type)
    if handler is None:
        # No executor for this task → non-retryable terminal failure.
        fail_execution(db, execution, execution.user_id,
                       reason=f"No handler for {execution.execution_type}/"
                              f"{execution.task_type}", retryable=False)
        raise WorkerRetryError("No handler registered for execution",
                               retryable=False)

    # Move into RUNNING (QUEUED→RUNNING or RETRYING→RUNNING allowed).
    if execution.status in ("QUEUED", "RETRYING", "PLANNING"):
        try:
            transition(db, execution, "RUNNING", execution.user_id)
        except ExecutionStateError:
            pass  # already RUNNING (recovered claim)
    elif execution.status != "RUNNING":
        raise WorkerRetryError(
            f"Execution in unexpected status {execution.status}",
            retryable=False)

    checkpoint = latest_checkpoint(db, execution)
    ctx = {
        "job": job,
        "execution": execution,
        "resume_step": checkpoint.step_number if checkpoint else 0,
        "step": 0,
    }
    try:
        result = handler(execution, db, ctx) or {}
        output = result.get("output_reference")
        cost = result.get("actual_cost", 0.0)
        tokens_in = result.get("input_tokens", 0)
        tokens_out = result.get("output_tokens", 0)
        complete_execution(db, execution, execution.user_id,
                           output_reference=output, actual_cost=cost,
                           input_tokens=tokens_in, output_tokens=tokens_out)
        save_checkpoint(db, execution, step_number=ctx["step"] + 1,
                        state={"done": True, "output_reference": output},
                        artifact_reference=output)
        complete_job(db, job)
        db.flush()
    except WorkerRetryError:
        raise
    except Exception as exc:  # noqa: BLE001 — worker boundary
        reason = f"{type(exc).__name__}: {exc}"[:2000]
        error_class = _classify_error(exc)
        if _retryable_failure(error_class):
            transition_retry(db, execution, reason)
            raise WorkerRetryError(reason, retryable=True)
        fail_execution(db, execution, execution.user_id,
                       reason=reason, retryable=False)
        raise WorkerRetryError(reason, retryable=False)


def transition_retry(db: Session, execution: AIExecution, reason: str) -> None:
    if execution.status in ("RUNNING", "WAITING_TOOL", "WAITING_APPROVAL"):
        execution.retry_count += 1
        try:
            transition(db, execution, "RETRYING", execution.user_id,
                       failure_reason=reason)
        except ExecutionStateError:
            pass
    db.flush()


def run_execution_worker_once(
    db: Session,
    worker_id: str,
    heartbeat: bool = True,
) -> Optional[object]:
    """Claim + run a single AI execution job for this worker."""
    if heartbeat:
        register_worker(db, worker_id, queue_name=QUEUE_NAME,
                        version="16")
        db.commit()
    job = run_once(
        db, QUEUE_NAME, worker_id,
        handler=lambda d, j: process_execution(d, j),
    )
    if heartbeat:
        touch_heartbeat(db, worker_id)
        db.commit()
    return job


def drain_executions(db: Session, worker_id: str, max_jobs: int = 10) -> dict:
    """Run up to ``max_jobs`` due executions (used by tests + CLI)."""
    processed = 0
    for _ in range(max_jobs):
        job = run_execution_worker_once(db, worker_id)
        if job is None:
            break
        processed += 1
    return {"processed": processed}


def recover_all(db: Session) -> dict:
    """Stale-worker + stale-execution + hard-timeout recovery sweep."""
    stale = recover_stale_workers(db)
    from ..services.ai_execution_service import (
        recover_stale_executions, timeout_hard)
    recovered_execs = recover_stale_executions(db, actor_id=0)
    timed_out = timeout_hard(db, actor_id=0)
    db.flush()
    return {
        **stale,
        "executions_recovered": len(recovered_execs),
        "executions_timed_out": len(timed_out),
    }


def shutdown_worker(db: Session, worker_id: str) -> dict:
    return graceful_shutdown(db, worker_id)


# ---------------------------------------------------------------------------
# Failure classification (shared with trace service)
# ---------------------------------------------------------------------------

def _classify_error(exc: Exception) -> str:
    from ..services.trace_service import classify_failure
    return classify_failure(exc)
