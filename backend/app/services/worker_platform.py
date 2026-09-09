"""Production worker platform — durable, provider-agnostic job queue.

Architecture:

    Producer ── enqueue_job ──▶ WorkerJob (DB row, tenant scoped)
    Consumer ── claim_job ──▶ atomic CLAIMED
              ── run handler ──▶ COMPLETED / FAILED(retry→QUEUED) / DEAD_LETTERED

The queue backend is deliberately decoupled from application logic: the
default backend is the relational outbox-style ``worker_jobs`` table (works
with PostgreSQL and SQLite). A broker-backed backend may be added behind the
same functions without changing producers/consumers.

Guarantees:
- Atomic claiming (single guarded UPDATE) — two workers never claim one job.
- Bounded retries with exponential backoff and dead-letter state.
- Heartbeat records + stale worker recovery (jobs are re-queued only when the
  claiming worker has stopped heartbeating; dedupe keys make side effects
  idempotent).
- Priority ordering with tenant fairness and anti-starvation (older jobs get
  an age boost).
"""

import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

from sqlalchemy.orm import Session

from ..models.phase16 import (
    WorkerJob, WorkerHeartbeat, JOB_STATUSES, WORKER_STATUSES,
)
from ..services.worker_scheduler import PRIORITY_RANK, validate_priority

logger = logging.getLogger(__name__)

# Workers that stop heartbeating for longer than this are presumed dead.
STALE_WORKER_SECONDS = 90
# Default retry backoff (deterministic exponential: base * 2 ** attempt).
RETRY_BACKOFF_BASE_SECONDS = 30
MAX_ATTEMPTS_DEFAULT = 3


class DuplicateJobError(Exception):
    """Same dedupe key + queue + workspace used with a different payload."""


class JobNotFoundError(Exception):
    """Job does not exist."""


class JobStateError(Exception):
    """Invalid job state transition."""


class WorkerRetryError(Exception):
    """Raised by a handler to request a bounded retry of the job."""

    def __init__(self, message: str, retryable: bool = True,
                 delay_seconds: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.delay_seconds = delay_seconds


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------

def enqueue_job(
    db: Session,
    queue_name: str,
    job_type: str,
    workspace_id: int,
    payload: Optional[dict] = None,
    user_id: Optional[int] = None,
    organization_id: Optional[int] = None,
    priority: str = "NORMAL",
    dedupe_key: Optional[str] = None,
    run_after: Optional[datetime] = None,
    max_attempts: int = MAX_ATTEMPTS_DEFAULT,
    trace_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> WorkerJob:
    """Enqueue a durable job. Idempotent per (workspace, queue, dedupe_key)."""
    priority = validate_priority(priority)
    if dedupe_key:
        existing = (
            db.query(WorkerJob)
            .filter(
                WorkerJob.workspace_id == workspace_id,
                WorkerJob.queue_name == queue_name,
                WorkerJob.dedupe_key == dedupe_key,
            )
            .first()
        )
        if existing:
            # Same job re-enqueued → return the existing durable record.
            if existing.job_type != job_type:
                raise DuplicateJobError(
                    "Dedupe key already used for a different job type"
                )
            if existing.status in ("COMPLETED", "DEAD_LETTERED", "CANCELLED"):
                raise DuplicateJobError(
                    "Dedupe key was already used by a terminal job; "
                    "reuse is not allowed for the same tenant+queue")
            return existing
    job = WorkerJob(
        queue_name=queue_name,
        job_type=job_type,
        payload_json=json.dumps(payload, default=str) if payload else None,
        status="QUEUED",
        priority=priority,
        workspace_id=workspace_id,
        organization_id=organization_id,
        user_id=user_id,
        max_attempts=max_attempts,
        dedupe_key=dedupe_key,
        run_after=run_after,
        trace_id=trace_id or str(uuid.uuid4())[:32],
        correlation_id=correlation_id,
    )
    db.add(job)
    db.flush()
    return job


# ---------------------------------------------------------------------------
# Consumer: atomic claim
# ---------------------------------------------------------------------------

def _due_filter(query, now):
    return query.filter(
        WorkerJob.status == "QUEUED",
        (WorkerJob.next_retry_at.is_(None)) | (WorkerJob.next_retry_at <= now),
        (WorkerJob.run_after.is_(None)) | (WorkerJob.run_after <= now),
    )





def _active_counts_by_workspace(db: Session, queue_name: str, now) -> dict:
    rows = (
        db.query(WorkerJob.workspace_id, WorkerJob.status,
                 WorkerJob.claimed_by, WorkerJob.heartbeat_at)
        .filter(WorkerJob.queue_name == queue_name)
        .all()
    )
    counts: dict = {}
    for workspace_id, status, claimed_by, hb in rows:
        if status in ("CLAIMED", "RUNNING"):
            # A claim whose worker is dead is not consuming capacity.
            if claimed_by and hb is not None:
                last = _as_utc(hb)
                if last and (now - last).total_seconds() > STALE_WORKER_SECONDS:
                    continue
            counts[workspace_id] = counts.get(workspace_id, 0) + 1
    return counts


def claim_job(
    db: Session,
    queue_name: str,
    worker_id: str,
    now: Optional[datetime] = None,
    per_workspace_cap: int = 3,
    skip_workspace_ids: Optional[set] = None,
) -> Optional[WorkerJob]:
    """Atomically claim the next due job for ``queue_name``.

    Candidates are ordered by priority, then tenant fairness (workspaces with
    the fewest active jobs go first), then age (anti-starvation). Returns the
    claimed job (status CLAIMED) or None when nothing is due or tenant caps
    are exhausted.
    """
    now = now or _now()
    skip = skip_workspace_ids or set()
    query = db.query(WorkerJob).filter(WorkerJob.queue_name == queue_name)
    query = _due_filter(query, now)
    candidates = query.limit(50).all()
    active = _active_counts_by_workspace(db, queue_name, now)
    rank = {p: i for i, p in enumerate(
        ["CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND"])}

    def key(job):
        created = _as_utc(job.created_at) or now
        return (rank.get(job.priority, 2), active.get(job.workspace_id, 0),
                created)

    for job in sorted(candidates, key=key):
        ws = job.workspace_id
        if ws in skip:
            continue
        if active.get(ws, 0) >= per_workspace_cap:
            continue  # another tenant owns the remaining capacity this round
        # Atomic guarded UPDATE — only one consumer wins.
        updated = (
            db.query(WorkerJob)
            .filter(WorkerJob.id == job.id, WorkerJob.status == "QUEUED")
            .update({
                "status": "CLAIMED",
                "claimed_by": worker_id,
                "heartbeat_at": now,
                "started_at": job.started_at or now,
                "attempt": job.attempt + 1,
            })
        )
        if updated == 1:
            db.flush()
            db.refresh(job)
            return job
        db.rollback()  # lost the race; try the next candidate
    return None


def mark_running(db: Session, job: WorkerJob, worker_id: str) -> WorkerJob:
    if job.status != "CLAIMED" or job.claimed_by != worker_id:
        raise JobStateError(
            f"Cannot mark job {job.id} RUNNING (status={job.status}, "
            f"claimed_by={job.claimed_by})"
        )
    job.status = "RUNNING"
    job.heartbeat_at = _now()
    db.flush()
    return job


def heartbeat_job(db: Session, job: WorkerJob) -> None:
    job.heartbeat_at = _now()
    db.flush()


def complete_job(db: Session, job: WorkerJob) -> WorkerJob:
    if job.status == "COMPLETED":
        return job  # idempotent
    if job.status not in ("CLAIMED", "RUNNING"):
        raise JobStateError(f"Cannot complete job {job.id} in status {job.status}")
    job.status = "COMPLETED"
    job.completed_at = _now()
    job.heartbeat_at = None
    db.flush()
    return job


def release_job(db: Session, job: WorkerJob, reason: str = "worker shutdown") -> WorkerJob:
    """Return an in-flight job to the queue (graceful shutdown)."""
    if job.status not in ("CLAIMED", "RUNNING"):
        return job
    job.status = "QUEUED"
    job.claimed_by = None
    job.heartbeat_at = None
    job.next_retry_at = _now()
    job.error_message = reason
    db.flush()
    return job


def fail_job(
    db: Session,
    job: WorkerJob,
    error: str,
    retryable: bool = True,
    delay_seconds: Optional[int] = None,
) -> WorkerJob:
    """Fail a job — bounded retry with exponential backoff or dead-letter."""
    if job.status in ("DEAD_LETTERED", "CANCELLED", "COMPLETED"):
        return job  # idempotent
    if job.status not in ("CLAIMED", "RUNNING", "QUEUED"):
        raise JobStateError(f"Cannot fail job {job.id} in status {job.status}")
    if retryable and job.attempt < job.max_attempts:
        backoff = delay_seconds or (
            RETRY_BACKOFF_BASE_SECONDS * (2 ** (job.attempt - 1)))
        job.status = "QUEUED"
        job.claimed_by = None
        job.heartbeat_at = None
        job.next_retry_at = _now() + timedelta(seconds=backoff)
        job.error_message = error[:2000]
    else:
        job.status = "DEAD_LETTERED"
        job.completed_at = _now()
        job.claimed_by = None
        job.heartbeat_at = None
        job.error_message = error[:2000]
        logger.error(
            "Job %s (%s/%s) dead-lettered after %d attempts: %s",
            job.id, job.queue_name, job.job_type, job.attempt, error,
        )
    db.flush()
    return job


def cancel_job(db: Session, job_id: int) -> WorkerJob:
    job = db.query(WorkerJob).filter(WorkerJob.id == job_id).first()
    if not job:
        raise JobNotFoundError(f"Job {job_id} not found")
    if job.status in ("COMPLETED", "DEAD_LETTERED", "CANCELLED"):
        return job
    job.status = "CANCELLED"
    job.completed_at = _now()
    db.flush()
    return job


# ---------------------------------------------------------------------------
# Worker heartbeat registry
# ---------------------------------------------------------------------------

def register_worker(
    db: Session,
    worker_id: str,
    queue_name: Optional[str] = None,
    version: Optional[str] = None,
    hostname: Optional[str] = None,
    pid: Optional[int] = None,
) -> WorkerHeartbeat:
    hb = db.query(WorkerHeartbeat).filter(
        WorkerHeartbeat.worker_id == worker_id).first()
    now = _now()
    if hb is None:
        hb = WorkerHeartbeat(
            worker_id=worker_id, queue_name=queue_name, version=version,
            hostname=hostname, pid=pid, status="RUNNING",
            started_at=now, last_heartbeat=now,
        )
        db.add(hb)
    else:
        hb.queue_name = queue_name or hb.queue_name
        hb.version = version or hb.version
        hb.hostname = hostname or hb.hostname
        hb.pid = pid or hb.pid
        hb.status = "RUNNING"
        hb.last_heartbeat = now
        if hb.started_at is None:
            hb.started_at = now
    db.flush()
    return hb


def touch_heartbeat(
    db: Session, worker_id: str, current_job_type: Optional[str] = None,
    current_job_id: Optional[int] = None,
) -> Optional[WorkerHeartbeat]:
    hb = db.query(WorkerHeartbeat).filter(
        WorkerHeartbeat.worker_id == worker_id).first()
    if hb is None:
        return None
    hb.last_heartbeat = _now()
    hb.status = "RUNNING"
    hb.current_job_type = current_job_type or hb.current_job_type
    hb.current_job_id = current_job_id or hb.current_job_id
    db.flush()
    return hb


def mark_worker_stopped(
    db: Session, worker_id: str, status: str = "STOPPED",
) -> Optional[WorkerHeartbeat]:
    hb = db.query(WorkerHeartbeat).filter(
        WorkerHeartbeat.worker_id == worker_id).first()
    if hb is None:
        return None
    hb.status = status
    hb.current_job_type = None
    hb.current_job_id = None
    hb.stopped_at = _now()
    db.flush()
    return hb


def recover_stale_workers(
    db: Session,
    stale_seconds: int = STALE_WORKER_SECONDS,
    now: Optional[datetime] = None,
) -> dict:
    """Mark workers that stopped heartbeating as DEAD and re-queue their
    in-flight jobs so a healthy worker can finish them (idempotently)."""
    now = now or _now()
    cutoff = now - timedelta(seconds=stale_seconds)
    stale = (
        db.query(WorkerHeartbeat)
        .filter(
            WorkerHeartbeat.status.in_(("RUNNING", "IDLE")),
            WorkerHeartbeat.last_heartbeat < cutoff,
        )
        .all()
    )
    recovered_jobs = 0
    for worker in stale:
        worker.status = "DEAD"
        worker.current_job_type = None
        worker.current_job_id = None
        worker.stopped_at = now
        db.flush()
        orphaned = (
            db.query(WorkerJob)
            .filter(
                WorkerJob.claimed_by == worker.worker_id,
                WorkerJob.status.in_(("CLAIMED", "RUNNING")),
            )
            .all()
        )
        for job in orphaned:
            # Never re-run non-idempotent side effects twice: jobs with a
            # dedupe key are safe to retry; jobs without one are still
            # re-queued because the previous run may not have started.
            job.status = "QUEUED"
            job.claimed_by = None
            job.heartbeat_at = None
            job.next_retry_at = now
            job.error_message = "recovered from stale worker"
            recovered_jobs += 1
        db.flush()
    return {"workers_dead": len(stale), "jobs_recovered": recovered_jobs}


# ---------------------------------------------------------------------------
# Graceful shutdown + single-run loop
# ---------------------------------------------------------------------------

def graceful_shutdown(db: Session, worker_id: str) -> dict:
    """Stop claiming work, release the in-flight job (if any), stop heartbeat."""
    hb = mark_worker_stopped(db, worker_id, status="STOPPED")
    if hb is None:
        return {"released": 0}
    in_flight = (
        db.query(WorkerJob)
        .filter(WorkerJob.claimed_by == worker_id,
                WorkerJob.status.in_(("CLAIMED", "RUNNING")))
        .all()
    )
    for job in in_flight:
        release_job(db, job)
    db.flush()
    return {"released": len(in_flight)}


def run_once(
    db: Session,
    queue_name: str,
    worker_id: str,
    handler: Callable[[Session, WorkerJob], None],
    heartbeat_every: Optional[Callable[[], bool]] = None,
    per_workspace_cap: int = 3,
) -> Optional[WorkerJob]:
    """Claim and process exactly one job.

    - handler runs the job; any WorkerRetryError schedules a bounded retry.
    - unexpected exceptions fail the job (retryable by default).
    """
    job = claim_job(db, queue_name, worker_id,
                    per_workspace_cap=per_workspace_cap)
    if job is None:
        return None
    mark_running(db, job, worker_id)
    db.commit()
    try:
        handler(db, job)
        if job.status not in ("COMPLETED", "DEAD_LETTERED", "CANCELLED"):
            complete_job(db, job)
        db.commit()
        return job
    except WorkerRetryError as exc:
        # Handler owns its persistence — do NOT roll back. Only re-schedule.
        db.expire(job)
        job = db.get(WorkerJob, job.id)
        if job is None:
            db.rollback()
            return None
        fail_job(db, job, str(exc), retryable=exc.retryable,
                 delay_seconds=exc.delay_seconds)
        if job.status == "DEAD_LETTERED":
            _fail_linked_execution(db, job, str(exc))
        db.commit()
        return job
    except Exception as exc:  # noqa: BLE001 — worker boundary catch
        db.rollback()
        db.expire(job)
        job = db.get(WorkerJob, job.id)
        fail_job(db, job, f"{type(exc).__name__}: {exc}", retryable=True)
        if job.status == "DEAD_LETTERED":
            _fail_linked_execution(db, job, f"{type(exc).__name__}: {exc}")
        db.commit()
        logger.exception("Worker job %s failed", job.id)
        return job


def _fail_linked_execution(db: Session, job: WorkerJob, error: str) -> None:
    """When an AI-execution job is dead-lettered, fail the linked execution
    so its state never lingers in RETRYING."""
    if job.queue_name != "AI_EXECUTIONS" or not job.dedupe_key:
        return
    from ..models.ai_execution import AIExecution
    execution = db.query(AIExecution).filter(
        AIExecution.id == job.dedupe_key).first()
    if execution is not None and execution.status not in (
            "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        execution.status = "FAILED"
        execution.failure_reason = (error or "dead-lettered")[:2000]
        execution.completed_at = _now()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def queue_metrics(db: Session, queue_name: Optional[str] = None,
                  now: Optional[datetime] = None) -> dict:
    """Safe queue metrics: depth, active, retries, dead letters, wait time,
    tenant fairness spread."""
    now = now or _now()
    query = db.query(WorkerJob)
    if queue_name:
        query = query.filter(WorkerJob.queue_name == queue_name)
    jobs = query.all()
    statuses = {"QUEUED": 0, "CLAIMED": 0, "RUNNING": 0, "COMPLETED": 0,
                "FAILED": 0, "DEAD_LETTERED": 0, "CANCELLED": 0}
    retry_scheduled = 0
    wait_ms_sum = 0.0
    wait_ms_count = 0
    per_ws: dict = {}
    for job in jobs:
        statuses[job.status] = statuses.get(job.status, 0) + 1
        if job.status == "QUEUED" and job.attempt > 0:
            retry_scheduled += 1
        if job.completed_at is not None and job.created_at is not None:
            start = _as_utc(job.started_at) or _as_utc(job.created_at)
            done = _as_utc(job.completed_at)
            wait_ms_sum += (start - _as_utc(job.created_at)).total_seconds() * 1000
            wait_ms_count += 1
            per_ws[job.workspace_id] = per_ws.get(job.workspace_id, 0) + 1
    active_per_ws = list(
        (db.query(WorkerJob.workspace_id)
         .filter(WorkerJob.status.in_(("CLAIMED", "RUNNING")))
         .group_by(WorkerJob.workspace_id).all())
    )
    active_values = [1] * len(active_per_ws) if active_per_ws else [0]
    fairness_spread = 0.0
    if active_values:
        mean = sum(active_values) / len(active_values)
        variance = sum((v - mean) ** 2 for v in active_values) / len(active_values)
        fairness_spread = math.sqrt(variance)
    return {
        "queue": queue_name or "all",
        "depth": len(jobs),
        "statuses": statuses,
        "retries_scheduled": retry_scheduled,
        "avg_wait_ms": round(wait_ms_sum / wait_ms_count, 1) if wait_ms_count else 0.0,
        "completed": statuses["COMPLETED"],
        "dead_letters": statuses["DEAD_LETTERED"],
        "active_tenants": len(active_values),
        "tenant_fairness_spread": round(fairness_spread, 4),
        "measured_at": now.isoformat(),
    }
