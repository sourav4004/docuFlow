"""Distributed worker fleet — Phase 17.

Adds the explicit-lease layer on top of the durable queue:

- every claimed job gets a lease row (owner worker, token, expiry)
- heartbeats extend the lease; workers that stop heartbeating have their
  leases expired and jobs re-queued (never double-executed for side effects —
  handlers remain idempotent and payloads stay untouched)
- weighted tenant fairness with anti-starvation ordering
- autoscaling signals and dead-letter administration
- production shutdown protocol (SIGTERM-aware entry point)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import signal
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..core.config import settings
from ..models.phase16 import WorkerJob, WorkerHeartbeat
from ..models.phase17 import JobLease
from . import worker_platform as wp

logger = logging.getLogger(__name__)

LEASE_DEFAULT_SECONDS = 300
LEASE_GRACE_SECONDS = 60
PRIORITY_ORDER = ["CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND"]
PRIORITY_WEIGHTS = {"CRITICAL": 16, "HIGH": 8, "NORMAL": 4, "LOW": 2,
                    "BACKGROUND": 1}
DEFAULT_VERSION = "17.0.0"

_app_version = getattr(settings, "app_version", None) or DEFAULT_VERSION


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# ---------------------------------------------------------------------------
# Worker identity
# ---------------------------------------------------------------------------

class WorkerIdentity:
    """Safe runtime identity — never exposes secrets or full environment."""

    def __init__(self, worker_id: Optional[str] = None,
                 queue_names: Optional[list[str]] = None,
                 version: Optional[str] = None):
        self.worker_id = worker_id or f"w-{uuid.uuid4().hex[:12]}"
        self.hostname = socket.gethostname()[:200]
        self.pid = os.getpid()
        self.version = version or _app_version
        self.queue_names = queue_names or ["default"]
        self.started_at = _now()

    def to_public(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "version": self.version,
            "queues": self.queue_names,
            "started_at": self.started_at.isoformat(),
            "runtime": platform.python_version(),
        }


def register_fleet_worker(db: Session, identity: WorkerIdentity,
                          queue_names: Optional[list[str]] = None) -> WorkerHeartbeat:
    """Register (or refresh) a worker heartbeat row."""
    queues = ",".join(queue_names or identity.queue_names)
    hb = db.query(WorkerHeartbeat).filter(
        WorkerHeartbeat.worker_id == identity.worker_id).first()
    now = _now()
    if hb is None:
        hb = WorkerHeartbeat(
            worker_id=identity.worker_id,
            queue_name=queues[:40],
            version=identity.version,
            hostname=identity.hostname,
            pid=identity.pid,
            status="RUNNING",
            started_at=now,
            last_heartbeat=now,
        )
        db.add(hb)
    else:
        hb.queue_name = queues[:40]
        hb.version = identity.version
        hb.hostname = identity.hostname
        hb.pid = identity.pid
        hb.status = "RUNNING"
        hb.last_heartbeat = now
    db.flush()
    return hb


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------

def acquire_lease(db: Session, job: WorkerJob, worker_id: str,
                  duration_seconds: int = LEASE_DEFAULT_SECONDS) -> JobLease:
    """Persist an explicit lease for a claimed job (idempotent per job)."""
    token = uuid.uuid4().hex
    now = _now()
    job.lease_token = token
    job.lease_expires_at = now + timedelta(seconds=duration_seconds)
    existing = db.query(JobLease).filter(JobLease.job_id == job.id).first()
    if existing is None:
        lease = JobLease(job_id=job.id, worker_id=worker_id,
                         lease_token=token,
                         expires_at=job.lease_expires_at,
                         heartbeat_at=now, created_at=now)
        db.add(lease)
        db.flush()
        return lease
    existing.worker_id = worker_id
    existing.lease_token = token
    existing.expires_at = job.lease_expires_at
    existing.heartbeat_at = now
    db.flush()
    return existing


def extend_lease(db: Session, job: WorkerJob, worker_id: str,
                 duration_seconds: int = LEASE_DEFAULT_SECONDS) -> bool:
    """Extend the lease owned by ``worker_id``. Returns False when the worker
    no longer owns the lease (job must abort)."""
    if job.lease_token is None or job.claimed_by != worker_id:
        return False
    now = _now()
    lease = db.query(JobLease).filter(JobLease.job_id == job.id).first()
    if lease is None or lease.worker_id != worker_id:
        return False
    job.lease_expires_at = now + timedelta(seconds=duration_seconds)
    job.heartbeat_at = now
    lease.expires_at = job.lease_expires_at
    lease.heartbeat_at = now
    db.flush()
    return True


def recover_expired_leases(db: Session, grace_seconds: int = LEASE_GRACE_SECONDS,
                           now: Optional[datetime] = None) -> dict:
    """Re-queue jobs whose lease expired without a terminal state.

    A lease is considered expired when ``expires_at`` is in the past and the
    owning worker's heartbeat is older than the grace window (or the worker is
    marked stopped/dead). Handlers stay idempotent, so re-execution after
    recovery is safe.
    """
    now = now or _now()
    cutoff = now - timedelta(seconds=grace_seconds)
    stale_workers = {
        row.worker_id
        for row in db.query(WorkerHeartbeat.worker_id)
        .filter((WorkerHeartbeat.last_heartbeat < cutoff)
                | (WorkerHeartbeat.status.in_(("DEAD", "STOPPED"))))
        .all()
    }
    expired = (
        db.query(JobLease)
        .filter(JobLease.expires_at < now)
        .all()
    )
    recovered = 0
    skipped = 0
    for lease in expired:
        job = db.query(WorkerJob).filter(WorkerJob.id == lease.job_id).first()
        if job is None:
            db.delete(lease)
            skipped += 1
            continue
        if job.status in ("COMPLETED", "DEAD_LETTERED", "CANCELLED"):
            # Terminal job — drop the stale lease row.
            db.delete(lease)
            skipped += 1
            continue
        if job.status not in ("CLAIMED", "RUNNING"):
            db.delete(lease)
            skipped += 1
            continue
        if lease.worker_id not in stale_workers:
            # Worker is still alive and may finish; leave the lease alone.
            skipped += 1
            continue
        # Owner is gone — re-queue (idempotent execution assumed).
        job.status = "QUEUED"
        job.claimed_by = None
        job.heartbeat_at = None
        job.lease_token = None
        job.lease_expires_at = None
        job.next_retry_at = now + timedelta(seconds=1)
        job.error_message = "lease recovered after worker loss"
        db.delete(lease)
        recovered += 1
    db.flush()
    return {"recovered": recovered, "skipped": skipped,
            "expired_leases": len(expired)}


# ---------------------------------------------------------------------------
# Weighted fair claim (fair scheduling 2.0)
# ---------------------------------------------------------------------------

def claim_weighted(db: Session, queue_name: str, worker_id: str,
                   per_workspace_cap: int = 3,
                   skip_workspace_ids: Optional[set] = None,
                   now: Optional[datetime] = None) -> Optional[WorkerJob]:
    """Fair claim with priority weights + waiting-time anti-starvation.

    Ranking combines: priority weight (dominant), tenant active-count (round
    robin), and job age (anti-starvation tie-breaker). A single noisy tenant
    cannot monopolize capacity beyond ``per_workspace_cap``.
    """
    now = now or _now()
    skip = skip_workspace_ids or set()
    # Fair candidate window: at most ``per_workspace_cap`` due jobs per
    # workspace (highest priority first). A noisy tenant that floods the queue
    # can never monopolize the window itself, so quieter tenants always stay
    # eligible regardless of absolute job counts.
    from sqlalchemy import case, func, or_, select
    t = WorkerJob.__table__
    priority_rank = case(
        (t.c.priority == "CRITICAL", 0),
        (t.c.priority == "HIGH", 1),
        (t.c.priority == "NORMAL", 2),
        (t.c.priority == "LOW", 3),
        else_=4,
    )
    rn = func.row_number().over(
        partition_by=t.c.workspace_id,
        order_by=(priority_rank.asc(), t.c.created_at.asc()),
    ).label("rn")
    inner = (
        select(t.c.id, rn)
        .where(
            t.c.queue_name == queue_name,
            t.c.status == "QUEUED",
            or_(t.c.run_after.is_(None), t.c.run_after <= now),
            or_(t.c.next_retry_at.is_(None), t.c.next_retry_at <= now),
        )
        .subquery()
    )
    candidates = (
        db.query(WorkerJob)
        .join(inner, WorkerJob.id == inner.c.id)
        .filter(inner.c.rn <= per_workspace_cap)
        .order_by(WorkerJob.created_at.asc())
        .limit(500)
        .all()
    )
    active = wp._active_counts_by_workspace(db, queue_name, now)

    def fairness_key(job: WorkerJob):
        weight = PRIORITY_WEIGHTS.get(job.priority, 4)
        created = _as_utc(job.created_at) or now
        age_ms = max(0.0, (now - created).total_seconds() * 1000.0)
        # priority dominates; then tenants with less active work; then oldest
        return (-weight, active.get(job.workspace_id, 0), -age_ms)

    for job in sorted(candidates, key=fairness_key):
        ws = job.workspace_id
        if ws in skip:
            continue
        if active.get(ws, 0) >= per_workspace_cap:
            continue
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
            acquire_lease(db, job, worker_id)
            return job
        db.rollback()
    return None


# ---------------------------------------------------------------------------
# Autoscaling signals
# ---------------------------------------------------------------------------

def autoscale_signals(db: Session, queue_name: Optional[str] = None,
                      now: Optional[datetime] = None) -> dict:
    """Metrics a controller can use to scale a worker fleet."""
    now = now or _now()
    q = db.query(WorkerJob)
    if queue_name:
        q = q.filter(WorkerJob.queue_name == queue_name)

    queued = q.filter(WorkerJob.status == "QUEUED").count()
    running = q.filter(WorkerJob.status.in_(("CLAIMED", "RUNNING"))).count()
    retrying = q.filter(
        WorkerJob.status == "QUEUED",
        WorkerJob.next_retry_at.isnot(None)).count()
    dead = q.filter(WorkerJob.status == "DEAD_LETTERED").count()

    oldest = (
        db.query(WorkerJob)
        .filter(WorkerJob.queue_name == queue_name if queue_name else True,
                WorkerJob.status == "QUEUED")
        .order_by(WorkerJob.created_at.asc())
        .first()
        if queue_name else db.query(WorkerJob).filter(
            WorkerJob.status == "QUEUED").order_by(
                WorkerJob.created_at.asc()).first()
    )
    oldest_age_seconds = None
    if oldest is not None and oldest.created_at is not None:
        oldest_age_seconds = max(
            0, int((now - _as_utc(oldest.created_at)).total_seconds()))

    worker_count = db.query(WorkerHeartbeat).filter(
        WorkerHeartbeat.status == "RUNNING").count()

    return {
        "queue_depth": queued,
        "retrying": retrying,
        "running": running,
        "dead_lettered": dead,
        "oldest_job_age_seconds": oldest_age_seconds,
        "worker_count": worker_count,
        "throughput_per_minute_avg": _throughput_estimate(db, queue_name),
        "failure_rate": _failure_rate(db, queue_name),
        "sampled_at": now.isoformat(),
    }


def _throughput_estimate(db: Session, queue_name: Optional[str]) -> float:
    """Completed jobs in the last 10 minutes / 10 — a stable rolling signal."""
    from sqlalchemy import func
    cutoff = _now() - timedelta(minutes=10)
    q = db.query(func.count(WorkerJob.id)).filter(
        WorkerJob.status == "COMPLETED", WorkerJob.completed_at >= cutoff)
    if queue_name:
        q = q.filter(WorkerJob.queue_name == queue_name)
    return round(q.scalar() or 0 / 10.0, 2)


def _failure_rate(db: Session, queue_name: Optional[str]) -> float:
    from sqlalchemy import func
    cutoff = _now() - timedelta(minutes=10)
    base = db.query(func.count(WorkerJob.id)).filter(
        WorkerJob.completed_at >= cutoff)
    failed = db.query(func.count(WorkerJob.id)).filter(
        WorkerJob.status == "DEAD_LETTERED",
        WorkerJob.completed_at >= cutoff)
    if queue_name:
        base = base.filter(WorkerJob.queue_name == queue_name)
        failed = failed.filter(WorkerJob.queue_name == queue_name)
    total = base.scalar() or 0
    if total == 0:
        return 0.0
    return round((failed.scalar() or 0) / total, 4)


# ---------------------------------------------------------------------------
# Dead-letter management
# ---------------------------------------------------------------------------

def list_dead_letters(db: Session, workspace_id: Optional[int] = None,
                      queue_name: Optional[str] = None, limit: int = 50,
                      offset: int = 0) -> dict:
    q = db.query(WorkerJob).filter(WorkerJob.status == "DEAD_LETTERED")
    if workspace_id:
        q = q.filter(WorkerJob.workspace_id == workspace_id)
    if queue_name:
        q = q.filter(WorkerJob.queue_name == queue_name)
    total = q.count()
    items = q.order_by(WorkerJob.completed_at.desc().nullslast()) \
        .offset(offset).limit(min(limit, 200)).all()
    return {
        "items": [
            {
                "id": j.id, "queue_name": j.queue_name, "job_type": j.job_type,
                "workspace_id": j.workspace_id, "attempt": j.attempt,
                "max_attempts": j.max_attempts,
                "error_message": (j.error_message or "")[:500],
                "completed_at": j.completed_at,
            }
            for j in items
        ],
        "total": total,
        "limit": min(limit, 200),
        "offset": offset,
    }


def requeue_dead_letter(db: Session, job_id: int, operator_user_id: int,
                        reset_attempts: bool = True,
                        audit: Optional[object] = None) -> WorkerJob:
    """Manually requeue a dead-lettered job. Idempotent — a job that is no
    longer DEAD_LETTERED is returned unchanged. Never replays automatically."""
    job = db.query(WorkerJob).filter(WorkerJob.id == job_id).first()
    if job is None:
        raise wp.JobNotFoundError(f"Job {job_id} not found")
    if job.status != "DEAD_LETTERED":
        return job
    job.status = "QUEUED"
    job.claimed_by = None
    job.heartbeat_at = None
    job.lease_token = None
    job.lease_expires_at = None
    job.completed_at = None
    job.next_retry_at = None
    if reset_attempts:
        job.attempt = 0
    job.error_message = f"manually requeued by operator #{operator_user_id}"
    db.flush()
    if audit is not None:
        try:
            audit(db, "WORKER_JOB_REQUEUED",
                  workspace_id=job.workspace_id,
                  organization_id=job.organization_id,
                  user_id=operator_user_id,
                  detail=f"job {job.id} ({job.queue_name}/{job.job_type})")
        except Exception:  # noqa: BLE001 — audit must never block recovery
            logger.exception("audit failure during DLQ requeue")
    return job


def abandon_dead_letter(db: Session, job_id: int,
                        operator_user_id: int) -> WorkerJob:
    job = db.query(WorkerJob).filter(WorkerJob.id == job_id).first()
    if job is None:
        raise wp.JobNotFoundError(f"Job {job_id} not found")
    if job.status == "DEAD_LETTERED":
        job.status = "CANCELLED"
        job.completed_at = _now()
        job.error_message = f"abandoned by operator #{operator_user_id}"
        db.flush()
    return job


# ---------------------------------------------------------------------------
# Shutdown protocol
# ---------------------------------------------------------------------------

class ShutdownManager:
    """SIGTERM-aware coordinator used by worker entry points."""

    def __init__(self, identity: WorkerIdentity):
        self.identity = identity
        self._stop = threading.Event()

    def request_stop(self, *_args) -> None:
        self._stop.set()
        logger.info("worker %s received shutdown signal",
                    self.identity.worker_id)

    def install(self) -> None:
        try:
            signal.signal(signal.SIGTERM, self.request_stop)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            pass
        try:
            signal.signal(signal.SIGINT, self.request_stop)
        except (ValueError, OSError):
            pass

    def stopping(self) -> bool:
        return self._stop.is_set()

    def wait(self, timeout: float = 0.5) -> None:
        self._stop.wait(timeout)

    def run(self, db_factory, tick_seconds: float = 1.0,
            per_round_seconds: float = 20.0) -> None:
        """Run a drain loop: claim → execute → ack until shutdown.

        ``db_factory`` must return a fresh Session per tick.
        """
        self.install()
        logger.info("worker %s started queues=%s",
                    self.identity.worker_id, self.identity.queue_names)
        while not self.stopping():
            try:
                db = db_factory()
                try:
                    wp.register_worker(
                        db, self.identity.worker_id,
                        queue_name=",".join(self.identity.queue_names)[:40],
                        version=self.identity.version,
                        hostname=self.identity.hostname,
                        pid=self.identity.pid)
                    db.commit()
                finally:
                    db.close()
            except Exception:  # noqa: BLE001
                logger.exception("heartbeat registration failed")
            for queue in self.identity.queue_names:
                if self.stopping():
                    break
                self._drain_queue(db_factory, queue, per_round_seconds)
            self.wait(tick_seconds)
        logger.info("worker %s shutdown complete", self.identity.worker_id)

    def _drain_queue(self, db_factory, queue_name: str,
                     per_round_seconds: float) -> None:
        deadline = time.time() + per_round_seconds
        while not self.stopping() and time.time() < deadline:
            db = db_factory()
            try:
                job = claim_weighted(db, queue_name, self.identity.worker_id,
                                     per_workspace_cap=3)
                if job is None:
                    return
                try:
                    wp.mark_running(db, job, self.identity.worker_id)
                    self._run_handler(db, job)
                    wp.complete_job(db, job)
                    db.commit()
                except Exception as exc:  # noqa: BLE001 — worker boundary
                    db.rollback()
                    db = db_factory()
                    job = db.query(WorkerJob).filter(
                        WorkerJob.id == job.id).first()
                    if job is not None:
                        wp.fail_job(db, job, str(exc)[:2000],
                                    retryable=True)
                        db.commit()
            finally:
                db.close()

    def _run_handler(self, db: Session, job: WorkerJob) -> None:
        from .worker_handlers import HANDLERS
        handler = HANDLERS.get(job.job_type)
        if handler is None:
            raise RuntimeError(f"no handler registered for {job.job_type!r}")
        payload = json.loads(job.payload_json) if job.payload_json else {}
        handler(db, payload)


# keep a module-level singleton for tests/CLI convenience
_shutdown_singleton: Optional[ShutdownManager] = None


def get_shutdown_manager(identity: Optional[WorkerIdentity] = None) -> ShutdownManager:
    global _shutdown_singleton
    if _shutdown_singleton is None:
        _shutdown_singleton = ShutdownManager(identity or WorkerIdentity())
    return _shutdown_singleton
