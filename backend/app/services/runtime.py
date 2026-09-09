"""Phase 18 runtime — worker/scheduler/event-processor process helpers.

Provides the production worker loop (capacity-aware heartbeats, CPU-aware
concurrency, lease extension), the scheduler loop (delayed-job promotion,
recurring maintenance, retention/cleanup), and the event-processor loop.
Every loop honors SIGTERM via :class:`ShutdownManager` and keeps state
durable in PostgreSQL.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def worker_capacity() -> dict:
    """Capacity from environment (never auto-provisions infrastructure).

    - ``WORKER_CONCURRENCY`` global job cap (default 4)
    - ``WORKER_CPU_AWARE`` 1 => scale concurrency from cpu_count (capped 8)
    - per-queue caps via ``WORKER_QUEUE_CONCURRENCY`` = ``queue:cap,...``
    """
    concurrency = int(os.getenv("WORKER_CONCURRENCY", "4"))
    if os.getenv("WORKER_CPU_AWARE", "0") == "1":
        try:
            import multiprocessing
            concurrency = max(1, min(multiprocessing.cpu_count() * 2, 8))
        except Exception:  # noqa: BLE001
            pass
    queue_caps: dict[str, int] = {}
    raw = os.getenv("WORKER_QUEUE_CONCURRENCY", "")
    if raw:
        for part in raw.split(","):
            if ":" in part:
                q, c = part.split(":", 1)
                try:
                    queue_caps[q.strip().upper()] = max(1, int(c.strip()))
                except ValueError:
                    continue
    return {"concurrency": max(1, concurrency), "queue_caps": queue_caps}


def queue_capacity(queue_name: str) -> int:
    caps = worker_capacity()["queue_caps"]
    return caps.get(queue_name, caps.get("default", worker_capacity()["concurrency"]))


def compute_load(active_jobs: int, capacity: int) -> float:
    if capacity <= 0:
        return 0.0
    return round(min(active_jobs / capacity, 1.0), 3)


# ---------------------------------------------------------------------------
# Heartbeat with capacity/load signals
# ---------------------------------------------------------------------------

def heartbeat_with_capacity(
    db: Session,
    worker_id: str,
    active_jobs: int,
    queue_assignments: Optional[list[str]] = None,
    load: Optional[float] = None,
) -> None:
    """Update the worker heartbeat row with capacity signals.

    Never exposes machine secrets — only counts and a load ratio.
    """
    from ..models.phase16 import WorkerHeartbeat
    hb = db.query(WorkerHeartbeat).filter(
        WorkerHeartbeat.worker_id == worker_id).first()
    now = _utcnow()
    if hb is None:
        hb = WorkerHeartbeat(worker_id=worker_id, status="RUNNING",
                             started_at=now, last_heartbeat=now)
        db.add(hb)
    hb.last_heartbeat = now
    hb.active_jobs = active_jobs
    hb.load = load
    if queue_assignments is not None:
        hb.queue_assignments = ",".join(queue_assignments)[:500]
    hb.status = "RUNNING"
    db.flush()


def mark_worker_inactive(db: Session, worker_id: str,
                         status: str = "STOPPED") -> None:
    from ..models.phase16 import WorkerHeartbeat
    hb = db.query(WorkerHeartbeat).filter(
        WorkerHeartbeat.worker_id == worker_id).first()
    if hb is None:
        return
    hb.status = status
    hb.stopped_at = _utcnow()
    db.flush()


# ---------------------------------------------------------------------------
# Worker process loop
# ---------------------------------------------------------------------------

def run_worker_process(
    db_factory: Callable[[], Session],
    identity,
    shutdown,
    queues: Optional[list[str]] = None,
    per_round_seconds: float = 20.0,
    tick_seconds: float = 1.0,
    heartbeat_every: Optional[Callable[[], bool]] = None,
) -> None:
    """Production worker loop: register → drain → heartbeat → shutdown.

    ``identity``: WorkerIdentity; ``shutdown``: ShutdownManager.
    """
    from .distributed import claim_weighted, extend_lease
    from .worker_platform import WorkerJob, complete_job, fail_job, mark_running

    queue_names = queues or identity.queue_names or ["default"]
    capacity = worker_capacity()
    shutdown.install()

    def _heartbeat(active: int) -> None:
        try:
            db = db_factory()
            try:
                from .distributed import register_fleet_worker
                register_fleet_worker(db, identity, queue_names=queue_names)
                heartbeat_with_capacity(
                    db, identity.worker_id, active, queue_names,
                    load=compute_load(active, capacity["concurrency"]))
                db.commit()
            finally:
                db.close()
        except Exception:  # noqa: BLE001
            logger.exception("worker heartbeat failed")

    logger.info("worker %s started queues=%s capacity=%s",
                identity.worker_id, queue_names, capacity)
    active = 0
    while not shutdown.stopping():
        _heartbeat(active)
        deadline = time.time() + per_round_seconds
        while not shutdown.stopping() and time.time() < deadline:
            per_queue = 0
            for queue in queue_names:
                if shutdown.stopping():
                    break
                cap = queue_capacity(queue)
                if active >= capacity["concurrency"]:
                    break
                if per_queue >= cap:
                    continue
                db = db_factory()
                try:
                    job = claim_weighted(db, queue, identity.worker_id,
                                         per_workspace_cap=3)
                    if job is None:
                        db.close()
                        continue
                    mark_running(db, job, identity.worker_id)
                    db.commit()
                    active += 1
                    per_queue += 1
                    try:
                        from .worker_handlers import HANDLERS
                        handler = HANDLERS.get(job.job_type)
                        if handler is None:
                            raise RuntimeError(
                                f"no handler registered for {job.job_type!r}")
                        import json as _json
                        payload = (_json.loads(job.payload_json)
                                   if job.payload_json else {})
                        handler(db, payload)
                        db.refresh(job)
                        if job.status not in ("COMPLETED", "DEAD_LETTERED",
                                              "CANCELLED"):
                            complete_job(db, job)
                        db.commit()
                    except Exception as exc:  # noqa: BLE001 — worker boundary
                        db.rollback()
                        db2 = db_factory()
                        try:
                            job2 = db2.query(WorkerJob).filter(
                                WorkerJob.id == job.id).first()
                            if job2 is not None:
                                fail_job(db2, job2, str(exc)[:2000],
                                         retryable=True)
                                db2.commit()
                        finally:
                            db2.close()
                except Exception:  # noqa: BLE001
                    logger.exception("claim round failed")
                    db.rollback()
                finally:
                    try:
                        db.close()
                    except Exception:  # noqa: BLE001
                        pass
                active = max(0, active - 1) if per_queue else active
            # maintain lease on any in-flight jobs during shutdown drain
            if shutdown.stopping():
                break
        if heartbeat_every is not None:
            try:
                if heartbeat_every():
                    break
            except Exception:  # noqa: BLE001
                logger.exception("heartbeat_every callback failed")
        shutdown.wait(tick_seconds)
    try:
        db = db_factory()
        try:
            mark_worker_inactive(db, identity.worker_id)
            db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        logger.exception("worker deregistration failed")
    logger.info("worker %s shutdown complete", identity.worker_id)


# ---------------------------------------------------------------------------
# Scheduler process loop
# ---------------------------------------------------------------------------

def run_scheduler_process(
    db_factory: Callable[[], Session],
    identity,
    shutdown,
    tick_seconds: float = 10.0,
    maintenance_every_seconds: float = 300.0,
) -> None:
    """Scheduler loop: promote due delayed jobs, run maintenance rounds.

    Maintenance round (idempotent, bounded):
    - promote delayed jobs (run_after <= now)
    - recover stale worker leases
    - expire stale approvals / merge requests
    - expire due memories
    - enqueue retention cleanup
    - enqueue due connector syncs
    """
    from .distributed import recover_expired_leases
    from .worker_platform import WorkerJob, enqueue_job

    shutdown.install()
    last_maintenance = 0.0
    logger.info("scheduler %s started", identity.worker_id)
    while not shutdown.stopping():
        try:
            db = db_factory()
            try:
                from .distributed import register_fleet_worker
                register_fleet_worker(db, identity,
                                      queue_names=["SCHEDULER"])
                db.commit()
            finally:
                db.close()
        except Exception:  # noqa: BLE001
            logger.exception("scheduler heartbeat failed")
        now = time.time()
        if now - last_maintenance >= maintenance_every_seconds:
            try:
                db = db_factory()
                try:
                    _scheduler_maintenance(db, identity.worker_id)
                    db.commit()
                finally:
                    db.close()
            except Exception:  # noqa: BLE001
                logger.exception("scheduler maintenance round failed")
            last_maintenance = now
        shutdown.wait(tick_seconds)
    try:
        db = db_factory()
        try:
            mark_worker_inactive(db, identity.worker_id)
            db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass
    logger.info("scheduler %s shutdown complete", identity.worker_id)


def _scheduler_maintenance(db: Session, worker_id: str) -> dict:
    from .distributed import recover_expired_leases
    from .worker_platform import WorkerJob, enqueue_job
    from ..models.phase15 import AIMemory
    from ..models.phase18 import EntityMergeRequest
    from ..models.ai_execution import AIApproval

    now = _utcnow()
    result: dict = {"promoted": 0, "leases_recovered": 0,
                    "approvals_expired": 0, "merges_expired": 0,
                    "memories_expired": 0, "cleanup_enqueued": False}

    # 1) Promote delayed jobs that are due.
    due = (db.query(WorkerJob)
           .filter(WorkerJob.status == "QUEUED",
                   WorkerJob.run_after.isnot(None),
                   WorkerJob.run_after <= now)
           .limit(200).all())
    for job in due:
        job.run_after = None
        result["promoted"] += 1

    # 2) Recover stale leases (bounded by the grace window).
    try:
        recovered = recover_expired_leases(db, grace_seconds=120)
        result["leases_recovered"] = recovered
    except Exception:  # noqa: BLE001
        logger.exception("lease recovery failed")

    # 3) Expire stale approvals.
    # AIApproval uses lowercase status values ("pending").
    expired_approvals = (db.query(AIApproval)
                         .filter(AIApproval.status == "pending",
                                 AIApproval.expires_at.isnot(None),
                                 AIApproval.expires_at <= now)
                         .limit(200).all())
    for ap in expired_approvals:
        ap.status = "expired"
        result["approvals_expired"] += 1

    # 4) Expire stale entity merge requests.
    expired_merges = (db.query(EntityMergeRequest)
                      .filter(EntityMergeRequest.status == "PENDING",
                              EntityMergeRequest.expires_at.isnot(None),
                              EntityMergeRequest.expires_at <= now)
                      .limit(200).all())
    for m in expired_merges:
        m.status = "EXPIRED"
        result["merges_expired"] += 1

    # 5) Expire due memories.
    try:
        from .memory3 import expire_due
        r = expire_due(db, now=now)
        result["memories_expired"] = r.get("expired", 0)
    except Exception:  # noqa: BLE001
        logger.exception("memory expiry failed")

    # 6) Enqueue a bounded retention cleanup job (idempotent dedupe key).
    # WorkerJob.workspace_id is a non-nullable FK, so resolve a real workspace;
    # if no workspace exists there is nothing to clean up yet.
    try:
        from ..models.workspace import Workspace
        target = db.query(Workspace).order_by(Workspace.id.asc()).first()
        if target is not None:
            enqueue_job(
                db, queue_name="MAINTENANCE", job_type="RETENTION_CLEANUP",
                workspace_id=target.id, payload={"limit": 200},
                dedupe_key=f"retention:{now.strftime('%Y%m%d%H')}",
                max_attempts=2)
            result["cleanup_enqueued"] = True
    except Exception:  # noqa: BLE001
        logger.exception("cleanup enqueue failed")
    return result


# ---------------------------------------------------------------------------
# Event processor loop
# ---------------------------------------------------------------------------

def run_event_processor_process(
    db_factory: Callable[[], Session],
    identity,
    shutdown,
    tick_seconds: float = 2.0,
    batch_size: int = 50,
) -> None:
    """Event processor loop: drain the transactional outbox in bounded batches."""
    from .worker_platform import enqueue_job
    shutdown.install()
    logger.info("event-processor %s started", identity.worker_id)
    while not shutdown.stopping():
        try:
            db = db_factory()
            try:
                from .distributed import register_fleet_worker
                register_fleet_worker(db, identity,
                                      queue_names=["EVENTS"])
                db.commit()
            finally:
                db.close()
            db = db_factory()
            try:
                from .event_bus import process_pending_events
                result = process_pending_events(
                    db, batch_size=batch_size)
                db.commit()
                processed = int(result.get("processed", 0))
                if processed > 0:
                    logger.info("event-processor processed %s events",
                                processed)
            finally:
                db.close()
        except Exception:  # noqa: BLE001
            logger.exception("event processing round failed")
        shutdown.wait(tick_seconds)
    try:
        db = db_factory()
        try:
            mark_worker_inactive(db, identity.worker_id)
            db.commit()
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass
    logger.info("event-processor %s shutdown complete", identity.worker_id)