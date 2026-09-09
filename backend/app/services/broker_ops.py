"""Phase 18 broker operations — production hardening.

Covers broker health (connection status, latency, queue depth, failures),
failover semantics (no silent job loss), delayed-job support, and the
Redis-adapter contract used when ``WORKER_BROKER=redis`` is configured.

The PostgreSQL broker remains the durability default; the Redis adapter is
exercised through an in-memory mock in tests so no Redis server is required.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

BROKER_NAMES = ("postgres", "redis")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def configured_broker() -> str:
    return os.getenv("WORKER_BROKER", "postgres").lower()


def validate_broker_config() -> None:
    """Fail fast on invalid broker configuration at startup."""
    broker = configured_broker()
    if broker not in BROKER_NAMES:
        raise RuntimeError(
            f"invalid WORKER_BROKER={broker!r}; expected postgres|redis")


# ---------------------------------------------------------------------------
# Broker health
# ---------------------------------------------------------------------------

def broker_health(db: Session) -> dict:
    """Safe health snapshot of the configured broker + queue depth."""
    from .broker import get_broker, BrokerUnavailable
    from .worker_platform import WorkerJob
    broker = configured_broker()
    health: dict = {
        "broker": broker,
        "status": "UP",
        "latency_ms": None,
        "queue_depth": 0,
        "errors": [],
    }
    if broker == "postgres":
        started = time.time()
        try:
            depth = db.query(WorkerJob).filter(
                WorkerJob.status.in_(("QUEUED", "CLAIMED", "RUNNING"))).count()
            health["queue_depth"] = depth
            health["latency_ms"] = round((time.time() - started) * 1000, 2)
        except Exception as exc:  # noqa: BLE001
            health["status"] = "DOWN"
            health["errors"].append(f"{type(exc).__name__}: {exc}")
        return health
    # Redis path — only when actually configured; probe cheaply.
    try:
        backend = get_broker("redis")
        started = time.time()
        ok = backend.ping()
        health["latency_ms"] = round((time.time() - started) * 1000, 2)
        if not ok:
            health["status"] = "DOWN"
    except BrokerUnavailable as exc:
        health["status"] = "DOWN"
        health["errors"].append(str(exc))
    except Exception as exc:  # noqa: BLE001
        health["status"] = "DOWN"
        health["errors"].append(f"{type(exc).__name__}: {exc}")
    return health


def queue_depth(db: Session, queue_name: Optional[str] = None,
                now: Optional[datetime] = None) -> dict:
    """Per-queue depth + oldest-job age (autoscaling signal)."""
    from .worker_platform import WorkerJob
    now = now or _utcnow()
    q = db.query(WorkerJob)
    if queue_name:
        q = q.filter(WorkerJob.queue_name == queue_name)
    rows = q.filter(WorkerJob.status.in_(("QUEUED", "CLAIMED", "RUNNING"))).all()
    per_queue: dict = {}
    oldest_age_s = None
    for job in rows:
        entry = per_queue.setdefault(job.queue_name, {"depth": 0, "oldest_age_s": None})
        entry["depth"] += 1
        created = job.created_at
        if created is not None:
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age = max(0.0, (now - created).total_seconds())
            if entry["oldest_age_s"] is None or age > entry["oldest_age_s"]:
                entry["oldest_age_s"] = round(age, 1)
            if oldest_age_s is None or age > oldest_age_s:
                oldest_age_s = round(age, 1)
    return {"per_queue": per_queue, "oldest_job_age_s": oldest_age_s}


# ---------------------------------------------------------------------------
# Delayed jobs / failover semantics
# ---------------------------------------------------------------------------

def promote_due_delayed(db: Session, limit: int = 200,
                        now: Optional[datetime] = None) -> int:
    """Promote delayed jobs whose run_after time has arrived.

    Called by the scheduler loop; idempotent and bounded.
    """
    from .worker_platform import WorkerJob
    now = now or _utcnow()
    due = (db.query(WorkerJob)
           .filter(WorkerJob.status == "QUEUED",
                   WorkerJob.run_after.isnot(None),
                   WorkerJob.run_after <= now)
           .limit(limit).all())
    for job in due:
        job.run_after = None
    return len(due)


def broker_failover_policy() -> dict:
    """Declarative failover semantics — never silently drop jobs."""
    return {
        "primary": configured_broker(),
        "on_unavailable": "stop_claiming_and_alert",  # never drop
        "requeue_on_reconnect": True,
        "visibility_timeout_s": int(os.getenv("WORKER_VISIBILITY_TIMEOUT", "300")),
        "dead_letter_after_attempts": int(os.getenv("WORKER_MAX_ATTEMPTS", "3")),
    }


# ---------------------------------------------------------------------------
# Redis adapter hardening helpers (contract-level, no server required)
# ---------------------------------------------------------------------------

class BrokerContract:
    """The interface every broker backend must satisfy.

    Used by tests to verify PostgresBroker / RedisBroker / fake adapters
    conform; production code never depends on a specific backend.
    """

    REQUIRED_METHODS = (
        "enqueue", "claim", "acknowledge", "retry", "dead_letter",
        "heartbeat", "cancel", "close",
    )

    @classmethod
    def verify(cls, backend) -> list[str]:
        missing = [m for m in cls.REQUIRED_METHODS
                   if not callable(getattr(backend, m, None))]
        return missing


def redis_available() -> bool:
    try:
        import redis  # noqa: F401
        return True
    except ImportError:
        return False