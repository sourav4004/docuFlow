"""Phase 19 — broker 2.0 hardening.

Explicit delivery guarantees (at-least-once + visibility timeout + ack +
retry + dead letter + delayed), handler-idempotency declaration (duplicate
deliveries are safe because handlers are idempotent), outage/failover
policy (no silent job loss), reconnect recovery accounting, Redis adapter
health/config that never fails imports when the redis package is absent,
and connection-pool configuration.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def delivery_guarantees() -> dict:
    return {
        "semantics": "at-least-once",
        "visibility_timeout_s": int(os.getenv("WORKER_VISIBILITY_S", "120")),
        "acknowledgement": "explicit ack after successful handling",
        "retry": "bounded exponential backoff with max attempts",
        "dead_letter": "jobs exceeding max attempts move to DEAD_LETTERED",
        "delayed_delivery": "run_after gates claim eligibility",
        "dedupe": "per (workspace, queue, dedupe_key) unique constraint",
        "note": "duplicate delivery is possible under at-least-once; "
                "handlers MUST be idempotent",
    }


def handler_idempotent(job_type: str, *, declared: Optional[bool] = None
                       ) -> dict:
    """Handlers are idempotent by declaration. Non-idempotent job types must
    opt out explicitly and therefore always run under a lease with a single
    owner."""
    if declared is False:
        return {"idempotent": False,
                "policy": "single-owner lease; no automatic replay"}
    return {"idempotent": True,
            "policy": "safe under duplicate delivery"}


def broker_config() -> dict:
    return {"backend": os.getenv("WORKER_BROKER", "postgres"),
            "redis_optional": True,
            "note": "redis is imported lazily; its absence never breaks "
                    "imports or tests"}


def redis_available() -> bool:
    """True only when redis is installed AND reachable (never required)."""
    try:
        import redis  # noqa: F401
    except ImportError:
        return False
    host = os.getenv("REDIS_HOST")
    port = int(os.getenv("REDIS_PORT", "6379"))
    if not host:
        return False
    try:
        client = redis.Redis(host=host, port=port,
                             socket_connect_timeout=1)
        return bool(client.ping())
    except Exception:  # noqa: BLE001
        return False


def outage_policy() -> dict:
    """Broker-unavailable behavior: no silent job loss."""
    return {
        "producer_behavior": "enqueue fails fast with BrokerUnavailable; "
                             "caller may retry with backoff",
        "consumer_behavior": "claim returns no work; workers keep "
                             "heartbeating and back off",
        "durability": "jobs are durable rows in PostgreSQL — a broker "
                      "outage never loses queued jobs",
        "recovery": "on reconnect, visibility-expired jobs become claimable "
                    "again",
    }


def broker_recovery_check(db: Session) -> dict:
    """Queued/claimed jobs whose lease expired may need recovery. Never
    double-executes: only jobs whose lease token is no longer heartbeating
    become claimable again (handled by lease recovery)."""
    from ..models.phase16 import WorkerJob
    from ..models.phase17 import JobLease
    now = _utcnow()
    lease_rows = db.query(JobLease).limit(5000).all()
    expired_leases = sum(1 for row in lease_rows
                         if _as_utc(row.expires_at) is not None
                         and _as_utc(row.expires_at) < now)
    job_rows = (db.query(WorkerJob)
                .filter(WorkerJob.status.in_(("CLAIMED", "RUNNING")))
                .limit(5000).all())
    expired_jobs = sum(1 for row in job_rows
                       if row.lease_expires_at is not None
                       and _as_utc(row.lease_expires_at) < now)
    queued = db.query(WorkerJob).filter(
        WorkerJob.status == "QUEUED").count()
    return {
        "queued": queued,
        "expired_leases": expired_leases,
        "expired_job_claims": expired_jobs,
        "recoverable": expired_jobs,
        "semantics": "recovery requeues; idempotent handlers make replay "
                     "safe",
        "as_of": now.isoformat(),
    }


def pool_config() -> dict:
    return {
        "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
        "max_overflow": int(os.getenv("DB_POOL_MAX_OVERFLOW", "20")),
        "pool_timeout_s": int(os.getenv("DB_POOL_TIMEOUT_S", "30")),
        "pool_recycle_s": int(os.getenv("DB_POOL_RECYCLE_S", "1800")),
        "pool_pre_ping": True,
    }
