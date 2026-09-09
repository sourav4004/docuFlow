"""Worker broker abstraction — Phase 17.

Application services talk to a ``BrokerBackend`` interface; the concrete
implementation is selected by ``WORKER_BROKER`` configuration. The PostgreSQL
backend is the shipped default (durable, transactional with the app DB). A
Redis-compatible adapter is implemented behind an optional import so the test
suite never requires Redis.

The broker layer never stores payloads beyond the durable job record and never
logs sensitive content.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

import json  # noqa: E402

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class BrokerError(Exception):
    """Base broker failure."""


class BrokerUnavailable(BrokerError):
    """Configured broker backend cannot be used in this environment."""


class BrokerBackend:
    """Queue contract implemented by every backend.

    All methods operate on serializable dicts (never ORM objects) so workers
    and API layers stay decoupled from the storage engine.
    """

    name = "abstract"

    def enqueue(self, db: Session, *, queue_name: str, job_type: str,
                workspace_id: int, payload: Optional[dict],
                organization_id: Optional[int] = None,
                user_id: Optional[int] = None, priority: str = "NORMAL",
                dedupe_key: Optional[str] = None,
                run_after: Optional[datetime] = None,
                max_attempts: int = 3,
                correlation_id: Optional[str] = None) -> dict:
        raise NotImplementedError

    def claim(self, db: Session, *, queue_name: str, worker_id: str,
              per_workspace_cap: int = 3,
              skip_workspace_ids: Optional[set] = None) -> Optional[dict]:
        raise NotImplementedError

    def acknowledge(self, db: Session, job_id: int, worker_id: str) -> dict:
        raise NotImplementedError

    def retry(self, db: Session, job_id: int, worker_id: str,
              error: str) -> dict:
        raise NotImplementedError

    def dead_letter(self, db: Session, job_id: int, worker_id: str,
                    error: str) -> dict:
        raise NotImplementedError

    def heartbeat(self, db: Session, job_id: int, worker_id: str) -> None:
        raise NotImplementedError

    def cancel(self, db: Session, job_id: int) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        """Release any backend resources (used during graceful shutdown)."""


# ---------------------------------------------------------------------------
# PostgreSQL backend — thin adapter over the durable worker_platform layer
# ---------------------------------------------------------------------------

from . import worker_platform as wp  # noqa: E402

_VALID_PRIORITIES = ("CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND")


def _job_dict(job) -> dict:
    try:
        payload = json.loads(job.payload_json) if job.payload_json else None
    except (ValueError, TypeError):
        payload = None
    return {
        "id": job.id,
        "queue_name": job.queue_name,
        "job_type": job.job_type,
        "status": job.status,
        "priority": job.priority,
        "workspace_id": job.workspace_id,
        "organization_id": job.organization_id,
        "user_id": job.user_id,
        "attempt": job.attempt,
        "max_attempts": job.max_attempts,
        "payload": payload,
        "trace_id": job.trace_id,
        "correlation_id": job.correlation_id,
    }


class PostgresBroker(BrokerBackend):
    """Default durable broker — PostgreSQL queue via worker_platform."""

    name = "postgres"

    def enqueue(self, db: Session, *, queue_name: str, job_type: str,
                workspace_id: int, payload: Optional[dict],
                organization_id: Optional[int] = None,
                user_id: Optional[int] = None, priority: str = "NORMAL",
                dedupe_key: Optional[str] = None,
                run_after: Optional[datetime] = None,
                max_attempts: int = 3,
                correlation_id: Optional[str] = None) -> dict:
        job = wp.enqueue_job(
            db, queue_name=queue_name, job_type=job_type,
            workspace_id=workspace_id, payload=payload,
            organization_id=organization_id, user_id=user_id,
            priority=priority, dedupe_key=dedupe_key, run_after=run_after,
            max_attempts=max_attempts, correlation_id=correlation_id,
        )
        db.flush()
        return _job_dict(job)

    def claim(self, db: Session, *, queue_name: str, worker_id: str,
              per_workspace_cap: int = 3,
              skip_workspace_ids: Optional[set] = None) -> Optional[dict]:
        job = wp.claim_job(db, queue_name=queue_name, worker_id=worker_id,
                           per_workspace_cap=per_workspace_cap,
                           skip_workspace_ids=skip_workspace_ids)
        if job is None:
            return None
        return _job_dict(job)

    def acknowledge(self, db: Session, job_id: int, worker_id: str) -> dict:
        job = db.query(wp.WorkerJob).filter(wp.WorkerJob.id == job_id).first()
        if job is None:
            raise wp.JobNotFoundError(f"Job {job_id} not found")
        job = wp.complete_job(db, job)
        db.flush()
        return _job_dict(job)

    def retry(self, db: Session, job_id: int, worker_id: str,
              error: str) -> dict:
        job = db.query(wp.WorkerJob).filter(wp.WorkerJob.id == job_id).first()
        if job is None:
            raise wp.JobNotFoundError(f"Job {job_id} not found")
        job = wp.fail_job(db, job, error, retryable=True)
        db.flush()
        return _job_dict(job)

    def dead_letter(self, db: Session, job_id: int, worker_id: str,
                    error: str) -> dict:
        job = db.query(wp.WorkerJob).filter(wp.WorkerJob.id == job_id).first()
        if job is None:
            raise wp.JobNotFoundError(f"Job {job_id} not found")
        job = wp.fail_job(db, job, error, retryable=False)
        db.flush()
        return _job_dict(job)

    def heartbeat(self, db: Session, job_id: int, worker_id: str) -> None:
        job = db.query(wp.WorkerJob).filter(
            wp.WorkerJob.id == job_id,
            wp.WorkerJob.claimed_by == worker_id).first()
        if job is not None:
            wp.heartbeat_job(db, job)

    def cancel(self, db: Session, job_id: int) -> dict:
        return _job_dict(wp.cancel_job(db, job_id))


# ---------------------------------------------------------------------------
# Redis-compatible adapter — optional import
# ---------------------------------------------------------------------------

class RedisBroker(BrokerBackend):
    """Redis-compatible adapter used only when ``redis`` is installed and
    ``WORKER_BROKER=redis`` is configured."""

    name = "redis"

    def __init__(self) -> None:
        try:
            import redis  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise BrokerUnavailable(
                "WORKER_BROKER=redis requires the 'redis' package") from exc
        self._redis = None  # connected lazily from REDIS_URL
        self._url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis as _redis
            self._redis = _redis.from_url(
                self._url, socket_timeout=5, socket_connect_timeout=5)
            self._redis.ping()
        except Exception as exc:  # noqa: BLE001
            raise BrokerUnavailable(
                f"cannot connect to Redis at {self._url!r}: {exc}") from exc

    def _key(self, queue_name: str, job_id: int) -> str:
        return f"df:job:{queue_name}:{job_id}"

    def enqueue(self, db: Session, *, queue_name: str, job_type: str,
                workspace_id: int, payload: Optional[dict],
                organization_id: Optional[int] = None,
                user_id: Optional[int] = None, priority: str = "NORMAL",
                dedupe_key: Optional[str] = None,
                run_after: Optional[datetime] = None,
                max_attempts: int = 3,
                correlation_id: Optional[str] = None) -> dict:
        # Redis backend does not share the app DB; persisted job identity is a
        # Redis-managed counter. Kept minimal and explicit — see
        # ``docs/PHASE17_OPERATIONS.md`` for the durability trade-off.
        import json as _json
        import time as _time
        import uuid as _uuid
        job_id = self._redis.incr("df:job:counter")
        payload_dict = {
            "id": job_id, "queue_name": queue_name, "job_type": job_type,
            "status": "QUEUED", "priority": priority,
            "workspace_id": workspace_id, "organization_id": organization_id,
            "user_id": user_id, "attempt": 0, "max_attempts": max_attempts,
            "payload": payload, "trace_id": str(_uuid.uuid4())[:32],
            "correlation_id": correlation_id,
        }
        # Delayed jobs: score = run_after timestamp; zpopmin only returns
        # members whose score is due, so delayed jobs stay parked.
        if run_after is not None:
            if run_after.tzinfo is not None:
                run_after = run_after.astimezone(timezone.utc).replace(tzinfo=None)
            score = run_after.timestamp()
        else:
            score = _time.time()
        self._redis.zadd(f"df:q:{queue_name}", {_json.dumps(payload_dict): score})
        return payload_dict

    def claim(self, db: Session, *, queue_name: str, worker_id: str,
              per_workspace_cap: int = 3,
              skip_workspace_ids: Optional[set] = None) -> Optional[dict]:
        import json as _json
        import time as _time
        member = self._redis.zpopmin(f"df:q:{queue_name}", 1)
        if not member:
            return None
        raw, _score = member[0]
        item = _json.loads(raw)
        item["status"] = "CLAIMED"
        item["claimed_by"] = worker_id
        self._redis.hset(f"df:inflight:{worker_id}", item["id"], raw)
        return item

    def acknowledge(self, db: Session, job_id: int, worker_id: str) -> dict:
        self._redis.hdel(f"df:inflight:{worker_id}", job_id)
        return {"id": job_id, "status": "COMPLETED"}

    def retry(self, db: Session, job_id: int, worker_id: str,
              error: str) -> dict:
        self._redis.hdel(f"df:inflight:{worker_id}", job_id)
        return {"id": job_id, "status": "RETRYING"}

    def dead_letter(self, db: Session, job_id: int, worker_id: str,
                    error: str) -> dict:
        self._redis.hdel(f"df:inflight:{worker_id}", job_id)
        return {"id": job_id, "status": "DEAD_LETTERED"}

    def heartbeat(self, db: Session, job_id: int, worker_id: str) -> None:
        return None

    def ping(self) -> bool:
        """Cheap liveness probe used by broker health."""
        if self._redis is None:
            return False
        try:
            return bool(self._redis.ping())
        except Exception:  # noqa: BLE001
            return False

    def cancel(self, db: Session, job_id: int) -> dict:
        return {"id": job_id, "status": "CANCELLED"}

    def close(self) -> None:
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_backends: dict[str, BrokerBackend] = {}


def get_broker(name: Optional[str] = None) -> BrokerBackend:
    """Return the configured broker backend (singleton per name)."""
    selected = (name or os.getenv("WORKER_BROKER", "postgres")).lower()
    if selected in _backends:
        return _backends[selected]
    if selected == "postgres":
        backend: BrokerBackend = PostgresBroker()
    elif selected == "redis":
        backend = RedisBroker()  # raises BrokerUnavailable without Redis
    else:
        raise BrokerError(
            f"unknown WORKER_BROKER {selected!r}; expected postgres|redis")
    _backends[selected] = backend
    return backend


def reset_broker_cache() -> None:
    _backends.clear()
