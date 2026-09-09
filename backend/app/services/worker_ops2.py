"""Phase 19 — worker platform 2.0 + scheduler 2.0.

Self-healing worker platform: health scoring, quarantine with operator
override and automatic recovery criteria, deterministic load/capacity
models, tenant-fair placement, safe job migration (lease-safe + audited),
queue backpressure with structured decisions, and autoscaling signals.

Scheduler 2.0: database-backed leader election with lease/heartbeat/takeover,
deterministic scheduled-job deduplication, and crash/reconnect recovery.

Everything is deterministic and DB-only — no OS-specific metrics leak into
decisions and no tests depend on machine introspection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

LEASE_GRACE_SECONDS = 90


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _recent_heartbeat(db: Session, worker_id: str, grace: int =
                      LEASE_GRACE_SECONDS):
    from ..models.phase16 import WorkerHeartbeat
    hb = (db.query(WorkerHeartbeat)
          .filter(WorkerHeartbeat.worker_id == worker_id)
          .order_by(WorkerHeartbeat.last_heartbeat.desc()).first())
    if hb is None or hb.last_heartbeat is None:
        return None
    if _utcnow() - _as_utc(hb.last_heartbeat) > timedelta(seconds=grace):
        return None
    return hb


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------

def quarantine_worker(db: Session, *, worker_id: str,
                      reason: Optional[str] = None,
                      failure_count: int = 1,
                      auto_recover_after_minutes: Optional[int] = None,
                      criteria: Optional[dict] = None):
    """Quarantine a repeatedly failing worker. Idempotent per worker."""
    from ..models.phase19 import WorkerQuarantine
    row = (db.query(WorkerQuarantine)
           .filter(WorkerQuarantine.worker_id == worker_id).first())
    if row is None:
        row = WorkerQuarantine(worker_id=worker_id)
        db.add(row)
    row.reason = reason or row.reason or "repeated failures"
    row.failure_count = failure_count
    row.status = "QUARANTINED"
    if auto_recover_after_minutes is not None:
        row.auto_recover_after = (_utcnow()
                                  + timedelta(minutes=auto_recover_after_minutes))
    row.recovery_criteria_json = json.dumps(criteria or {})
    row.updated_at = _utcnow()
    db.flush()
    return row


def operator_override(db: Session, *, worker_id: str,
                      operator_user_id: int,
                      note: Optional[str] = None) -> dict:
    """Operator override releases a quarantine for a specific worker."""
    from ..models.phase19 import WorkerQuarantine
    row = (db.query(WorkerQuarantine)
           .filter(WorkerQuarantine.worker_id == worker_id).first())
    if row is None:
        return {"status": "NOT_QUARANTINED", "worker_id": worker_id}
    if row.status == "OPERATOR_OVERRIDE":
        return {"status": "ALREADY_OVERRIDDEN", "worker_id": worker_id}
    row.status = "OPERATOR_OVERRIDE"
    row.operator_user_id = operator_user_id
    row.reason = note or row.reason
    row.updated_at = _utcnow()
    db.flush()
    return {"status": "OVERRIDDEN", "worker_id": worker_id}


def auto_recover_quarantines(db: Session, limit: int = 100) -> dict:
    """Recover quarantines whose auto-recovery time has arrived."""
    from ..models.phase19 import WorkerQuarantine
    now = _utcnow()
    candidates = (db.query(WorkerQuarantine)
                  .filter(WorkerQuarantine.status == "QUARANTINED",
                          WorkerQuarantine.auto_recover_after.isnot(None))
                  .limit(min(limit * 2, 1000)).all())
    due = [row for row in candidates
           if _as_utc(row.auto_recover_after) <= now]
    for row in due:
        row.status = "AUTO_RECOVERED"
        row.updated_at = now
    db.flush()
    return {"recovered": len(due)}


def quarantine_status(db: Session, worker_id: str) -> dict:
    from ..models.phase19 import WorkerQuarantine
    row = (db.query(WorkerQuarantine)
           .filter(WorkerQuarantine.worker_id == worker_id).first())
    if row is None:
        return {"quarantined": False, "worker_id": worker_id}
    blocked = row.status == "QUARANTINED" and (
        row.auto_recover_after is None
        or _as_utc(row.auto_recover_after) > _utcnow())
    return {"quarantined": blocked, "worker_id": worker_id,
            "status": row.status, "reason": row.reason,
            "failure_count": row.failure_count,
            "auto_recover_after": row.auto_recover_after}


def list_quarantines(db: Session, limit: int = 100) -> list:
    from ..models.phase19 import WorkerQuarantine
    return (db.query(WorkerQuarantine)
            .order_by(WorkerQuarantine.created_at.desc())
            .limit(min(limit, 500)).all())


# ---------------------------------------------------------------------------
# Health scoring / load scoring
# ---------------------------------------------------------------------------

def worker_health_score(db: Session, worker_id: str) -> float:
    """Deterministic 0..1 health: heartbeat freshness, job success rate,
    quarantine state."""
    from ..models.phase16 import WorkerJob
    hb = _recent_heartbeat(db, worker_id)
    if hb is None:
        return 0.0
    score = 1.0
    since = _utcnow() - timedelta(hours=1)
    jobs = (db.query(WorkerJob)
            .filter(WorkerJob.claimed_by == worker_id)
            .limit(1000).all())
    jobs = [j for j in jobs
            if j.created_at is not None
            and _as_utc(j.created_at) >= since]
    failed = sum(1 for j in jobs
                 if j.status in ("FAILED", "DEAD_LETTERED"))
    total = len(jobs)
    if total >= 3 and (failed / total) > 0.5:
        score -= 0.5
    qs = quarantine_status(db, worker_id)
    if qs["quarantined"]:
        score = 0.0
    return round(max(0.0, min(1.0, score)), 3)


def worker_load_score(db: Session, worker_id: str) -> dict:
    """Load inputs: active jobs, queue depth assigned, capacity utilization,
    recent error rate, latency. Deterministic; never OS-specific."""
    from ..models.phase16 import WorkerHeartbeat, WorkerJob
    from ..services.runtime import worker_capacity
    hb = _recent_heartbeat(db, worker_id)
    if hb is None:
        return {"worker_id": worker_id, "score": 1.0, "live": False}
    capacity = worker_capacity()
    cap = int(os.getenv("WORKER_CONCURRENCY", str(capacity["concurrency"])))
    cap = max(1, cap)
    active = max(0, int(hb.active_jobs or 0))
    utilization = min(active / cap, 1.0)
    since = _utcnow() - timedelta(minutes=15)
    recent = (db.query(WorkerJob)
              .filter(WorkerJob.claimed_by == worker_id)
              .limit(500).all())
    recent = [j for j in recent
              if j.completed_at is not None
              and _as_utc(j.completed_at) >= since]
    failed = sum(1 for j in recent if j.status in ("FAILED",
                                                   "DEAD_LETTERED"))
    error_rate = (failed / len(recent)) if recent else 0.0
    score = round(min(1.0, utilization * 0.6 + error_rate * 0.4), 3)
    return {"worker_id": worker_id, "score": score, "live": True,
            "active_jobs": active, "capacity": cap,
            "utilization": round(utilization, 3),
            "error_rate": round(error_rate, 3)}


def capacity_model(db: Session) -> dict:
    """Capacity: global concurrency, per-queue caps, per-tenant limits."""
    from ..services.runtime import worker_capacity
    caps = worker_capacity()
    return {
        "global_concurrency": caps["concurrency"],
        "queue_caps": caps["queue_caps"],
        "tenant_limit_active_jobs": int(
            os.getenv("WORKER_TENANT_LIMIT", "50")),
        "provider_capacity_ratio": float(
            os.getenv("PROVIDER_CAPACITY_RATIO", "0.9")),
        "live_workers": _live_worker_count(db),
    }


def _live_worker_count(db: Session) -> int:
    from ..models.phase16 import WorkerHeartbeat
    now = _utcnow()
    hbs = db.query(WorkerHeartbeat).limit(2000).all()
    return sum(1 for hb in hbs
               if hb.last_heartbeat is not None
               and now - _as_utc(hb.last_heartbeat)
               < timedelta(seconds=LEASE_GRACE_SECONDS))


# ---------------------------------------------------------------------------
# Placement + migration + backpressure
# ---------------------------------------------------------------------------

def placement_decision(db: Session, *, workspace_id: int,
                       queue_name: str, priority: str = "NORMAL",
                       required_capability: Optional[str] = None) -> dict:
    """Intelligent placement: tenant-fair, quarantine-aware, least-loaded.

    Deterministic: candidates are live, non-quarantined workers sorted by
    (queue assignment match desc, load asc, worker id asc). Never chooses a
    quarantined worker.
    """
    from ..models.phase16 import WorkerHeartbeat, WorkerJob
    from ..services.runtime import worker_capacity
    now = _utcnow()
    candidates = []
    hbs = (db.query(WorkerHeartbeat)
           .filter(WorkerHeartbeat.status.in_(
               ("RUNNING", "IDLE", "STARTING")))
           .limit(2000).all())
    live = [hb for hb in hbs
            if hb.last_heartbeat is not None
            and now - _as_utc(hb.last_heartbeat)
            < timedelta(seconds=LEASE_GRACE_SECONDS)]
    caps = worker_capacity()
    for hb in live:
        if quarantine_status(db, hb.worker_id)["quarantined"]:
            continue
        load = worker_load_score(db, hb.worker_id)["score"]
        assignments = [q.strip().upper() for q in
                       (hb.queue_assignments or "").split(",") if q.strip()]
        queue_cap = caps["queue_caps"].get(queue_name.upper(),
                                           caps["concurrency"])
        if int(hb.active_jobs or 0) >= max(1, queue_cap):
            continue
        matches = int(queue_name.upper() in assignments or not assignments)
        candidates.append((matches, load, hb.worker_id))
    if not candidates:
        return {"decision": "DEFER", "reason": "no eligible worker",
                "workspace_id": workspace_id}
    # tenant fairness — bound active jobs already claimed per workspace
    tenant_limit = int(os.getenv("WORKER_TENANT_LIMIT", "50"))
    active = (db.query(WorkerJob)
              .filter(WorkerJob.workspace_id == workspace_id,
                      WorkerJob.status.in_(("CLAIMED", "RUNNING")))
              .count())
    if active >= tenant_limit:
        return {"decision": "DEFER",
                "reason": "tenant capacity reached",
                "workspace_id": workspace_id}
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    return {"decision": "ASSIGN", "worker_id": candidates[0][2],
            "reason": "least-loaded eligible worker",
            "workspace_id": workspace_id, "eligible": len(candidates)}


def migrate_job(db: Session, *, job_id: int, reason: str,
                target_worker: Optional[str] = None) -> dict:
    """Safely migrate a job away from an unhealthy worker.

    Lease-safe: the explicit JobLease is released first, the claim fields on
    the job are cleared, and the job returns to QUEUED (or is directly handed
    to ``target_worker``). Duplicate execution is prevented by worker-level
    lease recovery rules: only the holder of the lease may run a job.
    """
    from ..models.phase16 import WorkerJob
    from ..models.phase17 import JobLease
    from ..models.phase19 import JobMigrationRecord
    job = db.query(WorkerJob).get(job_id)
    if job is None:
        raise ValueError("job not found")
    from_worker = job.claimed_by
    db.query(JobLease).filter(JobLease.job_id == job_id).delete()
    job.claimed_by = None
    job.heartbeat_at = None
    job.lease_expires_at = None
    job.lease_token = None
    if target_worker:
        job.claimed_by = target_worker
        job.lease_expires_at = _utcnow() + timedelta(seconds=60)
        job.status = "CLAIMED"
    else:
        job.status = "QUEUED"
        job.next_retry_at = _utcnow()
    db.add(JobMigrationRecord(job_id=job_id, from_worker=from_worker,
                              to_worker=target_worker, reason=reason,
                              status="MIGRATED"))
    db.flush()
    return {"job_id": job_id, "from_worker": from_worker,
            "to_worker": target_worker, "status": job.status,
            "migrated": True}


def backpressure_decision(db: Session, *, workspace_id: int,
                          queue_name: str,
                          estimated_cost: Optional[float] = None,
                          budget_remaining: Optional[float] = None) -> dict:
    """Queue backpressure: global, tenant, provider, budget, worker health.

    Returns a structured decision: ACCEPT / DEFER / REJECT (with a stable
    ``code`` so callers can return structured errors).
    """
    depth = _queue_depth(db, queue_name)
    caps = capacity_model(db)
    global_cap = int(os.getenv("WORKER_QUEUE_MAX_DEPTH", "10000"))
    if depth >= global_cap:
        return {"decision": "REJECT", "code": "REJECTED_BACKPRESSURE",
                "reason": "global queue depth at capacity",
                "queue": queue_name, "depth": depth}
    tenant_active = _tenant_active(db, workspace_id)
    tenant_limit = caps["tenant_limit_active_jobs"]
    if tenant_active >= tenant_limit:
        return {"decision": "DEFER", "code": "BACKPRESSURE_TENANT_LIMIT",
                "reason": "tenant active-job limit reached",
                "active": tenant_active, "limit": tenant_limit}
    if budget_remaining is not None and budget_remaining < 0:
        return {"decision": "REJECT", "code": "BUDGET_EXHAUSTED",
                "reason": "budget exhausted", "budget_remaining":
                budget_remaining}
    live = _live_worker_count(db)
    if live == 0:
        return {"decision": "DEFER", "code": "BACKPRESSURE_NO_WORKERS",
                "reason": "no live workers"}
    return {"decision": "ACCEPT", "code": None, "reason": "capacity available",
            "queue": queue_name, "depth": depth}


def _queue_depth(db: Session, queue_name: str) -> int:
    from ..models.phase16 import WorkerJob
    return (db.query(WorkerJob)
            .filter(WorkerJob.queue_name == queue_name,
                    WorkerJob.status.in_(("QUEUED", "CLAIMED", "RUNNING")))
            .count())


def _tenant_active(db: Session, workspace_id: int) -> int:
    from ..models.phase16 import WorkerJob
    return (db.query(WorkerJob)
            .filter(WorkerJob.workspace_id == workspace_id,
                    WorkerJob.status.in_(("QUEUED", "CLAIMED", "RUNNING")))
            .count())


def autoscale_signals(db: Session) -> dict:
    """Metrics for external autoscaling (never auto-provisions)."""
    from ..models.phase16 import WorkerJob, WorkerHeartbeat
    now = _utcnow()
    since = now - timedelta(minutes=5)
    queued = db.query(WorkerJob).filter(
        WorkerJob.status == "QUEUED").count()
    running = db.query(WorkerJob).filter(
        WorkerJob.status.in_(("CLAIMED", "RUNNING"))).count()
    done = sum(1 for j in
               (db.query(WorkerJob)
                .filter(WorkerJob.completed_at.isnot(None))
                .limit(2000).all())
               if _as_utc(j.completed_at) and _as_utc(j.completed_at)
               >= since)
    failed = sum(1 for j in
                 (db.query(WorkerJob)
                  .filter(WorkerJob.status == "FAILED")
                  .limit(2000).all())
                 if j.created_at is not None
                 and _as_utc(j.created_at) >= since)
    oldest = (db.query(WorkerJob)
              .filter(WorkerJob.status == "QUEUED")
              .order_by(WorkerJob.created_at.asc()).first())
    oldest_age_s = None
    if oldest is not None and oldest.created_at is not None:
        oldest_age_s = max(0.0, (now - _as_utc(oldest.created_at))
                           .total_seconds())
    live = _live_worker_count(db)
    total_5m = done + failed
    return {
        "queue_depth": queued,
        "running_jobs": running,
        "oldest_job_age_s": round(oldest_age_s, 1) if oldest_age_s
        is not None else 0.0,
        "throughput_5m": done,
        "failure_rate_5m": round(failed / total_5m, 3) if total_5m else 0.0,
        "live_workers": live,
        "as_of": now.isoformat(),
    }


# ---------------------------------------------------------------------------
# Scheduler 2.0 — leader election
# ---------------------------------------------------------------------------

def acquire_leadership(db: Session, *, leader_id: str,
                       hostname: Optional[str] = None,
                       pid: Optional[int] = None,
                       version: Optional[str] = None,
                       lease_seconds: int = 30) -> dict:
    """DB-backed leader election. Exactly one LEADER holds a valid lease."""
    from ..models.phase19 import SchedulerLeader
    now = _utcnow()
    rows = (db.query(SchedulerLeader)
            .filter(SchedulerLeader.status == "LEADER").all())
    for row in rows:
        if row.leader_id == leader_id:
            row.lease_until = now + timedelta(seconds=lease_seconds)
            row.last_heartbeat = now
            db.flush()
            return {"status": "LEADER", "leader_id": leader_id,
                    "renewed": True}
        if _as_utc(row.lease_until) is not None \
                and _as_utc(row.lease_until) > now:
            return {"status": "STANDBY", "leader_id": leader_id,
                    "leader": row.leader_id,
                    "reason": "active leader holds the lease"}
    # lease expired or absent → takeover
    for row in rows:
        row.status = "STALE"
    leader = SchedulerLeader(leader_id=leader_id, hostname=hostname, pid=pid,
                             version=version,
                             acquired_at=now,
                             lease_until=now + timedelta(seconds=lease_seconds),
                             last_heartbeat=now, status="LEADER")
    db.add(leader)
    db.flush()
    return {"status": "LEADER", "leader_id": leader_id, "takeover": True}


def renew_leadership(db: Session, *, leader_id: str,
                     lease_seconds: int = 30) -> bool:
    from ..models.phase19 import SchedulerLeader
    row = (db.query(SchedulerLeader)
           .filter(SchedulerLeader.leader_id == leader_id,
                   SchedulerLeader.status == "LEADER").first())
    if row is None:
        return False
    row.lease_until = _utcnow() + timedelta(seconds=lease_seconds)
    row.last_heartbeat = _utcnow()
    db.flush()
    return True


def release_leadership(db: Session, *, leader_id: str) -> dict:
    from ..models.phase19 import SchedulerLeader
    row = (db.query(SchedulerLeader)
           .filter(SchedulerLeader.leader_id == leader_id,
                   SchedulerLeader.status == "LEADER").first())
    if row is None:
        return {"status": "NOT_LEADER"}
    row.status = "RELEASED"
    db.flush()
    return {"status": "RELEASED", "leader_id": leader_id}


def recover_stale_leadership(db: Session) -> dict:
    """Expired leader leases become STALE and may be taken over."""
    from ..models.phase19 import SchedulerLeader
    now = _utcnow()
    leaders = (db.query(SchedulerLeader)
               .filter(SchedulerLeader.status == "LEADER")
               .limit(100).all())
    stale = [row for row in leaders
             if _as_utc(row.lease_until) is None
             or _as_utc(row.lease_until) < now]
    for row in stale:
        row.status = "STALE"
    db.flush()
    return {"stale_marked": len(stale)}


def scheduler_dedupe_key(*parts: str) -> str:
    """Deterministic scheduled-job key → prevents duplicate scheduled jobs."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def ensure_scheduled_job(db: Session, *, workspace_id: int,
                         organization_id: Optional[int],
                         queue_name: str, job_type: str,
                         dedupe_key: str, payload: Optional[dict] = None,
                         run_after: Optional[datetime] = None,
                         user_id: Optional[int] = None):
    """Idempotent scheduled enqueue with a deterministic dedupe key."""
    from .worker_platform import enqueue_job
    return enqueue_job(db, queue_name=queue_name, job_type=job_type,
                       workspace_id=workspace_id,
                       organization_id=organization_id, user_id=user_id,
                       payload=payload, dedupe_key=dedupe_key,
                       run_after=run_after)
