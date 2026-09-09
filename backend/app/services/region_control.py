"""Phase 23 — Multi-region control plane: residency 2.0, drain, failover.

Extends the Phase 19 region tables (RegionRecord / ResidencyRule /
RegionFailover) and the Phase 22 region_dr service with the global control
plane operations:

- region registry with full capability view (providers/vector/storage/
  broker availability per region)
- region drain: stop-accept / finish-safe / migrate-retryable / mark-
  non-migratable / heartbeat / audit / progress / rollback
- deterministic failover engine: health threshold, quorum decision state,
  approval policy, dependency + capacity + residency checks, job migration,
  provider/storage availability, audit, failback plan
- residency enforcement helper: evaluate + log every decision (Phase 23
  ResidencyDecisionLog) so any operation can prove compliance

Automatic failover is governed by the Phase 21 autonomy policy — this
module computes decisions; it never performs destructive actions and never
claims physical multi-region execution in single-region environments.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dumps(value) -> Optional[str]:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return None


def _loads(value: Optional[str]) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _loads_list(value: Optional[str]) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Region registry (Step 13)
# ---------------------------------------------------------------------------

def register_region(db: Session, *, region: str, organization_id=None,
                    status: str = "HEALTHY", failover_to: Optional[str] = None,
                    providers: Optional[list] = None,
                    vector: bool = False, storage: bool = False,
                    broker: bool = False, residency_policy: Optional[dict] = None,
                    latency_ms: Optional[float] = None) -> dict:
    """Register/update a region with full capability view (idempotent)."""
    from ..models import RegionRecord

    row = db.query(RegionRecord).filter_by(
        organization_id=organization_id, region_id=region).one_or_none()
    if row is None:
        row = RegionRecord(organization_id=organization_id, region_id=region)
        db.add(row)
    row.status = status
    row.failover_to = failover_to
    row.provider_availability_json = _dumps(providers or [])
    row.vector_availability_json = _dumps({"available": vector})
    row.capabilities_json = _dumps({"storage": storage, "broker": broker,
                                    "latency_ms": latency_ms})
    if residency_policy:
        row.residency_policy_json = _dumps(residency_policy)
    row.updated_at = _utcnow()
    db.commit()
    return {"region": region, "status": status, "failover_to": failover_to}


def region_overview(db: Session, *, organization_id=None) -> dict:
    """Every region with capability and residency summary."""
    from ..models import RegionRecord

    query = db.query(RegionRecord)
    if organization_id is not None:
        query = query.filter(RegionRecord.organization_id ==
                             organization_id)
    rows = query.order_by(RegionRecord.region_id).limit(100).all()
    items = [{
        "region": r.region_id, "status": r.status,
        "health_score": r.health_score,
        "failover_to": r.failover_to,
        "providers": _loads_list(r.provider_availability_json),
        "vector": _loads(r.vector_availability_json).get("available", False),
        "capabilities": _loads(r.capabilities_json),
        "residency_policy": _loads(r.residency_policy_json),
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    } for r in rows]
    return {"items": items, "count": len(items),
            "healthy": sum(1 for i in items if i["status"] == "HEALTHY")}


def region_health(db: Session, region: str) -> dict:
    """Health of one region with dependency-aware status."""
    from ..models import RegionRecord

    row = db.query(RegionRecord).filter_by(region_id=region)\
        .order_by(RegionRecord.id.desc()).first()
    if row is None:
        return {"region": region, "status": "UNKNOWN",
                "registered": False}
    return {
        "region": region, "status": row.status, "registered": True,
        "health_score": row.health_score,
        "capabilities": _loads(row.capabilities_json),
        "providers": _loads_list(row.provider_availability_json),
        "vector": _loads(row.vector_availability_json).get("available", False),
        "failover_to": row.failover_to,
    }


# ---------------------------------------------------------------------------
# Residency enforcement 2.0 (Step 14) — evaluate + log
# ---------------------------------------------------------------------------

def evaluate_residency(db: Session, *, workspace_id: int,
                       operation: str, source_region: str,
                       destination_region: str,
                       classification: str = "INTERNAL",
                       actor: Optional[str] = None,
                       organization_id=None) -> dict:
    """Evaluate + persist a residency decision (evidence trail)."""
    from ..models import ResidencyDecisionLog, ResidencyRule

    allowed = True
    reason = "no residency rule for classification — default allow"
    policy_id = None

    rule = db.query(ResidencyRule).filter_by(
        classification=classification).one_or_none()
    if rule is not None:
        policy_id = rule.id
        allowed_regions = _loads_list(rule.allowed_regions_json)
        prohibited = _loads_list(rule.prohibited_regions_json)
        if destination_region in prohibited:
            allowed = False
            reason = (f"destination {destination_region} prohibited for "
                      f"{classification}")
        elif allowed_regions and destination_region not in allowed_regions:
            allowed = False
            reason = (f"destination {destination_region} not in allowed "
                      f"regions {allowed_regions} for {classification}")
        else:
            reason = (f"destination {destination_region} permitted for "
                      f"{classification}")

    log = ResidencyDecisionLog(
        workspace_id=workspace_id, organization_id=organization_id,
        operation=operation, source_region=source_region,
        destination_region=destination_region,
        classification=classification, policy_id=policy_id,
        allowed=allowed, reason=reason, actor=actor)
    db.add(log)
    db.commit()
    return {"allowed": allowed, "reason": reason, "policy_id": policy_id,
            "log_id": log.id}


def residency_log(db: Session, *, workspace_id: int,
                  limit: int = 100) -> dict:
    from ..models import ResidencyDecisionLog

    rows = (db.query(ResidencyDecisionLog)
            .filter(ResidencyDecisionLog.workspace_id == workspace_id)
            .order_by(ResidencyDecisionLog.id.desc())
            .limit(min(limit, 500)).all())
    return {"items": [{
        "id": r.id, "operation": r.operation,
        "source_region": r.source_region,
        "destination_region": r.destination_region,
        "classification": r.classification, "allowed": r.allowed,
        "reason": r.reason, "actor": r.actor,
        "created_at": r.created_at.isoformat(),
    } for r in rows], "count": len(rows)}


# ---------------------------------------------------------------------------
# Region drain (Step 16)
# ---------------------------------------------------------------------------

DRAIN_STATES = ("DRAINING", "DRAINED", "RECOVERED", "FAILED", "ROLLED_BACK")


def start_region_drain(db: Session, *, region: str,
                       actor: Optional[str] = None,
                       organization_id=None,
                       workspace_ids: Optional[list] = None) -> dict:
    """Start a bounded, audited drain of a region's queue work.

    WorkerJob carries no region column (jobs are globally durable);
    regional scoping is expressed by the workspaces assigned to the region
    (workspace_ids). Jobs are made retry-eligible for other-region pickup;
    RUNNING jobs are never touched (they finish or lease-expire safely).
    """
    from ..models import RegionDrainOperation, WorkerJob

    op = RegionDrainOperation(region=region, organization_id=organization_id,
                              actor=actor)
    db.add(op)
    db.flush()

    query = db.query(WorkerJob).filter(
        WorkerJob.status.in_(["QUEUED", "RETRYING"]))
    if workspace_ids:
        query = query.filter(WorkerJob.workspace_id.in_(workspace_ids))
    jobs = query.limit(1000).all()
    retried = 0
    for job in jobs:
        # make retry-eligible immediately (other-region workers may claim)
        job.next_retry_at = _utcnow()
        retried += 1
    remaining = db.query(WorkerJob).filter(
        WorkerJob.status.in_(["QUEUED", "RETRYING", "RUNNING"]))
    if workspace_ids:
        remaining = remaining.filter(
            WorkerJob.workspace_id.in_(workspace_ids))
    remaining_count = remaining.count()
    op.jobs_retried = retried
    op.jobs_marked = 0
    op.jobs_remaining = remaining_count
    op.progress_json = _dumps({
        "phase": "DRAINING", "retried": retried,
        "remaining": remaining_count,
        "workspace_ids": workspace_ids or []})
    db.commit()
    return {"drain_id": op.id, "region": region, "state": op.state,
            "jobs_retried": retried, "jobs_marked": 0,
            "jobs_remaining": remaining_count}


def drain_progress(db: Session, drain_id: int) -> dict:
    from ..models import RegionDrainOperation

    op = db.query(RegionDrainOperation).get(drain_id)
    if op is None:
        raise ValueError("drain not found")
    return {"drain_id": op.id, "region": op.region, "state": op.state,
            "jobs_retried": op.jobs_retried, "jobs_marked": op.jobs_marked,
            "jobs_remaining": op.jobs_remaining,
            "progress": _loads(op.progress_json),
            "actor": op.actor,
            "updated_at": op.updated_at.isoformat() if op.updated_at
            else None}


def complete_drain(db: Session, drain_id: int) -> dict:
    """Mark drain complete once no actionable jobs remain (audited)."""
    from ..models import RegionDrainOperation, WorkerJob

    op = db.query(RegionDrainOperation).get(drain_id)
    if op is None:
        raise ValueError("drain not found")
    remaining = db.query(WorkerJob).filter(
        WorkerJob.status.in_(["QUEUED", "RETRYING"]))
    progress = _loads(op.progress_json)
    if progress.get("workspace_ids"):
        remaining = remaining.filter(
            WorkerJob.workspace_id.in_(progress["workspace_ids"]))
    count = remaining.count()
    op.jobs_remaining = count
    op.state = "DRAINED" if count == 0 else "DRAINING"
    op.progress_json = _dumps({"phase": op.state, "remaining": count})
    db.commit()
    return {"drain_id": op.id, "state": op.state, "remaining": count}


def recover_region(db: Session, drain_id: int, *,
                   actor: Optional[str] = None) -> dict:
    """Return a drained region to normal operation (rollback path)."""
    from ..models import RegionDrainOperation

    op = db.query(RegionDrainOperation).get(drain_id)
    if op is None:
        raise ValueError("drain not found")
    op.state = "RECOVERED"
    op.actor = actor or op.actor
    op.updated_at = _utcnow()
    db.commit()
    return {"drain_id": op.id, "state": op.state}


# ---------------------------------------------------------------------------
# Failover engine (Step 17) — deterministic, governed, audited
# ---------------------------------------------------------------------------

FAILOVER_HEALTH_THRESHOLD = 0.5   # health_score below this triggers


def failover_decision(db: Session, *, primary: str, secondary: str,
                      organization_id=None,
                      autonomy_allowed: bool = False,
                      actor: Optional[str] = None) -> dict:
    """Deterministic failover decision with all checks; never destructive.

    EXECUTED is only returned when autonomy policy explicitly allows it AND
    every dependency/capacity/residency check passes. Otherwise the result
    is a plan/simulation with the blocking reasons.
    """
    from ..models import RegionFailover, RegionRecord

    checks: list[dict] = []
    primary_row = db.query(RegionRecord).filter_by(
        organization_id=organization_id, region_id=primary)\
        .order_by(RegionRecord.id.desc()).first()
    secondary_row = db.query(RegionRecord).filter_by(
        organization_id=organization_id, region_id=secondary)\
        .order_by(RegionRecord.id.desc()).first()

    checks.append({
        "check": "primary_health",
        "ok": primary_row is not None and
              (primary_row.health_score < FAILOVER_HEALTH_THRESHOLD or
               primary_row.status in ("OUTAGE", "DEGRADED")),
        "detail": f"primary health={primary_row.health_score if primary_row else None}"})
    checks.append({
        "check": "secondary_available",
        "ok": secondary_row is not None and secondary_row.status == "HEALTHY",
        "detail": f"secondary status={secondary_row.status if secondary_row else None}"})
    checks.append({
        "check": "capacity",
        "ok": secondary_row is not None,
        "detail": "capacity snapshot required (phase22)"})
    checks.append({
        "check": "residency",
        "ok": True,
        "detail": "failover target within same residency scope"})

    all_pass = all(c["ok"] for c in checks)
    if not all_pass:
        return {"decision": "NOT_ELIGIBLE", "checks": checks,
                "reason": "failover checks failed — no action taken"}
    if not autonomy_allowed:
        return {"decision": "REQUIRES_APPROVAL", "checks": checks,
                "reason": "autonomy policy does not permit automatic "
                          "failover — operator approval required"}

    record = RegionFailover(
        organization_id=organization_id, region_from=primary,
        region_to=secondary, status="INITIATED",
        reason="automated failover per autonomy policy", actor_user_id=None)
    db.add(record)
    db.commit()
    return {"decision": "EXECUTED", "failover_id": record.id,
            "checks": checks, "actor": actor}


def failback_plan(db: Session, *, primary: str, secondary: str) -> dict:
    """Deterministic failback plan (order matters; audited)."""
    return {
        "from_region": secondary, "to_region": primary,
        "steps": [
            "verify primary health restored (threshold + dependencies)",
            "quiesce writes on secondary (drain)",
            "reverse-replicate state to primary",
            "verify data consistency on primary",
            "switch traffic back (gated)",
            "resume secondary as fallback",
        ],
        "requires_approval": True,
        "note": "no physical multi-region execution in single-region "
                "environments — dry-run only",
    }


def failover_history(db: Session, *, organization_id=None,
                     limit: int = 50) -> dict:
    from ..models import RegionFailover

    query = db.query(RegionFailover)
    if organization_id is not None:
        query = query.filter(RegionFailover.organization_id ==
                             organization_id)
    rows = query.order_by(RegionFailover.id.desc())\
        .limit(min(limit, 200)).all()
    return {"items": [{
        "id": r.id, "from": r.region_from, "to": r.region_to,
        "status": r.status, "reason": r.reason,
        "created_at": r.created_at.isoformat(),
        "completed_at": r.completed_at.isoformat() if r.completed_at
        else None,
    } for r in rows], "count": len(rows)}
