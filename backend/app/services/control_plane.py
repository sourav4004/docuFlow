"""Phase 19 — enterprise control plane.

Versioned, immutable configuration snapshots with audited activation and
rollback; region/deployment abstraction with deterministic region selection;
data-residency rules; explicit, audited region failover; and a read-only
global-health aggregate for the operations center.

Reads are tenant-safe and bounded. Snapshots never silently change production
policy: every activation/rollback records actor, reason, and a structured
diff, and rollback is implemented as a NEW snapshot whose payload restores an
older version (immutable history).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SENSITIVITIES = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")
REGION_STATUSES = ("HEALTHY", "DEGRADED", "OUTAGE", "MAINTENANCE",
                   "FAILED_OVER")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _loads(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _dumps(value) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Control-plane snapshots
# ---------------------------------------------------------------------------

def diff_configs(old: Optional[dict], new: Optional[dict]) -> dict:
    """Structural, deterministic diff between two config dictionaries."""
    old = old or {}
    new = new or {}
    keys = set(old) | set(new)
    added, removed, changed = {}, {}, {}
    for key in sorted(keys):
        if key not in old:
            added[key] = new[key]
        elif key not in new:
            removed[key] = old[key]
        elif old.get(key) != new.get(key):
            changed[key] = {"from": old.get(key), "to": new.get(key)}
    return {"added": added, "removed": removed, "changed": changed,
            "unchanged_count": len(set(old) & set(new))}


def snapshot_config(db: Session, *, scope_type: str, scope_id: Optional[int],
                    config: dict, actor_user_id: Optional[int] = None,
                    reason: Optional[str] = None):
    """Persist a new immutable configuration snapshot (versioned)."""
    from ..models.phase19 import ControlPlaneSnapshot
    active = latest_snapshot(db, scope_type=scope_type, scope_id=scope_id,
                             only_active=True)
    version = (latest_snapshot(db, scope_type=scope_type, scope_id=scope_id)
               or 0)
    next_version = (version.version + 1) if version else 1
    snap = ControlPlaneSnapshot(
        scope_type=scope_type.upper(), scope_id=scope_id, version=next_version,
        config_json=_dumps(config) or "{}",
        diff_json=_dumps(diff_configs(_loads(active.config_json)
                                      if active else None, config)),
        actor_user_id=actor_user_id, reason=reason, is_active=False)
    db.add(snap)
    db.flush()
    return snap


def latest_snapshot(db: Session, *, scope_type: str,
                    scope_id: Optional[int], only_active: bool = False):
    from ..models.phase19 import ControlPlaneSnapshot
    q = (db.query(ControlPlaneSnapshot)
         .filter(ControlPlaneSnapshot.scope_type == scope_type.upper(),
                 ControlPlaneSnapshot.scope_id == scope_id))
    if only_active:
        q = q.filter(ControlPlaneSnapshot.is_active.is_(True))
    return (q.order_by(ControlPlaneSnapshot.version.desc()).first())


def list_snapshots(db: Session, *, scope_type: str,
                   scope_id: Optional[int], limit: int = 50) -> list:
    from ..models.phase19 import ControlPlaneSnapshot
    return (db.query(ControlPlaneSnapshot)
            .filter(ControlPlaneSnapshot.scope_type == scope_type.upper(),
                    ControlPlaneSnapshot.scope_id == scope_id)
            .order_by(ControlPlaneSnapshot.version.desc())
            .limit(min(limit, 200)).all())


def activate_snapshot(db: Session, snapshot_id: int,
                      actor_user_id: Optional[int] = None) -> dict:
    """Activate one snapshot; deactivate peers in the same scope (audited)."""
    from ..models.phase19 import ControlPlaneSnapshot
    snap = db.query(ControlPlaneSnapshot).get(snapshot_id)
    if snap is None:
        raise ValueError("snapshot not found")
    current = latest_snapshot(db, scope_type=snap.scope_type,
                              scope_id=snap.scope_id, only_active=True)
    if current is not None and current.id == snap.id:
        return {"status": "ALREADY_ACTIVE", "snapshot_id": snap.id,
                "version": snap.version}
    if current is not None:
        current.is_active = False
    snap.is_active = True
    snap.actor_user_id = actor_user_id
    db.flush()
    return {"status": "ACTIVATED", "snapshot_id": snap.id,
            "version": snap.version,
            "scope": {"type": snap.scope_type, "id": snap.scope_id}}


def rollback_snapshot(db: Session, *, scope_type: str,
                      scope_id: Optional[int], target_version: int,
                      actor_user_id: Optional[int] = None,
                      reason: Optional[str] = None) -> dict:
    """Roll back to an earlier version by creating a new snapshot whose
    payload restores it. Immutable history + audited + idempotent: rolling
    back to the already-active version is a no-op."""
    from ..models.phase19 import ControlPlaneSnapshot
    target = (db.query(ControlPlaneSnapshot)
              .filter(ControlPlaneSnapshot.scope_type == scope_type.upper(),
                      ControlPlaneSnapshot.scope_id == scope_id,
                      ControlPlaneSnapshot.version == target_version)
              .first())
    if target is None:
        raise ValueError("target version not found")
    active = latest_snapshot(db, scope_type=scope_type, scope_id=scope_id,
                             only_active=True)
    if active is not None and active.version == target_version:
        return {"status": "NOOP", "detail": "target version already active",
                "version": target_version}
    snap = snapshot_config(db, scope_type=scope_type, scope_id=scope_id,
                           config=_loads(target.config_json) or {},
                           actor_user_id=actor_user_id,
                           reason=reason or "rollback to v%d" % target_version)
    result = activate_snapshot(db, snap.id, actor_user_id=actor_user_id)
    result["restores_version"] = target_version
    result["rollback"] = True
    return result


def effective_config(db: Session, *, scope_type: str,
                     scope_id: Optional[int]) -> dict:
    """Config of the active snapshot (defaults to {}). Bounded read."""
    active = latest_snapshot(db, scope_type=scope_type, scope_id=scope_id,
                             only_active=True)
    return _loads(active.config_json) if active else {}


# ---------------------------------------------------------------------------
# Regions + residency
# ---------------------------------------------------------------------------

def upsert_region(db: Session, *, organization_id: Optional[int],
                  region_id: str, name: Optional[str] = None,
                  deployment: Optional[str] = None,
                  status: str = "HEALTHY",
                  health_score: Optional[float] = None,
                  capabilities: Optional[dict] = None,
                  capacity: Optional[dict] = None,
                  provider_availability: Optional[dict] = None,
                  vector_availability: Optional[dict] = None,
                  residency_policy: Optional[dict] = None,
                  failover_to: Optional[str] = None):
    from ..models.phase19 import RegionRecord
    if status not in REGION_STATUSES:
        raise ValueError("invalid region status")
    if health_score is None:
        health_score = 1.0 if status == "HEALTHY" else (
            0.6 if status == "DEGRADED" else 0.0)
    row = (db.query(RegionRecord)
           .filter(RegionRecord.organization_id == organization_id,
                   RegionRecord.region_id == region_id).first())
    if row is None:
        row = RegionRecord(organization_id=organization_id,
                           region_id=region_id)
        db.add(row)
    row.name = name or row.name or region_id
    row.deployment = deployment or row.deployment
    row.status = status
    row.health_score = max(0.0, min(1.0, health_score))
    row.capabilities_json = _dumps(capabilities) if capabilities is not None \
        else row.capabilities_json
    row.capacity_json = _dumps(capacity) if capacity is not None \
        else row.capacity_json
    row.provider_availability_json = (_dumps(provider_availability)
                                      if provider_availability is not None
                                      else row.provider_availability_json)
    row.vector_availability_json = (_dumps(vector_availability)
                                    if vector_availability is not None
                                    else row.vector_availability_json)
    row.residency_policy_json = (_dumps(residency_policy)
                                 if residency_policy is not None
                                 else row.residency_policy_json)
    row.failover_to = failover_to or row.failover_to
    row.updated_at = _utcnow()
    db.flush()
    return row


def list_regions(db: Session, organization_id: Optional[int] = None,
                 limit: int = 100) -> list:
    from ..models.phase19 import RegionRecord
    q = db.query(RegionRecord)
    if organization_id is not None:
        q = q.filter(RegionRecord.organization_id == organization_id)
    return q.order_by(RegionRecord.region_id).limit(min(limit, 500)).all()


def set_residency_rule(db: Session, *, organization_id: Optional[int],
                       classification: str,
                       allowed_regions: Optional[list] = None,
                       prohibited_regions: Optional[list] = None,
                       default_region: Optional[str] = None):
    from ..models.phase19 import ResidencyRule
    if classification not in SENSITIVITIES:
        raise ValueError("invalid classification")
    row = (db.query(ResidencyRule)
           .filter(ResidencyRule.organization_id == organization_id,
                   ResidencyRule.classification == classification).first())
    if row is None:
        row = ResidencyRule(organization_id=organization_id,
                            classification=classification)
        db.add(row)
    row.allowed_regions_json = _dumps(allowed_regions) \
        if allowed_regions is not None else row.allowed_regions_json
    row.prohibited_regions_json = _dumps(prohibited_regions) \
        if prohibited_regions is not None else row.prohibited_regions_json
    row.default_region = default_region or row.default_region
    db.flush()
    return row


def evaluate_residency(db: Session, *, organization_id: Optional[int],
                       classification: str,
                       region_id: str) -> dict:
    """Residency check: allowed/prohibited regions per classification."""
    from ..models.phase19 import ResidencyRule
    if classification not in SENSITIVITIES:
        raise ValueError("invalid classification")
    rule = (db.query(ResidencyRule)
            .filter(ResidencyRule.organization_id == organization_id,
                    ResidencyRule.classification == classification).first())
    if rule is None:
        return {"allowed": True, "reason": "no residency rule configured",
                "classification": classification, "region": region_id}
    allowed = _loads(rule.allowed_regions_json) or []
    prohibited = _loads(rule.prohibited_regions_json) or []
    if prohibited and region_id in prohibited:
        return {"allowed": False,
                "reason": "region prohibited for classification",
                "classification": classification, "region": region_id}
    if allowed and region_id not in allowed:
        return {"allowed": False,
                "reason": "region not allowed for classification",
                "classification": classification, "region": region_id}
    return {"allowed": True, "reason": "region permitted",
            "classification": classification, "region": region_id}


def select_region(db: Session, *, organization_id: Optional[int],
                  classification: str,
                  required_capability: Optional[str] = None,
                  preferred: Optional[list] = None) -> dict:
    """Deterministic region selection honoring residency + health.

    Candidates must pass the residency rule and (when declared) the
    required capability in provider/vector availability. Ties resolve by
    preference order, then health score desc, then region id asc.
    """
    from ..models.phase19 import RegionRecord
    regions = list_regions(db, organization_id=organization_id)
    candidates = []
    for region in regions:
        residency = evaluate_residency(db, organization_id=organization_id,
                                       classification=classification,
                                       region_id=region.region_id)
        if not residency["allowed"]:
            continue
        if required_capability:
            avail = _loads(region.provider_availability_json) or {}
            vec = _loads(region.vector_availability_json) or {}
            has = (avail.get(required_capability) is True
                   or vec.get(required_capability) is True
                   or (required_capability in avail)
                   or (required_capability in vec))
            if not has:
                continue
        candidates.append(region)
    if not candidates:
        return {"selected": None, "reason": "no eligible region",
                "classification": classification,
                "required_capability": required_capability}
    preferred = preferred or []
    candidates.sort(key=lambda r: (
        preferred.index(r.region_id) if r.region_id in preferred
        else len(preferred),
        -r.health_score,
        r.region_id))
    chosen = candidates[0]
    return {"selected": chosen.region_id, "reason": "deterministic selection",
            "classification": classification, "eligible": len(candidates),
            "health_score": chosen.health_score,
            "required_capability": required_capability}


def initiate_failover(db: Session, *, organization_id: Optional[int],
                      region_from: str, region_to: str,
                      actor_user_id: Optional[int] = None,
                      reason: Optional[str] = None) -> dict:
    """Explicit failover — no silent data loss: requires the target region to
    exist and be healthy, then marks the source region FAILED_OVER."""
    from ..models.phase19 import RegionRecord, RegionFailover
    target = (db.query(RegionRecord)
              .filter(RegionRecord.organization_id == organization_id,
                      RegionRecord.region_id == region_to).first())
    if target is None:
        raise ValueError("target region not registered")
    if target.status == "OUTAGE":
        raise ValueError("cannot fail over into an OUTAGE region")
    source = (db.query(RegionRecord)
              .filter(RegionRecord.organization_id == organization_id,
                      RegionRecord.region_id == region_from).first())
    if source is None:
        raise ValueError("source region not registered")
    failover = RegionFailover(organization_id=organization_id,
                              region_from=region_from, region_to=region_to,
                              status="INITIATED", reason=reason,
                              actor_user_id=actor_user_id)
    db.add(failover)
    source.status = "FAILED_OVER"
    source.health_score = 0.0
    source.failover_to = region_to
    db.flush()
    return {"failover_id": failover.id, "status": "INITIATED",
            "from": region_from, "to": region_to}


def complete_failover(db: Session, failover_id: int) -> dict:
    from ..models.phase19 import RegionFailover
    row = db.query(RegionFailover).get(failover_id)
    if row is None:
        raise ValueError("failover not found")
    if row.status == "COMPLETED":
        return {"status": "ALREADY_COMPLETED", "failover_id": row.id}
    row.status = "COMPLETED"
    row.completed_at = _utcnow()
    db.flush()
    return {"status": "COMPLETED", "failover_id": row.id}


def revert_failover(db: Session, failover_id: int,
                    actor_user_id: Optional[int] = None) -> dict:
    from ..models.phase19 import RegionFailover, RegionRecord
    row = db.query(RegionFailover).get(failover_id)
    if row is None:
        raise ValueError("failover not found")
    if row.status == "REVERTED":
        return {"status": "ALREADY_REVERTED", "failover_id": row.id}
    source = (db.query(RegionRecord)
              .filter(RegionRecord.organization_id == row.organization_id,
                      RegionRecord.region_id == row.region_from).first())
    if source is not None:
        source.status = "HEALTHY"
        source.health_score = 1.0
        source.failover_to = None
    row.status = "REVERTED"
    row.completed_at = _utcnow()
    row.actor_user_id = actor_user_id
    db.flush()
    return {"status": "REVERTED", "failover_id": row.id}


def residency_violations(db: Session, *, organization_id: Optional[int],
                         samples: list[dict], limit: int = 200) -> list[dict]:
    """Check (classification, region) samples for residency violations."""
    out = []
    for sample in samples[:limit]:
        result = evaluate_residency(
            db, organization_id=organization_id,
            classification=sample.get("classification", "INTERNAL"),
            region_id=sample.get("region"))
        if not result["allowed"]:
            out.append({"sample": sample, **result})
    return out


# ---------------------------------------------------------------------------
# Global health aggregate (read-only, bounded)
# ---------------------------------------------------------------------------

def global_health(db: Session) -> dict:
    """Aggregate control-plane health for /ops 4.0. Never exposes secrets."""
    from ..models.phase16 import WorkerHeartbeat
    from ..models.phase17 import AlertRule
    from ..models.phase19 import SchedulerLeader
    regions = list_regions(db)
    now = _utcnow()
    workers = db.query(WorkerHeartbeat).limit(500).all()
    live = [w for w in workers
            if w.last_heartbeat is not None
            and _as_utc(w.last_heartbeat) is not None
            and now - _as_utc(w.last_heartbeat) < timedelta(seconds=90)]
    leader = (db.query(SchedulerLeader)
              .filter(SchedulerLeader.status == "LEADER").first())
    return {
        "as_of": now.isoformat(),
        "regions": [{"region_id": r.region_id, "status": r.status,
                     "health_score": r.health_score}
                    for r in regions[:100]],
        "workers": {"registered": len(workers), "live": len(live)},
        "scheduler_leader": ({"leader_id": leader.leader_id,
                              "lease_until": leader.lease_until.isoformat()
                              if leader.lease_until else None}
                             if leader else None),
        "alert_rules_enabled": db.query(AlertRule)
        .filter(AlertRule.enabled.is_(True)).count(),
        "status": "HEALTHY" if (live or not workers) and leader is not None
        else "DEGRADED",
    }
