"""Phase 20 — Notification intelligence.

Alert prioritization (informational/low/medium/high/critical), fingerprint
deduplication with occurrence counting, escalation rules (low->medium when
repeated, priority jumps), and per-user/workspace notification preference
categories.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import OpsAlertEvent, NotificationPreference

SEVERITIES = ("informational", "low", "medium", "high", "critical")
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}

VALID_CATEGORIES = (
    "quality", "cost", "incident", "provider", "ingestion", "security",
    "workflow", "knowledge", "budget", "experiment",
)

ESCALATION_RULES = {
    "low": ("repeat>=3", "medium"),
    "medium": ("repeat>=5", "high"),
    "high": ("repeat>=8", "critical"),
}


def _fingerprint(category: str, message: str) -> str:
    raw = f"{category}:{message}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def raise_alert(db: Session, *, category: str, severity: str,
                message: str,
                workspace_id: Optional[int] = None) -> OpsAlertEvent:
    """Prioritize + deduplicate: same fingerprint increments occurrences and
    may escalate severity per escalation rules."""
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity: {severity}")
    fp = _fingerprint(category, message)
    existing = db.query(OpsAlertEvent).filter_by(fingerprint=fp)\
        .filter(OpsAlertEvent.status.in_(["OPEN", "ACKNOWLEDGED"])).first()
    if existing is not None:
        existing.occurrence_count += 1
        existing.updated_at = datetime.now(timezone.utc)
        _apply_escalation(existing)
        db.flush()
        return existing
    row = OpsAlertEvent(category=category, severity=severity, fingerprint=fp,
                        message=message, workspace_id=workspace_id)
    db.add(row)
    db.flush()
    return row


def _apply_escalation(row: OpsAlertEvent) -> None:
    count = row.occurrence_count
    current_rank = SEVERITY_RANK[row.severity]
    for rule, (threshold, new_sev) in ESCALATION_RULES.items():
        if row.severity == rule or SEVERITY_RANK[rule] == current_rank:
            n = int(threshold.split(">=")[1])
            if count >= n and SEVERITY_RANK[new_sev] > SEVERITY_RANK[row.severity]:
                row.severity = new_sev


def list_alerts(db: Session, *, status: Optional[str] = None,
                severity: Optional[str] = None,
                category: Optional[str] = None,
                limit: int = 100) -> list[dict]:
    q = db.query(OpsAlertEvent)
    if status:
        q = q.filter(OpsAlertEvent.status == status)
    if severity:
        q = q.filter(OpsAlertEvent.severity == severity)
    if category:
        q = q.filter(OpsAlertEvent.category == category)
    rows = q.order_by(OpsAlertEvent.created_at.desc()).limit(limit).all()
    return [{"id": a.id, "category": a.category, "severity": a.severity,
             "message": a.message, "status": a.status,
             "occurrence_count": a.occurrence_count,
             "workspace_id": a.workspace_id, "created_at": a.created_at}
            for a in rows]


def acknowledge(db: Session, alert_id: int) -> dict:
    alert = db.get(OpsAlertEvent, alert_id)
    if alert is None:
        raise KeyError("alert not found")
    alert.status = "ACKNOWLEDGED"
    alert.updated_at = datetime.now(timezone.utc)
    return {"acknowledged": True, "alert_id": alert_id}


def set_preference(db: Session, *, user_id: int,
                   category: str, enabled: bool,
                   workspace_id: Optional[int] = None) -> NotificationPreference:
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unknown notification category: {category}")
    row = db.query(NotificationPreference).filter_by(
        user_id=user_id, workspace_id=workspace_id,
        category=category).first()
    if row is None:
        row = NotificationPreference(user_id=user_id,
                                     workspace_id=workspace_id,
                                     category=category, enabled=enabled)
        db.add(row)
    row.enabled = enabled
    row.updated_at = datetime.now(timezone.utc)
    db.flush()
    return row


def preferences(db: Session, *, user_id: int,
                workspace_id: Optional[int] = None) -> list[dict]:
    rows = db.query(NotificationPreference).filter_by(user_id=user_id)\
        .order_by(NotificationPreference.category).all()
    return [{"category": p.category, "enabled": p.enabled,
             "workspace_id": p.workspace_id} for p in rows]


def alert_summary(db: Session, *, limit: int = 500) -> dict:
    """Aggregate open alerts by severity and category for the ops center."""
    rows = db.query(OpsAlertEvent).filter_by(status="OPEN")\
        .order_by(OpsAlertEvent.created_at.desc()).limit(limit).all()
    by_sev: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for r in rows:
        by_sev[r.severity] = by_sev.get(r.severity, 0) + 1
        by_cat[r.category] = by_cat.get(r.category, 0) + 1
    return {"total_open": len(rows),
            "by_severity": dict(sorted(by_sev.items(),
                                       key=lambda kv: -SEVERITY_RANK[kv[0]])),
            "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1]))}