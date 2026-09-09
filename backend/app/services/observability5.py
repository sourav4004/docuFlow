"""Phase 20 — Observability 5.0.

Unified AI system health score (availability, latency, quality, cost, error
rate, queue health, provider health), a declarative service dependency
graph (API → broker → workers → provider → database → vector → connectors),
incident correlation (related failures grouped by fingerprint), incident
lifecycle (OPEN → INVESTIGATING → MITIGATED → RESOLVED → POSTMORTEM), and a
persisted incident timeline.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import Incident, IncidentEvent

INCIDENT_STATUSES = {
    "OPEN", "INVESTIGATING", "MITIGATED", "RESOLVED", "POSTMORTEM",
}

INCIDENT_TRANSITIONS = {
    ("OPEN", "INVESTIGATING"),
    ("OPEN", "MITIGATED"),
    ("INVESTIGATING", "MITIGATED"),
    ("INVESTIGATING", "RESOLVED"),
    ("MITIGATED", "RESOLVED"),
    ("MITIGATED", "INVESTIGATING"),
    ("RESOLVED", "POSTMORTEM"),
    ("RESOLVED", "MITIGATED"),
}

SERVICE_DEPENDENCY_GRAPH = {
    "api": ["broker", "database", "provider", "vector"],
    "broker": ["database"],
    "worker": ["broker", "provider", "vector", "database", "connectors"],
    "scheduler": ["broker", "database"],
    "event_processor": ["broker", "database"],
    "provider": ["database"],
    "vector": ["database"],
    "connectors": ["provider", "database"],
}


def _fingerprint(systems: list[str], summary: str) -> str:
    raw = ",".join(sorted(systems)) + ":" + (summary or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def dependency_graph() -> dict:
    return {"services": sorted(SERVICE_DEPENDENCY_GRAPH),
            "edges": [{"from": k, "to": v}
                      for k, vs in SERVICE_DEPENDENCY_GRAPH.items()
                      for v in vs]}


def ai_system_health(factors: dict) -> dict:
    """Unified health score (0..1) from weighted 0..1 factors:
    availability, latency, quality, cost, error_rate, queue_health,
    provider_health."""
    weights = {
        "availability": 0.25, "latency": 0.15, "quality": 0.15,
        "cost": 0.10, "error_rate": 0.15, "queue_health": 0.10,
        "provider_health": 0.10,
    }
    total = 0.0
    denom = 0.0
    signals = {}
    for k, w in weights.items():
        if k in factors:
            v = max(0.0, min(1.0, float(factors[k])))
            signals[k] = round(v, 4)
            total += v * w
            denom += w
    score = round((total / denom) if denom else 0.0, 4)
    status = "healthy" if score >= 0.8 else \
        ("degraded" if score >= 0.5 else "critical")
    return {"health_score": score, "status": status, "signals": signals,
            "weights": weights}


def correlate_incident(db: Session, *, title: str, severity: str,
                       affected_systems: list[str], summary: str,
                       fingerprint: Optional[str] = None) -> Incident:
    """Correlate failures into incidents: existing OPEN incident with the
    same fingerprint gets an event appended; otherwise a new incident."""
    fp = fingerprint or _fingerprint(affected_systems, summary)
    existing = db.query(Incident).filter_by(fingerprint=fp)\
        .filter(Incident.status.in_(["OPEN", "INVESTIGATING",
                                     "MITIGATED"])).first()
    if existing is not None:
        db.add(IncidentEvent(
            incident_id=existing.id, event_type="note",
            detail=f"Repeated correlation: {summary[:200]}"))
        existing.updated_at = __import__(
            "datetime").datetime.now(__import__("datetime").timezone.utc)
        db.flush()
        return existing
    inc = Incident(title=title, severity=severity,
                   affected_systems=json.dumps(affected_systems or []),
                   summary=summary, fingerprint=fp)
    db.add(inc)
    db.flush()
    db.add(IncidentEvent(incident_id=inc.id, event_type="detected",
                         detail=f"Incident opened: {summary[:200]}"))
    db.flush()
    return inc


def transition_incident(db: Session, *, incident_id: int, to_status: str,
                        actor_user_id: Optional[int] = None,
                        detail: Optional[str] = None) -> dict:
    """Validate + apply an incident lifecycle transition with a timeline
    event."""
    inc = db.get(Incident, incident_id)
    if inc is None:
        raise KeyError("incident not found")
    if to_status not in INCIDENT_STATUSES:
        raise ValueError(f"Unknown status: {to_status}")
    if (inc.status, to_status) not in INCIDENT_TRANSITIONS and \
            inc.status != to_status:
        raise ValueError(f"Transition {inc.status} -> {to_status} not "
                         "allowed")
    prev = inc.status
    inc.status = to_status
    from datetime import datetime, timezone
    inc.updated_at = datetime.now(timezone.utc)
    db.add(IncidentEvent(incident_id=incident_id,
                         event_type=to_status.lower(),
                         detail=detail or f"Status {prev} -> {to_status}",
                         actor_user_id=actor_user_id))
    db.flush()
    return {"incident_id": incident_id, "previous": prev,
            "new_status": to_status}


def list_incidents(db: Session, *, status: Optional[str] = None,
                   severity: Optional[str] = None,
                   limit: int = 100) -> list[dict]:
    q = db.query(Incident)
    if status:
        q = q.filter(Incident.status == status)
    if severity:
        q = q.filter(Incident.severity == severity)
    rows = q.order_by(Incident.created_at.desc()).limit(limit).all()
    return [{"id": i.id, "title": i.title, "severity": i.severity,
             "status": i.status,
             "affected_systems": json.loads(i.affected_systems or "[]"),
             "summary": i.summary, "created_at": i.created_at,
             "updated_at": i.updated_at} for i in rows]


def incident_timeline(db: Session, incident_id: int,
                      limit: int = 200) -> list[dict]:
    rows = db.query(IncidentEvent).filter_by(incident_id=incident_id)\
        .order_by(IncidentEvent.created_at.asc()).limit(limit).all()
    return [{"id": e.id, "event_type": e.event_type, "detail": e.detail,
             "actor_user_id": e.actor_user_id, "created_at": e.created_at}
            for e in rows]