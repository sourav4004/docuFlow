"""Phase 20 — Knowledge graph intelligence.

Graph health (orphan entities/relationships, conflicts, stale relationships,
low-confidence entities), graph drift over time, candidate graph
recommendations (entities/aliases/relationships — validation required), and
deterministic graph quality evaluation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import GraphHealth, GraphRecommendation
from ..models.knowledge_graph import Entity, EntityRelationship


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def graph_health(db: Session, *, workspace_id: int,
                 stale_days: int = 180) -> GraphHealth:
    """Compute and persist graph health for a workspace.

    Uses the actual Entity/EntityRelationship columns: relationships carry
    source_id/target_id and valid_until; entities carry last_seen_at.
    """
    entities = db.query(Entity).filter_by(workspace_id=workspace_id).all()
    rels = db.query(EntityRelationship).filter_by(
        workspace_id=workspace_id).all()
    entity_ids = {e.id for e in entities}
    orphan_rels = [r for r in rels
                   if r.source_id not in entity_ids
                   or r.target_id not in entity_ids]
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(days=stale_days)
    expired_rels = [r for r in rels
                    if r.valid_until is not None
                    and _as_utc(r.valid_until) < now]
    stale_rels = [r for r in rels
                  if r.observed_at is not None
                  and _as_utc(r.observed_at) < stale_cutoff
                  and r.is_current]
    # entities last seen long ago (no observed relationships recently)
    stale_entities = [e for e in entities
                      if e.last_seen_at is not None
                      and _as_utc(e.last_seen_at) < stale_cutoff]
    low_conf_entities = [e for e in entities
                         if getattr(e, "confidence", None) is None
                         or (isinstance(getattr(e, "confidence", None),
                                       (int, float))
                             and e.confidence < 0.5)]
    orphan_entities = [e for e in entities
                       if not any(r.source_id == e.id
                                  or r.target_id == e.id
                                  for r in rels)]
    total = max(len(entities), 1)
    signals = {
        "orphan_relationships": len(orphan_rels),
        "stale_relationships": len(stale_rels),
        "expired_relationships": len(expired_rels),
        "stale_entities": len(stale_entities),
        "low_confidence_entities": len(low_conf_entities),
        "orphan_entities": len(orphan_entities),
        "total_entities": len(entities),
        "total_relationships": len(rels),
    }
    score = 1.0 - (len(orphan_rels) * 0.3 + (len(stale_rels) +
                   len(expired_rels)) * 0.2 +
                   len(orphan_entities) * 0.1 +
                   len(low_conf_entities) * 0.1) / total
    score = round(max(score, 0.0), 4)
    row = GraphHealth(workspace_id=workspace_id,
                      metrics_json=_dumps(signals), score=score)
    db.add(row)
    db.flush()
    return row


def _as_utc(value):
    """Coerce naive datetimes to aware UTC for comparisons."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def health_summary(db: Session, *, workspace_id: int) -> Optional[dict]:
    row = db.query(GraphHealth).filter_by(workspace_id=workspace_id)\
        .order_by(GraphHealth.created_at.desc(),
                  GraphHealth.id.desc()).first()
    if row is None:
        return None
    return {"id": row.id, "score": row.score,
            "metrics": json.loads(row.metrics_json or "{}"),
            "created_at": row.created_at}


def graph_drift(db: Session, *, workspace_id: int) -> dict:
    """Relationship-change drift: compare the two most recent health
    snapshots."""
    rows = db.query(GraphHealth).filter_by(workspace_id=workspace_id)\
        .order_by(GraphHealth.created_at.desc()).limit(2).all()
    if len(rows) < 2:
        return {"drift": False, "reason": "insufficient history"}
    a, b = rows[0], rows[1]
    ma = json.loads(a.metrics_json or "{}")
    mb = json.loads(b.metrics_json or "{}")
    changed = sorted({
        k for k in set(ma) | set(mb)
        if ma.get(k) != mb.get(k) and k not in (
            "total_entities", "total_relationships", "stale_entities")})
    return {"drift": bool(changed), "changed_metrics": changed,
            "from_score": b.score, "to_score": a.score,
            "from_time": b.created_at, "to_time": a.created_at}


def recommend(db: Session, *, workspace_id: int, kind: str,
              candidate: str, rationale: Optional[str] = None) -> dict:
    if kind not in ("entity", "alias", "relationship"):
        raise ValueError(f"Unknown recommendation kind: {kind}")
    rec = GraphRecommendation(workspace_id=workspace_id, kind=kind,
                              candidate=candidate, rationale=rationale)
    db.add(rec)
    db.flush()
    return {"id": rec.id, "kind": kind, "candidate": candidate,
            "status": rec.status}


def generate_recommendations(db: Session, *, workspace_id: int) -> list[dict]:
    """Generate candidate aliases for repeated re-occurring entity names —
    all candidates require validation."""
    entities = db.query(Entity).filter_by(workspace_id=workspace_id).all()
    created = []
    name_counts: dict[str, int] = {}
    for e in entities:
        name = getattr(e, "name", None)
        if name:
            name_counts[name.lower()] = name_counts.get(name.lower(), 0) + 1
    for name, count in name_counts.items():
        if count > 1:
            created.append(recommend(
                db, workspace_id=workspace_id, kind="alias",
                candidate=f"Canonicalize '{name}' across {count} entities",
                rationale="Repeated entity name detected"))
    return created


def quality_evaluation(entities) -> dict:
    """Deterministic graph quality evaluation from a provided entity list
    (each with name/confidence/relationships)."""
    total = len(entities)
    if not total:
        return {"score": 0.0, "signals": {"no_entities": True}}
    have_name = sum(1 for e in entities if e.get("name"))
    have_rel = sum(1 for e in entities if e.get("relationships"))
    confidences = []
    for e in entities:
        c = e.get("confidence")
        if isinstance(c, (int, float)):
            confidences.append(c)
        elif c:
            confidences.append(1.0 if c == "HIGH" else
                               0.5 if c == "MEDIUM" else 0.2)
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    score = round((have_name / total) * 0.4 +
                  (have_rel / total) * 0.3 + avg_conf * 0.3, 4)
    return {"score": score,
            "signals": {"name_coverage": round(have_name / total, 4),
                        "relationship_coverage": round(have_rel / total, 4),
                        "avg_confidence": round(avg_conf, 4)}}