"""Knowledge graph 3.0 — canonicalization, relationship validity, change
detection and entity-centric timelines.

Canonicalization NEVER merges entities automatically: duplicate candidates
are surfaced with evidence, and merging requires an explicit human decision.
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.knowledge_graph import Entity, EntityRelationship
from ..models.phase16 import EntityChange, ENTITY_CHANGE_TYPES

_CHANGE_TYPES = set(ENTITY_CHANGE_TYPES)


def canonical_name(name: str) -> str:
    """Deterministic canonical form for entity matching (never destructive)."""
    text = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return re.sub(r"\s+", " ", text)


def register_entity(
    db: Session,
    workspace_id: int,
    name: str,
    entity_type: str,
    aliases: Optional[list[str]] = None,
    document_id: Optional[int] = None,
    confidence: float = 1.0,
) -> Entity:
    """Register an entity — matching on (workspace, canonical, type).

    Existing entities accumulate aliases and bump last_seen/source_count;
    nothing is deleted or merged.
    """
    norm = canonical_name(name)
    entity = (
        db.query(Entity)
        .filter(Entity.workspace_id == workspace_id,
                Entity.normalized_name == norm,
                Entity.entity_type == entity_type)
        .first()
    )
    now = datetime.now(timezone.utc)
    if entity is None:
        entity = Entity(workspace_id=workspace_id, name=name,
                        entity_type=entity_type,
                        normalized_name=norm, confidence=confidence,
                        first_seen_at=now, last_seen_at=now)
        db.add(entity)
        db.flush()
        # First sighting is always recorded on the entity timeline.
        db.add(EntityChange(
            workspace_id=workspace_id, entity_id=entity.id,
            change_type="NEW_ENTITY", document_id=document_id,
            new_value=name,
            evidence=f"document {document_id}" if document_id else None,
            confidence="HIGH"))
    else:
        existing_aliases = _alias_list(entity.aliases)
        merged = set(existing_aliases + (aliases or [])) - {name}
        entity.aliases = json.dumps(sorted(merged)) if merged else None
        entity.last_seen_at = now
        entity.source_count = (entity.source_count or 1) + 1
        entity.confidence = round(
            max(entity.confidence or 0, confidence) * 0.9 + confidence * 0.1, 3)
        db.flush()
    return entity


def _alias_list(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except (ValueError, TypeError):
        return []


def add_alias(db: Session, entity_id: int, alias: str) -> Entity:
    entity = db.query(Entity).filter(Entity.id == entity_id).first()
    if entity is None:
        raise ValueError("Entity not found")
    aliases = set(_alias_list(entity.aliases)) | {alias}
    aliases.discard(entity.name)
    entity.aliases = json.dumps(sorted(aliases)) if aliases else None
    db.flush()
    return entity


def duplicate_candidates(db: Session, workspace_id: int) -> list[dict]:
    """Entities that look like duplicates (same canonical name, different
    spelling/type) — surfaced with evidence; never auto-merged."""
    rows = (
        db.query(Entity)
        .filter(Entity.workspace_id == workspace_id,
                Entity.normalized_name.isnot(None))
        .all()
    )
    by_norm: dict = {}
    for e in rows:
        by_norm.setdefault(e.normalized_name, []).append(e)
    candidates = []
    for norm, group in by_norm.items():
        if len(group) < 2:
            continue
        primary = max(group, key=lambda e: e.source_count or 1)
        for other in group:
            if other.id == primary.id:
                continue
            candidates.append({
                "primary_entity_id": primary.id,
                "primary_name": primary.name,
                "candidate_id": other.id,
                "candidate_name": other.name,
                "canonical": norm,
                "reason": f"same canonical name '{norm}'",
                "confidence": round(min(primary.confidence or 0.5,
                                        other.confidence or 0.5), 3),
            })
    return candidates


def record_relationship(
    db: Session,
    workspace_id: int,
    source_id: int,
    target_id: int,
    relationship_type: str,
    confidence: float = 0.8,
    source_document_id: Optional[int] = None,
    evidence: Optional[str] = None,
    valid_from: Optional[datetime] = None,
    valid_until: Optional[datetime] = None,
) -> EntityRelationship:
    """Record a relationship with confidence/evidence/validity.

    When the same (source, target, type) already exists with different
    observation evidence, the old row is closed (valid_until = now,
    is_current = False) and an EntityChange event is recorded — relationships
    are never silently overwritten.
    """
    now = datetime.now(timezone.utc)
    prior = (
        db.query(EntityRelationship)
        .filter(EntityRelationship.workspace_id == workspace_id,
                EntityRelationship.source_id == source_id,
                EntityRelationship.target_id == target_id,
                EntityRelationship.relationship_type == relationship_type,
                EntityRelationship.is_current.is_(True))
        .first()
    )
    if prior is not None and prior.observed_at is not None:
        prior.is_current = False
        prior.valid_until = now
        db.add(EntityChange(
            workspace_id=workspace_id, entity_id=source_id,
            change_type="RELATIONSHIP_CHANGED",
            document_id=source_document_id,
            old_value=f"{relationship_type} -> {target_id}",
            new_value=f"{relationship_type} -> {target_id} (re-observed)",
            evidence=evidence, confidence="MEDIUM"))
    rel = EntityRelationship(
        workspace_id=workspace_id,
        source_id=source_id,
        target_id=target_id,
        relationship_type=relationship_type,
        confidence=round(max(0.0, min(1.0, confidence)), 3),
        source_document_id=source_document_id,
        valid_from=valid_from or now,
        valid_until=valid_until,
        observed_at=now,
        is_current=True,
    )
    db.add(rel)
    db.flush()
    return rel


def detect_entity_attribute_change(
    db: Session,
    entity: Entity,
    attribute: str,
    old_value,
    new_value,
    document_id: Optional[int] = None,
    evidence: Optional[str] = None,
) -> Optional[EntityChange]:
    """Record an entity attribute change when a value actually changed."""
    if old_value == new_value:
        return None
    change_type = ("RENAMED" if attribute == "name" else "ATTRIBUTE_CHANGED")
    event = EntityChange(
        workspace_id=entity.workspace_id, entity_id=entity.id,
        change_type=change_type, document_id=document_id,
        old_value=str(old_value) if old_value is not None else None,
        new_value=str(new_value) if new_value is not None else None,
        evidence=evidence, confidence="MEDIUM",
    )
    db.add(event)
    db.flush()
    return event


def entity_timeline(db: Session, entity_id: int, limit: int = 200) -> list[dict]:
    """Entity-centric timeline: changes + relationship lifecycle events."""
    entity = db.query(Entity).filter(Entity.id == entity_id).first()
    if entity is None:
        return []
    events = (
        db.query(EntityChange)
        .filter(EntityChange.entity_id == entity_id)
        .order_by(EntityChange.created_at.desc())
        .limit(limit)
        .all()
    )
    rels = (
        db.query(EntityRelationship)
        .filter((EntityRelationship.source_id == entity_id)
                | (EntityRelationship.target_id == entity_id))
        .order_by(EntityRelationship.observed_at.desc())
        .limit(limit)
        .all()
    )
    timeline = [
        {
            "type": f"entity_change:{e.change_type.lower()}",
            "at": e.created_at,
            "entity_id": e.entity_id,
            "document_id": e.document_id,
            "detail": f"{e.change_type}: {e.old_value} -> {e.new_value}",
        }
        for e in events
    ]
    for r in rels:
        direction = "out" if r.source_id == entity_id else "in"
        timeline.append({
            "type": f"relationship:{r.relationship_type.lower()}:{direction}",
            "at": r.observed_at,
            "entity_id": entity_id,
            "document_id": r.source_document_id,
            "detail": f"{r.relationship_type} confidence={r.confidence} "
                      f"current={r.is_current}",
        })
    timeline.sort(key=lambda item: item["at"], reverse=True)
    return timeline


def entity_summary(db: Session, workspace_id: int, entity_id: int) -> dict:
    """Entity-centric intelligence: entity, relationships, timeline, sources."""
    entity = db.query(Entity).filter(
        Entity.id == entity_id, Entity.workspace_id == workspace_id).first()
    if entity is None:
        raise ValueError("Entity not found in workspace")
    rels = (
        db.query(EntityRelationship)
        .filter(EntityRelationship.workspace_id == workspace_id,
                EntityRelationship.is_current.is_(True),
                (EntityRelationship.source_id == entity_id)
                | (EntityRelationship.target_id == entity_id))
        .all()
    )
    out = []
    for r in rels:
        other_id = r.target_id if r.source_id == entity_id else r.source_id
        other = db.query(Entity).filter(Entity.id == other_id).first()
        out.append({
            "related_entity_id": other_id,
            "related_name": other.name if other else None,
            "direction": "outgoing" if r.source_id == entity_id else "incoming",
            "relationship_type": r.relationship_type,
            "confidence": r.confidence,
            "evidence_document_id": r.source_document_id,
            "is_current": r.is_current,
            "valid_until": r.valid_until,
        })
    docs = {r["evidence_document_id"] for r in out if r["evidence_document_id"]}
    return {
        "entity": {
            "id": entity.id,
            "name": entity.name,
            "normalized_name": entity.normalized_name,
            "entity_type": entity.entity_type,
            "aliases": _alias_list(entity.aliases),
            "confidence": entity.confidence,
            "source_count": entity.source_count,
            "first_seen_at": entity.first_seen_at,
            "last_seen_at": entity.last_seen_at,
        },
        "relationships": out,
        "source_documents": sorted(docs),
        "timeline": entity_timeline(db, entity_id),
    }
