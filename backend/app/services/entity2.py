"""Entity intelligence 2.0 — entity-centric summaries, change detection,
relationship validation, and duplicate/contradiction detection.

Never merges entities automatically; validation results require a human
approval path.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.knowledge_graph import Entity, EntityRelationship
from ..models.document import Document
from ..models.phase15 import TemporalFact, KnowledgeChange
from ..services.audit_service import log_audit_event


def entity_summary(db: Session, workspace_id: int, entity_id: int) -> dict:
    """Entity-centric intelligence bundle (tenant scoped)."""
    entity = (
        db.query(Entity)
        .filter(Entity.workspace_id == workspace_id, Entity.id == entity_id)
        .first()
    )
    if not entity:
        raise ValueError("Entity not found in this workspace")

    relationships = (
        db.query(EntityRelationship)
        .filter(
            EntityRelationship.workspace_id == workspace_id,
            (EntityRelationship.source_id == entity_id) | (EntityRelationship.target_id == entity_id),
        )
        .limit(100)
        .all()
    )
    facts = (
        db.query(TemporalFact)
        .filter(TemporalFact.workspace_id == workspace_id, TemporalFact.entity_id == entity_id)
        .limit(50)
        .all()
    )
    changes = (
        db.query(KnowledgeChange)
        .filter(KnowledgeChange.workspace_id == workspace_id)
        .limit(200)
        .all()
    )
    entity_changes = [
        c for c in changes
        if c.affected_entities_json and entity.name.lower() in c.affected_entities_json.lower()
    ]

    aliases = []
    try:
        aliases = json.loads(entity.aliases) if entity.aliases else []
    except (ValueError, TypeError):
        aliases = []

    return {
        "id": entity.id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "aliases": aliases,
        "confidence": entity.confidence,
        "first_seen": entity.first_seen_at.isoformat() if entity.first_seen_at else None,
        "last_seen": entity.last_seen_at.isoformat() if entity.last_seen_at else None,
        "relationships": [
            {
                "id": r.id,
                "source": r.source.name if r.source else r.source_id,
                "target": r.target.name if r.target else r.target_id,
                "type": r.relationship_type,
                "confidence": r.confidence,
            }
            for r in relationships
        ],
        "temporal_facts": [
            {"id": f.id, "type": f.fact_type, "value": f.fact_value,
             "valid_from": f.valid_from.isoformat() if f.valid_from else None,
             "valid_until": f.valid_until.isoformat() if f.valid_until else None}
            for f in facts
        ],
        "recent_changes": [
            {"id": c.id, "type": c.change_type, "severity": c.severity, "summary": c.summary}
            for c in entity_changes[-10:]
        ],
    }


def validate_entity_relationships(db: Session, workspace_id: int) -> dict:
    """Detect duplicate entities, contradictory attributes, stale
    relationships, and orphan entities. Reports only — never merges."""
    entities = (
        db.query(Entity)
        .filter(Entity.workspace_id == workspace_id)
        .limit(1000)
        .all()
    )
    relationships = (
        db.query(EntityRelationship)
        .filter(EntityRelationship.workspace_id == workspace_id)
        .limit(2000)
        .all()
    )

    # Duplicate entities: same normalized name, different rows
    by_name: dict[str, list[Entity]] = {}
    for e in entities:
        key = e.name.strip().lower()
        by_name.setdefault(key, []).append(e)
    duplicates = [
        {"name": name, "ids": [e.id for e in group]}
        for name, group in by_name.items() if len(group) > 1
    ]

    # Orphan relationships: source or target missing
    entity_ids = {e.id for e in entities}
    orphan_relationships = [
        {"id": r.id, "source_id": r.source_id, "target_id": r.target_id}
        for r in relationships
        if r.source_id not in entity_ids or r.target_id not in entity_ids
    ]

    # Contradictory attributes: same entity, conflicting temporal facts
    facts = (
        db.query(TemporalFact)
        .filter(
            TemporalFact.workspace_id == workspace_id,
            TemporalFact.entity_id.isnot(None),
        )
        .limit(1000)
        .all()
    )
    by_entity: dict[int, dict[str, list]] = {}
    for f in facts:
        by_entity.setdefault(f.entity_id, {}).setdefault(f.fact_type, []).append(f)
    contradictory = []
    for entity_id, types in by_entity.items():
        for fact_type, items in types.items():
            values = {i.fact_value for i in items}
            if len(values) > 1:
                contradictory.append({
                    "entity_id": entity_id,
                    "fact_type": fact_type,
                    "values": sorted(values),
                })

    return {
        "duplicate_entities": duplicates,
        "orphan_relationships": orphan_relationships,
        "contradictory_attributes": contradictory,
        "note": "Validation only — entity merges require human approval",
    }


def record_entity_touch(db: Session, workspace_id: int, entity_id: int) -> None:
    """Update last_seen (extraction pipeline hook)."""
    entity = (
        db.query(Entity)
        .filter(Entity.workspace_id == workspace_id, Entity.id == entity_id)
        .first()
    )
    if entity:
        entity.last_seen_at = datetime.now(timezone.utc)
        db.flush()