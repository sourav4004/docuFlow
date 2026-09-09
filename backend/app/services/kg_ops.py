"""Phase 18 knowledge graph operations — merge workflow, change events,
temporal relationships, bounded traversal.

Ambiguous merges are NEVER automatic: they become EntityMergeRequest rows
requiring human approval with expiry. Traversal is depth-bounded to protect
the database.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase18 import EntityMergeRequest
from ..models.knowledge_graph import Entity, EntityRelationship
from ..models.phase16 import EntityChange

logger = logging.getLogger(__name__)

MERGE_EXPIRY_DAYS = 7
MAX_TRAVERSAL_DEPTH = 6
MAX_TRAVERSAL_NODES = 500


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _owned_entity(db: Session, workspace_id: int, entity_id: int) -> Entity:
    row = (db.query(Entity)
           .filter(Entity.id == entity_id,
                   Entity.workspace_id == workspace_id)
           .first())
    if row is None:
        raise KeyError(f"entity {entity_id} not found in workspace")
    return row


# ---------------------------------------------------------------------------
# Merge workflow (approval-gated)
# ---------------------------------------------------------------------------

def request_merge(db: Session, *, workspace_id: int,
                  source_entity_id: int, target_entity_id: int,
                  reason: Optional[str] = None,
                  evidence: Optional[dict] = None,
                  requested_by: Optional[int] = None,
                  expires_in_days: int = MERGE_EXPIRY_DAYS) -> EntityMergeRequest:
    """Create a merge request. Never merges automatically."""
    if source_entity_id == target_entity_id:
        raise ValueError("cannot merge an entity with itself")
    _owned_entity(db, workspace_id, source_entity_id)
    _owned_entity(db, workspace_id, target_entity_id)
    import json
    existing = (db.query(EntityMergeRequest)
                .filter(EntityMergeRequest.workspace_id == workspace_id,
                        EntityMergeRequest.status == "PENDING",
                        EntityMergeRequest.source_entity_id == source_entity_id,
                        EntityMergeRequest.target_entity_id == target_entity_id)
                .first())
    if existing is not None:
        return existing
    row = EntityMergeRequest(
        workspace_id=workspace_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        reason=(reason or "")[:1000],
        evidence_json=json.dumps(evidence or {})[:4000],
        requested_by=requested_by,
        expires_at=_utcnow() + timedelta(days=expires_in_days))
    db.add(row)
    db.flush()
    return row


def decide_merge(db: Session, *, workspace_id: int, merge_id: int,
                 decision: str, reviewer_id: int) -> dict:
    """APPROVE or REJECT a merge request. Approval performs the merge."""
    if decision not in ("APPROVE", "REJECT"):
        raise ValueError("decision must be APPROVE or REJECT")
    row = (db.query(EntityMergeRequest)
           .filter(EntityMergeRequest.id == merge_id,
                   EntityMergeRequest.workspace_id == workspace_id)
           .first())
    if row is None:
        raise KeyError(f"merge request {merge_id} not found")
    if row.status != "PENDING":
        raise ValueError(f"merge request already {row.status}")
    now = _utcnow()
    if row.expires_at is not None and _as_utc(row.expires_at) < now:
        row.status = "EXPIRED"
        db.flush()
        raise ValueError("merge request expired")
    if decision == "REJECT":
        row.status = "REJECTED"
        row.reviewed_by = reviewer_id
        row.reviewed_at = now
        db.flush()
        return {"merged": False, "status": "REJECTED"}
    # APPROVE — perform the merge within the workspace.
    source = _owned_entity(db, workspace_id, row.source_entity_id)
    target = _owned_entity(db, workspace_id, row.target_entity_id)
    # Re-point all relationships at the target.
    (db.query(EntityRelationship)
     .filter(EntityRelationship.workspace_id == workspace_id,
             EntityRelationship.source_id == source.id)
     .update({"source_id": target.id}))
    (db.query(EntityRelationship)
     .filter(EntityRelationship.workspace_id == workspace_id,
             EntityRelationship.target_id == source.id)
     .update({"target_id": target.id}))
    # Merge aliases (stored as JSON text).
    import json as _json
    def _alias_set(raw):
        if not raw:
            return set()
        try:
            value = _json.loads(raw)
            return set(value) if isinstance(value, list) else set()
        except Exception:  # noqa: BLE001
            return set()
    aliases = _alias_set(target.aliases)
    aliases.update(_alias_set(source.aliases))
    target.aliases = _json.dumps(sorted(aliases))
    # Record the change event, then soft-canonicalize: source keeps its row
    # flagged as merged so history remains intact (no destructive delete).
    db.add(EntityChange(
        workspace_id=workspace_id,
        entity_id=target.id,
        change_type="MERGED",
        evidence=(f"merged entity {source.id} into {target.id}"),
        created_at=now))
    import json as _json
    try:
        meta = _json.loads(source.metadata_json) if source.metadata_json else {}
    except Exception:  # noqa: BLE001
        meta = {}
    meta["merged_into"] = target.id
    source.metadata_json = _json.dumps(meta)[:4000]
    row.status = "APPROVED"
    row.reviewed_by = reviewer_id
    row.reviewed_at = now
    db.flush()
    return {"merged": True, "status": "APPROVED", "target_entity_id": target.id}


def list_merge_requests(db: Session, workspace_id: Optional[int] = None,
                        status: Optional[str] = None,
                        limit: int = 50) -> dict:
    q = db.query(EntityMergeRequest)
    if workspace_id is not None:
        q = q.filter(EntityMergeRequest.workspace_id == workspace_id)
    if status:
        q = q.filter(EntityMergeRequest.status == status)
    rows = q.order_by(EntityMergeRequest.created_at.desc()).limit(
        min(limit, 200)).all()
    return {"items": [{"id": r.id, "workspace_id": r.workspace_id,
                       "source_entity_id": r.source_entity_id,
                       "target_entity_id": r.target_entity_id,
                       "reason": r.reason, "status": r.status,
                       "requested_by": r.requested_by,
                       "reviewed_by": r.reviewed_by,
                       "expires_at": r.expires_at,
                       "created_at": r.created_at} for r in rows],
            "total": len(rows)}


# ---------------------------------------------------------------------------
# Temporal relationships
# ---------------------------------------------------------------------------

def add_temporal_relationship(
    db: Session, *, workspace_id: int, source_id: int, target_id: int,
    rel_type: str, valid_from: Optional[datetime] = None,
    valid_to: Optional[datetime] = None,
    confidence: float = 0.5,
    evidence: Optional[dict] = None) -> EntityRelationship:
    """Add a relationship with a validity period and evidence provenance."""
    if valid_from is not None and valid_to is not None and valid_from > valid_to:
        raise ValueError("valid_from must precede valid_to")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    _owned_entity(db, workspace_id, source_id)
    _owned_entity(db, workspace_id, target_id)
    import json
    row = EntityRelationship(
        workspace_id=workspace_id,
        source_id=source_id,
        target_id=target_id,
        relationship_type=rel_type[:100],
        confidence=confidence,
        valid_from=valid_from,
        valid_until=valid_to,
        metadata_json=json.dumps({"evidence": evidence or {},
                                  "source": "phase18"})[:4000])
    db.add(row)
    db.flush()
    return row


def relationships_active_at(db: Session, workspace_id: int,
                            entity_id: int, as_of: datetime) -> list[dict]:
    """Relationships valid ``as_of`` — the temporal query boundary."""
    _owned_entity(db, workspace_id, entity_id)
    rows = (db.query(EntityRelationship)
            .filter(EntityRelationship.workspace_id == workspace_id,
                    ((EntityRelationship.source_id == entity_id) |
                     (EntityRelationship.target_id == entity_id)),
                    EntityRelationship.valid_from.is_(None) |
                    (EntityRelationship.valid_from <= as_of),
                    EntityRelationship.valid_until.is_(None) |
                    (EntityRelationship.valid_until >= as_of))
            .limit(200).all())
    return [{"source_id": r.source_id, "target_id": r.target_id,
             "rel_type": r.relationship_type, "confidence": r.confidence,
             "valid_from": r.valid_from, "valid_until": r.valid_until}
            for r in rows]


# ---------------------------------------------------------------------------
# Bounded traversal
# ---------------------------------------------------------------------------

def traverse_entity(db: Session, *, workspace_id: int, entity_id: int,
                    max_depth: int = MAX_TRAVERSAL_DEPTH,
                    max_nodes: int = MAX_TRAVERSAL_NODES) -> dict:
    """BFS traversal with hard depth + node bounds."""
    if max_depth > MAX_TRAVERSAL_DEPTH:
        raise ValueError(
            f"traversal depth capped at {MAX_TRAVERSAL_DEPTH}")
    _owned_entity(db, workspace_id, entity_id)
    visited: dict[int, dict] = {}
    frontier = [entity_id]
    depth = 0
    while frontier and depth <= max_depth and len(visited) < max_nodes:
        next_frontier: list[int] = []
        for eid in frontier:
            if eid in visited or len(visited) >= max_nodes:
                continue
            entity = (db.query(Entity)
                      .filter(Entity.id == eid,
                              Entity.workspace_id == workspace_id)
                      .first())
            if entity is None:
                continue
            visited[eid] = {"id": entity.id, "name": entity.name,
                            "entity_type": entity.entity_type}
            rels = (db.query(EntityRelationship)
                    .filter(EntityRelationship.workspace_id == workspace_id,
                            ((EntityRelationship.source_id == eid) |
                             (EntityRelationship.target_id == eid)))
                    .limit(50).all())
            for r in rels:
                neighbor = r.target_id if r.source_id == eid else r.source_id
                if neighbor not in visited and neighbor not in next_frontier:
                    next_frontier.append(neighbor)
        frontier = next_frontier
        depth += 1
    return {"root": entity_id, "depth_reached": depth,
            "node_count": len(visited),
            "truncated": len(visited) >= max_nodes or bool(frontier),
            "entities": list(visited.values())}