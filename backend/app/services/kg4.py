"""Organization knowledge graph 4.0 — Phase 17.

- canonical entity representation with aliases/confidence/temporal validity
- deterministic entity-resolution *candidates* (never auto-merged)
- relationship suggestions requiring validation before persistence
- incompatible-relationship conflict detection
- organization-level aggregation across member workspaces
- bounded, workspace-isolated graph search with temporal filtering
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.knowledge_graph import Entity, EntityRelationship
from ..models.phase17 import EntityCandidate, RelationshipSuggestion

logger = logging.getLogger(__name__)

_NORM_RE = re.compile(r"[^a-z0-9]+")

MAX_TRAVERSAL_DEPTH = 4
MAX_TRAVERSAL_NODES = 200


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_name(name: str) -> str:
    return _NORM_RE.sub(" ", (name or "").lower()).strip()


# ---------------------------------------------------------------------------
# Canonical entities / resolution
# ---------------------------------------------------------------------------

def suggest_entity_candidates(db: Session, workspace_id: int,
                              entity_id: int) -> list[dict]:
    """Deterministic candidate matching: normalized-name and alias overlap.

    Produces PENDING candidates only — merging requires human approval.
    """
    entity = db.query(Entity).filter(
        Entity.id == entity_id, Entity.workspace_id == workspace_id).first()
    if entity is None:
        raise ValueError(f"Entity {entity_id} not found")
    name = normalize_name(entity.name)
    if not name:
        return []
    peers = db.query(Entity).filter(
        Entity.id != entity_id, Entity.workspace_id == workspace_id,
        Entity.id > entity_id).limit(500).all()
    results = []
    for peer in peers:
        peer_name = normalize_name(peer.name)
        score = 0.0
        method = None
        if peer_name and peer_name == name:
            score = 1.0
            method = "normalized_name"
        elif peer_name and (name in peer_name or peer_name in name):
            score = 0.7
            method = "name_overlap"
        else:
            aliases = _load_aliases(peer.aliases)
            if name in aliases:
                score = 0.8
                method = "alias_match"
        if score < 0.7 and peer_name:
            # token overlap handles legal suffixes (Ltd/Limited, Inc/Corp)
            name_tokens = set(name.split())
            peer_tokens = set(peer_name.split())
            common = name_tokens & peer_tokens
            if common and len(common) >= max(1, min(len(name_tokens),
                                                    len(peer_tokens)) - 1):
                score = 0.75
                method = "token_overlap"
        if score >= 0.7:
            existing = (
                db.query(EntityCandidate)
                .filter(EntityCandidate.entity_id == entity_id,
                        EntityCandidate.candidate_id == peer.id)
                .first()
            )
            if existing is None:
                db.add(EntityCandidate(
                    workspace_id=workspace_id, entity_id=entity_id,
                    candidate_id=peer.id, score=score, method=method,
                    reason=f"matched via {method}"))
                db.flush()
            results.append({"candidate_id": peer.id, "score": round(score, 3),
                            "method": method, "status": "PENDING"})
    return results


def _load_aliases(raw: Optional[str]) -> set[str]:
    try:
        return {normalize_name(a) for a in json.loads(raw or "[]")}
    except (ValueError, TypeError):
        return set()


def resolve_candidate(db: Session, workspace_id: int, candidate_id: int,
                      decision: str, resolved_by: int) -> dict:
    """Human decision on a resolution candidate (APPROVED / REJECTED)."""
    cand = db.query(EntityCandidate).filter(
        EntityCandidate.id == candidate_id,
        EntityCandidate.workspace_id == workspace_id).first()
    if cand is None:
        raise ValueError(f"Candidate {candidate_id} not found")
    if decision not in ("APPROVED", "REJECTED"):
        raise ValueError("decision must be APPROVED or REJECTED")
    # Defense in depth: both referenced entities must live in the candidate's
    # workspace — hand-crafted cross-workspace candidates are rejected even
    # though candidate creation already enforces this.
    entity = db.query(Entity).filter(
        Entity.id == cand.entity_id,
        Entity.workspace_id == cand.workspace_id).first()
    peer = db.query(Entity).filter(
        Entity.id == cand.candidate_id,
        Entity.workspace_id == cand.workspace_id).first()
    if entity is None or peer is None:
        raise ValueError(
            "candidate references an entity outside its workspace")
    cand.status = decision
    db.flush()
    return {"candidate_id": cand.id, "status": cand.status,
            "entity_id": cand.entity_id,
            "candidate_entity_id": cand.candidate_id}


# ---------------------------------------------------------------------------
# Relationship suggestions + conflict detection
# ---------------------------------------------------------------------------

def _document_ids(db: Session, entity_id: int, workspace_id: int) -> set[int]:
    """Document ids where the entity is linked as source or target."""
    rows = db.query(EntityRelationship).filter(
        EntityRelationship.workspace_id == workspace_id,
        (EntityRelationship.source_id == entity_id)
        | (EntityRelationship.target_id == entity_id)).limit(500).all()
    return {r.source_document_id for r in rows if r.source_document_id}


def suggest_relationships(db: Session, workspace_id: int,
                          entity_a_id: int, limit: int = 100) -> list[dict]:
    """Deterministic shared-evidence suggestion. Two entities linked to the
    same document are candidates for an (undirected) relationship."""
    a = db.query(Entity).filter(
        Entity.id == entity_a_id, Entity.workspace_id == workspace_id).first()
    if a is None:
        raise ValueError(f"Entity {entity_a_id} not found")
    docs_a = _document_ids(db, entity_a_id, workspace_id)
    if not docs_a:
        return []
    peers = db.query(Entity).filter(
        Entity.id != entity_a_id, Entity.workspace_id == workspace_id,
        Entity.id > entity_a_id).limit(500).all()
    suggestions = []
    for peer in peers:
        docs_b = _document_ids(db, peer.id, workspace_id)
        shared = docs_a & docs_b
        if not shared:
            continue
        confidence = min(0.95, 0.45 + 0.15 * len(shared))
        evidence = f"co-occurs in {len(shared)} shared document(s): " \
                   f"{sorted(shared)[:3]}"
        existing = (
            db.query(RelationshipSuggestion)
            .filter(RelationshipSuggestion.entity_a_id == entity_a_id,
                    RelationshipSuggestion.entity_b_id == peer.id)
            .first()
        )
        if existing is None:
            db.add(RelationshipSuggestion(
                workspace_id=workspace_id, entity_a_id=entity_a_id,
                entity_b_id=peer.id, relation_type="related_to",
                confidence=confidence, evidence=evidence,
                method="shared_evidence"))
            db.flush()
        suggestions.append({"entity_b_id": peer.id,
                            "confidence": round(confidence, 3),
                            "evidence": evidence, "status": "PENDING"})
        if len(suggestions) >= limit:
            break
    return suggestions


def apply_relationship_suggestion(db: Session, workspace_id: int,
                                  suggestion_id: int, relation_type: str,
                                  decided_by: int) -> dict:
    """Promote a validated suggestion into a real relationship (gated —
    never automatic)."""
    sug = db.query(RelationshipSuggestion).filter(
        RelationshipSuggestion.id == suggestion_id,
        RelationshipSuggestion.workspace_id == workspace_id).first()
    if sug is None:
        raise ValueError(f"Suggestion {suggestion_id} not found")
    existing = db.query(EntityRelationship).filter(
        EntityRelationship.workspace_id == workspace_id,
        EntityRelationship.source_id == sug.entity_a_id,
        EntityRelationship.target_id == sug.entity_b_id,
        EntityRelationship.relationship_type == relation_type).first()
    if existing is None:
        db.add(EntityRelationship(
            workspace_id=workspace_id,
            source_id=sug.entity_a_id,
            target_id=sug.entity_b_id,
            relationship_type=relation_type,
            confidence=sug.confidence,
            source_document_id=None,
            metadata_json=json.dumps({"kind": "suggestion",
                                      "suggestion_id": sug.id,
                                      "evidence": sug.evidence}),
        ))
    sug.status = "APPROVED"
    sug.relation_type = relation_type
    db.flush()
    return {"suggestion_id": sug.id, "status": "APPROVED",
            "relationship": relation_type}


def detect_relationship_conflicts(db: Session, workspace_id: int,
                                  entity_id: int) -> list[dict]:
    """Detect incompatible relationships on the same entity pair."""
    rels = db.query(EntityRelationship).filter(
        EntityRelationship.workspace_id == workspace_id,
        (EntityRelationship.source_id == entity_id)
        | (EntityRelationship.target_id == entity_id)).all()
    by_pair: dict[tuple[int, int], set[str]] = {}
    for r in rels:
        key = tuple(sorted((r.source_id, r.target_id)))
        by_pair.setdefault(key, set()).add(r.relationship_type)
    conflicts = []
    for key, types in by_pair.items():
        if len(types) >= 2:
            conflicts.append({
                "entity_pair": list(key),
                "relationship_types": sorted(types),
                "category": "multiple_conflicting_types",
                "message": "same entity pair carries multiple relationship "
                           "types and needs review",
            })
    return conflicts


# ---------------------------------------------------------------------------
# Organization aggregation + graph search
# ---------------------------------------------------------------------------

def organization_graph_summary(db: Session, organization_id: int,
                               workspace_ids: list[int]) -> dict:
    if not workspace_ids:
        return {"organization_id": organization_id, "workspaces": 0,
                "entities": 0, "documents": 0, "policies": 0,
                "deadlines": 0, "workflows": 0}
    from ..models.workspace import Workspace
    from ..models.document import Document
    from ..models.knowledge import Deadline
    from ..models.phase15 import PolicyStatement
    ws_count = db.query(Workspace).filter(
        Workspace.id.in_(workspace_ids),
        Workspace.organization_id == organization_id).count()
    entity_count = db.query(Entity).filter(
        Entity.workspace_id.in_(workspace_ids)).count()
    doc_count = db.query(Document).filter(
        Document.workspace_id.in_(workspace_ids)).count()
    policy_count = db.query(PolicyStatement).filter(
        PolicyStatement.workspace_id.in_(workspace_ids)).count()
    deadline_count = db.query(Deadline).filter(
        Deadline.workspace_id.in_(workspace_ids)).count()
    return {
        "organization_id": organization_id,
        "workspaces": ws_count,
        "entities": entity_count,
        "documents": doc_count,
        "policies": policy_count,
        "deadlines": deadline_count,
    }


def graph_search(db: Session, workspace_id: int, entity_id: int,
                 max_depth: int = MAX_TRAVERSAL_DEPTH,
                 temporal_as_of: Optional[datetime] = None) -> dict:
    """Bounded BFS traversal over relationships inside one workspace."""
    root = db.query(Entity).filter(
        Entity.id == entity_id, Entity.workspace_id == workspace_id).first()
    if root is None:
        raise ValueError(f"Entity {entity_id} not found in workspace")
    depth = min(max_depth, MAX_TRAVERSAL_DEPTH)
    visited: set[int] = set()
    edges: list[dict] = []
    seen_edges: set = set()
    frontier = [entity_id]
    seen_nodes = 0
    for _ in range(depth + 1):
        if not frontier or seen_nodes >= MAX_TRAVERSAL_NODES:
            break
        next_frontier: list[int] = []
        for current in frontier:
            if current in visited:
                continue
            visited.add(current)
            seen_nodes += 1
            rels = db.query(EntityRelationship).filter(
                EntityRelationship.workspace_id == workspace_id,
                (EntityRelationship.source_id == current)
                | (EntityRelationship.target_id == current)).limit(200).all()
            for r in rels:
                if temporal_as_of is not None:
                    if r.valid_from is not None and \
                            r.valid_from > temporal_as_of:
                        continue
                    if r.valid_until is not None and \
                            r.valid_until < temporal_as_of:
                        continue
                other = r.target_id if r.source_id == current else r.source_id
                edge_key = (min(current, other), max(current, other),
                            r.relationship_type, r.id)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append({
                    "from": current,
                    "to": other,
                    "type": r.relationship_type,
                    "confidence": round(r.confidence, 3)
                    if r.confidence is not None else None,
                    "explicit": True,
                    "evidence": (r.metadata_json or "")[:500],
                    "valid_from": r.valid_from,
                    "valid_until": r.valid_until,
                })
                if other not in visited:
                    next_frontier.append(other)
        frontier = next_frontier
    return {
        "root_entity_id": entity_id,
        "depth": depth,
        "visited_entities": sorted(visited),
        "edges": edges,
        "truncated": seen_nodes >= MAX_TRAVERSAL_NODES,
    }
