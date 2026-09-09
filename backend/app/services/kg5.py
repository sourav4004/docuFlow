"""Phase 19 — knowledge graph 5.0.

Deterministic entity canonicalization (aliases/normalization/confidence/
provenance with human approval), merge safety (authorization checked before
merge decisions; provenance preserved; merges audited and reversible in
record terms), relationship confidence/evidence tracking, temporal
relationship expiration, and a graph-consistency checker that detects orphan
relationships, cross-tenant edges, invalid references, contradictory
relationships and impossible temporal states.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

CONTRADICTION_PAIRS = (
    ("manages", "reports_to"), ("acquired", "divested"),
    ("employs", "employed_by"), ("owns", "owned_by"),
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _loads(raw: Optional[str]):
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _alias_set(entity) -> set:
    return {str(a).strip().lower() for a in _loads(entity.aliases)}

# Deterministic legal-form abbreviation expansion for canonical matching.
# A name is canonicalized to a comparable token sequence; e.g. "Acme Corp"
# and "ACME Corporation" both canonicalize to "acme corporation".
_ABBREVIATIONS = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "llc": "limited liability company", "co": "company",
    "intl": "international", "gmbh": "gmbh", "ag": "ag",
}


def normalize_name(name: str) -> str:
    tokens = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).split()
    return " ".join(_ABBREVIATIONS.get(tok, tok) for tok in tokens)


# ---------------------------------------------------------------------------
# Canonicalization 2.0
# ---------------------------------------------------------------------------

def canonical_candidates(db: Session, *, workspace_id: int,
                         entity_ids: list[int],
                         limit: int = 200) -> dict:
    """Deterministic candidate matching over normalized names + aliases.
    Ambiguous matches are candidates ONLY — never auto-merged."""
    from ..models.knowledge_graph import Entity
    from ..models.phase17 import EntityCandidate
    entities = (db.query(Entity)
                .filter(Entity.workspace_id == workspace_id,
                        Entity.id.in_(entity_ids[:500])).all())
    by_normalized = {}
    for entity in entities:
        norm = normalize_name(entity.name)
        for alias in _alias_set(entity):
            if alias:
                by_normalized.setdefault(normalize_name(alias), set())
                by_normalized[normalize_name(alias)].add(entity.id)
        by_normalized.setdefault(norm, set()).add(entity.id)
    pairs = []
    for norm, ids in by_normalized.items():
        ids = sorted(ids)
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j], "normalized_name", norm))
    created = 0
    for source_id, target_id, method, norm in pairs[:limit]:
        existing = (db.query(EntityCandidate)
                    .filter(EntityCandidate.workspace_id == workspace_id,
                            EntityCandidate.entity_id == source_id,
                            EntityCandidate.candidate_id == target_id,
                            EntityCandidate.status == "PENDING").first())
        if existing is None:
            db.add(EntityCandidate(
                workspace_id=workspace_id, entity_id=source_id,
                candidate_id=target_id,
                score=0.9 if method == "normalized_name" else 0.6,
                method=method, reason=f"shared normalized name {norm!r}",
                status="PENDING"))
            created += 1
    db.flush()
    return {"candidates_created": created, "entities_checked": len(entities)}


def can_authorize_merge(db: Session, *, workspace_id: int,
                        user_id: Optional[int]) -> bool:
    """Merge approval requires an OWNER/ADMIN role in the workspace."""
    if not user_id:
        return False
    from ..models.workspace import WorkspaceMember
    member = (db.query(WorkspaceMember)
              .filter(WorkspaceMember.workspace_id == workspace_id,
                      WorkspaceMember.user_id == user_id).first())
    return member is not None and member.role in ("OWNER", "ADMIN")


def merge_safety_check(db: Session, *, workspace_id: int,
                       source_entity_id: int,
                       target_entity_id: int) -> dict:
    """Pre-merge safety: same workspace, distinct entities, no cross-tenant
    reference, entity rows exist."""
    from ..models.knowledge_graph import Entity
    source = (db.query(Entity)
              .filter(Entity.id == source_entity_id).first())
    target = (db.query(Entity)
              .filter(Entity.id == target_entity_id).first())
    if source is None or target is None:
        return {"safe": False, "reason": "entity not found"}
    if source.workspace_id != workspace_id or \
            target.workspace_id != workspace_id:
        return {"safe": False,
                "reason": "cross-workspace merge rejected"}
    if source.id == target.id:
        return {"safe": False, "reason": "cannot merge entity with itself"}
    return {"safe": True, "source_workspace": source.workspace_id,
            "target_workspace": target.workspace_id}


def decide_merge_safe(db: Session, *, workspace_id: int, merge_id: int,
                      decision: str, reviewer_id: int) -> dict:
    """Authorization-gated merge decision (delegates to kg_ops after checks).
    Provenance is preserved: merged source keeps its row flagged merged."""
    if not can_authorize_merge(db, workspace_id=workspace_id,
                               user_id=reviewer_id):
        raise PermissionError("reviewer lacks OWNER/ADMIN role in workspace")
    from ..models.phase18 import EntityMergeRequest
    row = db.query(EntityMergeRequest).get(merge_id)
    if row is None or row.workspace_id != workspace_id:
        raise KeyError("merge request not found in workspace")
    safety = merge_safety_check(db, workspace_id=workspace_id,
                                source_entity_id=row.source_entity_id,
                                target_entity_id=row.target_entity_id)
    if not safety["safe"]:
        return {"merged": False, "status": "REJECTED",
                "reason": safety["reason"]}
    from .kg_ops import decide_merge
    return decide_merge(db, workspace_id=workspace_id, merge_id=merge_id,
                        decision=decision, reviewer_id=reviewer_id)


def expire_pending_merges(db: Session, *, workspace_id: Optional[int] = None,
                          limit: int = 100) -> dict:
    from ..models.phase18 import EntityMergeRequest
    q = db.query(EntityMergeRequest).filter(
        EntityMergeRequest.status == "PENDING",
        EntityMergeRequest.expires_at.isnot(None),
        EntityMergeRequest.expires_at < _utcnow())
    if workspace_id is not None:
        q = q.filter(EntityMergeRequest.workspace_id == workspace_id)
    rows = q.limit(limit).all()
    for row in rows:
        row.status = "EXPIRED"
    db.flush()
    return {"expired": len(rows)}


# ---------------------------------------------------------------------------
# Relationships — confidence + expiry
# ---------------------------------------------------------------------------

def set_relationship_confidence(db: Session, *, workspace_id: int,
                                relationship_id: int,
                                confidence: float,
                                evidence: Optional[str] = None) -> dict:
    from ..models.knowledge_graph import EntityRelationship
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be within [0, 1]")
    row = (db.query(EntityRelationship)
           .filter(EntityRelationship.id == relationship_id,
                   EntityRelationship.workspace_id == workspace_id).first())
    if row is None:
        raise KeyError("relationship not found")
    row.confidence = round(confidence, 4)
    metadata = _loads(row.metadata_json)
    if metadata is None:
        metadata = []
    if evidence:
        metadata.append({"evidence": str(evidence)[:500],
                         "recorded_at": _utcnow().isoformat()})
    row.metadata_json = json.dumps(metadata, default=str)[:4000]
    db.flush()
    return {"relationship_id": row.id, "confidence": row.confidence}


def expire_relationship(db: Session, *, workspace_id: int,
                        relationship_id: int,
                        at: Optional[datetime] = None) -> dict:
    from ..models.knowledge_graph import EntityRelationship
    at = _as_utc(at) or _utcnow()
    row = (db.query(EntityRelationship)
           .filter(EntityRelationship.id == relationship_id,
                   EntityRelationship.workspace_id == workspace_id).first())
    if row is None:
        raise KeyError("relationship not found")
    row.valid_until = at
    row.is_current = False
    db.flush()
    return {"relationship_id": row.id, "valid_until": at,
            "is_current": False}


def active_relationships(db: Session, *, workspace_id: int,
                         entity_id: Optional[int] = None,
                         limit: int = 200) -> list:
    from ..models.knowledge_graph import EntityRelationship
    q = db.query(EntityRelationship).filter(
        EntityRelationship.workspace_id == workspace_id,
        EntityRelationship.is_current.is_(True))
    if entity_id is not None:
        q = q.filter((EntityRelationship.source_id == entity_id) |
                     (EntityRelationship.target_id == entity_id))
    return q.limit(min(limit, 500)).all()


# ---------------------------------------------------------------------------
# Graph consistency checker
# ---------------------------------------------------------------------------

def graph_consistency(db: Session, *, workspace_id: int,
                      persist: bool = True,
                      dry_run: bool = True) -> dict:
    """Detect orphan relationships, cross-tenant edges, invalid entity
    references, contradictory relationships and impossible temporal states.
    Repairs are never automatic — issues are reported."""
    from ..models.knowledge_graph import Entity, EntityRelationship
    from ..models.phase19 import ConsistencyReport
    issues = []
    rels = (db.query(EntityRelationship)
            .filter(EntityRelationship.workspace_id == workspace_id)
            .limit(10000).all())
    # Resolve endpoints across ALL workspaces so edges pointing at entities
    # in another workspace are reported as cross-tenant, not as orphans.
    endpoint_ids = set()
    for rel in rels:
        endpoint_ids.add(rel.source_id)
        endpoint_ids.add(rel.target_id)
    entities_by_id = {}
    if endpoint_ids:
        for entity in db.query(Entity).filter(
                Entity.id.in_(list(endpoint_ids)[:20000])
        ).limit(20000).all():
            entities_by_id[entity.id] = entity
    by_pair = {}
    for rel in rels:
        # orphan / cross-workspace edges
        if rel.source_id not in entities_by_id or \
                rel.target_id not in entities_by_id:
            issues.append({"kind": "orphan_relationship",
                           "relationship_id": rel.id,
                           "detail": "endpoint entity missing"})
            continue
        src, tgt = entities_by_id[rel.source_id], \
            entities_by_id[rel.target_id]
        if src.workspace_id != workspace_id or \
                tgt.workspace_id != workspace_id:
            issues.append({"kind": "cross_tenant_edge",
                           "relationship_id": rel.id,
                           "detail": "relationship crosses workspace "
                                     "boundary"})
            continue
        # impossible temporal states
        if rel.valid_from is not None and rel.valid_until is not None \
                and _as_utc(rel.valid_from) > _as_utc(rel.valid_until):
            issues.append({"kind": "impossible_temporal",
                           "relationship_id": rel.id,
                           "detail": "valid_from after valid_until"})
        if rel.is_current:
            key = tuple(sorted((rel.source_id, rel.target_id)))
            by_pair.setdefault(key, []).append(rel)
    # contradictions: symmetric inverse pairs both current
    checked_pairs = set()
    for (a, b), pair_rels in by_pair.items():
        for rel in pair_rels:
            types = {r.relationship_type.lower() for r in pair_rels}
            for left, right in CONTRADICTION_PAIRS:
                if {left, right} <= types and (rel.id not in checked_pairs):
                    issues.append({"kind": "contradictory_relationships",
                                   "relationship_id": rel.id,
                                   "detail": f"both {left!r} and {right!r} "
                                             "are active between the same "
                                             "entities"})
                    checked_pairs.add(rel.id)
    if persist:
        db.add(ConsistencyReport(workspace_id=workspace_id,
                                 check_kind="graph_consistency",
                                 status="ISSUES" if issues else "CLEAN",
                                 issue_count=len(issues),
                                 issues_json=json.dumps(issues, default=str),
                                 dry_run=dry_run))
        db.flush()
    return {"workspace_id": workspace_id, "check": "graph_consistency",
            "clean": not issues, "issue_count": len(issues),
            "issues": issues[:200]}
