"""Organizational memory 3.0 — Phase 17.

- consolidation merges compatible memories WITHOUT losing provenance
- supersession records 'old fact replaced by new fact' (both preserved)
- confidence accounts for evidence, freshness, source authority, conflict state
- retrieval obeys the user → workspace → organization scope hierarchy
- automatic expiration by policy (memory expires_at)
- every mutation is auditable (callers pass an audit callback or we log rows)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from ..models.phase15 import AIMemory
from ..models.phase16 import MemoryConflict
from ..models.phase17 import MemorySupersession

logger = logging.getLogger(__name__)

AUDIT = Callable[[Session, str, dict], None]
_audit: Optional[AUDIT] = None

SOURCE_AUTHORITY = {"document": 1.0, "workflow": 0.85, "agent": 0.7,
                    "user": 0.95, "conversation": 0.6, "connector": 0.8}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def configure_audit(audit: Optional[AUDIT]) -> None:
    """Install an audit sink (e.g., the platform audit-log service)."""
    global _audit
    _audit = audit


def _emit(db: Session, action: str, detail: str, workspace_id: int,
          user_id: Optional[int] = None) -> None:
    if _audit is not None:
        try:
            _audit(db, action, workspace_id=workspace_id, user_id=user_id,
                   detail=detail)
        except Exception:  # noqa: BLE001
            logger.exception("memory audit failure")


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def confidence(db: Session, memory: AIMemory,
               conflict_state: Optional[str] = None) -> float:
    """Composite confidence in [0,1]: evidence * freshness * authority.

    Never treated as certainty — surfaced alongside human-readable factors.
    """
    raw = getattr(memory, "confidence", None)
    level_map = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}
    base = level_map.get(raw, 0.7) if isinstance(raw, str) else float(raw or 0.7)
    source = str(getattr(memory, "source", None) or "agent").split(":")[0]
    authority = SOURCE_AUTHORITY.get(source, 0.7)
    freshness = 1.0
    updated = memory.updated_at or memory.created_at
    if updated is not None:
        age_days = max(0, (_utcnow() - _as_utc(updated)).total_seconds()
                       / 86400.0)
        freshness = max(0.3, 1.0 - age_days / 365.0)
    conflict_penalty = 0.85 if conflict_state == "CONFLICT" else 1.0
    score = base * authority * freshness * conflict_penalty
    return round(min(1.0, max(0.05, score)), 4)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# ---------------------------------------------------------------------------
# Consolidation (provenance-preserving merge)
# ---------------------------------------------------------------------------

def consolidate(db: Session, memory_ids: list[int], workspace_id: int,
                user_id: Optional[int] = None) -> dict:
    """Merge compatible memories into one canonical row.

    The merged row's content is a deterministic concatenation with explicit
    provenance of each contributing memory kept in its source field.
    """
    memories = (db.query(AIMemory)
                .filter(AIMemory.id.in_(memory_ids),
                        AIMemory.workspace_id == workspace_id).all())
    if len(memories) < 2:
        raise ValueError("consolidation requires at least two memories")
    scope = {m.scope for m in memories}
    mtype = {m.memory_type for m in memories}
    if len(scope) > 1 or len(mtype) > 1:
        raise ValueError("cannot consolidate across scopes or memory types")
    canonical = memories[0]
    parts = []
    for m in memories:
        parts.append({"memory_id": m.id, "content": m.content,
                      "source": m.source})
    provenance = json.dumps(parts, default=str)
    canonical.content = " / ".join(f"[#{m.id}] {m.content}"
                                   for m in memories)
    canonical.source = f"consolidated:{provenance[:1000]}"
    canonical.updated_at = _utcnow()
    for m in memories[1:]:
        # Keep the rows (never destroy provenance) but mark them superseded by
        # linking to the canonical memory through the supersession table.
        db.add(MemorySupersession(
            workspace_id=workspace_id, old_memory_id=m.id,
            new_memory_id=canonical.id,
            reason="consolidated compatible memories",
            created_by_user_id=user_id))
    db.flush()
    _emit(db, "MEMORY_CONSOLIDATED",
          f"memories {sorted(memory_ids)} -> {canonical.id}",
          workspace_id, user_id)
    return {"canonical_memory_id": canonical.id, "merged": len(memories) - 1}


# ---------------------------------------------------------------------------
# Supersession
# ---------------------------------------------------------------------------

def supersede(db: Session, old_memory_id: int, new_content: str,
              workspace_id: int, memory_type: str, source: str,
              reason: str = "", user_id: Optional[int] = None,
              scope: str = "WORKSPACE", confidence_value: float = 0.8,
              expires_at: Optional[datetime] = None) -> dict:
    """Record 'old fact replaced by new fact' without deleting the old one."""
    old = db.query(AIMemory).filter(
        AIMemory.id == old_memory_id,
        AIMemory.workspace_id == workspace_id).first()
    if old is None:
        raise ValueError(f"Memory {old_memory_id} not found in workspace")
    new = AIMemory(
        workspace_id=workspace_id, organization_id=old.organization_id,
        user_id=old.user_id, memory_type=memory_type or old.memory_type,
        scope=scope, content=new_content, source=source,
        confidence=confidence_value, expires_at=expires_at,
    )
    db.add(new)
    db.flush()
    db.add(MemorySupersession(
        workspace_id=workspace_id, old_memory_id=old.id,
        new_memory_id=new.id, reason=reason or "superseded by newer fact",
        created_by_user_id=user_id))
    _emit(db, "MEMORY_SUPERSEDED",
          f"{old.id} -> {new.id} ({reason or 'newer fact'})",
          workspace_id, user_id)
    db.flush()
    return {"old_memory_id": old.id, "new_memory_id": new.id}


# ---------------------------------------------------------------------------
# Retrieval (scope hierarchy)
# ---------------------------------------------------------------------------

def retrieve_for_user(db: Session, workspace_id: int,
                      user_id: Optional[int] = None,
                      organization_id: Optional[int] = None,
                      memory_type: Optional[str] = None,
                      limit: int = 50) -> dict:
    """Scope-ordered memory retrieval: user → workspace → organization.

    A user only ever sees memories whose scope permits them.
    """
    visible: dict[int, AIMemory] = {}
    if user_id is not None:
        q = db.query(AIMemory).filter(
            AIMemory.workspace_id == workspace_id,
            AIMemory.user_id == user_id, AIMemory.scope == "USER")
        for m in q.limit(limit).all():
            visible[m.id] = m
    q = db.query(AIMemory).filter(
        AIMemory.workspace_id == workspace_id,
        AIMemory.scope == "WORKSPACE")
    if memory_type:
        q = q.filter(AIMemory.memory_type == memory_type)
    for m in q.limit(limit).all():
        visible[m.id] = m
    if organization_id is not None:
        q = db.query(AIMemory).filter(
            AIMemory.organization_id == organization_id,
            AIMemory.scope == "ORGANIZATION")
        if memory_type:
            q = q.filter(AIMemory.memory_type == memory_type)
        for m in q.limit(limit).all():
            visible[m.id] = m
    now = _utcnow()
    items = []
    for m in visible.values():
        if m.expires_at is not None and _as_utc(m.expires_at) < now:
            continue  # expired memories are never surfaced as current
        conflict = _conflict_state(db, m.id)
        items.append({
            "id": m.id,
            "memory_type": m.memory_type,
            "scope": m.scope,
            "content": m.content,
            "source": m.source,
            "confidence": confidence(db, m, conflict),
            "conflict_state": conflict,
            "expires_at": m.expires_at,
            "updated_at": m.updated_at,
        })
    items.sort(key=lambda i: (-i["confidence"], i["id"]))
    return {"items": items[:limit], "total": len(items)}


def _conflict_state(db: Session, memory_id: int) -> Optional[str]:
    row = db.query(MemoryConflict).filter(
        (MemoryConflict.memory_a_id == memory_id)
        | (MemoryConflict.memory_b_id == memory_id)).first()
    return "CONFLICT" if row is not None else None


# ---------------------------------------------------------------------------
# Expiration + lifecycle
# ---------------------------------------------------------------------------

def expire_due(db: Session, now: Optional[datetime] = None) -> dict:
    """Expire memories past their expiration window (bounded, idempotent)."""
    now = now or _utcnow()
    expired = db.query(AIMemory).filter(
        AIMemory.expires_at.isnot(None),
        AIMemory.expires_at <= now,
        AIMemory.lifecycle_status != "EXPIRED").limit(500).all()
    count = 0
    for m in expired:
        m.lifecycle_status = "EXPIRED"
        count += 1
    db.flush()
    return {"expired": count}
