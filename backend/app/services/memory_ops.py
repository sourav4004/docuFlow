"""Phase 18 memory platform — lifecycle, provenance, conflict queue, cleanup.

Memory lifecycle: candidate → validated → active → superseded → expired.

Every mutation is auditable; conflicting memories are never silently
overwritten — they surface as review items.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase15 import AIMemory
from ..models.phase16 import MemoryConflict

logger = logging.getLogger(__name__)

VALID_TRANSITIONS = {
    "candidate": {"validated", "expired", "deleted"},
    "validated": {"active", "candidate", "expired", "deleted"},
    "active": {"superseded", "expired", "deleted"},
    "superseded": {"expired", "deleted"},
    "expired": {"deleted"},
    "deleted": set(),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _owned_memory(db: Session, workspace_id: int, memory_id: int) -> AIMemory:
    row = (db.query(AIMemory)
           .filter(AIMemory.id == memory_id,
                   AIMemory.workspace_id == workspace_id)
           .first())
    if row is None:
        raise KeyError(f"memory {memory_id} not found in workspace")
    return row


def _status_of(memory: AIMemory) -> str:
    return (memory.lifecycle_status or "candidate").lower()


def can_transition(current: str, target: str) -> bool:
    return target in VALID_TRANSITIONS.get(current.lower(), set())


def transition_memory(db: Session, *, workspace_id: int, memory_id: int,
                      target: str, reason: Optional[str] = None,
                      user_id: Optional[int] = None) -> dict:
    """Server-side state machine for memory lifecycle."""
    memory = _owned_memory(db, workspace_id, memory_id)
    current = _status_of(memory)
    target = target.lower()
    if not can_transition(current, target):
        raise ValueError(
            f"invalid memory transition {current!r} -> {target!r}")
    old = memory.lifecycle_status
    memory.lifecycle_status = target.upper()
    if target == "expired":
        memory.expires_at = _utcnow()
    memory.updated_at = _utcnow()
    db.flush()
    _emit_memory_audit(db, workspace_id, memory_id, f"{old} -> {target}",
                       user_id, reason)
    return {"memory_id": memory_id, "from": old, "to": target.upper()}


def validate_memory(db: Session, *, workspace_id: int, memory_id: int,
                    evidence_ref: Optional[str] = None,
                    user_id: Optional[int] = None) -> dict:
    """candidate -> validated (or -> active when already validated)."""
    memory = _owned_memory(db, workspace_id, memory_id)
    current = _status_of(memory)
    target = "validated" if current == "candidate" else "active"
    if evidence_ref:
        memory.source = evidence_ref[:2000]
    return transition_memory(db, workspace_id=workspace_id,
                             memory_id=memory_id, target=target,
                             reason="validated with evidence", user_id=user_id)


def memory_conflicts(db: Session, workspace_id: int,
                     limit: int = 50) -> dict:
    """Conflict queue — surfaced, never auto-resolved."""
    rows = (db.query(MemoryConflict)
            .filter(MemoryConflict.workspace_id == workspace_id,
                    MemoryConflict.status == "OPEN")
            .order_by(MemoryConflict.created_at.desc())
            .limit(min(limit, 200)).all())
    return {"items": [
        {"id": c.id, "memory_a_id": c.memory_a_id,
         "memory_b_id": c.memory_b_id,
         "conflict_type": c.conflict_type,
         "description": c.description,
         "created_at": c.created_at}
        for c in rows], "total": len(rows)}


def expire_due_memories(db: Session, *, workspace_id: Optional[int] = None,
                        limit: int = 200,
                        now: Optional[datetime] = None) -> dict:
    """Safely expire memories past their expiration (bounded cleanup)."""
    now = now or _utcnow()
    q = db.query(AIMemory).filter(
        AIMemory.lifecycle_status.in_(("ACTIVE", "VALIDATED", "CANDIDATE")),
        AIMemory.expires_at.isnot(None),
        AIMemory.expires_at <= now)
    if workspace_id is not None:
        q = q.filter(AIMemory.workspace_id == workspace_id)
    rows = q.limit(limit).all()
    expired = 0
    for memory in rows:
        old = memory.lifecycle_status
        memory.lifecycle_status = "EXPIRED"
        memory.updated_at = now
        _emit_memory_audit(db, memory.workspace_id, memory.id,
                           f"{old} -> EXPIRED (scheduled)")
        expired += 1
    return {"expired": expired, "limit": limit}


def _emit_memory_audit(db: Session, workspace_id: int, memory_id: int,
                       detail: str, user_id: Optional[int] = None,
                       reason: Optional[str] = None) -> None:
    try:
        from .audit_service import log_action
        log_action(db, workspace_id=workspace_id, user_id=user_id,
                   action="memory.lifecycle",
                   detail=f"memory {memory_id}: {detail}"
                          + (f" ({reason})" if reason else ""))
    except Exception:  # noqa: BLE001 — audit is best-effort, never fatal
        logger.warning("memory audit write failed", exc_info=True)


def memory_scope_check(db: Session, workspace_id: int,
                       organization_id: Optional[int],
                       memory: AIMemory) -> bool:
    """Scope-hierarchy check: user → workspace → organization.

    A memory is visible only within its own workspace (and the organization
    only when the memory itself is organization-scoped).
    """
    if memory.workspace_id == workspace_id:
        return True
    if (organization_id is not None
            and memory.organization_id == organization_id
            and memory.scope in ("organization", "org")):
        return True
    return False