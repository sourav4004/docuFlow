"""Organizational memory 2.0 — persistent AI memory with governance.

Every memory has a scope (WORKSPACE/USER/DOCUMENT/CONVERSATION), source,
confidence, created/updated timestamps, and optional expiration. Memories are
never inferred across users; workspace memory is only readable within its
workspace. Admins can inspect, delete, export, expire, and reset memory.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase15 import AIMemory, MEMORY_TYPES, MEMORY_SCOPES
from ..models.workspace import Workspace
from ..services.audit_service import log_audit_event


class MemoryValidationError(Exception):
    """Raised on invalid memory writes."""


def _validate(user_id, workspace_id, scope, memory_type):
    if memory_type not in MEMORY_TYPES:
        raise MemoryValidationError(f"Unknown memory type: {memory_type}")
    if scope not in MEMORY_SCOPES:
        raise MemoryValidationError(f"Unknown memory scope: {scope}")
    if scope == "USER" and user_id is None:
        raise MemoryValidationError("USER-scoped memory requires a user")


def store_memory(
    db: Session,
    workspace_id: int,
    user_id: int,
    memory_type: str,
    content: str,
    scope: str = "WORKSPACE",
    source: Optional[str] = None,
    confidence: str = "MEDIUM",
    expires_at: Optional[datetime] = None,
    organization_id: Optional[int] = None,
    dedupe: bool = True,
) -> AIMemory:
    """Store a memory (deduplicated on identical type+scope+content)."""
    _validate(user_id, workspace_id, scope, memory_type)
    if not content or not content.strip():
        raise MemoryValidationError("Memory content cannot be empty")

    if dedupe:
        existing = (
            db.query(AIMemory)
            .filter(
                AIMemory.workspace_id == workspace_id,
                AIMemory.memory_type == memory_type,
                AIMemory.scope == scope,
                AIMemory.content == content.strip(),
            )
            .first()
        )
        if existing:
            existing.updated_at = datetime.now(timezone.utc)
            db.flush()
            return existing

    memory = AIMemory(
        workspace_id=workspace_id,
        organization_id=organization_id,
        user_id=user_id if scope == "USER" else None,
        memory_type=memory_type,
        scope=scope,
        content=content.strip(),
        source=source,
        confidence=confidence,
        expires_at=expires_at,
    )
    db.add(memory)
    db.flush()
    return memory


def list_memories(
    db: Session,
    workspace_id: int,
    user_id: Optional[int] = None,
    scope: Optional[str] = None,
    memory_type: Optional[str] = None,
    limit: int = 100,
) -> list[AIMemory]:
    """List memories visible in a workspace.

    USER-scoped memories are only visible to their owner; workspace-scoped
    memories are visible to every workspace member (isolation at row level).
    """
    query = db.query(AIMemory).filter(AIMemory.workspace_id == workspace_id)
    if user_id is not None:
        query = query.filter(
            (AIMemory.scope != "USER") | (AIMemory.user_id == user_id)
        )
    if scope:
        query = query.filter(AIMemory.scope == scope)
    if memory_type:
        query = query.filter(AIMemory.memory_type == memory_type)
    return query.order_by(AIMemory.updated_at.desc()).limit(limit).all()


def get_memory(db: Session, workspace_id: int, memory_id: int) -> Optional[AIMemory]:
    return (
        db.query(AIMemory)
        .filter(AIMemory.workspace_id == workspace_id, AIMemory.id == memory_id)
        .first()
    )


def delete_memory(db: Session, workspace_id: int, memory_id: int, actor_id: int) -> bool:
    memory = get_memory(db, workspace_id, memory_id)
    if not memory:
        return False
    db.delete(memory)
    db.flush()
    log_audit_event(
        db, event_type="memory", event_action="delete",
        user_id=actor_id, resource_type="memory", resource_id=memory_id,
        details=f"Memory deleted (type={memory.memory_type}, scope={memory.scope})",
    )
    return True


def expire_memories(db: Session, now: Optional[datetime] = None) -> int:
    """Expire memories past their expiration (purge pass)."""
    now = now or datetime.now(timezone.utc)
    expired = (
        db.query(AIMemory)
        .filter(AIMemory.expires_at.isnot(None), AIMemory.expires_at < now)
        .all()
    )
    for memory in expired:
        db.delete(memory)
    if expired:
        db.flush()
    return len(expired)


def export_memories(db: Session, workspace_id: int, user_id: Optional[int] = None) -> list[dict]:
    """Export visible memories as JSON-safe dicts (for governance)."""
    memories = list_memories(db, workspace_id, user_id=user_id, limit=1000)
    return [
        {
            "id": m.id,
            "type": m.memory_type,
            "scope": m.scope,
            "content": m.content,
            "source": m.source,
            "confidence": m.confidence,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "expires_at": m.expires_at.isoformat() if m.expires_at else None,
        }
        for m in memories
    ]


def reset_workspace_memory(db: Session, workspace_id: int, actor_id: int) -> int:
    """Delete all workspace memory (admin action, audited)."""
    memories = (
        db.query(AIMemory).filter(AIMemory.workspace_id == workspace_id).all()
    )
    count = len(memories)
    for memory in memories:
        db.delete(memory)
    if memories:
        db.flush()
    log_audit_event(
        db, event_type="memory", event_action="reset_workspace",
        user_id=actor_id, resource_type="workspace", resource_id=workspace_id,
        details=f"Workspace memory reset ({count} memories deleted)",
    )
    return count


def memory_summary(db: Session, workspace_id: int) -> dict:
    """Summary of memory by type/scope for the security/explainability UI."""
    from sqlalchemy import func
    rows = (
        db.query(AIMemory.scope, AIMemory.memory_type, func.count(AIMemory.id))
        .filter(AIMemory.workspace_id == workspace_id)
        .group_by(AIMemory.scope, AIMemory.memory_type)
        .all()
    )
    total = sum(count for _, _, count in rows)
    return {
        "total": total,
        "by_scope_type": [
            {"scope": scope, "type": mtype, "count": count}
            for scope, mtype, count in rows
        ],
    }