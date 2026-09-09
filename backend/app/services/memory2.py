"""Organizational memory 2.0 — conflicts, lifecycle, provenance, governance.

Conflicting memories are ALWAYS preserved and surfaced as ``MemoryConflict``
records — never silently overwritten. Lifecycle: ACTIVE → EXPIRED (date) /
SUPERSEDED (by a newer memory) / DELETED (soft, for governance audits).
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase15 import AIMemory, MEMORY_SCOPES, MEMORY_TYPES
from ..models.phase16 import MemoryConflict
from ..services.memory_service import store_memory, MemoryValidationError

LIFECYCLE_STATUSES = ("ACTIVE", "EXPIRED", "SUPERSEDED", "DELETED")

_AMOUNT_RE = re.compile(r"[\$€£]\s?(\d[\d,]*(?:\.\d+)?)|\b(\d[\d,]*(?:\.\d+)?)\s*%")
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_BOOL_TOKENS = (
    ("yes", "no"), ("true", "false"), ("approved", "rejected"),
    ("active", "inactive"), ("enabled", "disabled"), ("include", "exclude"),
)


def _signature(content: str) -> dict:
    """Deterministic content fingerprint for conflict comparison."""
    amounts = tuple(float((m.group(1) or m.group(2)).replace(",", ""))
                    for m in _AMOUNT_RE.finditer(content))
    dates = tuple(m.group(0) for m in _DATE_RE.finditer(content))
    lowered = content.lower()
    booleans = []
    for pos, neg in _BOOL_TOKENS:
        if pos in lowered and neg not in lowered:
            booleans.append(pos)
        elif neg in lowered and pos not in lowered:
            booleans.append(neg)
    return {"amounts": amounts, "dates": dates, "booleans": tuple(booleans)}


def memories_conflict(a: AIMemory, b: AIMemory) -> Optional[dict]:
    """Check two memories for a deterministic contradiction.

    Returns a description dict or None. Only same-scope memories about the
    same kind of fact can conflict; differing amounts/dates/boolean claims
    with otherwise similar content produce a CONTRADICTION.
    """
    if a.memory_type != b.memory_type:
        return None
    if a.scope != b.scope:
        return None
    if a.id == b.id:
        return None
    sa, sb = _signature(a.content), _signature(b.content)
    if sa["amounts"] and sb["amounts"] and sa["amounts"] != sb["amounts"]:
        if len(sa["amounts"]) == 1 and len(sb["amounts"]) == 1:
            return {
                "conflict_type": "CONTRADICTION",
                "description": (f"Conflicting amounts: '{a.content[:120]}' vs "
                                f"'{b.content[:120]}'"),
                "evidence": {"amount_a": sa["amounts"][0],
                             "amount_b": sb["amounts"][0]},
            }
    if sa["dates"] and sb["dates"] and sa["dates"] != sb["dates"]:
        if len(sa["dates"]) == 1 and len(sb["dates"]) == 1:
            return {
                "conflict_type": "CONTRADICTION",
                "description": (f"Conflicting dates: '{a.content[:120]}' vs "
                                f"'{b.content[:120]}'"),
                "evidence": {"date_a": sa["dates"][0], "date_b": sb["dates"][0]},
            }
    if sa["booleans"] and sb["booleans"] and sa["booleans"] != sb["booleans"]:
        if len(sa["booleans"]) == 1 and len(sb["booleans"]) == 1:
            return {
                "conflict_type": "CONTRADICTION",
                "description": (f"Conflicting status claims: "
                                f"'{a.content[:120]}' vs '{b.content[:120]}'"),
                "evidence": {"claim_a": sa["booleans"][0],
                             "claim_b": sb["booleans"][0]},
            }
    return None


def create_memory(
    db: Session,
    workspace_id: int,
    user_id: int,
    memory_type: str,
    scope: str,
    content: str,
    source: Optional[str] = None,
    confidence: str = "MEDIUM",
    organization_id: Optional[int] = None,
    expires_at: Optional[datetime] = None,
) -> tuple[AIMemory, Optional[MemoryConflict]]:
    """Store a memory (provenance required) and scan for conflicts.

    Returns (memory, conflict) — a conflict row is created when a same-scope
    memory already contradicts the new one; both rows are preserved.
    """
    if not source:
        raise MemoryValidationError("source is required for memory provenance")
    if memory_type not in MEMORY_TYPES:
        raise MemoryValidationError(f"Invalid memory type: {memory_type}")
    if scope not in MEMORY_SCOPES:
        raise MemoryValidationError(f"Invalid memory scope: {scope}")
    memory = store_memory(
        db, user_id=user_id, workspace_id=workspace_id, memory_type=memory_type,
        scope=scope, content=content, source=source, confidence=confidence,
        expires_at=expires_at,
    )
    conflict = None
    peers = (
        db.query(AIMemory)
        .filter(AIMemory.workspace_id == workspace_id,
                AIMemory.scope == scope,
                AIMemory.memory_type == memory_type,
                AIMemory.lifecycle_status == "ACTIVE",
                AIMemory.id != memory.id)
        .limit(500)
        .all()
    )
    if scope == "USER" or scope == "CONVERSATION":
        peers = [p for p in peers if p.user_id == user_id]
    for peer in peers:
        verdict = memories_conflict(memory, peer)
        if verdict is None:
            continue
        conflict = MemoryConflict(
            workspace_id=workspace_id,
            memory_a_id=peer.id,
            memory_b_id=memory.id,
            conflict_type=verdict["conflict_type"],
            description=verdict["description"],
            evidence_json=json.dumps(verdict["evidence"]),
            status="OPEN",
        )
        db.add(conflict)
        break  # one conflict record per new memory is enough to surface
    db.flush()
    return memory, conflict


def list_conflicts(db: Session, workspace_id: int,
                   status: Optional[str] = None,
                   limit: int = 100) -> list[MemoryConflict]:
    query = db.query(MemoryConflict).filter(
        MemoryConflict.workspace_id == workspace_id)
    if status:
        query = query.filter(MemoryConflict.status == status)
    return query.order_by(MemoryConflict.created_at.desc()).limit(
        min(limit, 200)).all()


def resolve_conflict(db: Session, conflict_id: int, actor_id: int,
                     keep_memory_id: int, note: Optional[str] = None,
                     workspace_id: Optional[int] = None) -> MemoryConflict:
    conflict = db.query(MemoryConflict).filter(
        MemoryConflict.id == conflict_id).first()
    if conflict is None:
        raise ValueError("Memory conflict not found")
    if workspace_id is not None and conflict.workspace_id != workspace_id:
        raise PermissionError("Conflict belongs to another workspace")
    if keep_memory_id not in (conflict.memory_a_id, conflict.memory_b_id):
        raise ValueError("keep_memory_id must reference one conflicting memory")
    loser = conflict.memory_a_id if keep_memory_id == conflict.memory_b_id \
        else conflict.memory_b_id
    superseded = db.query(AIMemory).filter(AIMemory.id == loser).first()
    if superseded is not None:
        superseded.lifecycle_status = "SUPERSEDED"
        superseded.supersedes_id = keep_memory_id
        winner = db.query(AIMemory).filter(AIMemory.id == keep_memory_id).first()
        if winner is not None:
            winner.supersedes_id = superseded.id
    conflict.status = "RESOLVED"
    conflict.resolved_by = actor_id
    conflict.resolved_at = datetime.now(timezone.utc)
    if note:
        conflict.description = (conflict.description or "") + f" | {note}"
    db.flush()
    return conflict


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def expire_memory(db: Session, memory_id: int, actor_id: int,
                  workspace_id: Optional[int] = None) -> AIMemory:
    memory = db.query(AIMemory).filter(AIMemory.id == memory_id).first()
    if memory is None:
        raise ValueError("Memory not found")
    if workspace_id is not None and memory.workspace_id != workspace_id:
        raise PermissionError("Memory belongs to another workspace")
    memory.lifecycle_status = "EXPIRED"
    memory.expires_at = datetime.now(timezone.utc)
    db.flush()
    return memory


def delete_memory_soft(db: Session, memory_id: int, actor_id: int,
                       workspace_id: Optional[int] = None) -> AIMemory:
    memory = db.query(AIMemory).filter(AIMemory.id == memory_id).first()
    if memory is None:
        raise ValueError("Memory not found")
    if workspace_id is not None and memory.workspace_id != workspace_id:
        raise PermissionError("Memory belongs to another workspace")
    memory.lifecycle_status = "DELETED"
    db.flush()
    return memory


def apply_expirations(db: Session, now: Optional[datetime] = None) -> int:
    """Expire memories past their expires_at (scheduled sweep)."""
    now = now or datetime.now(timezone.utc)
    due = (
        db.query(AIMemory)
        .filter(AIMemory.expires_at.isnot(None),
                AIMemory.expires_at <= now,
                AIMemory.lifecycle_status == "ACTIVE")
        .all()
    )
    for memory in due:
        memory.lifecycle_status = "EXPIRED"
    db.flush()
    return len(due)


def governance_export(
    db: Session,
    workspace_id: int,
    scope_filter: Optional[str] = None,
    user_filter: Optional[int] = None,
) -> list[dict]:
    """Scoped memory export for governance (never crosses workspaces)."""
    query = db.query(AIMemory).filter(AIMemory.workspace_id == workspace_id)
    if scope_filter:
        query = query.filter(AIMemory.scope == scope_filter)
    if user_filter is not None:
        query = query.filter(AIMemory.user_id == user_filter)
    rows = query.order_by(AIMemory.created_at.desc()).limit(5000).all()
    return [
        {
            "id": m.id,
            "memory_type": m.memory_type,
            "scope": m.scope,
            "user_id": m.user_id,
            "content": m.content,
            "source": m.source,
            "confidence": m.confidence,
            "lifecycle_status": m.lifecycle_status,
            "supersedes_id": m.supersedes_id,
            "created_at": m.created_at,
            "updated_at": m.updated_at,
            "expires_at": m.expires_at,
        }
        for m in rows
    ]


def governance_summary(db: Session, workspace_id: int) -> dict:
    rows = (
        db.query(AIMemory)
        .filter(AIMemory.workspace_id == workspace_id)
        .all()
    )
    by_status: dict = {}
    by_scope: dict = {}
    for m in rows:
        by_status[m.lifecycle_status] = by_status.get(m.lifecycle_status, 0) + 1
        by_scope[m.scope] = by_scope.get(m.scope, 0) + 1
    return {
        "total": len(rows),
        "by_lifecycle_status": by_status,
        "by_scope": by_scope,
        "conflicts_open": db.query(MemoryConflict).filter(
            MemoryConflict.workspace_id == workspace_id,
            MemoryConflict.status == "OPEN").count(),
    }
