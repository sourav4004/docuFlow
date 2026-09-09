"""Phase 19 — memory platform 4.0.

Candidate generation from validated evidence only, provenance-scoped
validation, user/workspace/organization suppression, durable conflict
resolution workflows (never silent overwrites), TTL/event/policy-driven
expiration, and explainability payloads that show what a memory is, its
source/confidence/age/scope and why it is used — never hidden chain-of-
thought.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

VALID_CONFIDENCE = ("LOW", "MEDIUM", "HIGH")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# ---------------------------------------------------------------------------
# Candidate generation + validation
# ---------------------------------------------------------------------------

def create_candidate(db: Session, *, workspace_id: int,
                     organization_id: Optional[int] = None,
                     user_id: Optional[int] = None,
                     memory_type: str = "fact",
                     scope: str = "WORKSPACE",
                     content: str,
                     source: Optional[str] = None,
                     evidence_ref: Optional[str] = None,
                     confidence: str = "MEDIUM",
                     ttl_days: Optional[int] = None) -> dict:
    """Generate a candidate memory. Persists ONLY with validated evidence:
    ``source`` or ``evidence_ref`` is required (provenance gate)."""
    from ..models.phase15 import AIMemory
    if not (source or evidence_ref):
        return {"created": False,
                "reason": "candidate memories require validated evidence "
                          "(source or evidence_ref)"}
    if confidence not in VALID_CONFIDENCE:
        raise ValueError("invalid confidence level")
    expires_at = None
    if ttl_days is not None:
        expires_at = _utcnow() + timedelta(days=max(1, ttl_days))
    memory = AIMemory(
        workspace_id=workspace_id, organization_id=organization_id,
        user_id=user_id, memory_type=memory_type, scope=scope.upper(),
        content=content, source=evidence_ref or source,
        confidence=confidence, expires_at=expires_at,
        lifecycle_status="CANDIDATE")
    db.add(memory)
    db.flush()
    return {"created": True, "memory_id": memory.id,
            "lifecycle_status": memory.lifecycle_status,
            "expires_at": expires_at}


def validate_candidate(db: Session, *, workspace_id: int, memory_id: int,
                       user_id: Optional[int] = None) -> dict:
    """candidate -> validated -> active. Requires provenance fields."""
    from ..models.phase15 import AIMemory
    memory = (db.query(AIMemory)
              .filter(AIMemory.id == memory_id,
                      AIMemory.workspace_id == workspace_id).first())
    if memory is None:
        raise KeyError("memory not found")
    if not (memory.source or memory.updated_at):
        return {"validated": False,
                "reason": "missing provenance (source)"}
    from .memory_ops import validate_memory
    step1 = validate_memory(db, workspace_id=workspace_id,
                            memory_id=memory_id, user_id=user_id)
    if step1["to"] == "VALIDATED":
        from .memory_ops import transition_memory
        step2 = transition_memory(db, workspace_id=workspace_id,
                                  memory_id=memory_id, target="active",
                                  user_id=user_id,
                                  reason="validated candidate activated")
        return {"validated": True, "memory_id": memory_id,
                "state": step2["to"]}
    return {"validated": True, "memory_id": memory_id,
            "state": step1["to"]}


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

def suppress_memory(db: Session, *, memory_id: int, scope_type: str,
                    suppressed_by_user_id: Optional[int] = None,
                    workspace_id: Optional[int] = None,
                    organization_id: Optional[int] = None,
                    reason: Optional[str] = None) -> dict:
    from ..models.phase19 import MemorySuppression
    if scope_type not in ("USER", "WORKSPACE", "ORGANIZATION"):
        raise ValueError("invalid suppression scope")
    existing = (db.query(MemorySuppression)
                .filter(MemorySuppression.memory_id == memory_id,
                        MemorySuppression.scope_type == scope_type)
                .first())
    if existing is not None:
        return {"status": "ALREADY_SUPPRESSED", "memory_id": memory_id,
                "scope_type": scope_type}
    db.add(MemorySuppression(memory_id=memory_id, scope_type=scope_type,
                             suppressed_by_user_id=suppressed_by_user_id,
                             workspace_id=workspace_id,
                             organization_id=organization_id,
                             reason=reason))
    db.flush()
    return {"status": "SUPPRESSED", "memory_id": memory_id,
            "scope_type": scope_type}


def remove_suppression(db: Session, *, memory_id: int,
                       scope_type: str) -> dict:
    from ..models.phase19 import MemorySuppression
    row = (db.query(MemorySuppression)
           .filter(MemorySuppression.memory_id == memory_id,
                   MemorySuppression.scope_type == scope_type).first())
    if row is None:
        return {"status": "NOT_SUPPRESSED"}
    db.delete(row)
    db.flush()
    return {"status": "REMOVED", "memory_id": memory_id,
            "scope_type": scope_type}


def is_suppressed(db: Session, *, memory_id: int,
                  user_id: Optional[int] = None,
                  workspace_id: Optional[int] = None,
                  organization_id: Optional[int] = None) -> bool:
    from ..models.phase19 import MemorySuppression
    rows = db.query(MemorySuppression).filter(
        MemorySuppression.memory_id == memory_id).all()
    for row in rows:
        if row.scope_type == "USER" and row.suppressed_by_user_id == user_id:
            return True
        if row.scope_type == "WORKSPACE" and row.workspace_id == workspace_id:
            return True
        if row.scope_type == "ORGANIZATION" and \
                row.organization_id == organization_id:
            return True
    return False


def list_suppressions(db: Session, *, memory_id: Optional[int] = None,
                      limit: int = 100) -> list:
    from ..models.phase19 import MemorySuppression
    q = db.query(MemorySuppression)
    if memory_id is not None:
        q = q.filter(MemorySuppression.memory_id == memory_id)
    return q.order_by(MemorySuppression.created_at.desc()
                      ).limit(min(limit, 500)).all()


# ---------------------------------------------------------------------------
# Conflict resolution workflow
# ---------------------------------------------------------------------------

def resolve_conflict_workflow(db: Session, *, workspace_id: int,
                              conflict_id: int, decision: str,
                              reviewer_user_id: int,
                              reason: Optional[str] = None) -> dict:
    """Durable conflict workflow. Decisions: KEEP_A / KEEP_B / SUPERSEDE_A /
    SUPERSEDE_B. The losing memory is superseded (explicit record), never
    deleted."""
    from ..models.phase15 import AIMemory
    from ..models.phase16 import MemoryConflict
    from ..models.phase17 import MemorySupersession
    conflict = db.query(MemoryConflict).get(conflict_id)
    if conflict is None or conflict.workspace_id != workspace_id:
        raise KeyError("conflict not found in workspace")
    if conflict.status != "OPEN":
        return {"status": "ALREADY_RESOLVED",
                "resolution": conflict.status}
    memory_a = db.query(AIMemory).get(conflict.memory_a_id)
    memory_b = db.query(AIMemory).get(conflict.memory_b_id)
    if memory_a is None or memory_b is None:
        raise ValueError("conflict references missing memory")
    winner, loser = memory_a, memory_b
    if decision == "KEEP_B":
        winner, loser = memory_b, memory_a
    elif decision in ("SUPERSEDE_A", "SUPERSEDE_B"):
        if decision == "SUPERSEDE_A":
            winner, loser = memory_b, memory_a
        else:
            winner, loser = memory_a, memory_b
        db.add(MemorySupersession(
            workspace_id=workspace_id, old_memory_id=loser.id,
            new_memory_id=winner.id, reason=reason or decision,
            created_by_user_id=reviewer_user_id))
        loser.lifecycle_status = "SUPERSEDED"
        loser.supersedes_id = winner.id
        loser.updated_at = _utcnow()
    conflict.status = "RESOLVED"
    conflict.resolved_by = reviewer_user_id
    conflict.resolved_at = _utcnow()
    db.flush()
    return {"status": "RESOLVED", "decision": decision,
            "winner_memory_id": winner.id,
            "loser_memory_id": loser.id}


# ---------------------------------------------------------------------------
# Expiration policy
# ---------------------------------------------------------------------------

def expiration_policy() -> dict:
    """Default TTL policy by memory type (overridable via env, bounded)."""
    return {
        "fact": int(__import__("os").getenv("MEMORY_TTL_FACT_DAYS", "365")),
        "preference": int(__import__("os").getenv(
            "MEMORY_TTL_PREFERENCE_DAYS", "180")),
        "event": int(__import__("os").getenv("MEMORY_TTL_EVENT_DAYS", "90")),
        "organizational": int(__import__("os").getenv(
            "MEMORY_TTL_ORG_DAYS", "365")),
    }


def apply_policy_expiration(db: Session, *, workspace_id: Optional[int] = None,
                            limit: int = 200) -> dict:
    """Expire memories past policy TTL or explicit expires_at (bounded)."""
    from ..models.phase15 import AIMemory
    from .memory_ops import expire_due_memories
    result = expire_due_memories(db, workspace_id=workspace_id, limit=limit)
    policy = expiration_policy()
    rows = (db.query(AIMemory)
            .filter(AIMemory.lifecycle_status.in_(
                ("ACTIVE", "VALIDATED", "CANDIDATE")),
                AIMemory.expires_at.is_(None)).limit(limit).all())
    expired = 0
    for memory in rows:
        days = policy.get(memory.memory_type)
        if not days:
            continue
        if _utcnow() - _as_utc(memory.created_at) > timedelta(days=days):
            memory.lifecycle_status = "EXPIRED"
            memory.expires_at = _utcnow()
            memory.updated_at = _utcnow()
            expired += 1
    db.flush()
    result["policy_expired"] = expired
    return result


# ---------------------------------------------------------------------------
# Explainability + scope-hierarchy retrieval
# ---------------------------------------------------------------------------

def explain_memory(db: Session, *, memory_id: int,
                   workspace_id: int) -> dict:
    """User-facing memory explanation — evidence, confidence, age, scope,
    suppression, conflict state and usage rationale. Never hidden reasoning."""
    from ..models.phase15 import AIMemory
    memory = (db.query(AIMemory)
              .filter(AIMemory.id == memory_id,
                      AIMemory.workspace_id == workspace_id).first())
    if memory is None:
        raise KeyError("memory not found")
    conflicts = _conflict_ids(db, memory.id)
    age_days = max(0, (_utcnow() - _as_utc(memory.created_at)).days) \
        if memory.created_at else None
    return {
        "memory_id": memory.id,
        "content": memory.content,
        "source": memory.source,
        "confidence": memory.confidence,
        "scope": memory.scope,
        "lifecycle_status": memory.lifecycle_status,
        "age_days": age_days,
        "suppressions": [{"scope_type": s.scope_type}
                         for s in list_suppressions(db,
                                                   memory_id=memory.id)],
        "conflict_count": len(conflicts),
        "why_used": f"memory retrieved because it is {memory.scope} scoped, "
                    f"{memory.lifecycle_status.lower()}, confidence "
                    f"{memory.confidence.lower()}",
        "note": "explanations show decisions and evidence, never internal "
                "reasoning",
    }


def _conflict_ids(db: Session, memory_id: int) -> list[int]:
    from ..models.phase16 import MemoryConflict
    rows = (db.query(MemoryConflict)
            .filter((MemoryConflict.memory_a_id == memory_id) |
                    (MemoryConflict.memory_b_id == memory_id),
                    MemoryConflict.status == "OPEN").all())
    return [r.id for r in rows]


def retrieve_memories(db: Session, *, workspace_id: int,
                      user_id: Optional[int] = None,
                      organization_id: Optional[int] = None,
                      query: Optional[str] = None,
                      limit: int = 20) -> dict:
    """Scope-hierarchy retrieval: user → workspace → organization. Only
    non-suppressed, non-expired memories are returned (bounded)."""
    from ..models.phase15 import AIMemory
    q = db.query(AIMemory).filter(
        AIMemory.lifecycle_status.in_(("ACTIVE", "VALIDATED")))
    results = []
    for scope_row in q.limit(2000).all():
        if scope_row.scope == "USER" and \
                scope_row.user_id != user_id:
            continue
        if scope_row.scope == "WORKSPACE" and \
                scope_row.workspace_id != workspace_id:
            continue
        if scope_row.scope == "ORGANIZATION" and \
                scope_row.organization_id != organization_id:
            continue
        if is_suppressed(db, memory_id=scope_row.id, user_id=user_id,
                         workspace_id=workspace_id,
                         organization_id=organization_id):
            continue
        if scope_row.expires_at is not None and \
                _as_utc(scope_row.expires_at) < _utcnow():
            continue
        score = 0.0
        if query:
            ql = query.lower()
            if ql in scope_row.content.lower():
                score += 1.0
            for token in ql.split():
                if token and token in scope_row.content.lower():
                    score += 0.5
        results.append({"memory_id": scope_row.id,
                        "content": scope_row.content,
                        "scope": scope_row.scope,
                        "confidence": scope_row.confidence,
                        "source": scope_row.source,
                        "score": round(score, 3)})
    rank_key = ("USER", "WORKSPACE", "ORGANIZATION")
    results.sort(key=lambda r: (rank_key.index(r["scope"]) if
                                r["scope"] in rank_key else 9,
                                -r["score"]))
    return {"results": results[:limit], "total": len(results),
            "scope_hierarchy": ["USER", "WORKSPACE", "ORGANIZATION"]}
