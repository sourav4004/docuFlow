"""Phase 20 — Memory intelligence.

Memory quality (usage, correctness, conflicts, expiration, user suppression,
provenance), memory decay detection, safe consolidation (only when
confidence/provenance rules permit), conflict-queue routing to review, and
user controls (inspect, suppress, delete where permitted, view provenance).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import MemoryIntelligence
from ..models.phase15 import AIMemory


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def record_quality(db: Session, *, memory_id: int, workspace_id: int,
                   metrics: dict) -> MemoryIntelligence:
    row = MemoryIntelligence(memory_id=memory_id,
                             workspace_id=workspace_id,
                             metrics_json=_dumps(metrics))
    db.add(row)
    db.flush()
    return row


def memory_quality(db: Session, *, workspace_id: int,
                   limit: int = 500) -> dict:
    """Aggregate memory quality across a workspace."""
    rows = db.query(AIMemory).filter_by(workspace_id=workspace_id)\
        .limit(limit).all()
    total = len(rows)
    active = sum(1 for r in rows if r.lifecycle_status == "ACTIVE")
    expired = sum(1 for r in rows
                  if r.lifecycle_status in ("EXPIRED", "SUPERSEDED"))
    suppressed = sum(1 for r in rows
                     if getattr(r, "lifecycle_status", "ACTIVE") == "SUPPRESSED")
    no_source = sum(1 for r in rows if not r.source)
    low_conf = sum(1 for r in rows if r.confidence in ("LOW", None))
    return {
        "total": total, "active": active,
        "expired_or_superseded": expired,
        "suppressed": suppressed,
        "missing_provenance": no_source,
        "low_confidence": low_conf,
        "health_fraction": round((active / total) if total else 1.0, 4),
    }


def decay_report(db: Session, *, workspace_id: int,
                 stale_days: int = 90) -> list[dict]:
    """Memories approaching/past expiry or last updated long ago."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=stale_days)
    rows = db.query(AIMemory).filter(
        AIMemory.workspace_id == workspace_id,
        AIMemory.lifecycle_status == "ACTIVE").limit(500).all()
    out = []
    for m in rows:
        updated = _as_utc(m.updated_at)
        expires = _as_utc(m.expires_at)
        reason = None
        if updated is not None and updated < cutoff:
            reason = f"not updated in {stale_days}+ days"
        elif expires is not None and expires < now:
            reason = "past expiration"
        elif expires is not None and expires - now < timedelta(days=7):
            reason = f"expires in {(expires - now).days + 1}d"
        if reason:
            out.append({"memory_id": m.id, "content": (m.content or "")[:120],
                        "reason": reason, "expires_at": m.expires_at,
                        "updated_at": m.updated_at})
    return out


def _as_utc(value):
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def consolidation_candidates(db: Session, *, workspace_id: int,
                             max_distance: float = 0.2) -> list[dict]:
    """Merge-compatible memories only when confidence/provenance rules
    permit: same source type, EXPERIMENTAL threshold via content similarity
    (deterministic token overlap)."""
    rows = db.query(AIMemory).filter_by(
        workspace_id=workspace_id,
        lifecycle_status="ACTIVE").limit(500).all()
    candidates = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a.memory_type != b.memory_type:
                continue
            if (a.confidence in ("LOW", None) or
                    b.confidence in ("LOW", None)):
                continue
            sim = _content_similarity(a.content or "", b.content or "")
            if sim >= (1.0 - max_distance):
                candidates.append({
                    "memory_a": a.id, "memory_b": b.id,
                    "content_similarity": round(sim, 4),
                    "confidence_a": a.confidence,
                    "confidence_b": b.confidence,
                    "requires_review": True,
                })
    return candidates[:50]


def _content_similarity(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(len(ta | tb), 1)


def conflict_queue(db: Session, *, workspace_id: int,
                   limit: int = 100) -> list[dict]:
    """Route uncertain memory conflicts to review (existing conflict rows
    and low-confidence duplicate candidates)."""
    candidates = consolidation_candidates(db, workspace_id=workspace_id)
    rows = db.query(AIMemory).filter_by(workspace_id=workspace_id)\
        .order_by(AIMemory.updated_at.desc()).limit(limit).all()
    uncertain = [{"memory_id": m.id, "confidence": m.confidence,
                  "content": (m.content or "")[:120],
                  "source": m.source}
                 for m in rows if m.confidence in ("LOW", None)]
    return {"conflict_candidates": candidates[:20],
            "uncertain_memories": uncertain[:20]}


def user_controls(db: Session, *, memory_id: int, workspace_id: int,
                  action: str, user_id: Optional[int] = None) -> dict:
    """User controls over their own memories: inspect, suppress, delete
    (where permitted). Never crosses workspace boundaries."""
    m = db.query(AIMemory).filter_by(id=memory_id,
                                     workspace_id=workspace_id).first()
    if m is None:
        raise KeyError("memory not found")
    if action == "inspect":
        return {
            "memory_id": m.id, "content": m.content, "source": m.source,
            "confidence": m.confidence, "scope": m.scope,
            "memory_type": m.memory_type, "created_at": m.created_at,
            "expires_at": m.expires_at,
            "lifecycle_status": m.lifecycle_status,
            "supersedes_id": m.supersedes_id,
        }
    if action == "suppress":
        m.lifecycle_status = "SUPPRESSED"
        db.flush()
        return {"memory_id": m.id, "lifecycle_status": "SUPPRESSED"}
    if action == "delete":
        if user_id is not None and m.user_id is not None \
                and m.user_id != user_id:
            raise PermissionError("Cannot delete another user's memory")
        db.delete(m)
        db.flush()
        return {"deleted": True, "memory_id": memory_id}
    raise ValueError(f"Unknown action: {action}")