"""Phase 20 — Feedback intelligence.

Unified feedback normalization across search/RAG/citations/summaries/
extraction/agents/workflows, deterministic quality filtering (noise,
contradictions, duplicates, abuse), an authorized feedback→golden-dataset
builder, and governance: feedback becomes a golden example only with
explicit authorization.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import FeedbackEvent

VALID_SOURCES = {
    "search", "rag", "citation", "summary", "extraction", "agent",
    "workflow",
}


def _fingerprint(source: str, workspace_id: int,
                 target_id: Optional[str], comment: Optional[str]) -> str:
    raw = f"{source}:{workspace_id}:{target_id or ''}:{(comment or '').strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def record_feedback(db: Session, *, workspace_id: int,
                    source: str, rating: Optional[int] = None,
                    comment: Optional[str] = None,
                    target_id: Optional[str] = None,
                    user_id: Optional[int] = None) -> FeedbackEvent:
    """Record normalized feedback from any source. Rating: -1/0/+1."""
    if source not in VALID_SOURCES:
        raise ValueError(f"Unknown feedback source: {source}")
    if rating not in (None, -1, 0, 1):
        raise ValueError("Rating must be -1, 0, or 1")
    row = FeedbackEvent(workspace_id=workspace_id, user_id=user_id,
                        source=source, target_id=target_id, rating=rating,
                        comment=comment)
    db.add(row)
    db.flush()
    return row


def feedback_quality(db: Session, *, workspace_id: int,
                     limit: int = 1000) -> list[dict]:
    """Flag low-quality feedback across a workspace. Flags:
    duplicate / contradictory / abusive. Never silently drops feedback —
    flags are stored in quality_flags JSON for review."""
    rows = db.query(FeedbackEvent).filter_by(workspace_id=workspace_id)\
        .order_by(FeedbackEvent.created_at.desc()).limit(limit).all()
    by_target: dict[str, list] = {}
    comments: dict[str, int] = {}
    for r in rows:
        if r.target_id:
            by_target.setdefault(r.target_id, []).append(r)
        key = (r.comment or "").strip().lower()
        if key:
            comments[key] = comments.get(key, 0) + 1
    out = []
    for r in rows:
        flags = []
        if r.target_id and len(by_target.get(r.target_id, [])) > 1:
            ratings = {x.rating for x in by_target[r.target_id]}
            if 1 in ratings and -1 in ratings:
                flags.append("contradictory")
        key = (r.comment or "").strip().lower()
        if key and comments.get(key, 0) > 1:
            flags.append("duplicate")
        if (r.comment or "") and len(r.comment) < 15 and \
                comments.get(key, 0) >= 3:
            flags.append("abusive")
        if not flags and r.comment and len(r.comment) > 200:
            flags.append("noisy")
        if flags:
            r.quality_flags = json.dumps(flags)
        out.append({"id": r.id, "source": r.source, "rating": r.rating,
                    "comment": (r.comment or "")[:120],
                    "target_id": r.target_id, "flags": flags,
                    "status": r.status, "created_at": r.created_at})
    return out


def feedback_summary(db: Session, *, workspace_id: int,
                     limit: int = 1000) -> dict:
    rows = db.query(FeedbackEvent).filter_by(workspace_id=workspace_id)\
        .order_by(FeedbackEvent.created_at.desc()).limit(limit).all()
    total = len(rows)
    positive = sum(1 for r in rows if r.rating == 1)
    negative = sum(1 for r in rows if r.rating == -1)
    flagged = sum(1 for r in rows if r.quality_flags)
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "flagged": flagged,
        "positive_rate": (positive / total) if total else 0.0,
        "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
    }


def promote_to_golden(db: Session, *, feedback_id: int,
                      dataset_name: str, authorized_by: int,
                      reason: Optional[str] = None) -> dict:
    """Governed promotion: only authorized reviewers may turn a feedback
    event into a golden evaluation example."""
    row = db.get(FeedbackEvent, feedback_id)
    if row is None:
        raise KeyError("feedback not found")
    if row.status != "RECEIVED":
        raise ValueError(
            f"Only RECEIVED feedback can be promoted (status={row.status})")
    if not authorized_by:
        raise PermissionError("Promotion requires explicit authorization")
    from ..models.phase20 import ExperimentDataset
    existing = db.query(ExperimentDataset).filter_by(
        name=dataset_name).first()
    items = []
    if existing is not None:
        items = json.loads(existing.items_json or "[]")
    else:
        existing = ExperimentDataset(
            name=dataset_name, kind="golden", domain="rag",
            items_json="[]")
        db.add(existing)
        db.flush()
    items.append({
        "source": row.source, "comment": row.comment,
        "rating": row.rating, "target_id": row.target_id,
        "workspace_id": row.workspace_id,
        "promoted_by": authorized_by, "reason": reason,
    })
    existing.items_json = json.dumps(items, default=str)
    row.status = "GOLDEN"
    return {"dataset_id": existing.id, "feedback_id": row.id,
            "status": "GOLDEN", "examples": len(items)}