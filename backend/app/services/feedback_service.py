"""AI feedback service — submission and workspace-level analytics.

Feedback is stored per workspace and never used to train models
automatically. Analytics are tenant scoped.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.search_intel import AIFeedback

VALID_RATINGS = ("thumbs_up", "thumbs_down")
VALID_CATEGORIES = ("correct", "incorrect", "missing_info", "citation_issue", None)


def submit_feedback(
    db: Session,
    workspace_id: int,
    user_id: int,
    rating: str,
    execution_id: Optional[str] = None,
    category: Optional[str] = None,
    comment: Optional[str] = None,
) -> AIFeedback:
    """Record user feedback on an AI answer."""
    if rating not in VALID_RATINGS:
        raise ValueError(f"Invalid rating: {rating}")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category: {category}")
    if comment and len(comment) > 2000:
        comment = comment[:2000]

    feedback = AIFeedback(
        workspace_id=workspace_id,
        user_id=user_id,
        execution_id=execution_id,
        rating=rating,
        category=category,
        comment=comment,
    )
    db.add(feedback)
    db.flush()
    return feedback


def workspace_feedback_analytics(db: Session, workspace_id: int, days: int = 30) -> dict:
    """Aggregate feedback analytics for a workspace (tenant scoped)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    items = (
        db.query(AIFeedback)
        .filter(
            AIFeedback.workspace_id == workspace_id,
            AIFeedback.created_at >= cutoff,
        )
        .all()
    )
    total = len(items)
    if total == 0:
        return {
            "total_feedback": 0,
            "acceptance_rate": None,
            "thumbs_up": 0,
            "thumbs_down": 0,
            "categories": {},
            "recent_comments": [],
        }

    thumbs_up = sum(1 for f in items if f.rating == "thumbs_up")
    categories: dict = {}
    for f in items:
        key = f.category or "unspecified"
        categories[key] = categories.get(key, 0) + 1

    return {
        "total_feedback": total,
        "acceptance_rate": round(thumbs_up / total, 3),
        "thumbs_up": thumbs_up,
        "thumbs_down": total - thumbs_up,
        "categories": categories,
        "recent_comments": [
            {"id": f.id, "rating": f.rating, "comment": f.comment, "created_at": f.created_at}
            for f in items
            if f.comment
        ][-20:],
    }