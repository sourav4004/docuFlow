"""Phase 20 — Retrieval self-improvement.

Query-quality analysis, deterministic retrieval-failure classification,
persisted recommendations (never auto-applied), and an explicit feedback
loop that enriches evaluation datasets instead of silently changing
production ranking.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import (
    RetrievalFailure, RetrievalRecommendation, SearchQualityEvent,
)

FAILURE_CLASSES = [
    "missing_document", "poor_chunking", "poor_keyword_match",
    "poor_vector_match", "metadata_mismatch", "graph_miss",
    "stale_knowledge", "permission_restriction",
]

RECOMMENDATION_CATEGORIES = [
    "chunk_size", "overlap", "retrieval_weights", "reranking",
    "metadata", "synonyms", "embedding_model",
]


def _hash(text: str) -> str:
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:16]


def record_search_event(db: Session, *, workspace_id: int, query: str,
                        event_type: str, result_count: Optional[int] = None,
                        latency_ms: Optional[float] = None,
                        query_preview: Optional[str] = None) -> SearchQualityEvent:
    ev = SearchQualityEvent(
        workspace_id=workspace_id, query_hash=_hash(query),
        query_preview=(query_preview or query)[:300],
        event_type=event_type, result_count=result_count,
        latency_ms=latency_ms)
    db.add(ev)
    db.flush()
    return ev


def classify_failure(db: Session, *, workspace_id: int, query: str,
                     failure_class: str, detail: Optional[str] = None,
                     query_preview: Optional[str] = None) -> RetrievalFailure:
    if failure_class not in FAILURE_CLASSES:
        raise ValueError(f"Unknown failure class: {failure_class}")
    f = RetrievalFailure(
        workspace_id=workspace_id, query_hash=_hash(query),
        query_preview=(query_preview or query)[:300],
        failure_class=failure_class, detail=detail)
    db.add(f)
    db.flush()
    return f


def failure_summary(db: Session, *, workspace_id: int,
                    limit: int = 100) -> dict:
    rows = db.query(RetrievalFailure).filter_by(workspace_id=workspace_id)\
        .order_by(RetrievalFailure.created_at.desc()).limit(limit).all()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.failure_class] = counts.get(r.failure_class, 0) + 1
    return {"total": len(rows),
            "by_class": dict(sorted(counts.items(),
                                    key=lambda kv: -kv[1])),
            "recent": [{"id": r.id, "failure_class": r.failure_class,
                        "query_preview": r.query_preview,
                        "created_at": r.created_at} for r in rows[:20]]}


def generate_recommendations(db: Session, *, workspace_id: int) -> list[dict]:
    """Generate deterministic recommendations from recorded failures.

    Each recommendation is persisted in PROPOSED state and requires
    evaluation before any activation.
    """
    summary = failure_summary(db, workspace_id=workspace_id)
    by_class = summary["by_class"]
    created = []
    if by_class.get("poor_chunking", 0) >= 2:
        created.append(_recommend(
            db, workspace_id, "chunk_size",
            "Evaluate smaller chunk size for workspaces with repeated "
            "poor_chunking failures",
            f"{by_class['poor_chunking']} poor_chunking failures recorded"))
    if by_class.get("poor_keyword_match", 0) >= 2:
        created.append(_recommend(
            db, workspace_id, "synonyms",
            "Add synonyms/aliases for queries with repeated keyword misses",
            f"{by_class['poor_keyword_match']} keyword misses recorded"))
    if by_class.get("metadata_mismatch", 0) >= 1:
        created.append(_recommend(
            db, workspace_id, "metadata",
            "Review document metadata extraction for metadata_mismatch "
            "failures", f"{by_class['metadata_mismatch']} mismatches"))
    if by_class.get("poor_vector_match", 0) >= 2:
        created.append(_recommend(
            db, workspace_id, "embedding_model",
            "Evaluate a different embedding model for repeated vector "
            "misses", f"{by_class['poor_vector_match']} vector misses"))
    if by_class.get("stale_knowledge", 0) >= 1:
        created.append(_recommend(
            db, workspace_id, "retrieval_weights",
            "Re-evaluate freshness weighting given stale_knowledge hits",
            f"{by_class['stale_knowledge']} stale hits"))
    return created


def _recommend(db: Session, workspace_id: int, category: str,
               suggestion: str, rationale: str) -> dict:
    rec = RetrievalRecommendation(
        workspace_id=workspace_id, category=category,
        suggestion=suggestion, rationale=rationale)
    db.add(rec)
    db.flush()
    return {"id": rec.id, "category": category, "suggestion": suggestion,
            "status": rec.status}


def list_recommendations(db: Session, *, workspace_id: int,
                         status: Optional[str] = None,
                         limit: int = 100) -> list[dict]:
    q = db.query(RetrievalRecommendation).filter_by(
        workspace_id=workspace_id)
    if status:
        q = q.filter(RetrievalRecommendation.status == status)
    rows = q.order_by(RetrievalRecommendation.created_at.desc())\
        .limit(limit).all()
    return [{"id": r.id, "category": r.category, "suggestion": r.suggestion,
             "rationale": r.rationale, "status": r.status,
             "created_at": r.created_at} for r in rows]


def query_quality_analysis(db: Session, *, workspace_id: int,
                           limit: int = 200) -> dict:
    """Aggregate search-quality events into deterministic insights."""
    rows = db.query(SearchQualityEvent).filter_by(
        workspace_id=workspace_id)\
        .order_by(SearchQualityEvent.created_at.desc()).limit(limit).all()
    total = len(rows)
    zero = sum(1 for r in rows if r.event_type == "zero_result")
    reformulated = sum(1 for r in rows if r.event_type == "reformulated")
    clicked = sum(1 for r in rows if r.event_type == "clicked")
    abandoned = sum(1 for r in rows if r.event_type == "abandoned")
    latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    return {
        "total_events": total,
        "zero_result_rate": (zero / total) if total else 0.0,
        "reformulation_rate": (reformulated / total) if total else 0.0,
        "click_rate": (clicked / total) if total else 0.0,
        "abandoned_rate": (abandoned / total) if total else 0.0,
        "avg_latency_ms": avg_latency,
        "poor_queries": [
            {"query_preview": r.query_preview, "event_type": r.event_type,
             "created_at": r.created_at}
            for r in rows if r.event_type in ("zero_result", "abandoned")
        ][:10],
    }


def feedback_to_dataset(db: Session, *, workspace_id: int,
                        query: str, relevant_ids: list[str],
                        dataset_name: str,
                        created_by: Optional[int] = None) -> dict:
    """Explicit user feedback becomes a curated evaluation example.

    Never silently alters production ranking — it only enriches an evaluation
    dataset that requires authorization before use as a golden example.
    """
    from ..models.phase20 import ExperimentDataset
    existing = db.query(ExperimentDataset).filter_by(
        name=dataset_name).first()
    items = []
    if existing is not None:
        items = json.loads(existing.items_json or "[]")
    else:
        existing = ExperimentDataset(
            name=dataset_name, kind="curated", domain="retrieval",
            items_json="[]")
        db.add(existing)
        db.flush()
    items.append({"query": query, "relevant": relevant_ids,
                  "source": "user_feedback", "workspace_id": workspace_id})
    existing.items_json = json.dumps(items, default=str)
    return {"dataset_id": existing.id, "examples": len(items),
            "status": "curated"}