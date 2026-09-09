"""Phase 18 search platform — unified planner, scope enforcement,
explainability, and authorized analytics.

Search analytics never stores raw query text — only a hash plus safe
metrics — and is always workspace-scoped.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase18 import SearchAnalyticsEvent

logger = logging.getLogger(__name__)

MODES = ("keyword", "vector", "hybrid", "graph", "memory", "metadata")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:24]


def unified_plan(query: str, filters: Optional[dict] = None,
                 scopes: Optional[list[str]] = None) -> dict:
    """Deterministic planner choosing retrieval mode + scope.

    Never exposes chain-of-thought — only the chosen plan.
    """
    scopes = scopes or ["document"]
    f = filters or {}
    if f.get("entity_id"):
        mode = "graph"
    elif f.get("memory"):
        mode = "memory"
    elif f.get("metadata"):
        mode = "metadata"
    elif f.get("collection_id") or len((query or "").split()) <= 2:
        mode = "hybrid"
    else:
        mode = "hybrid"
    if mode not in MODES:
        mode = "hybrid"
    return {
        "mode": mode,
        "scopes": scopes,
        "filters": f,
        "explanation": {
            "interpreted_query": (query or "").strip()[:200],
            "retrieval_mode": mode,
            "scope": scopes,
            "filters_applied": sorted(f.keys()),
        },
    }


def explain_results(plan: dict, results: list[dict], top: int = 3) -> dict:
    """Per-result rationale (scores and source) — safe, no chain-of-thought."""
    return {
        "plan": plan["explanation"],
        "total": len(results),
        "top_results": [
            {"document_id": r.get("document_id"),
             "chunk_id": r.get("chunk_id"),
             "score": r.get("score"),
             "mode": r.get("mode", plan["mode"]),
             "reason": f"score {r.get('score', 0):.3f} from "
                       f"{r.get('source', 'document')}"}
            for r in results[:top]],
    }


def record_search(db: Session, *, workspace_id: int,
                  organization_id: Optional[int],
                  query: str, mode: str, latency_ms: int,
                  result_count: int,
                  useful_signal: Optional[bool] = None) -> SearchAnalyticsEvent:
    """Authorized, non-sensitive search analytics record."""
    row = SearchAnalyticsEvent(
        workspace_id=workspace_id,
        organization_id=organization_id,
        query_hash=query_hash(query),
        retrieval_mode=mode[:20],
        latency_ms=latency_ms,
        result_count=result_count,
        zero_results=(result_count == 0),
        useful_signal=useful_signal,
    )
    db.add(row)
    db.flush()
    return row


def search_analytics(db: Session, workspace_id: int,
                     days: int = 30,
                     limit: int = 500) -> dict:
    """Aggregate search metrics for the workspace (bounded)."""
    from datetime import timedelta
    since = _utcnow() - timedelta(days=days)
    rows = (db.query(SearchAnalyticsEvent)
            .filter(SearchAnalyticsEvent.workspace_id == workspace_id,
                    SearchAnalyticsEvent.created_at >= since)
            .order_by(SearchAnalyticsEvent.created_at.desc())
            .limit(min(limit, 2000)).all())
    zero = sum(1 for r in rows if r.zero_results)
    latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
    mode_counts: dict = {}
    query_counts: dict = {}
    for r in rows:
        mode_counts[r.retrieval_mode] = mode_counts.get(r.retrieval_mode, 0) + 1
        if r.query_hash:
            query_counts[r.query_hash] = query_counts.get(r.query_hash, 0) + 1
    p95 = None
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1,
                          int(round(0.95 * (len(ordered) - 1))))]
    return {
        "searches": len(rows),
        "zero_result_rate": round(zero / len(rows), 3) if rows else None,
        "p95_latency_ms": p95,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1)
        if latencies else None,
        "by_mode": mode_counts,
        "top_queries_by_hash": sorted(query_counts.items(),
                                      key=lambda kv: -kv[1])[:10],
    }