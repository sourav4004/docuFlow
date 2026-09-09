"""Phase 19 — search 5.0.

Unified search planning combining keyword/vector/metadata/graph/memory/
freshness/scope, user-facing explainability (filters, ranking factors,
scope, source categories — never chain-of-thought), search-quality
analytics, and safe per-user personalization that never leaks data across
scopes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SOURCE_CATEGORIES = ("keyword", "vector", "metadata", "graph", "memory",
                     "versions", "connector", "workspace")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def query_hash(query: str) -> str:
    import hashlib
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:32]


def planner2(query: str, filters: Optional[dict] = None,
             scope: str = "workspace") -> dict:
    """Combine retrieval modes + scope + freshness intent into one plan."""
    filters = filters or {}
    lowered = (query or "").lower()
    sources = ["keyword"]
    strategy = "keyword"
    if len((query or "").split()) >= 3 or " " in (query or ""):
        sources.append("vector")
        strategy = "hybrid"
    if filters.get("entity") or _entity_like(lowered):
        sources.append("graph")
    if any(word in lowered for word in ("remember", "preference", "what do "
                                        "we know")):
        sources.append("memory")
    if filters.get("version") or any(word in lowered for word in
                                     ("old version", "previous version",
                                      "revision")):
        sources.append("versions")
    if filters.get("source"):
        sources.append("connector" if filters["source"] == "connector"
                       else "workspace")
    sources = [s for s in SOURCE_CATEGORIES if s in sources] or ["keyword"]
    return {
        "query": query, "strategy": strategy, "sources": sources,
        "scope": scope, "freshness_boost": bool(filters.get("fresh"))
        or any(word in lowered for word in ("current", "latest", "recent")),
        "ranking_factors": ["relevance", "authority", "freshness"],
        "filters": filters,
    }


def _entity_like(text: str) -> bool:
    return any(word in text for word in ("who", "what company", "org chart",
                                         "reporting"))


def explain_search(plan: dict, results: list[dict], top: int = 3) -> dict:
    """Explainable search result summary (never chain-of-thought)."""
    categories = {}
    for result in results:
        source = result.get("source_category") or plan["sources"][0]
        categories[source] = categories.get(source, 0) + 1
    return {
        "scope": plan.get("scope"),
        "filters": plan.get("filters"),
        "retrieval_mode": plan.get("strategy"),
        "sources_used": plan["sources"],
        "ranking_factors": plan.get("ranking_factors"),
        "result_categories": categories,
        "why_top_result": {
            "reason": "highest combined relevance + authority + freshness "
                      "score in scope",
            "top_result_id": results[0].get("id") if results else None,
        },
        "note": "explanation covers scope/filters/mode/scores — not "
                "internal reasoning",
    }


def record_quality(db: Session, *, workspace_id: int,
                   organization_id: Optional[int] = None,
                   query: Optional[str] = None,
                   retrieval_mode: Optional[str] = None,
                   latency_ms: Optional[int] = None,
                   result_count: Optional[int] = None,
                   useful_signal: Optional[bool] = None) -> dict:
    """Search-quality event — hashes only; never stores query text."""
    from ..models.phase18 import SearchAnalyticsEvent
    event = SearchAnalyticsEvent(
        workspace_id=workspace_id, organization_id=organization_id,
        query_hash=query_hash(query) if query else None,
        retrieval_mode=retrieval_mode, latency_ms=latency_ms,
        result_count=result_count, zero_results=(result_count == 0),
        useful_signal=useful_signal)
    db.add(event)
    db.flush()
    return {"event_id": event.id, "zero_results": event.zero_results}


def quality_summary(db: Session, *, workspace_id: int,
                    days: int = 30) -> dict:
    from ..models.phase18 import SearchAnalyticsEvent
    since = _utcnow() - timedelta(days=days)
    rows = (db.query(SearchAnalyticsEvent)
            .filter(SearchAnalyticsEvent.workspace_id == workspace_id,
                    SearchAnalyticsEvent.created_at >= since)
            .limit(5000).all())
    zero = sum(1 for r in rows if r.zero_results)
    useful = [r for r in rows if r.useful_signal is not None]
    latencies = sorted(r.latency_ms for r in rows if r.latency_ms
                       is not None)
    def pct(p):
        if not latencies:
            return None
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx]
    return {"events": len(rows),
            "zero_result_rate": round(zero / len(rows), 4) if rows else 0.0,
            "useful_rate": round(sum(1 for r in useful if r.useful_signal)
                                 / len(useful), 4) if useful else None,
            "latency_p95_ms": pct(0.95),
            "unique_queries": len({r.query_hash for r in rows
                                   if r.query_hash})}


# ---------------------------------------------------------------------------
# Personalization (workspace-scoped, never cross-scope)
# ---------------------------------------------------------------------------

def get_profile(db: Session, *, user_id: int, workspace_id: int) -> dict:
    from ..models.phase19 import SearchPersonalizationProfile
    profile = (db.query(SearchPersonalizationProfile)
               .filter(SearchPersonalizationProfile.user_id == user_id,
                       SearchPersonalizationProfile.workspace_id ==
                       workspace_id).first())
    if profile is None:
        return {"user_id": user_id, "workspace_id": workspace_id,
                "preferences": {}, "recent_intents": []}
    return {"user_id": profile.user_id,
            "workspace_id": profile.workspace_id,
            "preferences": _loads(profile.preferences_json) or {},
            "recent_intents": _loads(profile.recent_intents_json) or []}


def update_profile(db: Session, *, user_id: int, workspace_id: int,
                   preferences: Optional[dict] = None,
                   intent: Optional[str] = None) -> dict:
    from ..models.phase19 import SearchPersonalizationProfile
    profile = (db.query(SearchPersonalizationProfile)
               .filter(SearchPersonalizationProfile.user_id == user_id,
                       SearchPersonalizationProfile.workspace_id ==
                       workspace_id).first())
    if profile is None:
        profile = SearchPersonalizationProfile(user_id=user_id,
                                               workspace_id=workspace_id,
                                               preferences_json="{}",
                                               recent_intents_json="[]")
        db.add(profile)
    current_prefs = _loads(profile.preferences_json) or {}
    if preferences is not None:
        current_prefs.update(preferences)
        profile.preferences_json = json.dumps(current_prefs)[:4000]
    if intent:
        intents = _loads(profile.recent_intents_json) or []
        intents = ([intent] + [i for i in intents if i != intent])[:5]
        profile.recent_intents_json = json.dumps(intents)[:4000]
    profile.updated_at = _utcnow()
    db.flush()
    return {"user_id": user_id, "workspace_id": workspace_id,
            "preferences": current_prefs}


def personalize_query(profile: dict, query: str) -> dict:
    """Safe personalization: boosts terms from the user's OWN recent intents
    in their OWN workspace. Never leaks private information across scopes."""
    intents = profile.get("recent_intents") or []
    boosted = []
    lowered = query.lower()
    for intent in intents:
        tokens = [t for t in str(intent).lower().split() if len(t) > 3]
        for token in tokens:
            if token in lowered or any(token in w for w in lowered.split()):
                boosted.append(token)
    return {"query": query, "boosted_terms": list(dict.fromkeys(boosted)),
            "personalized": bool(boosted),
            "note": "personalization only uses the user's own workspace "
                    "history"}


def _loads(raw: Optional[str]):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None
