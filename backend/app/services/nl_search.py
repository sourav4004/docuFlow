"""Natural language search — intent detection, filters, explanations, saved searches, alerts.

Keyword search is preserved; NL search adds deterministic intent/filter
parsing and an optional AI answer layer.
"""

import json
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.search_intel import SavedSearch, SearchAlert
from ..models.notification import Notification
from ..services.audit_service import log_audit_event

# Intent patterns (deterministic, no LLM required). Patterns match word
# prefixes so plurals/inflections ("expire", "renewal", "policies", "risks")
# are detected.
INTENT_PATTERNS = [
    ("deadline", r"\b(expir\w*|renew\w*|deadline\w*|due|effectiv\w*)\b"),
    ("policy", r"\b(polic\w*|procedur\w*|guideline\w*)\b"),
    ("risk", r"\b(risk\w*|obligation\w*|liabilit\w*)\b"),
    ("summary", r"\b(summar\w*|overview\w*|what about)\b"),
    ("compare", r"\b(compare\w*|difference\w*|versus|\bvs)\b"),
    ("entity", r"\b(compan\w*|person\w*|vendor\w*|employee\w*)\b"),
]

# Natural language date/filter phrases -> structured filters
FILTER_PATTERNS = [
    (r"\blast month\b", {"updated": "last_month"}),
    (r"\bthis month\b", {"updated": "this_month"}),
    (r"\bmodified (in )?the last (\d+) days?\b", {"updated_days": "2"}),
    (r"\brecent\b", {"updated_days": "30"}),
    (r"\bcontracts?\b", {"document_type": "contract"}),
    (r"\bpolicies?\b", {"document_type": "policy"}),
    (r"\binvoices?\b", {"document_type": "invoice"}),
]

DATE_FILTER_MAP = {
    "last_month": "updated >= date('now','start of month','-1 month') and updated < date('now','start of month')",
    "this_month": "updated >= date('now','start of month')",
}


def detect_intent(query: str) -> list[str]:
    """Deterministic intent detection from query text."""
    lower = query.lower()
    return [name for name, pattern in INTENT_PATTERNS if re.search(pattern, lower)]


def extract_filters(query: str) -> dict:
    """Extract structured filters from natural language (never invented)."""
    filters: dict = {}
    lower = query.lower()
    for pattern, mapping in FILTER_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            if "updated_days" in mapping:
                try:
                    mapping["updated_days"] = str(int(match.group(2)))
                except (IndexError, ValueError):
                    pass
            filters.update(mapping)
    return filters


def explain_matches(query: str, result_kind: str) -> list[str]:
    """User-facing reasons a result matched (no hidden chain-of-thought)."""
    reasons = []
    intents = detect_intent(query)
    for intent in intents[:2]:
        reasons.append(f"Semantic match on {intent} intent")
    if result_kind == "keyword":
        reasons.append("Keyword match in document text")
    elif result_kind == "metadata":
        reasons.append("Metadata match")
    elif result_kind == "entity":
        reasons.append("Entity match")
    else:
        reasons.append("Relevance match")
    filters = extract_filters(query)
    if filters:
        reasons.append("Natural-language filters applied")
    return reasons


# ---------------------------------------------------------------
# Saved searches + alerts
# ---------------------------------------------------------------

def create_saved_search(
    db: Session,
    workspace_id: int,
    owner_id: int,
    name: str,
    query: str,
    filters: Optional[dict] = None,
    is_shared: bool = False,
) -> SavedSearch:
    saved = SavedSearch(
        workspace_id=workspace_id,
        owner_id=owner_id,
        name=name,
        query=query,
        filters_json=json.dumps(filters) if filters else None,
        is_shared=is_shared,
    )
    db.add(saved)
    db.flush()
    return saved


def list_saved_searches(db: Session, workspace_id: int, owner_id: int) -> list[SavedSearch]:
    """Personal + workspace-shared saved searches."""
    return (
        db.query(SavedSearch)
        .filter(
            SavedSearch.workspace_id == workspace_id,
            (SavedSearch.owner_id == owner_id) | (SavedSearch.is_shared.is_(True)),
        )
        .order_by(SavedSearch.created_at.desc())
        .all()
    )


def delete_saved_search(db: Session, saved_search: SavedSearch) -> None:
    db.delete(saved_search)
    db.flush()


def create_search_alert(db: Session, workspace_id: int, saved_search_id: int) -> SearchAlert:
    """Create an alert for a saved search (one per search — unique constraint)."""
    alert = SearchAlert(
        workspace_id=workspace_id,
        saved_search_id=saved_search_id,
        is_active=True,
    )
    db.add(alert)
    db.flush()
    return alert


def run_search_alerts(
    db: Session,
    workspace_id: int,
    match_counts: dict[int, int],
    user_ids: Optional[list[int]] = None,
) -> list[Notification]:
    """Record new matches for active alerts and notify (deduplicated per alert).

    Args:
        match_counts: saved_search_id -> current total match count.
    """
    now = datetime.now(timezone.utc)
    alerts = (
        db.query(SearchAlert)
        .filter(SearchAlert.workspace_id == workspace_id, SearchAlert.is_active.is_(True))
        .all()
    )
    notifications = []
    for alert in alerts:
        current = match_counts.get(alert.saved_search_id, 0)
        previous = alert.last_match_count or 0
        alert.last_checked_at = now
        new_matches = max(current - previous, 0)
        alert.last_match_count = current
        if new_matches <= 0:
            continue
        # Dedup: one notification per alert per run
        marker = f"search-alert:{alert.id}"
        already = (
            db.query(Notification)
            .filter(Notification.notification_type == "search_alert", Notification.resource_id == marker)
            .first()
        )
        if already:
            continue
        for user_id in user_ids or []:
            n = Notification(
                user_id=user_id,
                title="New search match",
                message=f"{new_matches} new document(s) matched your saved search.",
                notification_type="search_alert",
                resource_type="search_alert",
                resource_id=marker,
            )
            db.add(n)
            notifications.append(n)
    if notifications:
        db.flush()
    return notifications