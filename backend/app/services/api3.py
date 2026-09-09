"""Phase 19 — API platform 3.0.

Standardized structured errors (code/message/request_id/retryability),
bounded pagination helpers, idempotency standardization for retry-safe
writes, rate-limit tiers, and abuse detection events.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Errors + pagination
# ---------------------------------------------------------------------------

def error_payload(code: str, message: str, *, request_id: Optional[str],
                  retryable: bool = False,
                  status: Optional[int] = None,
                  details: Optional[dict] = None) -> dict:
    """Structured error contract: code, message, request_id, retryability."""
    return {"error": {"code": code, "message": message,
                      "request_id": request_id or "unknown",
                      "retryable": retryable, "status": status,
                      "details": details or {}}}


def page_params(limit: Optional[int], offset: Optional[int],
                max_limit: int = 200) -> dict:
    """Consistent bounded pagination."""
    limit = limit if limit is not None else 50
    offset = offset if offset is not None else 0
    limit = max(1, min(int(limit), max_limit))
    offset = max(0, int(offset))
    return {"limit": limit, "offset": offset}


def request_hash(method: str, path: str, body: Optional[dict]) -> str:
    raw = json.dumps(body or {}, sort_keys=True, default=str)
    return hashlib.sha256(
        f"{method}|{path}|{raw}".encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def idempotency_check(db: Session, *, workspace_id: int,
                      key: str, method: str, path: str,
                      body: Optional[dict] = None,
                      user_id: Optional[int] = None) -> dict:
    """Standard idempotency gate: replay detection + conflict detection.
    Returns existing response when the same request was already applied."""
    from ..models.idempotency import IdempotencyKey
    existing = (db.query(IdempotencyKey)
                .filter(IdempotencyKey.workspace_id == workspace_id,
                        IdempotencyKey.key == key).first())
    digest = request_hash(method, path, body)
    if existing is not None:
        if existing.request_hash != digest:
            return {"decision": "CONFLICT",
                    "reason": "same idempotency key, different request"}
        return {"decision": "REPLAY",
                "response": existing.response_json,
                "status": existing.response_status}
    row = IdempotencyKey(workspace_id=workspace_id, user_id=user_id, key=key,
                         request_hash=digest,
                         expires_at=_utcnow() + timedelta(hours=24))
    db.add(row)
    db.flush()
    return {"decision": "NEW", "record_id": row.id}


def idempotency_complete(db: Session, *, workspace_id: int, key: str,
                         response_json: str, status_code: int) -> None:
    from ..models.idempotency import IdempotencyKey
    row = (db.query(IdempotencyKey)
           .filter(IdempotencyKey.workspace_id == workspace_id,
                   IdempotencyKey.key == key).first())
    if row is not None:
        row.response_json = response_json
        row.response_status = status_code
        db.flush()


# ---------------------------------------------------------------------------
# Rate-limit tiers + abuse detection
# ---------------------------------------------------------------------------

def rate_limit_tiers() -> dict:
    """Configurable rate-limit tiers (requests/minute default)."""
    import os
    def env(name, default):
        return int(os.getenv(name, str(default)))
    return {
        "anonymous": env("RATE_LIMIT_ANONYMOUS", 20),
        "user": env("RATE_LIMIT_USER", 300),
        "workspace": env("RATE_LIMIT_WORKSPACE", 2000),
        "organization": env("RATE_LIMIT_ORGANIZATION", 10000),
        "api_key": env("RATE_LIMIT_API_KEY", 600),
        "operator": env("RATE_LIMIT_OPERATOR", 5000),
    }


def detect_abuse(db: Session, *, workspace_id: Optional[int] = None,
                 user_id: Optional[int] = None,
                 api_key_id: Optional[int] = None,
                 failures_in_window: int = 0,
                 requests_in_window: int = 0,
                 window_minutes: int = 5,
                 enumeration_hint: bool = False,
                 scope_violation_hint: bool = False,
                 persist: bool = True) -> dict:
    """API abuse detection: repeated failures, excessive requests,
    enumeration, scope violations. Persists durable events (no secrets)."""
    from ..models.phase19 import ApiAbuseEvent
    flagged = []
    if failures_in_window >= 20:
        flagged.append("REPEATED_FAILURES")
    if requests_in_window >= rate_limit_tiers()["api_key"] * \
            max(1, window_minutes // 60 + 1):
        flagged.append("EXCESSIVE_REQUESTS")
    if enumeration_hint and requests_in_window >= 30:
        flagged.append("ENUMERATION")
    if scope_violation_hint:
        flagged.append("SCOPE_VIOLATION")
    if persist:
        for kind in flagged:
            db.add(ApiAbuseEvent(api_key_id=api_key_id, user_id=user_id,
                                 workspace_id=workspace_id, kind=kind,
                                 detail=f"abuse window {window_minutes}m",
                                 window_start=_utcnow()))
        db.flush()
    return {"flagged": bool(flagged), "signals": flagged,
            "blocked": "SCOPE_VIOLATION" in flagged
            or "REPEATED_FAILURES" in flagged}


def recent_abuse(db: Session, *, api_key_id: Optional[int] = None,
                 user_id: Optional[int] = None,
                 since_minutes: int = 30,
                 limit: int = 100) -> list:
    from ..models.phase19 import ApiAbuseEvent
    since = _utcnow() - timedelta(minutes=since_minutes)
    q = db.query(ApiAbuseEvent).filter(ApiAbuseEvent.created_at >= since)
    if api_key_id is not None:
        q = q.filter(ApiAbuseEvent.api_key_id == api_key_id)
    if user_id is not None:
        q = q.filter(ApiAbuseEvent.user_id == user_id)
    return q.order_by(ApiAbuseEvent.created_at.desc()
                      ).limit(min(limit, 500)).all()
