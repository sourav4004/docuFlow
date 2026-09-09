"""Phase 18 provider production layer — timeouts, retries, rate limits,
percentile latency metrics.

All policies are deterministic and configuration-driven; no provider call
ever retries indefinitely.
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase18 import ProviderCallMetric

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUTS = {
    "connect_seconds": 10.0,
    "read_seconds": 60.0,
    "total_seconds": 120.0,
}
MAX_RETRIES = 5
MAX_BACKOFF_SECONDS = 60.0

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def timeout_policy() -> dict:
    """Configurable connect/read/total timeouts from environment."""
    return {
        "connect_seconds": float(os.getenv("PROVIDER_CONNECT_TIMEOUT",
                                           DEFAULT_TIMEOUTS["connect_seconds"])),
        "read_seconds": float(os.getenv("PROVIDER_READ_TIMEOUT",
                                        DEFAULT_TIMEOUTS["read_seconds"])),
        "total_seconds": float(os.getenv("PROVIDER_TOTAL_TIMEOUT",
                                         DEFAULT_TIMEOUTS["total_seconds"])),
    }


def retry_policy(status_code: Optional[int] = None,
                 attempt: int = 0,
                 retry_after: Optional[float] = None) -> dict:
    """Bounded exponential backoff with jitter.

    - non-retryable 4xx (400/401/403/404/422) never retry
    - 429 honors ``Retry-After`` when present
    - 5xx/408/429 retry up to ``MAX_RETRIES``
    """
    if attempt >= MAX_RETRIES:
        return {"retryable": False, "reason": "max attempts reached"}
    if status_code is not None and status_code not in RETRYABLE_STATUS:
        return {"retryable": False,
                "reason": f"non-retryable status {status_code}"}
    base = min(2.0 * (2 ** max(0, attempt)), MAX_BACKOFF_SECONDS)
    jitter = random.uniform(0.8, 1.2)  # noqa: S311 — retry jitter only
    delay = round(base * jitter, 3)
    if status_code == 429 and retry_after is not None:
        delay = max(delay, min(float(retry_after), MAX_BACKOFF_SECONDS))
    return {"retryable": True, "delay_seconds": delay,
            "attempt": attempt + 1, "max_attempts": MAX_RETRIES}


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS


def classify_error(exc: BaseException, status_code: Optional[int] = None) -> str:
    """Normalized failure classification for observability."""
    name = type(exc).__name__.lower()
    if status_code == 429 or "ratelimit" in name or "rate limit" in name:
        return "rate_limit"
    if status_code in (408,) or "timeout" in name:
        return "timeout"
    if status_code in (500, 502, 503, 504):
        return "provider"
    if status_code in (400, 422):
        return "validation"
    if status_code in (401, 403):
        return "authorization"
    return "unknown"


# ---------------------------------------------------------------------------
# Latency metrics (p50/p95/p99)
# ---------------------------------------------------------------------------

def record_call(db: Session, *, provider: str, model: Optional[str],
                ok: bool, latency_ms: Optional[float],
                error_class: Optional[str] = None,
                now: Optional[datetime] = None) -> ProviderCallMetric:
    row = ProviderCallMetric(provider=provider, model=model, ok=ok,
                             latency_ms=latency_ms,
                             error_class=error_class,
                             created_at=now or _utcnow())
    db.add(row)
    db.flush()
    return row


def _percentile(sorted_values: list[float], pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    idx = max(0, min(len(sorted_values) - 1,
                     int(round((pct / 100.0) * (len(sorted_values) - 1)))))
    return round(sorted_values[idx], 1)


def _as_utc(value) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def latency_percentiles(db: Session, *, provider: Optional[str] = None,
                        minutes: int = 60,
                        limit: int = 5000) -> dict:
    """p50/p95/p99 over the last ``minutes`` (bounded sample)."""
    cutoff = _utcnow()
    try:
        from datetime import timedelta
        cutoff = cutoff - timedelta(minutes=minutes)
    except Exception:  # noqa: BLE001
        pass
    q = db.query(ProviderCallMetric).filter(
        ProviderCallMetric.created_at >= cutoff)
    if provider:
        q = q.filter(ProviderCallMetric.provider == provider)
    rows = q.order_by(ProviderCallMetric.created_at.desc()).limit(limit).all()
    latencies = sorted([r.latency_ms for r in rows
                        if r.latency_ms is not None and r.ok])
    all_latencies = sorted([r.latency_ms for r in rows
                            if r.latency_ms is not None])
    ok_count = sum(1 for r in rows if r.ok)
    total = len(rows)
    return {
        "provider": provider or "*",
        "samples": total,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "overall_p50_ms": _percentile(all_latencies, 50),
        "success_rate": round(ok_count / total, 4) if total else None,
        "error_rate": round((total - ok_count) / total, 4) if total else None,
        "error_classes": _error_class_counts(rows),
    }


def _error_class_counts(rows) -> dict:
    counts: dict = {}
    for r in rows:
        if not r.ok and r.error_class:
            counts[r.error_class] = counts.get(r.error_class, 0) + 1
    return counts


def provider_health_dashboard(db: Session, limit: int = 50) -> dict:
    """Per-provider aggregate health for the ops dashboard."""
    from .provider_platform import provider_status
    statuses = provider_status(db, limit=limit)
    items = []
    for h in statuses:
        pct = latency_percentiles(db, provider=h.provider, minutes=1440)
        items.append({
            "provider": h.provider,
            "model": h.model,
            "status": h.status,
            "circuit_state": h.circuit_state,
            "success_count": h.success_count,
            "failure_count": h.failure_count,
            "avg_latency_ms": h.avg_latency_ms,
            "last_error": (h.last_error or "")[:200],
            "last_checked_at": h.last_checked_at,
            "p95_ms": pct["p95_ms"],
            "p99_ms": pct["p99_ms"],
            "success_rate": pct["success_rate"],
        })
    return {"providers": items, "total": len(items)}