"""Phase 19 — performance 2.0.

Request deduplication for safe identical concurrent requests (single-flight
registry), cache-3.0 audit helpers (TTL/invalidation/tenant isolation/
memory bounds), and cache-stampede protection (regenerate once, serve stale
while others wait).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request deduplication 2.0 (single-flight, in-process registry)
# ---------------------------------------------------------------------------

class SingleFlightRegistry:
    """Deduplicates identical concurrent requests per key.

    Safe use is caller-scoped: only READ/idempotent operations may be
    deduplicated across requests.
    """

    def __init__(self) -> None:
        self._active: dict[str, float] = {}

    def try_claim(self, key: str, *, now: Optional[float] = None,
                  ttl_s: float = 30.0) -> dict:
        now = now if now is not None else time.monotonic()
        started = self._active.get(key)
        if started is not None and now - started < ttl_s:
            return {"decision": "RUNNING", "key": key,
                    "joined": True, "started_at": started}
        self._active[key] = now
        return {"decision": "LEADER", "key": key,
                "joined": False, "started_at": now}

    def complete(self, key: str) -> None:
        self._active.pop(key, None)

    def active_count(self) -> int:
        return len(self._active)


def dedupe_policy(*, idempotent: bool) -> dict:
    if not idempotent:
        return {"dedupe_allowed": False,
                "reason": "non-idempotent requests are never deduplicated"}
    return {"dedupe_allowed": True, "reason": "safe identical requests join "
            "the in-flight execution"}


# ---------------------------------------------------------------------------
# Cache 3.0 audit + stampede protection
# ---------------------------------------------------------------------------

def cache_audit(*, ttl_s: Optional[int], tenant_isolated: bool,
                invalidated_on_write: bool,
                max_entries: Optional[int]) -> dict:
    """Audit one cache configuration."""
    issues = []
    if ttl_s is None or ttl_s <= 0:
        issues.append("missing or non-positive TTL")
    if not tenant_isolated:
        issues.append("cache is not tenant-isolated")
    if not invalidated_on_write:
        issues.append("no invalidation on write configured")
    if max_entries is None or max_entries <= 0:
        issues.append("cache has no memory bound")
    return {"healthy": not issues, "issues": issues}


def stampede_protect(cache: dict, key: str, *, ttl_s: float,
                     now: Optional[float] = None) -> dict:
    """Regenerate once; everyone else either receives the in-flight
    regeneration marker or serves the last value until it completes."""
    now = now if now is not None else time.monotonic()
    entry = cache.get(key)
    if entry is None:
        return {"decision": "REGENERATE", "reason": "cache miss",
                "key": key}
    value, created, regenerating = entry
    if regenerating:
        return {"decision": "WAIT_FOR_REGEN", "key": key,
                "value": value, "reason": "another request is regenerating"}
    if now - created > ttl_s:
        cache[key] = (value, created, True)
        return {"decision": "REGENERATE_WITH_STALE",
                "key": key, "value": value,
                "reason": "stale value served while one request "
                          "regenerates"}
    return {"decision": "HIT", "key": key, "value": value}


def complete_regeneration(cache: dict, key: str, value,
                          now: Optional[float] = None) -> None:
    now = now if now is not None else time.monotonic()
    cache[key] = (value, now, False)


def cache_stats(cache: dict, limit: int = 1000) -> dict:
    """Hit/eviction/memory-bound stats for the provided registry."""
    entries = list(cache.items())[:limit]
    stale = sum(1 for _, (_, created, regen) in entries if regen)
    return {"entries": len(entries),
            "regenerating": stale,
            "memory_bound": len(cache) <= limit,
            "note": "tenant isolation must be enforced by cache key design"}
