"""In-memory sliding window rate limiter.

Provides per-key rate limiting with configurable windows and limits.
Keys are typically IP addresses (unauthenticated) or user IDs (authenticated).

This implementation is single-process suitable for the current development
architecture. For multi-process/distributed deployments, a Redis-backed
implementation would be needed.
"""

import time
import logging
from collections import defaultdict
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """Configuration for rate limiting."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        auth_requests_per_minute: int = 30,
        auth_requests_per_hour: int = 500,
        upload_requests_per_minute: int = 5,
        message_requests_per_minute: int = 20,
        login_requests_per_minute: int = 10,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.auth_requests_per_minute = auth_requests_per_minute
        self.auth_requests_per_hour = auth_requests_per_hour
        self.upload_requests_per_minute = upload_requests_per_minute
        self.message_requests_per_minute = message_requests_per_minute
        self.login_requests_per_minute = login_requests_per_minute


# Default config
DEFAULT_CONFIG = RateLimitConfig()


class SlidingWindowCounter:
    """Sliding window rate limit counter."""

    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check_and_record(self, key: str, window_seconds: int, max_requests: int) -> tuple[bool, int]:
        """Check if request is allowed and record it.

        Returns:
            (allowed, retry_after_seconds)
        """
        now = time.time()
        cutoff = now - window_seconds

        # Clean old entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        if len(self._requests[key]) >= max_requests:
            # Calculate when the oldest request in the window will expire
            oldest = self._requests[key][0]
            retry_after = int(oldest + window_seconds - now) + 1
            return False, max(retry_after, 1)

        self._requests[key].append(now)
        return True, 0

    def cleanup(self, max_age: int = 7200):
        """Remove entries older than max_age seconds."""
        cutoff = time.time() - max_age
        empty_keys = []
        for key, timestamps in self._requests.items():
            self._requests[key] = [t for t in timestamps if t > cutoff]
            if not self._requests[key]:
                empty_keys.append(key)
        for key in empty_keys:
            del self._requests[key]


# Global counter instance
_counter = SlidingWindowCounter()


def _get_client_key(request: Request, user_id: Optional[int] = None) -> str:
    """Get the rate limit key for a request."""
    if user_id is not None:
        return f"user:{user_id}"

    # API key requests are limited per-key (revoked/expired keys never reach here)
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        import hashlib
        token = auth_header[7:].strip()
        return f"apikey:{hashlib.sha256(token.encode()).hexdigest()[:16]}"

    # Use X-Forwarded-For if available (behind proxy), else direct IP
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    return f"ip:{ip}"


def _classify_endpoint(path: str, method: str) -> str:
    """Classify endpoint for rate limit selection."""
    if path.startswith("/auth/login") or path.startswith("/auth/register"):
        return "login"
    if path.startswith("/documents") and method == "POST":
        return "upload"
    if path.endswith("/messages") and method == "POST":
        return "message"
    return "default"


def check_rate_limit(
    request: Request,
    user_id: Optional[int] = None,
) -> tuple[bool, int]:
    """Check rate limit for a request.

    Returns:
        (allowed, retry_after_seconds)
    """
    key = _get_client_key(request, user_id)
    path = request.url.path
    method = request.method
    category = _classify_endpoint(path, method)

    if category == "login":
        return _counter.check_and_record(f"{key}:login", 60, DEFAULT_CONFIG.login_requests_per_minute)
    elif category == "upload":
        return _counter.check_and_record(f"{key}:upload", 60, DEFAULT_CONFIG.upload_requests_per_minute)
    elif category == "message":
        return _counter.check_and_record(f"{key}:message", 60, DEFAULT_CONFIG.message_requests_per_minute)
    else:
        # Default: check per-minute and per-hour
        allowed, retry = _counter.check_and_record(f"{key}:min", 60, DEFAULT_CONFIG.requests_per_minute)
        if not allowed:
            return False, retry
        return _counter.check_and_record(f"{key}:hour", 3600, DEFAULT_CONFIG.requests_per_hour)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that applies rate limiting."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting when disabled or for health checks and root
        from ..core.config import settings
        if not settings.rate_limit_enabled or request.url.path in ("/", "/health", "/docs", "/openapi.json"):
            return await call_next(request)

        # Try to get user ID from session cookie (if available)
        user_id = None
        try:
            from ..core.auth import get_session
            from ..core.database import SessionLocal
            db = SessionLocal()
            try:
                session = get_session(request, db)
                if session:
                    user_id = session.user_id
            finally:
                db.close()
        except Exception:
            pass

        allowed, retry_after = check_rate_limit(request, user_id)

        if not allowed:
            logger.warning("Rate limit exceeded for %s on %s", _get_client_key(request, user_id), request.url.path)
            return Response(
                content='{"detail":"Too many requests. Please try again shortly."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(DEFAULT_CONFIG.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        return response
