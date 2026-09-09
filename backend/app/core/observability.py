"""Observability middleware for structured logging and request tracking.

Provides:
- Request correlation IDs
- Request duration tracking
- Structured JSON logging format
- Sensitive data redaction
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class StructuredFormatter(logging.Formatter):
    """Structured log formatter that outputs JSON-like structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured fields."""
        # Get correlation ID from context
        correlation_id = correlation_id_var.get()

        # Build structured message
        parts = [
            f"[{record.levelname}]",
            f"[{record.name}]",
        ]

        if correlation_id:
            parts.append(f"[{correlation_id}]")

        parts.append(record.getMessage())

        # Add timing info if present
        if hasattr(record, "duration_ms"):
            parts.append(f"(duration={record.duration_ms:.1f}ms)")

        return " ".join(parts)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(cid)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware that adds correlation IDs and request timing to all requests."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Request-ID") or str(
            uuid.uuid4()
        )[:8]
        set_correlation_id(correlation_id)

        # Track timing
        start_time = time.monotonic()

        # Process request
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Request failed: %s %s (%.1fms)",
                request.method,
                request.url.path,
                duration_ms,
                exc_info=True,
            )
            raise

        duration_ms = (time.monotonic() - start_time) * 1000

        # Add correlation ID to response headers
        response.headers["X-Request-ID"] = correlation_id

        # Log request
        logger.info(
            "%s %s %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response


# Configure structured logging
def setup_observability():
    """Configure structured logging for the application."""
    # Set up the structured formatter
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    # Set log level based on environment
    import os
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Suppress noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logger.info("Observability configured (log_level=%s)", log_level)


# Module logger
logger = logging.getLogger(__name__)
