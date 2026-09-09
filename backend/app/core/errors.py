"""Unified error model (Phase 24, Steps 31/38/49).

Every error response in DocuFlow now has the same JSON shape:

    {"error": {"code": "<machine-readable>", "message": "<safe>", "request_id": "..."}}

Rules enforced here:
- Internal details (stack traces, SQL, exception class names) NEVER reach
  the client — they are logged server-side with the correlation ID.
- Malformed/invalid payloads produce structured 4xx, never a raw 500.
- Domain errors raised by services carry a machine-readable code so the
  frontend and API consumers can branch on ``error.code``.
- The correlation ID matches the ``X-Request-ID`` response header so an
  operator can join a user-visible error to the server log in one step.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.observability import get_correlation_id

logger = logging.getLogger(__name__)

# Exception classes that are treated as *expected* domain conditions.
# Raising one produces the mapped status with a safe client message.
DOMAIN_ERROR_STATUS = {
    "validation": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "not_found": status.HTTP_404_NOT_FOUND,
    "conflict": status.HTTP_409_CONFLICT,
    "unauthorized": status.HTTP_401_UNAUTHORIZED,
    "forbidden": status.HTTP_403_FORBIDDEN,
    "rate_limited": status.HTTP_429_TOO_MANY_REQUESTS,
    "unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
}


class DomainError(Exception):
    """A domain-level error with a machine-readable code and safe message."""

    def __init__(self, code: str, message: str, *, status_code: int | None = None):
        if status_code is None:
            status_code = DOMAIN_ERROR_STATUS.get(code, status.HTTP_400_BAD_REQUEST)
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _error_payload(code: str, message: str) -> dict:
    request_id = get_correlation_id()
    error = {
        "code": code,
        "message": message,
        **({"request_id": request_id} if request_id else {}),
    }
    # Backward-compatible envelope: the legacy ``detail`` key is preserved so
    # every existing consumer keeps working, while new consumers read the
    # structured ``error`` object.
    return {"detail": message, "error": error}


def install_error_handlers(app: FastAPI) -> None:
    """Install the unified handlers. Idempotent."""

    @app.exception_handler(DomainError)
    async def _domain_error_handler(request: Request, exc: DomainError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
        # Keep existing {detail: ...} shape semantics but make it uniform:
        # detail may be a string or an already-structured dict.
        detail = exc.detail
        code = "http_error"
        if isinstance(detail, dict):
            code = str(detail.get("code", code))
            detail = detail.get("message", code)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(code, str(detail)),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        # Structured 422 with a coarse field summary — never echo raw input.
        fields = []
        for err in exc.errors()[:10]:
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            fields.append(loc or "body")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_payload(
                "validation_error",
                "Request validation failed for: " + ", ".join(fields)
                if fields
                else "Request validation failed.",
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        # Log the full detail server-side; return a uniform, safe 500.
        logger.exception(
            "Unhandled exception on %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload("internal_error", "Internal server error."),
        )
