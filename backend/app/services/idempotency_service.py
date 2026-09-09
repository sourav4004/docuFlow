"""Idempotency service.

Prevents duplicate expensive operations when clients retry. A stored
response is returned for repeated requests with the same key. If the same
key is reused with a different request payload, the request is rejected
(conflict) rather than silently producing a second result.
"""

import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from ..models.idempotency import IdempotencyKey


class IdempotencyConflictError(Exception):
    """Same idempotency key used with a different request body."""


def _request_hash(body: dict) -> str:
    return hashlib.sha256(
        json.dumps(body or {}, sort_keys=True, default=str).encode()
    ).hexdigest()


def get_or_record(
    db: Session,
    workspace_id: int,
    user_id: Optional[int],
    key: str,
    request_body: dict,
    ttl_seconds: int = 86400,
) -> Tuple[bool, Optional[IdempotencyKey]]:
    """Attempt to register an idempotent operation.

    Returns (is_new, record):
      - is_new=True: caller should execute the operation, then call
        complete() to store the response.
      - is_new=False: a record exists — callers must return the stored
        response (or raise conflict if request hash differs).

    Raises:
        IdempotencyConflictError: same key used with a different body.
    """
    now = datetime.now(timezone.utc)
    record = (
        db.query(IdempotencyKey)
        .filter(
            IdempotencyKey.workspace_id == workspace_id,
            IdempotencyKey.key == key,
        )
        .first()
    )
    if record is None:
        record = IdempotencyKey(
            workspace_id=workspace_id,
            user_id=user_id,
            key=key,
            request_hash=_request_hash(request_body),
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        db.add(record)
        db.flush()
        return True, record

    # Expired records are treated as new
    if record.expires_at is not None:
        expires = record.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if now > expires:
            record.request_hash = _request_hash(request_body)
            record.response_status = None
            record.response_json = None
            record.expires_at = now + timedelta(seconds=ttl_seconds)
            db.flush()
            return True, record

    if record.request_hash != _request_hash(request_body):
        raise IdempotencyConflictError(
            "Idempotency key reused with a different request body"
        )
    return False, record


def complete(
    db: Session,
    record: IdempotencyKey,
    response_status: int,
    response: dict,
) -> None:
    """Store the response for a completed idempotent operation."""
    record.response_status = response_status
    record.response_json = json.dumps(response, default=str)
    db.flush()


def get_stored_response(record: IdempotencyKey) -> Tuple[int, dict]:
    """Retrieve a stored idempotent response."""
    try:
        return record.response_status or 200, json.loads(record.response_json or "{}")
    except (ValueError, TypeError):
        return record.response_status or 200, {}