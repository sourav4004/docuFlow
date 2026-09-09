"""Worker handler registry — shared dispatch table for worker entry points.

Handlers are resolved lazily to avoid import cycles; every handler receives a
``db`` Session plus the parsed job payload and is expected to persist its own
state (worker layers own retry/dead-letter bookkeeping).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

HANDLERS: dict[str, object] = {}


def get_handler(job_type: str):
    """Return the callable for ``job_type`` (registering built-ins lazily)."""
    if not HANDLERS:
        _register_builtins()
    return HANDLERS.get(job_type)


def _register_builtins() -> None:
    from ..services import event_bus as eb
    HANDLERS["EVENT_PROCESS"] = lambda db, payload: eb.process_pending_events(
        db, max_events=int(payload.get("max_events", 50))
    )

    try:
        from ..services.ingestion2 import advance_ingestion_run
        HANDLERS["INGESTION_ADVANCE"] = (
            lambda db, payload: advance_ingestion_run(
                db, int(payload["run_id"])))
    except Exception:  # noqa: BLE001
        logger.exception("ingestion handler unavailable")

    try:
        from ..services.federation import run_connector_sync
        HANDLERS["CONNECTOR_SYNC"] = (
            lambda db, payload: run_connector_sync(
                db, int(payload["source_id"])))
    except Exception:  # noqa: BLE001
        logger.exception("connector handler unavailable")

    try:
        from ..services.vector_backfill import run_backfill_batch
        HANDLERS["VECTOR_BACKFILL_BATCH"] = (
            lambda db, payload: run_backfill_batch(
                db, int(payload["run_id"])))
    except Exception:  # noqa: BLE001
        logger.exception("vector backfill handler unavailable")
