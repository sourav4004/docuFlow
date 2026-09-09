"""Event processor process entry point.

Run with::

    python -m app.event_processor [--tick-seconds 2] [--batch-size 50]

Drains the transactional knowledge outbox in bounded batches. Fully
independent of the API and worker processes.
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DocuFlow event processor process")
    parser.add_argument("--tick-seconds", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    from app.core.database import SessionLocal
    from app.services.distributed import WorkerIdentity, get_shutdown_manager
    from app.services import runtime

    identity = WorkerIdentity(queue_names=["EVENTS"])
    shutdown = get_shutdown_manager(identity)
    runtime.run_event_processor_process(
        SessionLocal, identity, shutdown,
        tick_seconds=args.tick_seconds,
        batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())