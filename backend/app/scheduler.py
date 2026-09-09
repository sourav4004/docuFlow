"""Scheduler process entry point.

Run with::

    python -m app.scheduler [--tick-seconds 10] [--maintenance-every-seconds 300]

Responsible for delayed-job promotion, recurring maintenance (lease recovery,
approval/merge expiry, memory expiry, retention cleanup enqueue) — never for
per-job business logic.
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuFlow scheduler process")
    parser.add_argument("--tick-seconds", type=float, default=10.0)
    parser.add_argument("--maintenance-every-seconds", type=float,
                        default=300.0)
    args = parser.parse_args()

    from app.core.database import SessionLocal
    from app.services.distributed import WorkerIdentity, get_shutdown_manager
    from app.services import runtime

    identity = WorkerIdentity(queue_names=["SCHEDULER"])
    shutdown = get_shutdown_manager(identity)
    runtime.run_scheduler_process(
        SessionLocal, identity, shutdown,
        tick_seconds=args.tick_seconds,
        maintenance_every_seconds=args.maintenance_every_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())