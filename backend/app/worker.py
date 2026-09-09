"""Worker process entry point.

Run with::

    python -m app.worker [--queues AI_TASKS,DOCUMENT_PROCESSING] [--once]

Starts a distributed worker that registers its identity, heartbeats with
capacity/load signals, drains configured queues under weighted fairness,
and shuts down gracefully on SIGTERM/SIGINT.
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

logger = logging.getLogger(__name__)

DEFAULT_QUEUES = ("AI_TASKS", "DOCUMENT_PROCESSING", "AGENTS", "WORKFLOWS",
                  "MAINTENANCE", "EVENTS")


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuFlow worker process")
    parser.add_argument("--queues", default=",".join(DEFAULT_QUEUES),
                        help="comma-separated queue names to drain")
    parser.add_argument("--once", action="store_true",
                        help="claim and process exactly one job, then exit")
    parser.add_argument("--per-round-seconds", type=float, default=20.0)
    parser.add_argument("--tick-seconds", type=float, default=1.0)
    args = parser.parse_args()

    from app.core.database import SessionLocal
    from app.services.distributed import WorkerIdentity, get_shutdown_manager
    from app.services import runtime

    queues = [q.strip().upper() for q in args.queues.split(",") if q.strip()]
    identity = WorkerIdentity(queue_names=queues)

    if args.once:
        # Single-job mode: claim one job from the first queue and exit.
        from app.services.worker_platform import run_once
        from app.services.worker_handlers import get_handler
        db = SessionLocal()
        try:
            for queue in queues:
                job = run_once(db, queue, identity.worker_id,
                               lambda s, j: (get_handler(j.job_type) or
                                             (lambda *a: None))(s, j))
                if job is not None:
                    logger.info("processed job %s (%s)", job.id, job.job_type)
                    return 0
            logger.info("no work available")
            return 0
        finally:
            db.close()

    shutdown = get_shutdown_manager(identity)
    runtime.run_worker_process(
        SessionLocal, identity, shutdown,
        queues=queues,
        per_round_seconds=args.per_round_seconds,
        tick_seconds=args.tick_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())