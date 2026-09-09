#!/usr/bin/env python
"""Legacy workspace backfill CLI.

Assigns documents/collections with a NULL workspace to the single unambiguous
workspace of their owner. Ambiguous owners are reported and NEVER guessed.

Examples:
    # Preview (default: dry run — nothing is written):
    python -m scripts.backfill_workspaces --dry-run --batch-size 100

    # Execute in bounded batches:
    python -m scripts.backfill_workspaces --kind document_workspace \
        --batch-size 100

    # Resume a partial run from the last processed record id:
    python -m scripts.backfill_workspaces --resume 5000

Exit codes: 0 success, 1 error, 2 invalid args.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill legacy workspace assignments (safe, audited)")
    parser.add_argument("--kind", choices=["document_workspace",
                                           "collection_workspace"],
                        default="document_workspace")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only — no writes (default when no flag)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--resume", type=int, default=None,
                        help="Resume from record id (inclusive)")
    parser.add_argument("--workspace", type=int, default=None,
                        help="Restrict scan to this workspace's org (unused "
                             "for legacy scan; kept for operator familiarity)")
    parser.add_argument("--user", type=int, default=None,
                        help="Restrict scan to one owner user id")
    parser.add_argument("--run-to-completion", action="store_true",
                        help="Continue batches until the run finishes")
    args = parser.parse_args()

    if args.batch_size < 1 or args.batch_size > 500:
        print("error: batch-size must be 1..500", file=sys.stderr)
        return 2

    # The CLI defaults to a DRY RUN unless the operator explicitly disables it.
    dry_run = args.dry_run or not args.run_to_completion
    if dry_run and not args.dry_run:
        print("note: defaulting to --dry-run (no writes). Pass "
              "--run-to-completion to execute.")

    try:
        from app.core.database import SessionLocal
        from app.models.phase16 import BackfillRun
        from app.models.user import User
        from app.services.backfill_service import (
            create_run, execute_batch, run_complete,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot load app: {exc}", file=sys.stderr)
        return 1

    db = SessionLocal()
    try:
        operator = db.query(User).order_by(User.id.asc()).first()
        if operator is None:
            print("error: no operator user exists", file=sys.stderr)
            return 1
        run = create_run(
            db, created_by=operator.id, kind=args.kind, dry_run=dry_run,
            user_id=args.user, batch_size=args.batch_size,
            resume_from=args.resume,
        )
        db.commit()
        print(f"run #{run.id} kind={run.kind} dry_run={run.dry_run} "
              f"batch_size={run.batch_size}")
        stats = {"processed": 0, "assigned": 0, "skipped": 0,
                 "ambiguous": 0, "failed": 0}
        while True:
            batch = execute_batch(db, run)
            for key in stats:
                stats[key] += batch[key]
            db.commit()
            print(f"  batch: processed={batch['processed']} "
                  f"assigned={batch['assigned']} "
                  f"ambiguous={batch['ambiguous']} "
                  f"skipped={batch['skipped']} run_status={run.status}")
            if batch["batch_finished"]:
                break
            if args.batch_size and stats["processed"] >= 100000:
                print("warning: safety cap reached; run paused at "
                      f"cursor {run.cursor_id}", file=sys.stderr)
                break
        print(f"done: {stats}")
        return 0
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
