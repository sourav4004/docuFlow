# Phase 24 Operations — Production Stabilization & Release Baseline

Phase 24 is the final major engineering phase. It converts the feature-
complete platform (Phases 0–23) into stable, maintainable, deployment-ready
software. No new product features were added. Everything here is audit,
hardening, and honest-baseline work.

## What Phase 24 changed

### Test infrastructure
- **`tests/conftest.py` (new)** — global pytest config. Auto-marks the five
  legacy pgvector suites with `env_pgvector` and converts ONLY the exact
  `type "vector" does not exist` error into an explicit, documented skip.
  Any other failure still fails loudly. When pgvector is installed the
  marker is inert and the suites run natively.
- **`backend/run_tests.sh` / `backend/run_tests.bat`** — reusable validation
  runners. The exit code is always pytest's (PIPESTATUS / direct redirect),
  so filtered output can never mask a failure. Verified: pass→0,
  fail→nonzero, no-tests→5, collection error→4.
- **Legacy test cleanup** — 36 pre-existing failures eliminated:
  - 28 pgvector-dependent → environment-gated (above)
  - 6 stale `UPLOADED` status assertions → `QUEUED` (durable pipeline is
    intentional; documents are queued immediately)
  - 2 shared-identity pollution (`usera@test.com` on the real DB) → unique
    per-run identities; hardcoded `user_id: 2` replaced with the actual
    registered id
  - 1 legacy unconditional skip (`test_phase9.test_migration_head`) → real
    `alembic current` validation against the application PostgreSQL

### Backend hardening
- **Unified error model** (`app/core/errors.py`, wired in `app/main.py`):
  every error response now carries `{"detail", "error": {code, message,
  request_id?}}`. Legacy `detail` consumers keep working; new consumers get
  machine-readable codes. Unhandled exceptions return a safe uniform 500 —
  stack traces never reach the client — and log server-side with the
  correlation ID. Malformed JSON → structured 422.
- **Session idle timeout** (`app/core/auth.py`, `config.session_idle_timeout`,
  migration `031_session_idle_timeout`): sessions now expire after
  inactivity in addition to absolute expiry. Default 0 (disabled) for
  compatibility; production should set e.g. 86400.
- **Readiness broker honesty** (`app/api/health.py`): `/health/ready` now
  reports broker state — `postgres (redis not configured)`, `degraded: redis
  unreachable, postgres active`, or `redis available`.

### Frontend hardening
- **`components/ui/primitives.tsx` (new)** — shared design system:
  StatusBadge (shape + text, never color-only), Loading (aria-live),
  Skeleton/SkeletonCard, ErrorState (role=alert + retry), EmptyState,
  Card, Button. Accessible by default (focus-visible rings).
- **`lib/api.ts`** — unified `ApiError` class parsing the backend structured
  envelope (`code`, `request_id`), with `isAuthError / isForbidden /
  isNotFound / isRateLimited / isServerError` helpers. Backward compatible
  with legacy error shapes.
- **`lib/format.ts` (new)** — single source for bytes/dates/relative-time
  formatting and the canonical document status vocabulary (removes six
  duplicated helper implementations).
- **Accessibility fixes** — DocumentList loading/error states now announce
  via `role="status"` / `role="alert"` + `aria-live`.
- **`next.config.ts`** — `output: "standalone"` to match the production
  Dockerfile (the standalone COPY was previously relying on an implicit
  default; builds now provably emit it).

## Test exit-code policy (mandatory)

All authoritative test results must come from pytest's own exit status:

```bash
cd backend
./run_tests.sh tests/test_auth.py        # specific files
./run_tests.sh                           # full suite
```

Never report success from `pytest ... | tail` style pipelines — the shell
reports the LAST command's status, not pytest's. The runners handle this;
CI must invoke the runners or raw `python -m pytest`.

## Deployment checklist (see also FINAL_ACTIVATION_CHECKLIST.md)

1. Environment: `SECRET_KEY` (required in production — startup refuses
   otherwise), `DATABASE_URL`, `CORS_ORIGINS`, `COOKIE_SECURE=true`,
   `SESSION_IDLE_TIMEOUT` (recommended), `RATE_LIMIT_ENABLED=true`.
2. `alembic upgrade head` (single head, currently `031`).
3. Start API (uvicorn, graceful shutdown verified), workers, scheduler,
   event processor — each has its own Docker target/compose service.
4. Health verification: `/health/live` (liveness), `/health/ready`
   (DB + config + broker), `/worker-health` (fleet).
5. Smoke: register → login → upload → QUEUED → process → READY → search →
   RAG answer with citations.
6. Rollback: `alembic downgrade` one revision is safe for 031 (column drop);
   container redeploy of the previous image tag; PostgreSQL broker remains
   authoritative if Redis is removed.

## Backup / DR readiness

- PostgreSQL: use `pg_dump`/WAL archiving per environment; document RPO/RTO
  assumptions (see FINAL_ACTIVATION_CHECKLIST item 5 for measurement).
- Object storage: bucket versioning + lifecycle; restore verification is
  part of the Phase 23 storage migration verification flow.
- Config/secrets: environment references only; `.env` files are never
  committed and must be backed up through the secret manager.
- Restore: provision empty DB → restore dump → `alembic current` must match
  pre-incident head → verify tenant isolation probes → replay outbox if the
  event processor was mid-flight.

## Honest capability labels (current environment)

| Component | Label |
|---|---|
| PostgreSQL 18.4 | REAL (validated) |
| JSON vector fallback | REAL (validated) |
| pgvector native | UNAVAILABLE (gated suites self-enable) |
| PostgreSQL broker | REAL (validated) |
| Redis broker | UNAVAILABLE (adapter validated via failure path) |
| AI providers (real) | UNAVAILABLE (fake provider validated; gates ready) |
| S3 | UNAVAILABLE (local storage REAL; S3 path implemented) |
| Multi-region | SIMULATED (logic validated; no physical regions) |
| Browser E2E | UNAVAILABLE (no browser automation in env) |
| Soak / long-load | BOUNDED LOCAL VALIDATION only |
