# Phase 17 Operations Guide — Distributed AI + Enterprise Knowledge Cloud

This guide covers the Phase 17 production surfaces: the distributed worker
platform, broker selection, ingestion, vector backfill, connector sync,
workflow orchestration, and the enterprise governance controls.

## 1. Distributed Worker Platform

### Broker selection

`WORKER_BROKER` selects the queue backend:

- `WORKER_BROKER=postgres` (default) — durable broker backed by the `worker_jobs`
  table; transactional with the application database. No extra service.
- `WORKER_BROKER=redis` — Redis-compatible adapter used only when the `redis`
  package is installed and a `REDIS_URL` is reachable. Jobs are held in a
  Redis zset with claim/dead-letter semantics. Tests never require Redis.

Broker consumers are application services talking to the `BrokerBackend`
interface (`app/services/broker.py`) — no service is coupled to one storage
engine.

### Worker lifecycle

Every worker registers a heartbeat row (`worker_heartbeats`) with a unique
`worker_id`, safe hostname/pid/version, queue assignments, and timestamps.
Claims are atomic (guarded UPDATE on `worker_jobs.status`) and every claimed
job gets an explicit lease row (`job_leases`) owned by the claiming worker.

Lifecycle states: `QUEUED → CLAIMED → RUNNING → COMPLETED`, with
`RETRYING → DEAD_LETTERED` on bounded retry exhaustion and `CANCELLED` on
operator/agent cancellation. Heartbeats extend the lease; a worker that stops
heartbeating past the grace window has its leases expired and its jobs
re-queued by `recover_expired_leases` — handlers must remain idempotent
because a recovered job may be executed again by a healthy worker.

### Running a worker

```bash
# from backend/
python -m app.services.distributed --queue default   # drains one queue
```

Production shutdown is SIGTERM-aware (`ShutdownManager`): stop claiming,
finish the in-flight job, release resources, terminate the heartbeat. A
recovered job is never automatically replayed for dangerous side effects;
dead-letter administration is a manual, audited action through the API.

### Fair scheduling 2.0

`claim_weighted` ranks candidates by priority weight, per-tenant active-job
count (round robin), and job age (anti-starvation). The candidate window is
partitioned per workspace and capped by `per_workspace_cap`, so a single noisy
tenant can never monopolize the claim window — verified by the noisy-neighbor
and queue-saturation tests.

### Autoscaling signals

`GET /autoscale-signals` exposes queue depth, oldest job age, running jobs,
throughput, failure rate, and worker count for an external controller. The
platform does not provision infrastructure automatically.

## 2. Durable Ingestion

The ingestion pipeline persists per-stage rows with stage idempotency keys:

```
UPLOAD → VALIDATE → EXTRACT → OCR → LAYOUT → CHUNK → EMBED → INDEX
       → INTELLIGENCE → READY
```

- A failed stage never repeats completed stages (`retry_ingestion` resets only
  the failed stage; completed stages keep their `attempts` unchanged).
- `advance_ingestion_run` is the worker entry point and is idempotent.
- Runs expose progress (stage, percentage, retry count, errors) through
  `ingestion_progress` and bounded, offset pagination via `list_runs`.

API: `POST /ingestion`, `POST /ingestion/{run_id}/advance`,
`POST /ingestion/{run_id}/retry`, `GET /ingestion`.

## 3. Vector Backfill + Rebuild

`app/services/vector_backfill.py` embeds chunks missing vectors in idempotent
batches:

- `preview_backfill` reports work without writing (dry-run default).
- `start_backfill` executes a bounded batch; dimension mismatches are recorded
  per chunk and never silently mixed.
- `rebuild_vectors` regenerates embeddings per document atomically after a
  model/dimension change.

The JSON cosine fallback stays active when pgvector is unavailable; every
backfill run reports `native_pgvector` truthfully.

API: `GET /vector-backfill/preview`, `POST /vector-backfill`.

## 4. Knowledge Federation

Connectors (`app/services/federation.py`) sync external items through a driver
seam. Syncs are incremental and content-hash deduplicated — repeated syncs
never duplicate data; deleted items are tombstoned, never hard-deleted while
enabled. Sources declare scopes, permissions, allowed domains, retention, and
a credential **reference** — secrets are never stored in these tables.

API: `POST /connectors`, `GET /connectors`,
`POST /connectors/{source_id}/sync`, `GET /connectors/{source_id}/syncs`.

## 5. Workflow Orchestration

`app/services/workflow3.py` persists runs with an immutable definition
snapshot. Nodes execute in dependency order; runs support pause/resume,
workflow/node timeouts (`check_timeouts`), bounded per-node retry policies,
and compensation metadata that distinguishes reversible/irreversible side
effects. Runs in a paused `control_state` are never idle-timed out.

API: `POST /workflow-runs`, `POST /workflow-runs/{id}/advance|pause|resume`,
`GET /workflow-runs`.

## 6. Enterprise Governance

- Policy hierarchy: organization → workspace → feature → execution; the most
  restrictive applicable rule wins (model/tool allowlists intersect, budgets
  take the minimum, sensitivity takes the maximum restriction).
- Retention assignments are workspace- or org-scoped; workspace overrides org.
- Legal holds protect entity rows from the bounded cleanup worker — entities
  under an active hold are never auto-deleted, and holds must be explicitly
  released by an operator.
- Cleanup runs are bounded (≤200 rows) and audited.

API: `POST /governance/rules`, `GET /governance/check-model`,
`POST /governance/retention`, `POST /governance/cleanup`,
`POST /legal-holds`, `POST /legal-holds/{id}/release`.

## 7. AI Execution + Agent Durability

AI executions run through the durable queue with atomic claims, leases,
heartbeats, and checkpoints (Phase 15/16). Phase 17 agents additionally
validate plan DAGs (cycles rejected), enforce per-execution budgets (tokens,
cost, time, tool calls) before every step, and persist every step as a
checkpoint so a crash/restart can resume from the last completed step.
Human handoffs pause an execution in `WAITING_APPROVAL`; cancellation is
durable and closes open handoffs.

## 8. Observability

- `worker_heartbeats` + `job_leases` give worker/lease visibility.
- `autoscale_signals` and queue metrics expose depth/throughput/failure.
- Alert rules (`app/services/alerts.py`) evaluate named metrics with cooldowns
  and persist tenant-scoped events: queue backlog, provider outage, error
  rate, budget exhaustion, worker failure.
- Cost anomalies/forecasts are labeled estimates, never guarantees.

## 9. Environment Variables

| Variable | Purpose |
|----------|---------|
| `WORKER_BROKER` | `postgres` (default) or `redis` |
| `REDIS_URL` | Redis connection when `WORKER_BROKER=redis` |
| `OPENAI_API_KEY`/`*_BASE_URL`/model vars | Optional real provider config |

## 10. Recovery Runbook

1. **Worker outage** — workers stop heartbeating; after lease expiry + grace,
   `POST /workers/recover-stale` re-queues their jobs; healthy workers pick
   them up (handlers are idempotent).
2. **Queue backlog** — inspect `/ops/queue-metrics` + `/autoscale-signals`,
   scale workers, watch dead letters at `/dead-letters`.
3. **Dead-lettered job** — inspect, then requeue (reset attempts) or abandon
   through `POST /dead-letters/{id}/requeue|abandon`.
4. **Provider outage** — circuit breaker trips; fallback provider is selected
   by capability/sensitivity policy; check provider health under `/ops`.
5. **Database recovery** — restore from backup, run
   `alembic upgrade head` (single head `024_phase17_cloud`), verify
   authorization after restore (tenant isolation is row-scoped).
6. **pgvector absent** — the JSON cosine fallback is active; vector backfill
   reports `native_pgvector: false` and continues without the extension.
