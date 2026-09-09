# Phase 18 — Operations Guide

DocuFlow Phase 18 turns the platform into an operationally scalable enterprise AI
cloud: separate runtime processes, a hardened broker abstraction, a production
pgvector path, durable ingestion at scale, knowledge federation, governed RAG,
resumable agents, durable workflows, cost controls, and enterprise observability.

This guide covers the operational surface introduced by Phase 18. It complements
the Phase 17 operations guide.

---

## 1. Runtime Processes

Phase 18 separates the application into four independently startable processes:

| Process | Entry point | Responsibility |
|---------|-------------|----------------|
| API | `uvicorn app.main:app` | HTTP API, auth, tenant isolation |
| Worker | `python -m app.worker` | Claim + process durable jobs under weighted fairness |
| Scheduler | `python -m app.scheduler` | Delayed-job promotion, maintenance rounds, retention enqueue, connector syncs |
| Event processor | `python -m app.event_processor` | Drain the transactional outbox in bounded batches |

Each worker registers a persistent identity (unique id, hostname, pid, version,
startup time), heartbeats with capacity/load signals, and deregisters on graceful
shutdown. Stale workers (heartbeat expiry) are detected and their leases
recovered by any live worker or the scheduler — bounded by a grace window so no
non-idempotent side effect is replayed twice.

### Graceful shutdown

- `SIGTERM`/`SIGINT` stops claiming new work.
- In-flight jobs are completed or released.
- Worker heartbeat is marked inactive; leases are released for safe re-claim.
- All DB connections are closed before exit.

### Capacity

Workers advertise concurrency (`WORKER_CONCURRENCY`, default 4). Fair scheduling
(`claim_weighted`) partitions the candidate window per tenant so one noisy tenant
cannot starve others; priority, per-tenant caps, and global caps are all
enforced server-side.

---

## 2. Broker

The broker abstraction (`app/services/broker.py`) exposes a uniform interface:
enqueue, claim, acknowledge, retry, dead-letter, visibility timeout, heartbeat,
cancellation, and delayed jobs.

| Backend | Selection | Notes |
|---------|-----------|-------|
| PostgreSQL | `WORKER_BROKER=postgres` (default) | Durable `worker_jobs` table; transactional with the application DB |
| Redis-compatible | `WORKER_BROKER=redis` | Optional; requires the `redis` package and a reachable server. Lazy-imported so tests never need Redis |

Broker health (`GET /broker/health`) reports backend, connection status, latency,
and queue depth. If the broker becomes unavailable, jobs remain durable in the
queue (PostgreSQL backend) or in Redis; nothing is silently dropped. On
reconnect, queued jobs are claimable again.

---

## 3. Vector Platform

Phase 18 keeps the JSON cosine fallback and adds the production pgvector path.

- **Detection** (`detect_pgvector`): extension availability, version, and index
  support are probed at runtime.
- **Vector model registry** (`vector_models`): provider, model, dimensions,
  version, and creation time are tracked. `GET /vector/models` lists them;
  embedding versioning prevents mixing incompatible dimensions.
- **Dimension enforcement**: embeddings whose dimensions do not match the active
  model are rejected.
- **Index operations** (`POST /vector/index-ops`): safe create/rebuild/validate
  with configurable parameters (`VECTOR_INDEX_PARAMS`).
- **Health** (`GET /vector/health`): backend, active model, chunk count, vector
  coverage percentage.
- **Backfill** (`app/services/vector_backfill.py`): idempotent, batch, dry-run
  capable, resumable, with failure reporting. Never runs automatically.

### Current environment

pgvector is **not installed** in the development environment; the JSON cosine
fallback is active and fully functional. All pgvector-specific behavior is
implemented behind extension detection and the environment limitation is
reported honestly in the Phase 18 report.

---

## 4. Providers

- **Configuration** is environment-based (`OPENAI_API_KEY`, `OPENAI_BASE_URL`,
  model, embedding model/dimensions, timeouts, retries). Development runs
  without credentials using the deterministic fake provider; production
  requires explicit configuration.
- **Timeout policy**: connect/read/total timeouts are configurable.
- **Retry policy**: bounded exponential backoff; 429 responses honor
  `Retry-After` when present and safe.
- **Circuit breaker**: closed → open → half-open → closed, with consecutive
  failure thresholds and half-open probe recovery.
- **Fallback**: primary → retry → circuit breaker → secondary → final failure,
  respecting capability, sensitivity, budget, and organization policy.
- **Usage accounting**: input/output/embedding tokens and estimated cost are
  recorded per execution (provider calls surfaced on `/ops`).

---

## 5. Ingestion at Scale

- Ingestion is queue-driven (`DOCUMENT_PROCESSING`); every stage has its own
  idempotency key, owner lease, and retry state.
- Independent documents process concurrently; batches isolate failures (one
  poison document does not corrupt the batch).
- `GET /ingestion/batch-progress` reports total/completed/failed/retrying and
  percentage.
- **Poison documents**: documents failing repeatedly are moved to a
  quarantined/dead-letter state (`GET /ingestion/poison`, admin resolution via
  `POST /ingestion/poison/{id}/resolve`). They are never retried
  automatically.
- Per-stage retry policies and resource limits (memory, CPU, archive size)
  prevent runaway processing.

---

## 6. Knowledge Federation

- Connectors run on worker queues (`connector sync` jobs) with credential
  references only — never plaintext secrets.
- Every connector carries organization/workspace/resource scope, enforced
  server-side.
- Sync is incremental (cursor, content hash, modified time, tombstones),
  idempotent (repeated sync never duplicates), and resumable after
  interruption. Conflicts never silently overwrite local data.
- `GET /connectors/health` reports last sync, lag, throughput, and errors;
  per-connector rate limits respect external service limits.

---

## 7. RAG 6.0

- A retrieval planner selects the strategy (keyword/vector/metadata/graph/
  memory) from query intent.
- Evidence is ranked by relevance, authority, freshness, completeness, and
  contradiction; sufficiency determines whether an answer is grounded.
- Claim matrices track claim → evidence → confidence → contradiction; citation
  coverage is calculated; citations are validated to actually support claims.
- Conflict-aware answers surface meaningful conflicts; temporal answers support
  current/historical/as-of/between-date queries.
- When evidence is insufficient the system **refuses** rather than
  hallucinating, and bounded answer repair only uses retrieved evidence.

---

## 8. Agents and Workflows

- Agent plans persist as DAGs; every step is authorized independently and
  budgets (cost/tokens/time/tool calls) are verified before each step.
- Checkpoints are persisted before and after side effects; executions resume
  after worker restart, provider outage, or DB reconnect.
- Human handoff uses a durable `WAITING_APPROVAL` state; cancellation is
  durable and race-safe; failed executions can enter a dead-letter state.
- Workflow DAGs reject cycles, execute dependencies in order, run safe parallel
  branches, and persist every node execution. Node/workflow/idle timeouts,
  per-node retry policies, compensation classification, pause/resume, and
  replay of safe steps are supported.

---

## 9. Governance

- Policy hierarchy: organization → workspace → feature → execution; the most
  restrictive applicable policy wins.
- Model/provider/tool allowlists restrict what organizations may use.
- Sensitivity routing (`PUBLIC/INTERNAL/CONFIDENTIAL/RESTRICTED`) sends data
  only to approved providers; data minimization limits context.
- Retention applies to traces, prompts, artifacts, memories, executions, and
  documents; legal holds are non-destructive and block automatic deletion.
- Cleanup jobs are bounded, durable (`MAINTENANCE` queue, idempotent hourly
  dedupe key), and auditable.

---

## 10. Cost Controls

- Cost ledger attribution by organization, workspace, user, feature, model,
  provider, and execution (`GET /cost/attribution`).
- Budget policies: soft (approval threshold), hard (block), and downgrade.
  `POST /cost/budget-check` enforces them before execution.
- Forecasting (`GET /cost/projection`) is explicitly labeled an estimate with
  assumptions; anomaly detection flags sudden spikes.

---

## 11. Observability

- Correlation IDs propagate across request/execution/workflow/job/provider/
  tool boundaries.
- SLO snapshots record availability, latency percentiles, error rate, and
  queue age; `GET /observability/slo` reports compliance vs targets.
- Alerts are rule-driven and configurable (queue backlog, provider outage,
  high error rate, budget exhaustion, worker failure, DB latency).
- `/ops` (operations console) surfaces worker fleet, broker, queue, providers,
  vector platform, ingestion, connectors, SLOs, cost, search analytics, and
  disaster-recovery status. Sensitive payloads are redacted.

---

## 12. Disaster Recovery

- `GET /dr/backup-inventory`: declarative inventory of database backup method/
  schedule/retention, object storage, configuration, and migration state.
  Secrets are referenced, never stored.
- `GET /dr/validate-restore`: non-destructive restore validation — migration
  head sync and authorization integrity (orphan check).
- `GET /dr/tenant-isolation`: verifies tenant boundaries survive restore.

Runbooks: on database failure restore from the validated backup and verify the
migration head; on broker failure the PostgreSQL queue remains durable; on
worker failure stale leases are recovered by other workers; on provider failure
circuit breakers route to fallbacks with bounded retries.

---

## 13. Environment Variables (Phase 18 additions)

| Variable | Purpose | Default |
|----------|---------|---------|
| `WORKER_BROKER` | `postgres` or `redis` | `postgres` |
| `WORKER_CONCURRENCY` | Per-worker job concurrency | `4` |
| `WORKER_QUEUES` | Comma-separated queue assignments | all standard queues |
| `REDIS_URL` | Redis connection string (when broker=redis) | unset |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | Production provider config | unset (fake provider in dev) |
| `BACKUP_METHOD` / `BACKUP_SCHEDULE` / `BACKUP_RETENTION_DAYS` | Backup inventory metadata | `pg_dump` / `daily` / `14` |
| `OBJECT_STORAGE_BUCKET` | Object-storage flag for inventory | unset |
| `SECRET_MANAGER` | Secrets are env/secret-manager based | unset |

No production secrets are ever read from source; development defaults are safe
and production requires explicit configuration.