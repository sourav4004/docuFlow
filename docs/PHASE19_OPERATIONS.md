# Phase 19 — Operations Guide

DocuFlow Phase 19 turns the Phase 18 distributed enterprise AI cloud into a
governed, self-healing, globally addressable enterprise AI platform: an
enterprise control plane with versioned configuration, multi-region readiness,
worker self-healing and quarantine, scheduler leader election, broker delivery
guarantees, capability-driven provider routing, cost reservations and
forecasting, vector lifecycle management, durable ingestion with resource
governance, RAG 7.0 evidence gating, resumable agents and workflows, AI safety
8.0, governance 4.0, SLO enforcement, search quality, enterprise analytics,
data-consistency checking, DR readiness scoring, and the /ops 4.0 console.

This guide covers the operational surface introduced by Phase 19. It
complements the Phase 17 and Phase 18 operations guides.

---

## 1. Control Plane

`app/services/control_plane.py` is the central control-plane abstraction:

- **Tenant/workspace health** — `global_health()` reports API, database, broker,
  worker, provider, and vector status with per-component `healthy`/`degraded`/
  `down` states.
- **Regions** — `upsert_region` / `list_regions` register deployments with a
  status, health score, and failover target. Single-region deployments are the
  default; multi-region is represented without requiring physical regions.
- **Data residency** — `set_residency_rule` stores per-classification allowed /
  prohibited regions and a default region; `evaluate_residency` gates any
  operation that touches a region.
- **Failover** — `initiate_failover` performs an audited, idempotent region
  transition: it refuses unknown regions, refuses failover to the same region,
  and records the transition in the audit trail with actor and reason.

### Configuration snapshots

Control-plane configuration is versioned and immutable:

- `snapshot_config(scope_type, scope_id, config, actor, reason)` writes a new
  immutable version; only one snapshot per scope is active at a time.
- `activate_snapshot` activates a specific version; `rollback_snapshot` reverts
  to an earlier version (target version must exist).
- Every change is audited with actor, reason, timestamp, and the scope it
  applies to. Rollback is idempotent — rolling back to the already-active
  version is a no-op.

Never change production policy silently: always go through a snapshot + activate
(or rollback) so there is a diffable, reversible audit trail.

---

## 2. Workers, Scheduler, Broker

### Worker self-healing (`app/services/worker_ops2.py`)

- **Health scoring** — a deterministic 0–100 score from heartbeat freshness,
  failure history, active-job load, and queue starvation signals.
- **Quarantine** — workers whose repeated-failure threshold is exceeded are
  automatically quarantined with a reason, failure count, and optional
  auto-recovery time. Operators can override quarantine
  (`operator_override`), which is audited.
- **Job migration** — `migrate_job` requeues a job owned by an unhealthy worker
  with an audit record; the lease is released atomically so no duplicate
  execution occurs.
- **Backpressure** — `backpressure_decision` returns
  `allow`/`defer`/`reject` based on global, tenant, provider, and budget
  capacity, plus worker health.
- **Autoscale signals** — `autoscale_signals` exposes queue depth, running and
  retrying counts, dead-lettered jobs, oldest-job age, worker count, and
  failure rate for external autoscalers.

### Scheduler 2.0

- **Leader election** — `acquire_leadership` performs an atomic DB-backed
  claim with a lease expiry and heartbeat; `renew_leadership` extends it.
  Only the leader runs scheduled work, so no duplicate scheduled jobs.
- **Deduplication** — scheduled jobs carry deterministic keys; enqueue is a
  no-op if the same key is already queued/running.
- **Recovery** — a crashed leader's lease expires and a new leader takes over;
  jobs are re-enqueued only if they were never completed (durable state first).

### Broker 2.0 (`app/services/broker2.py`)

- **Delivery guarantees** — documented at-least-once semantics with a
  visibility timeout; handlers must be idempotent across duplicate deliveries.
- **Outage policy** — `outage_policy()` states that when the broker is
  unavailable, jobs are never silently dropped: enqueue surfaces an error and
  the queue remains durable in PostgreSQL; on reconnect, queued jobs are
  recovered (`broker_recovery_check` verifies recoverable state).
- **Redis adapter** — production-strengthened but fully optional: pooled
  connection, timeout, reconnect, health check, and graceful degradation. The
  `redis` package is imported lazily inside functions only; with no Redis
  server or package, the PostgreSQL broker remains the tested default.
- **Connection pool** — `pool_config()` exposes pool size, overflow, timeout,
  recycle, and health-check settings.

---

## 3. AI Provider Platform 2.0

`app/services/provider_ops2.py`:

- **Capability matrix** — a static matrix plus a persisted registry
  (`register_capability`) tracking completion, streaming, embeddings,
  structured output, tool calling, vision, context window, max output,
  embedding dimensions, per-1k-token costs, and latency class.
- **Model routing 3.0** — `route_model` selects a model deterministically from
  candidates using task, required capability, sensitivity, region, budget,
  latency target, quality target, and provider health. Modes: `cheapest`,
  `fastest`, `quality`, `balanced`. Routing never silently violates budget or
  governance policy — constraints fail closed.
- **Admission control** — `admission_check` verifies health, capability,
  policy, budget, rate limit, region, and sensitivity before a provider call.
- **Load shedding** — `load_shedding` returns a structured `defer`/`reject`
  decision with reason when capacity is exhausted; low-priority work is
  deferred before high-priority work.
- **Fallback 2.0** — primary → bounded retry → circuit breaker → secondary →
  degraded deterministic mode → final failure. Capability and policy
  constraints are respected at every hop.
- **Shadow testing** — `shadow_config` provides a safe architecture for
  comparing providers using synthetic/redacted inputs by default; production
  payloads are never copied into shadow runs.

---

## 4. AI Cost Platform 4.0

`app/services/cost4.py`:

- **Estimation** — `estimate_cost` computes a preflight estimate from model
  rates and expected tokens.
- **Reservation** — `reserve_budget` durably reserves budget for an execution
  (fail-closed if the budget would be exceeded); `release_reservation` releases
  unused funds on completion with an audit record.
- **Reconciliation** — `reconcile_reservation` compares estimated vs actual
  usage/cost vs reservation and flags discrepancies.
- **Forecasting** — `forecast2` projects daily/weekly/monthly cost by
  workspace, organization, model, provider, workflow, and agent. All forecasts
  are labeled estimates.
- **Anomaly detection** — `detect_anomalies` flags sudden spikes, unusual
  per-user/workflow/provider cost, token explosions, and repeated-failure cost.
- **Optimization recommendations** — `optimization_recommendations` suggests
  model downgrades, caching, batching, prompt reduction, embedding reuse, and
  workflow changes. Recommendations never change production policy
  automatically.

---

## 5. Vector Platform 2.0

`app/services/vector_ops2.py`:

- **Model lifecycle** — `set_lifecycle` transitions embedding models through
  `active` → `deprecated` → `migration-required` → `retired`, recorded with a
  reason and decider.
- **Compatibility** — `compatibility_check` rejects any embedding whose model,
  dimensions, version, provider, or distance metric differs from the
  collection's registered model. Incompatible embeddings are never mixed.
- **Rebuild planner** — `rebuild_plan` estimates batches, dry-run support,
  pause/resume, retry, progress, and a failure report; embeddings are only
  replaced when the target model is compatible.
- **Coverage monitoring** — `compute_coverage` tracks total chunks, embedded
  chunks, stale and failed embeddings, and model distribution.
- **Drift detection** — `embedding_drift` reports the fraction of chunks on
  deprecated/retired models so operators can schedule rebuilds.
- **Retrieval benchmarks** — `benchmark_retrieval` compares keyword, vector,
  hybrid, metadata, graph, and memory modes with precision, recall, MRR,
  latency, and citation coverage.

pgvector remains optional: native vectors are used only when the extension is
present; otherwise the JSON cosine fallback remains active.

---

## 6. Ingestion Platform 3.0

`app/services/ingestion_ops3.py`:

- **Resource governor** — `resource_governor()` exposes hard limits on file
  size, page count, extraction time, OCR work, embedding work, and total
  execution time. Violations fail with a safe, human-readable reason.
- **Stage priorities** — ingestion stages carry a priority used by queue
  scheduling.
- **Stage DAG validation** — `validate_ingestion_dag` rejects cycles and
  unknown stages before execution.
- **Stage checkpoints** — `checkpoint_stage` persists per-stage state so a
  crash resumes from the last completed stage.
- **Resume** — `resume_ingestion` recovers interrupted documents after worker
  crash, provider outage, process restart, or timeout.
- **Quality score** — `quality_score` produces per-document indicators:
  extraction completeness, OCR quality, metadata completeness, chunk quality,
  and embedding coverage.
- **Failure explanations** — `failure_reason` maps failures to safe,
  human-readable reasons without stack traces or secrets.

---

## 7. Federation / Knowledge Graph / Memory

### Federation 2.0 (`app/services/federation_ops2.py`)

- **Connector contract** — discover, fetch, incremental sync, tombstones,
  health, rate limits, retry, cursor, and checkpoint.
- **Checkpoints** — `save_checkpoint` persists sync cursors and content
  hashes; `resume_from_checkpoint` resumes interrupted syncs.
- **Exactly-once effects** — where exactly-once is impossible, effects are
  idempotent (content-hash keyed upserts), so duplicate deliveries never
  duplicate work.
- **Conflict engine** — `list_conflicts` / `mark_resolved` track
  externally-modified, deleted/recreated, metadata, and content conflicts.
  Local data is never silently overwritten.
- **Security** — connector access is scoped to tenant, organization, workspace,
  and resource; credentials are stored as references only; region policy is
  enforced.

### Knowledge graph 5.0 (`app/services/kg5.py`)

- Canonicalization with aliases, normalization, confidence, and provenance;
  merges require authorization, preserve provenance, and are audited.
- Relationship confidence + evidence, and temporal expiration.
- `graph_consistency` detects orphan relationships, cross-tenant edges,
  invalid entity references, contradictory relationships, and impossible
  temporal states.

### Memory 4.0 (`app/services/memory4.py`)

- Candidates are generated from validated evidence only; every memory carries
  provenance, scope, confidence, source, and timestamp.
- Suppression by user, workspace, or organization policy; durable conflict
  workflows; TTL / event-driven / supersession / policy expiration.
- `explain_memory` shows what memory exists, its source, confidence, age,
  scope, and why it is being used — without hidden chain-of-thought.

---

## 8. RAG 7.0

`app/services/rag7.py` implements evidence-first answering:

- **Intent planning** — query intent classification (factual, comparative,
  analytical, temporal, procedural, investigative, multi-document,
  entity-centric) drives a retrieval plan across keyword, vector, metadata,
  graph, memory, and document-version sources.
- **Evidence diversity/authority** — near-duplicate chunks are de-duplicated;
  evidence is ranked by authority, freshness, relevance, completeness, and
  confidence.
- **Sufficiency gating** — `evidence_sufficiency` decides whether evidence is
  sufficient *before* generation; insufficient evidence triggers a refusal
  rather than a hallucination.
- **Claim matrix 2.0** — claims are tracked with supporting and contradictory
  evidence, confidence, and citations; citation correctness and coverage are
  validated.
- **Conflict-aware answers** — conflicting sources surface the conflict and
  source differences instead of false certainty.
- **Temporal reasoning** — effective/expiration dates, version, publication
  date, and current validity are respected.
- **Bounded repair** — `repair_answer` repairs unsupported claims using
  retrieved evidence only, with a hard iteration bound.
- **Quality score** — `rag_quality_score` reports evidence sufficiency,
  citation coverage, citation correctness, contradiction level, and confidence.

---

## 9. Agents / Workflows / Actions

### Agent 4.0 (`app/services/agent4.py`)

- Plan validation (permissions, dependencies, budgets, timeouts, tool
  availability, tenant scope, risk) and simulation before side effects.
- Resource budgets (tokens, cost, steps, time, tool calls, output size) and
  checkpoints before/after side effects and external calls.
- Resume from the last valid checkpoint; race-safe, idempotent, audited
  cancellation; durable human handoffs (`WAITING_APPROVAL`, `WAITING_REVIEW`,
  `WAITING_INPUT`); dead letters with reason, retry count, last checkpoint,
  and safe-replay eligibility.

### Workflow 4.0 (`app/services/workflow4.py`)

- DAG validation, simulation, per-node checkpoints, safe parallel branches,
  join validation (incomplete joins never proceed), timeout recovery,
  bounded exponential retries, compensation classification
  (reversible / irreversible / compensatable / non-compensatable), and replay
  of idempotent-safe nodes only.

### Action platform 3.0 (`app/services/action3.py`)

- Risk engine classifies actions LOW / MEDIUM / HIGH / CRITICAL from data
  sensitivity, external side effects, destructiveness, resource count,
  reversibility, and tenant scope.
- Approval policy: auto-allow low risk, approval-required medium/high,
  blocked critical. Expired approvals cannot execute.
- Action previews show target, operation, affected resources, risk, expected
  effect, and required approval; every action is audited (requester, approver,
  action, target, policy, timestamp, outcome).

---

## 10. AI Safety 8.0 & Governance 4.0

### Safety 8.0 (`app/services/safety8.py`)

- `detect_injection` defends against direct/indirect injection, document text,
  OCR, metadata, filenames, hidden/encoded content, multi-step instructions,
  tool output, and connector content.
- `detect_exfiltration` blocks cross-tenant data access, secret extraction,
  system-prompt extraction, credential extraction, and unauthorized connector
  access.
- `validate_file` enforces MIME (declared vs magic bytes), archive structure,
  decompression ratio, file size, filename, and path safety.
- `sanitize_output` removes unsafe HTML, script injection, dangerous links,
  credential leakage, and internal metadata leakage.

### Governance 4.0 (`app/services/governance4.py`)

- Policy priority: organization → workspace → user, most-restrictive-wins.
- Model allowlist/denylist, provider allowlist, tool allowlists by
  organization/workspace/sensitivity/role.
- `minimize_context` removes unnecessary fields and redacts sensitive values
  before provider calls; `routing_decision` respects sensitivity classification
  (PUBLIC/INTERNAL/CONFIDENTIAL/RESTRICTED) and region restrictions.
- Legal holds (`is_held`) protect data from retention cleanup.

---

## 11. Observability, SLOs, Search, Analytics

### Observability 4.0 (`app/services/observability4.py`)

- Trace spans with correlation across API requests, queue jobs, workers,
  provider calls, retrieval, RAG, agents, workflows, tools, and notifications.
- `slo_health` evaluates configurable SLO windows (API latency/availability,
  worker completion, provider success, ingestion, RAG, workflows) and computes
  burn-rate style indicators; breach states are exposed to /ops.
- Alert deduplication keys alerts by fingerprint so repeated occurrences do
  not storm.

### Search 5.0 (`app/services/search5.py`)

- Planner combines keyword, vector, metadata, graph, memory, freshness, and
  scope; `explain_search` exposes filters and ranking factors without
  chain-of-thought; `quality_summary` tracks zero-result searches,
  reformulations, click-through, quality, and latency. Personalization stays
  within workspace scope.

### Enterprise analytics (`app/services/analytics2.py`)

- Organization and workspace analytics are aggregate-only: usage, AI requests,
  documents, storage, cost, search, workflows. `privacy_guard` enforces
  minimum-cell thresholds so private user data never appears in aggregates.

---

## 12. Data Consistency & DR

### Data consistency (`app/services/data_consistency.py`)

- `run_consistency_checks` (dry-run by default) checks documents, chunks,
  embeddings, collections, graph, memory, actions, workflows, usage, orphan
  records, and cross-tenant integrity. Repairs are always dry-run-first,
  bounded, auditable, and reversible where possible — suspicious data is never
  destroyed automatically.

### DR 2.0 (`app/services/dr2.py`)

- `record_backup` tracks backup scope, reference, migration head, and
  timestamp; `readiness_score` produces a deterministic 0–100 readiness score
  from backups, migration sync, and isolation checks; `restore_validation`
  re-checks migration head and tenant isolation after a restore.
- Runbook: detect outage → fail over region → restore → validate (migration
  head + tenant isolation) → rollback if needed → post-incident review.

---

## 13. API Platform 3.0

`app/services/api3.py`:

- **Idempotency** — `idempotency_check` / `idempotency_record` give
  side-effecting endpoints durable idempotency keys with replay detection
  (`replayed` flag), so retries are safe.
- **Structured errors** — `standard_error` returns `{code, message,
  request_id, retryable}`; internals are never exposed.
- **Pagination** — list endpoints use consistent bounded pagination with
  limits and cursors.
- **Rate limits** — tiered limits by user, workspace, organization, API key,
  and operator; `abuse_detection` flags repeated failures, suspicious key
  usage, excessive requests, enumeration, and scope violations.

---

## 14. Frontend Operations (/ops 4.0)

The /ops page now includes the Phase 19 control-plane panels:

- Global health (API, DB, broker, workers, providers) and scheduler leader
- Control-configuration snapshot versions
- Regions with status/health/failover targets
- Quarantined workers
- Broker backend, delivery guarantee, Redis availability, recovery status
- Provider capability registry and routing modes
- Vector coverage (total/embedded/failed chunks) and stale-embedding drift
- Ingestion resource governor limits and SLO health
- Consistency check results and DR readiness score
- RAG evaluation history and search quality

All panels load softly: a failing endpoint never blanks the page, the Refresh
button re-pulls everything, and loading/empty/error states are handled per
panel. Table markup uses semantic `<table>`/`<thead>`/`<th>` for screen
readers, and all statuses render as color-coded badges with text labels.

---

## 15. Environment & Configuration

- **pgvector** — optional. Detection gates the native path; JSON cosine
  fallback remains the default when the extension is absent.
- **Redis** — optional, lazily imported. The PostgreSQL broker is the tested
  default; `WORKER_BROKER=redis` activates the adapter only when a Redis
  server is reachable.
- **Providers** — no credentials required in development; the deterministic
  fake provider keeps the app fully usable. Real credentials are read from
  environment variables only, never from source.
- **Migrations** — exactly one Alembic head (`026_phase19_cloud2`); run
  `alembic upgrade head` on deploy.

### Runtime processes (unchanged from Phase 18)

| Process | Entry point |
|---------|-------------|
| API | `uvicorn app.main:app` |
| Worker | `python -m app.worker` |
| Scheduler | `python -m app.scheduler` |
| Event processor | `python -m app.event_processor` |