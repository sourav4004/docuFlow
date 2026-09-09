# Phase 23 Operations — Global AI Cloud Platform

Phase 23 turns DocuFlow from a production-operations platform into a
deployable global AI cloud platform: a unified capability control plane,
production provider gate with admission control, real cost reconciliation,
vector/model migration with zero-downtime search coexistence, multi-region
residency enforcement, broker/storage migration safety, and API platform 4.0
(idempotency, dedup, tenant-scoped cache, v1/v2 coexistence). Everything
extends Phases 0–22 — no duplicate abstractions.

## Capability control plane (`ops23`)

`GET /ops23/capabilities` — full registry (seeded on first read).
`GET /ops23/capabilities/{name}` — one component.
`POST /ops23/capabilities/check` — run detection (one or all).
`POST /ops23/capabilities/refresh` — re-detect + persist.

States are honest: REAL / SIMULATED / NOT_CONFIGURED / DEGRADED / UNAVAILABLE.
Never labeled a simulation as production validation.

## Global health + dependency graph (Steps 18–19)

- `GET /ops23/global-health` — capabilities + regions + degraded components
  + database/broker states, overall HEALTHY/DEGRADED/UNHEALTHY.
- `GET /ops23/dependencies` — component dependency graph (HARD/SOFT edges).
- `GET /ops23/dependencies/{component}/impact` — upstream/downstream impact,
  blast radius. Unknown components → 404.
- `GET /ops23/readiness`, `GET /ops23/degraded-components`.

## Provider production gate (Steps 6–9)

- `POST /ops23/providers/{kind}/readiness` — synthetic capability matrix
  (completion, streaming, embeddings, structured output, tools, multimodal)
  over the deterministic fake provider. Real credentials are never required
  and tenant data is never sent.
- `POST /ops23/routing/decide` — deterministic routing 3.0: capability,
  readiness score, sensitivity prohibitions, residency rules, context bound.
  Every decision (ROUTED or REJECTED) is persisted with reasons.
- `POST /ops23/routing/admission` — pre-flight gate: routing + budget.
- `POST /ops23/cost/reconcile` — estimated vs actual usage/cost with
  classification: OK / OVERBILLING_RISK / UNDERBILLING_RISK /
  ANOMALOUS_USAGE / MISSING_USAGE.
- `GET /ops23/cost/reconcile/{workspace_id}` — summary by classification.

## Vector activation + model migration (Steps 10–12)

- `GET /ops23/vector/activation` — native pgvector validation when REAL;
  JSON fallback reported honestly otherwise.
- Model migration lifecycle: register immutable version → dual generation
  (bounded batches, per-document status) → coverage verification → quality
  comparison → promotion gate → retirement plan. Rollback supported.
- Shadow comparisons (`POST /ops23/vector/shadow-compare`) accumulate
  evidence; the promotion gate requires a minimum comparison count and a
  better-share threshold, and always requires human approval.

## Multi-region + residency (Steps 13–17)

- `POST /ops23/regions` — register/update region with capability view.
- `POST /ops23/residency/evaluate` — per-operation decision with evidence
  log (request, source, destination, classification, decision, reason,
  actor). Illegal routing is BLOCKED and recorded.
- Drain lifecycle: start → progress → complete, with recovery (rollback).
  Regional scoping is expressed by workspace assignment (WorkerJob is
  globally durable and carries no region column).
- Failover: deterministic decision with health/capacity checks; EXECUTED
  only when autonomy policy explicitly allows it and every check passes.

## Broker + storage migration safety (Steps 3–5)

- Broker migration states: PLANNED → DRAINING → DRAINED → VERIFYING →
  VERIFIED → ACTIVATING → ACTIVE, with FAILED → re-plan and ROLLED_BACK
  rollback. Illegal transitions are rejected. Queue verification reports
  `no_job_loss_risk` before activation.
- Storage migration: plan (bounded batches) → run (checksum verified,
  resumable, idempotent) → progress. Dry-run first; no source deletion
  until verification; orphan cleanup always requires governance.

## API platform 4.0 (Steps 39–45)

- Idempotency keys with replay/in-flight/conflict semantics; FAILED allows
  clean retry. Tenant-scoped: the same key in another workspace is a
  different request.
- Tenant cache 3.0: versioned keys, workspace-scoped invalidation,
  stampede-safe `get_or_load`, stats.
- `GET /ops23/api/compatibility` — v1/v2 coexistence metadata.
- `GET /ops23/api/contract-matrix` — the 10-case contract matrix used by
  the API suites.

## Knowledge maintenance + freshness + drift (Steps 23–25)

- Freshness: FRESH / AGING / STALE / EXPIRED / UNKNOWN with reasons.
- Drift detection with explainable findings per workspace.
- Maintenance scan produces an auditable, bounded, resumable PLAN; safe
  actions only, destructive actions require approval.

## Webhooks + scheduler + review center (Steps 35, 48, 51)

- Webhook reliability 3.0: HMAC-SHA256 `t=...,v1=...` signatures with
  timestamp replay protection, exponential backoff, endpoint auto-disable
  after repeated failures, dead-letter management, operator re-enable.
- Scheduler 3.0: exactly-once claim per due bucket (duplicate claims are
  SKIPPED_DUPLICATE), finish-with-status, per-run audit.
- Unified review queue: typed items (ai_action, policy_conflict,
  knowledge_conflict, extraction_uncertainty, provider_issue,
  security_finding, workflow_approval, agent_handoff) with
  APPROVE / REJECT / REQUEST_CHANGES / DELEGATE / EXPIRE — all audited.

## Security operations 4.0 (Steps 36–38)

- Continuous scan corpora (injection, exfiltration) with deduplicated
  findings on the Phase 21 SEV0–4 incident model.
- Autonomy abuse detection: budget bypass, scope escalation, recursive
  execution — all blocked + recorded.
- Tool safety: allowlist, argument validation, execution budget, timeout,
  scope check, output sanitization (secret stripping + length cap).
- Action limits: per-operation / per-workspace / per-org / global emergency.
- Emergency stop: scoped (ALL / AI_ACTIONS / AGENTS / WORKFLOWS /
  AUTONOMOUS_RECOVERY), audited activation + lift.
- Security Center 3.0: signal recording, threshold-based incident creation,
  correlation, playbooks, classification snapshots + drift.

## Evaluation 2.0 + promotion gates (Steps 56–57)

- Reproducible evaluation runs (dataset version, model, provider, config,
  environment, metrics, latency, cost, security findings).
- Deterministic promotion gates: quality, security, latency, cost,
  regression. Promotion always requires explicit authorization; rollback
  supported.

## Search 6.0 + streams (Steps 52–53, 62)

- Ranking explainability (weighted attribution), document-diversity caps,
  self-evaluation that auto-creates improvement proposals on degradation
  (proposals only — human approval required for production changes).
- Durable tenant-scoped stream events with monotonic `seq` cursors,
  after-seq recovery, bounded reads, polling fallback.

## Worker platform 3.0 + fairness (Steps 46–47)

- Fair claim: priority-dominant ranking with tenant round-robin and
  age anti-starvation; per-workspace cap prevents noisy-neighbor monopoly.
- Leases, heartbeat, dead-letter listing/requeue (idempotent, never
  automatic), worker health score, quarantine workflow, autoscale signals.

## Data consistency engine (Step 43)

Bounded checks (orphan chunks, cross-tenant edges, memory integrity,
execution integrity) → report → repair plan. Plans are dry-run by default,
never destructive, and flag-for-review unless trivially safe.

## Environment limitations (honest)

- **pgvector**: NOT available in this environment (JSON fallback active).
  Native vector validation is reported as unavailable, never simulated.
- **Redis**: not configured; PostgreSQL broker remains authoritative.
- **Real AI providers**: no credentials configured; the deterministic fake
  provider is used everywhere and labeled as such.
- **Object storage**: local backend (REAL, LocalStorageBackend); S3-ready
  interface, signed URLs not claimed.
- **Multi-region**: single real region; drain/failover validated by
  deterministic simulation only.
- **Browser E2E / soak**: not run in this environment.

## Troubleshooting

- Capability shows NOT_CONFIGURED → run `POST /ops23/capabilities/check`.
- Routing REJECTED → inspect `reasons` in the persisted decision.
- Admission blocked → check the failing gate (`budget`, `routing`).
- Promotion gate failing → accumulate shadow comparisons; better-share and
  minimum counts must pass; human approval always required.
- Emergency: `POST /ops23/...` emergency stop via safety10 service, scoped,
  audited; lift after mitigation.
