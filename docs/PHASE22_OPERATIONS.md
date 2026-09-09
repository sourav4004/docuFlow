# Phase 22 Operations — Real-World Production AI Cloud + Continuous Operations

Phase 22 hardens DocuFlow from a governed-autonomous platform into a
production cloud: honest capability detection, production vector/provider
platforms, continuous evaluation with promotion gates, multi-region DR,
observability 2.0, and chaos/load harnesses. Everything builds directly on
Phases 0–21 — no duplicate abstractions.

## Capability registry (`ops22/infrastructure`)

The platform detects what is REAL in the environment and reports it honestly:

- **PostgreSQL** — REAL when the configured server responds.
- **pgvector** — REAL only when the `vector` extension is actually installed.
  Otherwise the JSON fallback is reported (never claimed as native).
- **Redis** — REAL only when a Redis server answers; otherwise PostgreSQL
  broker fallback remains active.
- **Provider credentials** — presence checks only (no values are ever
  logged or returned).

Everything downstream keys off this registry: benchmarks run natively on
pgvector when available and in clearly-labeled simulated mode otherwise.

## Vector production platform (`ops22/vector/*`)

- Coverage snapshots and embedding-version drift detection (drift ratio is
  the share of chunks whose embedding metadata lags the active version).
- Per-document atomic rebuild (`POST /ops22/vector/rebuild`) — bounded,
  idempotent, scoped to one workspace/document.
- Bounded benchmark (`POST /ops22/vector/benchmark/{workspace_id}`) —
  native timing on pgvector; deterministic simulated timings otherwise,
  always labeled with the backend actually used.
- Dimension-safety check validates that registered embedding models match
  the stored chunk dimension before any new ingestion path activates.
- Backfill preview (`GET /ops22/vector/backfill/preview`) is read-only and
  bounded; execution remains governed by the Phase 21 autonomy guard.

## Provider validation & cost reconciliation (`ops22/providers/*`)

- Failure-mode matrix: each provider kind is validated against timeout,
  429, 5xx, and malformed-response modes (deterministic harness; REAL calls
  only when credentials exist).
- Health rollup across circuit state, consecutive failures, and latency.
- Cost reconciliation compares recorded spend estimates against sampled
  call metrics and reports variance honestly (estimated vs reconciled).

## Continuous evaluation & promotion gates (`ops22/evaluations/*`)

- Evaluation executions persist dataset version, model, provider,
  configuration, metrics, environment, and status — reproducible by design.
- Regression detection compares the latest execution against the prior
  baseline per dataset with deterministic thresholds.
- Promotion requires passing all configured gates (quality floor,
  regression bound, latency bound, cost bound). Rollback restores the
  previous promoted configuration; both are audited.

## Multi-region & DR (`ops22/regions/*`, `ops22/dr/*`)

- Region registry with residency rules; residency guard blocks illegal
  cross-region routing and records every block.
- Failover plan + deterministic dry-run simulation. REAL failover is
  attempted only when actual multi-region infrastructure exists — never
  claimed otherwise.
- Backup health, restore drill (dry-run), and DR report summarize RTO/RPO
  targets versus what has actually been observed.

## Observability 2.0 & SLO 2.0 (`ops22/traces/*`, `ops22/slos/*`)

- Unified trace construction with configurable sampling, PII redaction
  before persistence, and per-stage latency breakdown.
- Error and cost correlation on traces.
- SLO definitions per domain, error-budget computation, burn-rate detection
  with persisted burn events, and automatic incident creation at thresholds.
- Worker runtime: heartbeat verification, lease recovery, weighted
  tenant fairness, dead-letter recovery, bounded backpressure, load
  shedding, graceful shutdown.

## Security scans & cost guard (`ops22/security/scan`, `ops22/cost/*`)

- Continuous security scans aggregate the prompt-injection, exfiltration,
  tool-abuse, and SSRF corpora and persist per-suite results with honest
  pass/fail counts.
- Cost guard estimates before execution and returns
  ALLOW / BLOCK / APPROVAL_REQUIRED against workspace budgets.
- Cost anomaly detection and forecasting per workspace.

## Operating loops & chaos/load harness (`ops22/loops/*`, `ops22/chaos`, `ops22/load`)

- **Self-heal loop** — detection → diagnosis → governed recovery →
  verification, one audited run per invocation.
- **Autonomy operating loop** — turns detected improvement candidates into
  Phase 20 proposals (never auto-activates them).
- **Chaos harness** — bounded, deterministic probes for provider failure
  modes, worker loss, broker reconnect, scheduler leadership loss, and
  database error handling; every probe is recorded.
- **Load harness** — bounded RAG / ingestion / search volume runs with
  recorded throughput and latency, honest about simulated execution.

## Streams (`ops22/streams/{stream}`)

Persisted, bounded, tenant-scoped operational event streams (`ops`,
`recovery`, `security`, `eval`, `chaos`, `load`) for the operations UI.

## Honest limitations

- pgvector is NOT installed on the current PostgreSQL server; the vector
  platform reports `json_fallback` and all benchmarks are simulated.
- Redis is not reachable; the PostgreSQL broker remains authoritative.
- No real provider credentials are configured; provider validation runs
  the deterministic harness only.
- Multi-region is simulated (single real region); failover remains a
  dry-run capability.

These are reported as environment facts — never as production validation.
