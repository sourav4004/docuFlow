# Phase 21 Operations — Autonomous Enterprise AI + Self-Healing Knowledge Cloud

Phase 21 turns DocuFlow into a governed-autonomous platform. The central
principle: **AI may detect, diagnose, recommend, evaluate, simulate, and
recover pre-approved low-risk failures — but production behavior changes only
through the autonomy guard, which enforces policy, risk classification,
budgets, cooldowns, and audit on every operation.**

## Autonomy control plane

- **Policies** — per (workspace, operation type, risk level): autonomy level,
  approval requirement, budget limit, hourly execution limit, cooldown, audit
  flag. Managed via `POST /ops21/autonomy/policies`.
- **Levels** — `OBSERVE`, `RECOMMEND`, `AUTO_LOW_RISK`, `AUTO_APPROVAL`,
  `MANUAL_ONLY`. Transitions are server-validated (no skip-level promotion);
  every change writes an `autonomy_transitions` audit row.
- **Guard** — `guard_operation()` is the only execution path. It records
  `ALLOWED` / `REQUIRES_APPROVAL` / `BLOCKED` with reason, honors emergency
  stop (scoped per subsystem), budgets, hourly limits, cooldowns, and
  idempotency keys. Simulation (`POST /ops21/autonomy/simulate`) answers
  "would this be allowed?" with zero side effects.
- **Audit** — every decision persists in `autonomous_operations` with actor,
  source (AI/SYSTEM/OPERATOR), input, decision, result, and rollback info.

## Self-healing

- Health aggregation across api/db/broker/workers/scheduler/event processor/
  provider/vector/ingestion/connector/cache with states `HEALTHY`,
  `DEGRADED`, `UNHEALTHY`, `RECOVERING`, `UNKNOWN`.
- Failure detection for worker stalls, queue buildup, provider/db/broker/
  connector/ingestion/vector/cache failures.
- Recovery playbooks are persisted and refuse destructive actions outright.
  Automatic execution requires: approved playbook + LOW risk + autonomy
  policy permitting + no emergency stop. Attempts are idempotent, bounded by
  cooldowns, and escalate to incidents after `max_attempts`.

## Self-diagnosis

`diagnose()` correlates failures, ranks hypotheses with explicit evidence and
deterministic confidence (capped at 95%, capped at 40% for small samples),
and renders operator-readable reports. Reports contain evidence, confidence,
and concise reasons — never hidden reasoning.

## Adaptive subsystems

- **Knowledge** — health dimensions (freshness, completeness, embeddings,
  metadata, entities, graph, memory, citations), incident detection, bounded
  recovery plans executed only through the guard.
- **Ingestion** — quality samples, drop-threshold anomaly detection,
  adaptation candidates.
- **Retrieval / RAG / Search** — continuous evaluation metrics, drift
  snapshots, bounded tuning candidates; promotion requires passing gates AND
  an `ALLOWED` autonomy decision.
- **Model/provider autopilot** — performance sampling, model and provider
  drift, zero-side-effect routing simulation with sensitivity/region/policy/
  budget/capability checks; routing promotion is governed.

## Cost autopilot

Live spend + linear forecasts per scope, anomaly detection, pre-execution
cost guard (`ALLOWED`/`REQUIRES_APPROVAL`/`BLOCKED`), and exactly four
pre-approved optimizations (`cache_reuse`, `context_reduction`, `batching`,
`cheaper_model`) — each audited with why it occurred. Cost-aware scheduling
preserves tenant fairness: budget pressure lowers priority but never starves.

## Agent & workflow autonomy

Plans are risk-classified (tools, sensitive data, external writes, destructive
ops, financial impact), simulated dry, optionally optimized (dedup/expensive
calls) under policy, and handed to human review when high risk. Failed runs
get durable idempotent recovery (`RETRY`/`CHECKPOINT`/`RESUME`/`ROLLBACK`/
`HANDOFF`) and dead-letter capture. Workflows get the same governed path with
a dedicated risk engine.

## Safety 10.0 & Security Center 3.0

- Injection corpus: 8 vectors (document/OCR/metadata/connector/tool-output/
  encoded/multilingual/indirect), all detected and blocked.
- Exfiltration corpus: 6 scenarios (tenant/workspace boundaries, restricted
  docs, credentials, internal config, output transfer), all blocked.
- Tool safety: allowlist, argument validation, budget, timeout, scope checks;
  output sanitization redacts secrets before they reach model context.
- Limits: per-operation, per-workspace-hour, plus global emergency stop
  (`POST /ops21/safety/emergency-stop`, audited, liftable).
- Security health scoring, threshold-driven security incidents, subject
  correlation, API abuse signals with adaptive rate-limit recommendations.

## Data governance 6.0

Classification snapshots with drift detection, policy impact analysis
(minimization for confidential/restricted), and residency violation recording
when data lands outside allowed regions. Data minimization checks verify only
required fields reach providers.

## Incidents 2.0

SEV0–SEV4 model, threshold-driven auto-creation, persisted timelines,
structured postmortem DRAFTS (`DRAFT_REQUIRES_HUMAN_REVIEW`) finalized only by
a human, and approved learnings converted into tests/monitoring rules/
recovery playbooks/evaluation cases — never executed automatically.

## Maintenance, backup, DR

Maintenance plans support dry runs, respect legal holds, and require approval
plus a prior dry run before destructive execution. Backup health reports
UNKNOWN honestly when no infrastructure exists; restore validation records are
explicitly flagged SIMULATED unless a real restore occurred. DR simulation
validates decision logic only.

## Multi-region

Region health/capacity tracking, residency guard blocking illegal cross-region
routing, dry-run failover simulation (residency + capacity), and governed
failover execution (HIGH risk → approval).

## Personal AI control

Per-user autonomy settings clamped to workspace policy, memory controls
(inspect/suppress/reset/delete gated by personal settings, all audited), and
an activity feed whose explanations contain evidence/policy only.

## Ops surface

`/ops21/*` — 91 endpoints, all authenticated, workspace writes owner/admin
only, bounded pagination, structured errors. Frontend: **Autonomy Center** at
`/autonomy` (linked from `/ops`).
