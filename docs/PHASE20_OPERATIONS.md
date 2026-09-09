# Phase 20 — Operations Guide

DocuFlow Phase 20 evolves the governed, self-healing platform from Phase 19
into a **self-improving enterprise AI knowledge operating system**: the system
measures its own quality, detects knowledge/AI/operational degradation,
proposes improvements, evaluates them offline, and — only with explicit,
audited authorization — promotes and rolls back approved configurations.
Production behavior is never changed by an ungoverned AI suggestion.

This guide complements `PHASE18_OPERATIONS.md` and `PHASE19_OPERATIONS.md`.

---

## 1. Architecture at a glance

| Area | Module |
|------|--------|
| Improvement control plane | `app/services/improvement_platform.py` |
| Experiment platform | `app/services/improvement_platform.py` |
| AI quality 3.0 | `app/services/quality3.py` |
| Retrieval self-improvement | `app/services/retrieval_intel.py` |
| RAG self-improvement | `app/services/rag_intel.py` |
| Knowledge health | `app/services/knowledge_intel.py` |
| Policy intelligence | `app/services/policy_intel.py` |
| Provider/model intelligence | `app/services/provider_intel.py` |
| Cost intelligence 5.0 | `app/services/cost_intel.py` |
| Agent / workflow intelligence | `app/services/agent_intel.py`, `app/services/workflow_intel2.py` |
| Memory / graph intelligence | `app/services/memory_intel.py`, `app/services/graph_intel.py` |
| Search intelligence | `app/services/search_intel.py` |
| Feedback intelligence | `app/services/feedback_intel.py` |
| Governance 5.0 | `app/services/governance5.py` |
| AI safety 9.0 | `app/services/safety9.py` |
| Observability 5.0 + incidents | `app/services/observability5.py` |
| SLO 2.0 / error budgets | `app/services/slo2.py` |
| Notification intelligence | `app/services/notifications_intel.py` |
| Versioned reporting | `app/services/reporting3.py` |
| API / DB intelligence | `app/services/api_intel.py`, `app/services/db_intel.py` |
| API surface | `app/api/ops20.py` (prefix `/ops20`) |
| UI | `/ops` (operations), `/ai-control` (AI control center) |

All Phase 20 tables live in `app/models/phase20.py`; schema changes ship in
migration `027_add_phase20_knowledge_os.py` (head `027_phase20_knowledge_os`).

---

## 2. Self-improvement control plane

`improvement_platform.py` implements the generic improvement model. Domains:
retrieval, RAG, model routing, provider routing, prompt/configuration,
workflow, ingestion, search, cost, latency, knowledge quality, agent, memory,
graph.

Proposal lifecycle (validated server-side, every transition audited):

```
PROPOSED → EVALUATING → APPROVAL_REQUIRED → APPROVED → STAGED → ACTIVE
    └──────────┬────────────────────────────────────────────────┘
               ├→ SUPERSEDED        ├→ ROLLED_BACK       └→ REJECTED
```

- AI-generated proposals (`author_source="ai"`) always start `PROPOSED` and can
  never self-activate.
- Transitions are recorded in `improvement_audits` with actor, timestamp,
  previous/new state, reason, and evidence.
- `ImprovementTransition` rows allow operators to declare custom transitions
  (for example a review gate) without code changes.

Operational commands: `imp.create_proposal`, `imp.transition`,
`imp.list_proposals`, `imp.proposal_audit_trail`. API:
`POST /ops20/improvements`, `POST /ops20/improvements/{id}/transition`,
`GET /ops20/improvements/{id}/audit`.

## 3. Experiment platform

- **Datasets** — `experiment_datasets` with kinds `golden` / `synthetic` /
  `anonymized` / `curated`.
- **Experiments** — `experiments` store an immutable `config_fingerprint`; the
  config of a running experiment is never mutated. Isolation: experiments
  never read or write production behavior.
- **Runs** — `experiment_runs` persist metrics, model, provider, cost,
  latency, dataset, environment, status.
- **Comparison** — `compare_runs` is a deterministic paired-delta comparison;
  verdicts are `CANDIDATE_BETTER` / `BASELINE_BETTER` / `INCONCLUSIVE`.
  Comparisons below a minimum sample size are always `INCONCLUSIVE` — no
  significance is claimed without data.
- **Promotion gate** — `promotion_eligible` enforces quality/cost/latency
  thresholds over completed runs. Promotion itself still requires explicit
  human authorization through the improvement lifecycle.

API: `/ops20/datasets`, `/ops20/experiments`,
`/ops20/experiments/{id}/runs`, `/ops20/experiments/{id}/compare`,
`/ops20/experiments/{id}/gate`.

## 4. AI quality platform

`quality3.py`:

- **Scorecards** per domain (retrieval, RAG, citations, extraction,
  summarization, search, agents, workflows) with dimensions: correctness,
  completeness, groundedness, citation coverage/correctness, confidence,
  latency, cost, refusal accuracy.
- **Thresholds** are configurable per scorecard.
- **Regression detection** (`detect_regression`) compares a scorecard against
  its baseline and reports regressions; it **never deploys a fix**.
- **Trends** — daily/weekly/monthly aggregation in `quality_trends`.
- **Alerts** — deduplicated quality degradation alerts in `quality_alerts`.

API: `/ops20/quality/scorecards`, `/ops20/quality/regression`,
`/ops20/quality/trends`, `/ops20/quality/alerts`.

## 5. Retrieval / RAG self-improvement

`retrieval_intel.py` classifies retrieval failures (missing document, poor
chunking, poor keyword/vector match, metadata mismatch, graph miss, stale
knowledge, permission restriction), analyzes query quality (zero results,
reformulations, abandoned searches), and generates persisted
recommendations in **PROPOSED** state — evaluation is required before
activation. `rag_intel.py` records RAG/claim-level failures (unsupported
answer, incomplete evidence, incorrect citation, conflicts, temporal
mismatch, overconfidence, unnecessary refusal) and evaluation pipelines whose
promotion requires quality/cost/latency/security gates.

API: `/ops20/retrieval/failures`, `/ops20/retrieval/recommendations`,
`/ops20/retrieval/recommendations/generate`, `/ops20/retrieval/query-quality`,
`/ops20/rag/failures`, `/ops20/rag/recommendations`, `/ops20/rag/pipelines`.

## 6. Knowledge intelligence

`knowledge_intel.py`:

- **Freshness** — per document/connector/entity/memory/graph last-observed
  times.
- **Staleness + drift** — stale content, changing terminology, entity and
  relationship changes.
- **Completeness** — metadata, embeddings, entities, relationships, summaries,
  citations coverage.
- **Health scores** — persisted `knowledge_health` rows per scope with
  explainable contributing signals.
- **Gap detection** — questions repeatedly asked with weak evidence become
  `knowledge_gap_insights` with recommendations. The engine **never invents
  knowledge** — recommendations name missing documents, metadata, connector
  syncs, or document refreshes.

API: `/ops20/knowledge/freshness`, `/ops20/knowledge/health`,
`/ops20/knowledge/gaps`. `policy_intel.py` versions policies and detects
drift; `policy_intel.simulate` answers *"would this operation be allowed?"*
with zero side effects.

## 7. Document intelligence

`knowledge_intel.change_intelligence` classifies document changes (formatting,
metadata, minor/major content, policy-relevant, numerical, deadline, entity,
relationship), computes change impact across summaries/embeddings/citations/
workflows/graph/memories, and plans safe reprocessing. Destructive
reprocessing is never executed automatically.

API: `POST /ops20/documents/changes`, `GET /ops20/documents/{id}/health`.

## 8. Model / provider intelligence

`provider_intel.py`:

- Per-model and per-provider performance profiles (latency, success rate,
  cost, quality, context/tool/structured-output support).
- Candidate routing recommendations persisted as **PROPOSED**.
- Deterministic routing simulation over historical/synthetic workloads.
- Provider anomaly detection (latency spikes, error spikes, unexpected cost,
  quality degradation).

API: `/ops20/providers/profiles`,
`/ops20/providers/routing-recommendations`,
`/ops20/providers/routing-recommendations/generate`,
`/ops20/providers/simulate-routing`, `/ops20/providers/anomalies/detect`.

## 9. Cost intelligence

`cost_intel.py` maintains per-workspace/organization/model/provider/workflow/
agent cost baselines, detects cost drift, measures token efficiency
(input/output/retrieval-context/repeated-context), and evaluates candidate
optimizations (smaller model, caching, batching, prompt/context reduction)
as experiments. Cost optimization is always shown against its quality
tradeoff — never optimized while silently destroying quality.

API: `/ops20/cost/baselines`, `/ops20/cost/drift`,
`/ops20/cost/token-efficiency`, `/ops20/cost/optimization-candidates`.

## 10. Agent / workflow intelligence

- `agent_intel.py` — success metrics, deterministic failure classification
  (planning, authorization, tool, provider, knowledge, timeout, policy,
  human approval), plan-quality scoring (unnecessary steps, redundant tools,
  excessive cost/latency, failed dependencies), candidate plan optimizations
  (evaluation required), and an agent safety score (policy violations,
  rejected tools, unsafe attempts, approval rate).
- `workflow_intel2.py` — workflow success analytics, bottleneck detection,
  failure hotspots, optimization recommendations, and side-effect-free
  simulation. Named `workflow_intel2` to avoid the pre-existing Phase 14
  `workflow_intel` (definition validation + versioned workflow drafts).

API: `/ops20/agents/intelligence`, `/ops20/workflows/intelligence`.

## 11. Memory / graph intelligence

- `memory_intel.py` — memory quality aggregates, decay reports, safe
  consolidation candidates (same source type, adequate confidence, content
  similarity — always review-required), conflict queues, and per-user
  controls (inspect / suppress / delete only where permitted; never across
  workspaces).
- `graph_intel.py` — graph health (orphan entities/relationships, stale and
  expired relationships, low-confidence entities), drift between health
  snapshots, candidate entity/alias/relationship recommendations
  (PROPOSED only), and deterministic graph quality evaluation.

API: `/ops20/memory/quality`, `/ops20/memory/{id}/controls`,
`/ops20/graph/health`.

## 12. Search intelligence + feedback

- `search_intel.py` — search quality scoring (zero results, reformulations,
  clicks, abandonment), controlled ranking experiments via the experiment
  platform, safe explanations (matched metadata, semantic similarity,
  freshness, source authority — never hidden reasoning), and personalization
  safety that cannot cross workspace/org scopes.
- `feedback_intel.py` — unified feedback from search/RAG/citations/
  summaries/extraction/agents/workflows; quality flags (noisy, contradictory,
  duplicate, abuse); and **governed golden promotion**: feedback becomes a
  golden evaluation example only with explicit authorization.

API: `/ops20/search/intelligence`, `/ops20/search/events`, `/ops20/feedback`,
`/ops20/feedback/quality`, `/ops20/feedback/{id}/promote`,
`/ops20/feedback/summary`.

## 13. Governance 5.0

`governance5.py`:

- Every production AI configuration change requires owner + reason; optional
  evaluation/approval references are recorded.
- Model / tool / provider / data / workflow policies are versioned in
  `policy_versions` with human-readable diffs (`governance_diff`).
- `effective_policy` merges organization → workspace → user policies with
  most-restrictive-wins semantics (allowlists intersect, denylists union,
  caps take the minimum, retention the maximum).
- `simulate` answers allow/block questions with **no side effects**.

API: `/ops20/governance/changes`, `/ops20/governance/effective`,
`/ops20/governance/simulate`, `/ops20/governance/audit`,
`/ops20/policy/versions`, `/ops20/policy/drift`, `/ops20/policy/simulate`,
`/ops20/policy/conflicts`.

## 14. AI safety 9.0

`safety9.py` provides a deterministic regression suite:

- **Prompt injection corpus** — direct, indirect, document, OCR, metadata,
  encoded (base64/hex), multilingual, tool output, connector content,
  malicious workflow instructions, and hidden-content vectors. Runs through
  the layered `ai_security2.detect_injection`; the corpus currently detects
  all cases (14/14).
- **Exfiltration suite** — credential/secret/system-prompt/cross-tenant
  extraction attempts.
- **Tool-abuse checks** — unauthorized tools, excessive calls, argument
  injection, recursion, scope escalation, destructive attempts (never
  permitted merely because an AI requested them).
- **Output security** — XSS, unsafe links, credential leakage, malicious
  markup.
- **Safety scorecard** — measurable 0..1 score from injection/exfil/tool/
  output detection rates.

API: `/ops20/safety/injection-corpus`, `/ops20/safety/exfiltration`,
`/ops20/safety/tool-abuse`, `/ops20/safety/output`.

## 15. Observability 5.0, incidents, SLOs

- `observability5.py` — unified AI system health score (availability,
  latency, quality, cost, error rate, queue health, provider health), a
  declarative service dependency graph, incident correlation by fingerprint
  (repeated failures append to the open incident instead of storming), and an
  incident lifecycle `OPEN → INVESTIGATING → MITIGATED → RESOLVED →
  POSTMORTEM` with a persisted timeline.
- `slo2.py` — SLO history, error-budget computation, error-budget policy
  (surface warning / recommend freeze / require operator review — it never
  auto-blocks unrelated operations), and reliability scores.
- `notifications_intel.py` — alert prioritization (informational → critical),
  fingerprint deduplication with occurrence counts, escalation rules
  (low→medium at 3 repeats, medium→high at 5, high→critical at 8), and
  per-user/workspace notification category preferences.

API: `/ops20/incidents`, `/ops20/incidents/{id}/timeline`,
`/ops20/incidents/{id}/transition`, `/ops20/health/score`,
`/ops20/dependency-graph`, `/ops20/slo/history`, `/ops20/slo/error-budget`,
`/ops20/alerts`, `/ops20/alerts/raise`, `/ops20/alerts/summary`,
`/ops20/notifications/preferences`.

## 16. Reporting

`reporting3.py` — immutable, versioned reports (`report_versions`) of kind
`ai_quality`, `knowledge_health`, `governance`, `reliability`, `cost`.
Each new report for the same scope increments the version; stored content is
never regenerated or mutated.

API: `/ops20/reports`.

## 17. API / database intelligence

- `api_intel.py` — per-endpoint health metrics (requests, errors, auth
  failures, latency), unstable-endpoint detection, contract validation
  (structured errors, validation, pagination, authorization, idempotency),
  idempotency audit across side-effecting routes, and abuse detection
  (enumeration, brute force, scope probing, abnormal key usage).
- `db_intel.py` — DB health metrics, query-regression detection against
  baselines, index-effectiveness review, high-growth table monitoring, and
  migration-head reporting.

API: `/ops20/api/health`, `/ops20/api/abuse`, `/ops20/database/health`,
`/ops20/database/metrics`.

## 18. UI

- `/ops` — Phase 19 operations console with a new **AI Control Center** link.
- `/ai-control` — the Phase 20 control center: system health, dependency
  graph, incidents, alerts, improvement proposals, experiments, SLO history,
  governance changes, knowledge health + gaps, AI quality scorecards, AI
  safety detection rates, agent/workflow/memory/graph/search intelligence,
  feedback summary, API/DB health, and versioned reports. Reads fail soft
  (one failing panel never blanks the page) and every list shows loading,
  empty, and error states.

## 19. Migration & schema

Migration `027_add_phase20_knowledge_os` creates the Phase 20 tables
(children before parents in the downgrade). Validation performed: single
head `027_phase20_knowledge_os`, full downgrade → upgrade roundtrip on
PostgreSQL, no orphaned/duplicate route registration.

## 20. Troubleshooting

- **A proposal cannot be activated.** Verify the state path — activation
  requires `APPROVED → STAGED → ACTIVE`, and every transition is server
  validated. Check `GET /ops20/improvements/{id}/audit` for the actor/reason
  trail.
- **An experiment comparison is INCONCLUSIVE.** Sample sizes below the
  minimum are intentionally inconclusive; add more runs with
  `sample_size`-bearing metrics.
- **Alert storms.** Alerts deduplicate by fingerprint and escalate on repeat
  counts; acknowledge or resolve to stop further grouping, or disable a
  category under `/ops20/notifications/preferences`.
- **Incident reopened unexpectedly.** `MITIGATED` incidents intentionally
  allow `INVESTIGATING`; correlation appends new failures to open incidents
  with the same fingerprint.
- **Retrieval recommendations appear without effect.** Recommendations are
  PROPOSED by design; route them through an experiment and the improvement
  lifecycle before any activation.
- **Safety corpus regressions.** The corpus runs in CI
  (`test_phase20_feedback_governance.py::TestSafety9`); a drop in the
  detection rate means the layered detectors regressed — fix the detector,
  never weaken the corpus.
