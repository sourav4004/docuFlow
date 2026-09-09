# Phase 16 — Production AI Operating Platform

Phase 16 turns the Phase 15 "Autonomous Knowledge Operating System" foundation into a
production-grade platform: durable workers, a config-driven real-provider gateway,
vector/embedding production hardening, page-level multimodal intelligence, semantic
policy contradiction detection, a safe legacy workspace backfill, RAG 4.0, agent
resumability, approval 2.0, cost guardrails, webhook reliability, and an observability
layer — all while keeping pgvector optional (JSON fallback preserved) and never
requiring real provider credentials for tests.

## Worker platform

- **Queue backend**: durable PostgreSQL-backed queue via the existing `worker_jobs`
  table (no external broker required). `QUEUE_BACKEND=postgres` is the only shipped
  backend; the provider interface (`app/services/worker_platform.py`) keeps broker
  adapters pluggable without coupling application logic to one broker.
- **Lifecycle**: `QUEUED → CLAIMED → RUNNING → CHECKPOINTED → COMPLETED`, or
  `RETRYING → FAILED → DEAD_LETTERED`, plus `CANCELLED`. Claiming is atomic
  (`UPDATE ... WHERE status='QUEUED'` guarded by a visibility timeout), so two
  workers can never process the same job.
- **Heartbeats**: `worker_heartbeats` rows track worker id, status, current job,
  start time, and last heartbeat. Stale workers (heartbeat older than a configurable
  interval) have their claims requeued.
- **Fairness**: jobs are selected per-tenant with bounded concurrency and priority
  ordering (CRITICAL → BACKGROUND); starvation protection rounds tenants. Per-tenant
  and global concurrency caps are enforced server-side.
- **Metrics**: queue depth/wait/running/retrying/dead per queue and per tenant are
  exposed through the authenticated admin `GET /ops/queue-metrics` endpoint.
- **Graceful shutdown**: `stop()` halts claiming, drains safe in-flight work,
  checkpoints, and releases claims.
- **AI execution worker**: `app/services/ai_execution_worker.py` connects Phase 15
  `AIExecution` records to the durable worker; execution steps are checkpointed and
  resume after recoverable failure via `resume_execution`.

## AI provider platform

- `app/services/provider_platform.py` provides a **config-driven OpenAI-compatible
  gateway**: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, plus timeout/retry/temperature/
  max-tokens settings. Prompts with sensitive content are never logged.
- **Capability registry** (`model_capabilities`) records per-model text/vision/tools/
  structured-output/streaming support, context window, max output, embedding
  dimensions, and pricing. Routing consults capabilities before selection.
- **Provider health** (`provider_health`) tracks rolling success/failure counts,
  latency, and a circuit breaker (closed/half-open/open). Fallback chains never retry
  indefinitely and never route RESTRICTED data to a policy-disallowed provider.
- Without configuration the deterministic fake provider is used, so the entire suite
  runs with zero credentials.

## Vector platform

- `detect_pgvector()` reports pgvector availability honestly at runtime
  (`GET /ops/vector-status`). pgvector absent → JSON cosine fallback remains active.
- **Dimension validation** prevents mismatched embeddings; **embedding cache 2.0**
  keys on provider+model+dimensions+normalized content hash and is tenant-safe.

## Multimodal document intelligence

- Page-level abstraction (`document_pages`, `page_regions`): page number, text,
  layout/tables/images, OCR regions with bounding boxes and per-block confidence.
- OCR and table/layout intelligence are provider-abstracted with deterministic local
  implementations; no third-party credentials required.
- Multimodal retrieval returns text chunks, table regions, and page regions with
  source/page provenance; image reasoning stays behind approved provider/tool paths
  with MIME, size, sensitivity, and timeout enforcement.

## Policy intelligence

- `app/services/policy2.py` normalizes policy statements into semantic structures
  (subject/action/condition/threshold/unit/timeframe/exception/scope/source) and
  detects real contradictions: numeric threshold overlap, date conflicts, negation,
  incompatible frequency/scope/obligations. Superficial textual differences are not
  reported as conflicts.
- Every conflict carries statement A/B, sources, category, confidence, and detection
  method, and is pushed into the human review queue (never auto-resolved).

## Legacy workspace backfill

- `app/services/backfill_service.py` + `GET/POST /admin/backfill/*` (org-admin only)
  + `scripts/backfill_workspaces.py` CLI. Deterministic, idempotent, resumable,
  batch-based, dry-run capable, and fully audited via backfill run/batch records.
- Ambiguous ownership is never guessed — the record is left unchanged and surfaced in
  the run's failures report.

## Knowledge, RAG 4.0, agents

- Evidence sufficiency, claim–evidence matrix, confidence from evidence quality/
  coverage/contradiction state, temporal ("as of") filtering, and bounded answer
  repair (unsupported claims only; evidence is never invented).
- Entity canonicalization (aliases/confidence/source counts, no destructive merging),
  relationship confidence/evidence/validity, entity change detection and timelines.
- Memory 2.0 adds confidence/provenance/conflicts (both sides preserved, never
  silently overwritten) and a full lifecycle (active/expired/superseded/deleted) with
  governance endpoints.
- Agent checkpoints, resume-after-crash, cancellation propagation, and dead-letter
  inspection/retry/abandon are supported.

## Governance, security, observability

- Approval 2.0: risk-based approval policies, approval expiration, full audit
  (requester/reviewer/decision/policy/action hash), and action preview.
- Webhooks: SSRF gate before any delivery, atomic claiming, outbox drain worker,
  signed deliveries with bounded retry/backoff; synchronous delivery never happens in
  request handlers.
- Traces (`TraceSpan`) record request → execution → provider/tool calls with
  correlation ids; `GET /ops/traces` and `/ops/trace-summary` are admin-scoped.
  Sensitive prompts are not stored by default.
- Worker/provider/vector status and gateway status are exposed to owners and
  organization admins through `/ops/*` (queue metrics, provider status, capabilities,
  vector status); `worker_ops.py` enforces the org-admin/owner gate server-side.

## Frontend

- New `/ops` Operations route (12 total routes) shows queue depth by class, tenant
  load, provider health, circuit state, and the honest vector-backend report; errors
  surface the admin/owner requirement when applicable. No stack traces or secrets are
  ever rendered.

## Running the worker

```bash
cd backend
venv/Scripts/python.exe -m alembic upgrade head
# drain background jobs (single process):
venv/Scripts/python.exe - <<'PY'
from app.main import app  # noqa
from app.services.worker_platform import run_once
run_once()  # claims/executes one batch of due jobs; call repeatedly or loop
PY
```

Operator backfill (never destructive by default):

```bash
cd backend
venv/Scripts/python.exe scripts/backfill_workspaces.py --dry-run --batch-size 50
venv/Scripts/python.exe scripts/backfill_workspaces.py --batch-size 50   # explicit run
```

## Tests

- `tests/test_phase16_worker.py` — 34 tests: queue lifecycle, atomic claiming,
  heartbeats, stale recovery, fairness, metrics, execution worker resume/cancel.
- `tests/test_phase16_provider.py` — 39 tests: gateway behavior, capability
  registry, routing, circuit breaker, fallback, vector compat/dimensions, cache.
- `tests/test_phase16_knowledge.py` — 91 tests: multimodal pages, policy semantics
  and contradictions, memory 2.0, RAG 4.0 evidence/repair, entity 3.0, backfill,
  approvals, traces.
- `tests/test_phase16_api.py` — 48 tests: API auth/authorization gates, SSRF webhook
  guards, error semantics (401/403/404/409/422), pagination bounds.

All 212 Phase 16 tests pass with deterministic fake providers; no credentials needed.
