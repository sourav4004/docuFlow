# FINAL ACTIVATION CHECKLIST (post-Phase 24)

Everything in this list is intentionally **deferred** because the target
infrastructure does not exist in the current environment. Each item lists the
exact activation steps, the tests that exercise it, and the honest status
labels to use afterwards. Do not mark any item complete without real
infrastructure evidence.

Status vocabulary: `REAL` (validated against the actual infrastructure),
`FALLBACK` (deterministic substitute validated), `UNAVAILABLE`
(infrastructure absent; never claim validated).

Priority order below is the execution order for the final cycle.

---

## 1. pgvector activation  — current: UNAVAILABLE

Local PostgreSQL 18.4 has no `vector` extension in `pg_available_extensions`.

Steps:
1. Install the pgvector extension matching the PostgreSQL build
   (Windows: use a build with pgvector, or move to a Linux/容器 image).
2. `CREATE EXTENSION vector;` on the DocuFlow database.
3. Run `alembic upgrade head` — the vector column migration applies natively.
4. Run the gated suites — they self-enable, no code change:
   - `tests/test_vector_search.py`
   - `tests/test_hybrid_retrieval.py`
   - `tests/test_retrieval.py`
   - `tests/test_retrieval_quality.py`
   - `tests/test_message_sources.py`
   (gate logic: `tests/conftest.py`, marker `env_pgvector`)
5. Run Phase 22 `vector_activation` benchmark; record `backend: REAL`.

Then label: `REAL` for native operators; keep `FALLBACK` claims only for the
JSON path.

## 2. Redis activation — current: UNAVAILABLE

Steps:
1. Provision Redis (or Valkey) and set `REDIS_URL`
   (`redis://` or `rediss://`; credentials only via the URL).
2. Set `WORKER_BROKER=redis` and `BROKER_FAILOVER_AUTO=false` initially.
3. Validate: `tests/test_phase18_worker.py` Redis server contract +
   `tests/test_phase24_reliability.py::TestBrokerContract` (the unavailable
   path auto-skips when Redis is reachable).
4. Run the Phase 23 broker migration dry-run, then a supervised switch.
5. Verify duplicate-protection: `dedupe_key` idempotency across a
   postgres→redis switch.

Then label: `REAL`. Until then PostgreSQL broker is authoritative (`REAL`).

## 3. Real AI providers — current: UNAVAILABLE (fake provider validated)

Steps:
1. Configure provider credentials via environment variables only — never in
   the DB or code. Verify `configured_providers()` reports them without
   printing secrets.
2. Run the Phase 23 provider production gate: completion, streaming,
   embeddings, structured output, tool calling, multimodal.
3. Run the failure matrix against the real API (429/timeout/5xx classes).
4. Record readiness scores with `simulated: false`.
5. Confirm the AI data boundary (sensitivity/residency checks) applies to
   the real provider ids.

Then label: `REAL`. The deterministic fake provider stays the default for
dev/test (`FALLBACK` path, validated).

## 4. Object storage (S3) activation — current: local storage validated, S3 UNAVAILABLE

Steps:
1. Set `STORAGE_BACKEND=s3` plus bucket/region/credentials via environment.
2. Run the Phase 23 storage migration plan dry-run, then batched migration
   with checksum verification.
3. Validate signed URLs, multipart-ready uploads, orphan detection.
4. Verify restore/verification flow and audit trail.

Then label: `REAL` for S3; local disk path remains `REAL` for development.

## 5. Production infrastructure & multi-region — current: SIMULATED

Steps:
1. Provision separate region deployments; register them in the region
   registry with residency rules.
2. Run residency guard tests against real provider/storage regions.
3. Run failover simulation first, then a governed, approved, drained
   failover exercise with RTO/RPO measurement.
4. Only after a real exercise: label `REAL`; otherwise `SIMULATED`.

## 6. Remaining legacy test cleanup — current: DONE in Phase 24

All 36 pre-existing failures were fixed or explicitly gated in Phase 24:
28 pgvector-gated (self-enable per item 1), 6 status updates, 2
shared-identity fixes. No legacy skips remain besides the two documented
environment-conditional skips (`redis installed` adapter test).

## 7. Full regression gate (rerun after each activation step)

```bash
cd backend
./run_tests.sh              # exit code is the authoritative result
```

Requirements: zero failures; skips only from the gates in items 1–4 while
their infrastructure is absent; classification of any new failure.

## 8. Frontend final polish & browser E2E — browser automation UNAVAILABLE

Steps:
1. Run the app locally (`uvicorn` + `next dev`) and walk the E2E product
   flows list (signup→upload→search→conversation→citations→collections→
   versions→AI suggestion→approval→workflow dry-run→notification→export→
   delete/restore).
2. Add Playwright (dev-dependency only) and codify the same flows.
3. Keyboard/focus/screen-reader pass on the converted pages.
4. Visual polish pass: spacing/typography consistency on /ops and
   /ai-control dense panels.

Then label: `REAL` for browser E2E; unit/contract coverage is already `REAL`.

## 9. Load / soak — current: bounded harnesses validated

Steps:
1. Run the Phase 22 chaos harness + load profiles against production-shaped
   infrastructure for ≥24h.
2. Record real throughput/latency curves; compare against SLO thresholds.
3. Only then label `REAL`; otherwise `BOUNDED LOCAL VALIDATION`.

## 10. Final security audit & deployment

1. Re-run secret scan (`grep` corpus in PHASE24_OPERATIONS.md).
2. Re-run the security suites (Phase 24 + Phases 20–23 corpora).
3. Rotate any credential that ever touched a non-production store.
4. Execute the deployment checklist in PHASE24_OPERATIONS.md §Deployment.
5. Verify liveness/readiness probes, structured errors, correlation IDs in
   the deployed environment.
