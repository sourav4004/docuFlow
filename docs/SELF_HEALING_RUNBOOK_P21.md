# Self-Healing Runbook (Phase 21)

All automatic recovery is governed by autonomy policy. If any scenario loops
or escalates unexpectedly, activate the emergency stop first, then diagnose.

## Preconditions

- Workspace autonomy policy for `recovery` exists (`AUTO_LOW_RISK` enables
  automatic recovery for approved low-risk playbooks).
- Playbook is approved and risk `LOW`; destructive actions can never be in a
  playbook (service refuses them).
- Attempts are idempotent (safe to retry), cooldown-gated, and escalate after
  `max_attempts`.

## Scenario: worker stall

1. Detect: `POST /ops21/health/failures` with
   `{"worker": {"heartbeat_age_seconds": 400}}` → `worker_stall`.
2. Diagnose: `POST /ops21/diagnosis` with the stall signals → ranked
   hypotheses with evidence/confidence.
3. Recover: approved playbook `restart_worker_state` (LOW risk) runs
   automatically under an `AUTO_LOW_RISK` recovery policy; otherwise the
   attempt is `POLICY_BLOCKED` and awaits an operator.
4. Verify: `POST /ops21/workers/health` returns `HEALTHY` after logical
   recovery; `GET /ops21/recovery/attempts` shows the audit.

## Scenario: broker degradation

1. Detect: `POST /ops21/broker/health` with reconnect/timeout counts →
   `DEGRADED` snapshot plus a recovery plan.
2. Recover: `reconnect_broker` / `requeue_safe_jobs` (safe-auto set).
   Duplicate consumption is verified idempotent.
3. Escalate: repeated failure → attempt `ESCALATED` → operator incident.

## Scenario: scheduler leader loss

1. Detect: `POST /ops21/scheduler/health` with `has_leader=false` →
   `DEGRADED`.
2. Recovery is bounded: missed schedules are recovered up to a hard cap;
   duplicate firing is prevented by leader deduplication.

## Scenario: ingestion degradation

1. Detect: `POST /ops21/ingestion/anomalies` compares the newest quality
   sample against the recent window (drop ≥ threshold → anomaly).
2. Recommend: an ingestion adaptation candidate is generated (proposal only).
3. Evaluate + promote: `/ops21/candidates/{id}/evaluate` then
   `/ops21/candidates/{id}/promote` — promotion requires passing gates AND an
   `ALLOWED` autonomy decision.

## Scenario: knowledge embedding gaps

1. Detect: `POST /ops21/knowledge/health` (embedding coverage < 80) and
   `POST /ops21/knowledge/incidents`.
2. Plan: `POST /ops21/knowledge/recovery-plans` (proposal, LOW risk for
   `rebuild_embeddings`).
3. Execute: `/ops21/knowledge/recovery-plans/{id}/execute` — governed by the
   guard; audited via `ops.knowledge_recovery` events.

## Escalation contract

- Repeated recovery attempts beyond `max_attempts` produce an `ESCALATED`
  attempt — automatic looping is impossible.
- Severe failures create incidents automatically (threshold rules), and
  incidents accumulate timeline entries (detection → diagnosis → recovery →
  resolution) plus a postmortem draft pending human review.

## Cache invalidation

`POST /ops21/cache/invalidate` is tenant-safe: only keys prefixed with the
caller's workspace id are invalidated; foreign keys are reported and skipped.
