# Incident Runbook

DocuFlow correlates related failures into incidents and drives them through
`OPEN → INVESTIGATING → MITIGATED → RESOLVED → POSTMORTEM`. Every transition
is validated server-side and appended to the incident timeline.

Related API: `POST /ops20/incidents`, `GET /ops20/incidents`,
`POST /ops20/incidents/{id}/transition`, `GET /ops20/incidents/{id}/timeline`.

---

## 1. Detection

- Alerts are prioritized (`informational` → `critical`) and deduplicated by
  fingerprint; repeated alerts increment `occurrence_count` and escalate
  severity (low→medium at 3 repeats, medium→high at 5, high→critical at 8).
  `GET /ops20/alerts` and `GET /ops20/alerts/summary`.
- Failures with the same affected systems + summary correlate into a single
  open incident (new events append rather than storming).
- System health: `GET /ops20/health/score?factors={…}` (availability,
  latency, quality, cost, error rate, queue health, provider health);
  `GET /ops20/dependency-graph` shows service dependencies for blast-radius
  reasoning.

## 2. Common incident playbooks

### 2.1 Provider outage (503/429/latency spike)

1. Check provider profiles + anomalies:
   `GET /ops20/providers/profiles`, `POST /ops20/providers/anomalies/detect`.
2. Confirm routing fallback is healthy (Phase 18 provider platform fallback
   chain; circuit breaker state under `/ops19`).
3. Open an incident on `provider`, transition to `INVESTIGATING`.
4. Watch error budgets: `GET /ops20/slo/error-budget?name=…` — an exhausted
   budget recommends a freeze for the SLO's own scope and requires operator
   review; it never auto-blocks unrelated operations.
5. After recovery, transition `MITIGATED → RESOLVED`; write the postmortem
   (`POSTMORTEM`).

### 2.2 Database incident

1. `GET /ops20/database/health` — latest health metrics, migration head,
   high-growth tables.
2. Compare query latency against baseline (`dbi.query_regression` in tests /
   replay via `POST /ops20/database/metrics`).
3. If schema/migration trouble: `alembic heads` must show the single head
   `027_phase20_knowledge_os`; never hand-edit `alembic_version`.

### 2.3 Worker / broker degradation

1. Phase 19 worker fleet + quarantine readouts (`/ops19/workers`,
   `/ops19/quarantines`, `/ops19/broker`).
2. Check queue health and dead letters (`/ops19/health`, `/ops19/broker`).
3. Quarantined workers recover automatically per their recovery criteria, or
   via operator override.

### 2.4 Quality regression

1. `GET /ops20/quality/scorecards`, `POST /ops20/quality/regression` —
   regressions surface but never auto-deploy.
2. Knowledge health: `GET /ops20/knowledge/health`, `/ops20/knowledge/gaps`.
3. Feedback: `GET /ops20/feedback/summary` and flagged feedback
   (`/ops20/feedback/quality`).
4. If a live improvement caused the regression, roll it back
   (`POST /ops20/improvements/{id}/transition` → `ROLLED_BACK`) per the
   self-improvement runbook.

### 2.5 Security incident

1. Rerun the safety suite: `/ops20/safety/injection-corpus`,
   `/ops20/safety/exfiltration`, `/ops20/safety/tool-abuse`.
2. API abuse signals: `GET /ops20/api/abuse` (enumeration, scope probing,
   abnormal key volume); check `GET /ops20/api/health` for auth-failure
   spikes.
3. Raise a `security` alert and route to the security owner; do not paste
   secrets or raw credentials anywhere in incident notes.

## 3. Timeline hygiene

- Every transition writes an event with actor, timestamp, and detail.
- Use `detail` to record evidence (alert IDs, run IDs, evaluation refs) —
  never hidden chain-of-thought and never secrets.
- Keep the timeline append-only: use `note`-style events by correlating the
  same incident rather than editing history.

## 4. Postmortem

- Transition `RESOLVED → POSTMORTEM` with a documented summary.
- Reference the rollback (if any) from the improvement lifecycle and the
  governed policy change that reverted production configuration
  (`GET /ops20/governance/audit`).
- Publish an immutable versioned report (`POST /ops20/reports`, kind
  `reliability` or `ai_quality`) capturing the incident's measurements.

## 5. Communication

- Operators see live status in `/ops` and `/ai-control`.
- Per-category notification preferences control who is notified
  (`POST /ops20/notifications/preferences`); escalation rules raise severity
  for repeated alerts automatically.
