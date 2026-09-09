# Self-Improvement Runbook

How a change moves through the governed self-improvement lifecycle — from a
proposal to an actively deployed (and if needed rolled back) configuration.
**AI may propose, evaluate, simulate, and prepare. Only authorized humans
promote to production.**

Lifecycle: `PROPOSED → EVALUATING → APPROVAL_REQUIRED → APPROVED → STAGED →
ACTIVE` with terminal/side exits `SUPERSEDED`, `ROLLED_BACK`, `REJECTED`.

---

## 1. Proposal

**Who:** any user (human-authored) or the system itself (AI-generated).

**How:** `POST /ops20/improvements` with `domain`, `title`, `problem`,
`proposed_change`, optional evidence/expected benefit/risk/estimated cost/
evaluation requirements, and `author_source` (`human` or `ai`).

**Guarantees:** every proposal starts `PROPOSED`. AI proposals never
auto-activate. Creation writes the first audit row.

**Checks before continuing**

```bash
curl -s -b cookies "$API/ops20/improvements/{id}/audit"   # actor + reason trail
```

## 2. Evaluation

Transition to `EVALUATING` when a reviewer starts work.

Run an offline evaluation instead of touching production:

1. Create a dataset — `POST /ops20/datasets` (kind `golden` / `synthetic` /
   `anonymized` / `curated`).
2. Create an experiment — `POST /ops20/experiments` (config is frozen by a
   config fingerprint).
3. Record baseline and candidate runs — `POST /ops20/experiments/{id}/runs`.
4. Compare — `POST /ops20/experiments/{id}/compare` (deterministic paired
   deltas; results below the minimum sample size are `INCONCLUSIVE`).
5. Check the promotion gate — `GET /ops20/experiments/{id}/gate?min_quality=…`
   (quality/cost/latency thresholds over completed runs).

Evaluation must be side-effect free: experiments never read or write
production behavior.

## 3. Approval

Transition `APPROVAL_REQUIRED → APPROVED` only with an explicit human
decision and a recorded reason. For governance-sensitive changes also record
the change in `POST /ops20/governance/changes` (owner, reason, policy, and
optional evaluation/approval references) and verify the versioned policy
diff (`GET /ops20/governance/audit`).

Use the no-side-effect simulator first when in doubt:
`POST /ops20/governance/simulate` (or `/ops20/policy/simulate`) answers
"would this operation be allowed under the policy?" — it never mutates
anything.

## 4. Staging

Transition `APPROVED → STAGED`. In staging:

- Run the change against synthetic/redacted workloads only.
- Compare quality, latency, cost, and citation correctness against the
  baseline (see the AI quality scorecards and reports: `POST /ops20/reports`).
- Confirm the AI safety scorecard is unaffected
  (`GET /ops20/safety/injection-corpus`, `/ops20/safety/exfiltration`).

## 5. Activation

Transition `STAGED → ACTIVE`. Activation is the only step that changes
production behavior and it must have been preceded by the full approval path
(server-side validation rejects any shortcut).

After activation:

- Watch the AI control center (`/ai-control`): system health, SLO history,
  error budgets, incidents, and alerts.
- Watch per-agent/per-workflow intelligence readouts
  (`/ops20/agents/intelligence`, `/ops20/workflows/intelligence`) for
  regressions in success rate, latency, or cost.
- Watch cost drift and token efficiency (`/ops20/cost/drift`,
  `/ops20/cost/token-efficiency`).

## 6. Monitoring

- Quality regressions are detected automatically
  (`/ops20/quality/regression`) but **never auto-deploy fixes** — they raise
  deduplicated alerts (`/ops20/alerts`).
- Knowledge drift and gaps surface under `/ops20/knowledge/*`.
- Incident correlation groups related failures
  (`POST /ops20/incidents`, then `POST /ops20/incidents/{id}/transition`
  through `INVESTIGATING → MITIGATED → RESOLVED`).

## 7. Rollback

Transition `ACTIVE → ROLLED_BACK` with the reason (e.g. quality regression,
budget breach, incident).

Rollback rules:

- Rollback is authorized, audited (actor + timestamp + reason recorded in the
  improvement audit trail), and idempotent — a second transition to
  `ROLLED_BACK` is a no-op or invalid, never a double effect.
- If the change was a governed policy, record the reverted version in
  `POST /ops20/governance/changes`; the human-readable diff shows exactly
  what changed back.
- Re-open or reference the related incident for the post-incident review
  (`POST /ops20/incidents/{id}/transition` to `POSTMORTEM`).

## 8. Review

- Terminal states: `ROLLED_BACK`, `REJECTED`, `SUPERSEDED`.
- Every lifecycle step is visible in
  `GET /ops20/improvements/{id}/audit` — no silent production changes.

---

## Escalation quick reference

| Symptom | Action |
|---------|--------|
| Experiment keeps returning INCONCLUSIVE | Increase run count / sample size before claiming significance |
| Gate says not eligible | Compare run metrics against the gate thresholds |
| Proposal stuck in APPROVAL_REQUIRED | Missing human approval — do not shortcut the transition |
| Regression after activation | Roll back with reason; open an incident; rerun the evaluation |
| Alerts repeating | Acknowledge or resolve, adjust notification preference, investigate root cause |
