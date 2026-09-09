# Autonomy Runbook (Phase 21)

The governed-autonomy lifecycle. Every step is audited; simulation and
activation never happen without explicit records.

## Lifecycle: OBSERVE → RECOMMEND → SIMULATE → APPROVE → ACTIVATE → MONITOR → ROLLBACK

1. **OBSERVE** — create the policy at the most restrictive level:

   ```
   POST /ops21/autonomy/policies
   { "workspace_id": W, "operation_type": "knowledge.recovery.embedding_gap",
     "autonomy_level": "OBSERVE" }
   ```

2. **RECOMMEND** — once signals look stable, advance one level at a time
   (skip-level promotion is rejected server-side):

   ```
   POST /ops21/autonomy/level
   { "workspace_id": W, "policy_id": P, "new_level": "RECOMMEND",
     "reason": "2 clean weeks" }
   ```

3. **SIMULATE** — zero-side-effect preview of the guard decision:

   ```
   POST /ops21/autonomy/simulate
   { "workspace_id": W, "operation_type": "knowledge.recovery.embedding_gap",
     "risk_level": "LOW" }
   ```

   Response includes `would_auto_execute`, the would-be decision, and the
   reason. No rows are written.

4. **APPROVE** — for candidate activations, gates must pass first
   (`/ops21/candidates/{id}/evaluate`), then promotion goes through the
   guard; under `RECOMMEND`/`AUTO_APPROVAL` policies the decision is
   `REQUIRES_APPROVAL` until an operator executes it via an `ALLOWED` policy.

5. **ACTIVATE** — the guard records the decision and the execution result
   (`execute_allowed` persists result + rollback info). Idempotency keys make
   retries safe.

6. **MONITOR** — `GET /ops21/autonomy/operations` is the audit surface: every
   decision with actor, source (AI/SYSTEM/OPERATOR), and reason.

7. **ROLLBACK** — executed operations carry rollback info; the contract is
   `mark_rolled_back` (only `SUCCEEDED` operations can roll back).

## Emergency stop

```
POST /ops21/safety/emergency-stop
{ "workspace_id": W, "scope": "ALL" | "AI_ACTIONS" | "AGENTS" |
  "WORKFLOWS" | "AUTONOMOUS_RECOVERY", "reason": "..." }
POST /ops21/safety/emergency-stop/{stop_id}/lift?workspace_id=W
```

An active stop is an absolute veto over automatic execution for the covered
subsystem (simulation reports the blocked decision). Lifting requires an
owner and is audited.

## Level semantics

| Level | Automatic execution |
|---|---|
| OBSERVE | never — record only |
| RECOMMEND | never — decisions are proposals |
| AUTO_LOW_RISK | only LOW-risk operations the policy covers |
| AUTO_APPROVAL | LOW + MEDIUM risk within pre-approved scope |
| MANUAL_ONLY | never — operator action required |

Transitions allowed: OBSERVE→{RECOMMEND, MANUAL_ONLY};
RECOMMEND→{OBSERVE, AUTO_LOW_RISK, MANUAL_ONLY};
AUTO_LOW_RISK→{RECOMMEND, AUTO_APPROVAL, MANUAL_ONLY};
AUTO_APPROVAL→{AUTO_LOW_RISK, MANUAL_ONLY}; MANUAL_ONLY→{OBSERVE, RECOMMEND}.
