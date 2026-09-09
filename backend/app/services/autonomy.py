"""Phase 21 — Autonomous operations control plane.

Governed autonomy: persisted policies per (workspace, operation type, risk
level), five autonomy levels with server-validated transitions, a centralized
action guard that no subsystem may bypass, zero-side-effect simulation, and a
full audit trail for every decision.

The guard is the ONLY path by which any subsystem may execute an autonomous
operation. It resolves policy, checks emergency stop, enforces execution
limits + cooldowns + budget, and records the decision — even when blocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models.phase21 import (
    AutonomyPolicy, AutonomyTransition, AutonomousOperation, EmergencyStop,
)

AUTONOMY_LEVELS = ["OBSERVE", "RECOMMEND", "AUTO_LOW_RISK", "AUTO_APPROVAL",
                   "MANUAL_ONLY"]

# Server-validated level transitions. Progression to more autonomy is
# always explicit; demotion is always allowed.
_ALLOWED_LEVEL_TRANSITIONS = {
    "OBSERVE": {"RECOMMEND", "MANUAL_ONLY"},
    "RECOMMEND": {"OBSERVE", "AUTO_LOW_RISK", "MANUAL_ONLY"},
    "AUTO_LOW_RISK": {"RECOMMEND", "AUTO_APPROVAL", "MANUAL_ONLY"},
    "AUTO_APPROVAL": {"AUTO_LOW_RISK", "MANUAL_ONLY"},
    "MANUAL_ONLY": {"OBSERVE", "RECOMMEND"},
}

# Risk levels eligible for automatic execution under AUTO_LOW_RISK.
_AUTO_LOW_RISK_LEVELS = {"LOW"}
_AUTO_APPROVAL_LEVELS = {"LOW", "MEDIUM"}

RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

_DECISIONS = ("ALLOWED", "REQUIRES_APPROVAL", "BLOCKED")


def _bounded_json(value: Any, limit: int = 6000) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


def get_policy(db: Session, workspace_id: int, operation_type: str,
               risk_level: str = "LOW") -> Optional[AutonomyPolicy]:
    """Resolve the most specific policy for an operation.

    Falls back: exact (op, risk) -> (op, any risk) -> None.
    """
    row = (db.query(AutonomyPolicy)
           .filter_by(workspace_id=workspace_id, operation_type=operation_type,
                      risk_level=risk_level)
           .first())
    if row is not None:
        return row
    return (db.query(AutonomyPolicy)
            .filter_by(workspace_id=workspace_id,
                       operation_type=operation_type)
            .order_by(AutonomyPolicy.id.asc())
            .first())


def create_policy(db: Session, workspace_id: int, operation_type: str,
                  risk_level: str = "LOW",
                  autonomy_level: str = "RECOMMEND",
                  requires_approval: bool = True,
                  budget_limit_usd: Optional[float] = None,
                  execution_limit_per_hour: Optional[int] = None,
                  cooldown_seconds: int = 300,
                  requires_audit: bool = True,
                  organization_id: Optional[int] = None,
                  created_by: Optional[int] = None,
                  actor: str = "operator") -> AutonomyPolicy:
    """Create (or update) an autonomy policy with an audited transition."""
    if autonomy_level not in AUTONOMY_LEVELS:
        raise ValueError(f"invalid autonomy level: {autonomy_level}")
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"invalid risk level: {risk_level}")
    row = (db.query(AutonomyPolicy)
           .filter_by(workspace_id=workspace_id,
                      operation_type=operation_type, risk_level=risk_level)
           .first())
    previous = row.autonomy_level if row else None
    if row is None:
        row = AutonomyPolicy(
            workspace_id=workspace_id, organization_id=organization_id,
            operation_type=operation_type, risk_level=risk_level,
            autonomy_level=autonomy_level, requires_approval=requires_approval,
            budget_limit_usd=budget_limit_usd,
            execution_limit_per_hour=execution_limit_per_hour,
            cooldown_seconds=cooldown_seconds,
            requires_audit=requires_audit, created_by=created_by)
        db.add(row)
        db.flush()
    else:
        row.autonomy_level = autonomy_level
        row.requires_approval = requires_approval
        row.budget_limit_usd = budget_limit_usd
        row.execution_limit_per_hour = execution_limit_per_hour
        row.cooldown_seconds = cooldown_seconds
        row.requires_audit = requires_audit
        db.flush()
    db.add(AutonomyTransition(
        policy_id=row.id, workspace_id=workspace_id,
        previous_level=previous, new_level=autonomy_level, actor=actor,
        reason="policy_created" if previous is None else "policy_updated"))
    db.commit()
    return row


def set_autonomy_level(db: Session, policy: AutonomyPolicy, new_level: str,
                       actor: str = "operator",
                       reason: Optional[str] = None) -> AutonomyPolicy:
    """Validate and persist an autonomy level transition (server-side only)."""
    if new_level not in AUTONOMY_LEVELS:
        raise ValueError(f"invalid autonomy level: {new_level}")
    if new_level == policy.autonomy_level:
        return policy
    allowed = _ALLOWED_LEVEL_TRANSITIONS.get(policy.autonomy_level, set())
    if new_level not in allowed:
        raise ValueError(
            f"transition {policy.autonomy_level} -> {new_level} not allowed")
    previous = policy.autonomy_level
    policy.autonomy_level = new_level
    db.flush()
    db.add(AutonomyTransition(
        policy_id=policy.id, workspace_id=policy.workspace_id,
        previous_level=previous, new_level=new_level, actor=actor,
        reason=reason))
    db.commit()
    return policy


def emergency_stop_active(db: Session, workspace_id: int,
                          scope: str = "ALL") -> bool:
    """True when an active emergency stop covers the requested scope."""
    rows = (db.query(EmergencyStop)
            .filter_by(workspace_id=workspace_id, active=True).all())
    for row in rows:
        if row.scope == "ALL" or row.scope == scope:
            return True
    return False


def _stop_scope_for(operation_type: str) -> str:
    """Map an operation type to its emergency-stop subsystem scope."""
    if operation_type.startswith("agent."):
        return "AGENTS"
    if operation_type.startswith("workflow."):
        return "WORKFLOWS"
    recovery_markers = ("recovery", "adaptation.", "routing.",
                        "cost.optimize.", "graph.repair.")
    if any(marker in operation_type for marker in recovery_markers):
        return "AUTONOMOUS_RECOVERY"
    return "AI_ACTIONS"


def _within_execution_limit(db: Session, workspace_id: int,
                            operation_type: str,
                            limit: Optional[int]) -> tuple[bool, int]:
    recent = (db.query(func.count(AutonomousOperation.id))
              .filter(AutonomousOperation.workspace_id == workspace_id,
                      AutonomousOperation.operation_type == operation_type,
                      AutonomousOperation.simulated.is_(False),
                      AutonomousOperation.created_at
                      >= datetime.now(timezone.utc) - timedelta(hours=1))
              .scalar() or 0)
    if limit is not None and recent >= limit:
        return False, int(recent)
    return True, int(recent)


def _within_cooldown(db: Session, workspace_id: int, operation_type: str,
                     cooldown_seconds: int) -> bool:
    if cooldown_seconds <= 0:
        return True
    last = (db.query(AutonomousOperation)
            .filter(AutonomousOperation.workspace_id == workspace_id,
                    AutonomousOperation.operation_type == operation_type,
                    AutonomousOperation.simulated.is_(False),
                    AutonomousOperation.decision == "ALLOWED",
                    AutonomousOperation.created_at
                    >= datetime.now(timezone.utc)
                    - timedelta(seconds=cooldown_seconds))
            .first())
    return last is None


def simulate_operation(db: Session, workspace_id: int, operation_type: str,
                       risk_level: str = "LOW",
                       estimated_cost_usd: float = 0.0) -> dict:
    """Zero-side-effect evaluation of what the guard would decide.

    Answers exactly one operator question: would this operation be
    automatically allowed under the current policy?
    """
    policy = get_policy(db, workspace_id, operation_type, risk_level)
    level = policy.autonomy_level if policy else "RECOMMEND"
    stop = emergency_stop_active(db, workspace_id,
                                 _stop_scope_for(operation_type))

    checks: dict[str, bool] = {}
    checks["no_emergency_stop"] = not stop
    if level == "OBSERVE" or level == "RECOMMEND":
        auto = False
        reason = (f"autonomy level {level} does not execute operations")
    elif level == "MANUAL_ONLY":
        auto = False
        reason = "autonomy level MANUAL_ONLY requires explicit operator action"
    elif level == "AUTO_LOW_RISK":
        auto = risk_level in _AUTO_LOW_RISK_LEVELS
        reason = ("low-risk automatic execution" if auto
                  else f"risk {risk_level} exceeds AUTO_LOW_RISK scope")
    elif level == "AUTO_APPROVAL":
        auto = risk_level in _AUTO_APPROVAL_LEVELS
        reason = ("pre-approved risk scope" if auto
                  else f"risk {risk_level} requires explicit approval")
    else:  # pragma: no cover - level set is closed
        auto = False
        reason = "unknown autonomy level"

    # Emergency stop is an absolute veto over automatic execution.
    if stop and auto:
        auto = False
        reason = "emergency stop active for this subsystem"

    if policy is not None:
        if policy.budget_limit_usd is not None \
                and estimated_cost_usd > policy.budget_limit_usd:
            auto = False
            reason = (f"estimated cost ${estimated_cost_usd:.2f} exceeds "
                      f"budget limit ${policy.budget_limit_usd:.2f}")
        limit_ok, recent = _within_execution_limit(
            db, workspace_id, operation_type, policy.execution_limit_per_hour)
        checks["execution_limit"] = limit_ok
        if not limit_ok:
            auto = False
            reason = (f"execution limit reached ({recent}/"
                      f"{policy.execution_limit_per_hour} per hour)")
        checks["cooldown_clear"] = _within_cooldown(
            db, workspace_id, operation_type, policy.cooldown_seconds)
        if not checks["cooldown_clear"]:
            auto = False
            reason = "cooldown window active"
    checks["risk_in_policy_scope"] = auto or level in ("OBSERVE", "RECOMMEND",
                                                       "MANUAL_ONLY")
    return {
        "operation_type": operation_type,
        "risk_level": risk_level,
        "policy_id": policy.id if policy else None,
        "policy_level": level,
        "would_auto_execute": auto,
        "decision": "ALLOWED" if auto else (
            "REQUIRES_APPROVAL" if level in ("RECOMMEND", "AUTO_LOW_RISK",
                                             "AUTO_APPROVAL") else "BLOCKED"),
        "reason": reason,
        "checks": checks,
        "simulated": True,
    }


def guard_operation(db: Session, workspace_id: int, operation_type: str,
                    risk_level: str = "LOW", actor: str = "system",
                    source: str = "SYSTEM",
                    input_payload: Optional[dict] = None,
                    estimated_cost_usd: float = 0.0,
                    idempotency_key: Optional[str] = None,
                    simulated: bool = False) -> AutonomousOperation:
    """Centralized autonomy enforcement — the only execution path.

    Records the decision (allowed, approval-required, or blocked) with full
    audit context. Repeated calls with the same idempotency key return the
    original decision instead of duplicating side effects.
    """
    key = idempotency_key or (
        f"{operation_type}:{workspace_id}:{actor}:{risk_level}")
    existing = (db.query(AutonomousOperation)
                .filter_by(workspace_id=workspace_id, idempotency_key=key)
                .first())
    if existing is not None:
        return existing

    simulation = simulate_operation(db, workspace_id, operation_type,
                                    risk_level, estimated_cost_usd)
    decision = simulation["decision"]
    reason = simulation["reason"]
    policy = get_policy(db, workspace_id, operation_type, risk_level)

    # Simulation-only calls never persist as real operations.
    op = AutonomousOperation(
        workspace_id=workspace_id, operation_type=operation_type,
        risk_level=risk_level, actor=actor, source=source,
        policy_id=policy.id if policy else None,
        policy_level=simulation["policy_level"],
        input_payload=_bounded_json(input_payload),
        decision=decision, decision_reason=reason,
        status="DECIDED", simulated=simulated,
        idempotency_key=key)
    db.add(op)
    db.commit()
    return op


def execute_allowed(db: Session, op: AutonomousOperation,
                    result: Optional[dict] = None,
                    rollback_info: Optional[dict] = None,
                    status: str = "SUCCEEDED") -> AutonomousOperation:
    """Record the execution result of an ALLOWED autonomous operation."""
    if op.decision != "ALLOWED":
        raise ValueError("only ALLOWED operations can be executed")
    op.status = status
    op.result = _bounded_json(result)
    op.rollback_info = _bounded_json(rollback_info)
    db.commit()
    return op


def mark_rolled_back(db: Session, op: AutonomousOperation,
                     reason: Optional[str] = None) -> AutonomousOperation:
    """Mark an executed operation rolled back (reversibility contract)."""
    if op.status != "SUCCEEDED":
        raise ValueError("only SUCCEEDED operations can be rolled back")
    op.status = "ROLLED_BACK"
    if reason:
        op.decision_reason = f"rolled back: {reason}"[:500]
    db.commit()
    return op


def list_operations(db: Session, workspace_id: int, limit: int = 50,
                    offset: int = 0,
                    operation_type: Optional[str] = None) -> list[AutonomousOperation]:
    """Bounded, paginated operation history (audit surface)."""
    limit = max(1, min(int(limit), 200))
    q = db.query(AutonomousOperation).filter_by(workspace_id=workspace_id)
    if operation_type:
        q = q.filter(AutonomousOperation.operation_type == operation_type)
    return (q.order_by(AutonomousOperation.id.desc())
            .offset(max(0, int(offset))).limit(limit).all())
