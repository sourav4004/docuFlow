"""Automation engine 2.0 — workflow transaction safety, compensation,
failure recovery, simulator, version diff, and risk classification.

The workflow engine itself stays declarative; this service adds the durable
execution semantics: per-node execution records (never double-run), side
effect compensation metadata (reversible vs irreversible), bounded retries,
a side-effect-free simulator, version diffs, and deterministic risk
classification.
"""

import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase15 import (
    WorkflowNodeExecution, WorkflowCompensation, ProviderHealth,
)
from ..models.workflow_version import WorkflowVersion
from ..services.audit_service import log_audit_event

RETRYABLE_ERROR_MARKERS = ("timeout", "temporary", "rate limit", "connection", "5")
MAX_AUTO_RETRIES = 3


class WorkflowReliabilityError(Exception):
    """Raised on invalid reliability operations."""


# ---------------------------------------------------------------------------
# Transaction safety — per-node execution records
# ---------------------------------------------------------------------------

def begin_node_execution(
    db: Session,
    workspace_id: int,
    workflow_execution_id: str,
    workflow_id: str,
    node_id: str,
    input_data: Optional[dict] = None,
) -> WorkflowNodeExecution:
    """Record a node attempt.

    The unique (execution, node, attempt) constraint guarantees a node can
    never be executed twice unintentionally: a retry must explicitly open a
    NEW attempt.
    """
    existing = (
        db.query(WorkflowNodeExecution)
        .filter(
            WorkflowNodeExecution.workflow_execution_id == workflow_execution_id,
            WorkflowNodeExecution.node_id == node_id,
            WorkflowNodeExecution.status.in_(("RUNNING", "COMPLETED")),
        )
        .first()
    )
    if existing:
        raise WorkflowReliabilityError(
            f"Node {node_id} already executed in this run (status={existing.status})"
        )

    attempt = 1
    last = (
        db.query(WorkflowNodeExecution)
        .filter(
            WorkflowNodeExecution.workflow_execution_id == workflow_execution_id,
            WorkflowNodeExecution.node_id == node_id,
        )
        .order_by(WorkflowNodeExecution.attempt.desc())
        .first()
    )
    if last:
        attempt = last.attempt + 1

    record = WorkflowNodeExecution(
        workflow_execution_id=workflow_execution_id,
        workflow_id=workflow_id,
        workspace_id=workspace_id,
        node_id=node_id,
        attempt=attempt,
        input_hash=hashlib.sha256(json.dumps(input_data or {}, sort_keys=True, default=str).encode()).hexdigest(),
        status="RUNNING",
        started_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.flush()
    return record


def complete_node_execution(
    db: Session,
    record: WorkflowNodeExecution,
    output_reference: Optional[str] = None,
) -> WorkflowNodeExecution:
    record.status = "COMPLETED"
    record.output_reference = output_reference
    record.completed_at = datetime.now(timezone.utc)
    db.flush()
    return record


def fail_node_execution(db: Session, record: WorkflowNodeExecution, error: str) -> WorkflowNodeExecution:
    record.status = "FAILED"
    record.error_message = error
    record.completed_at = datetime.now(timezone.utc)
    db.flush()
    return record


# ---------------------------------------------------------------------------
# Compensation metadata
# ---------------------------------------------------------------------------

REVERSIBLE_SIDE_EFFECTS = ("metadata_update", "document_move")
IRREVERSIBLE_SIDE_EFFECTS = ("notification", "email", "webhook", "delete", "external_write")


def record_side_effect(
    db: Session,
    workspace_id: int,
    workflow_execution_id: str,
    node_id: str,
    side_effect_type: str,
    metadata: Optional[dict] = None,
) -> WorkflowCompensation:
    """Record a side effect with explicit reversibility.

    Irreversible effects (a sent notification cannot be unsent) are recorded
    so the system never pretends they were undone.
    """
    reversible = side_effect_type in REVERSIBLE_SIDE_EFFECTS
    record = WorkflowCompensation(
        workflow_execution_id=workflow_execution_id,
        workspace_id=workspace_id,
        node_id=node_id,
        side_effect_type=side_effect_type,
        reversible=reversible,
        status="RECORDED",
        metadata_json=json.dumps(metadata, default=str) if metadata else None,
    )
    db.add(record)
    db.flush()
    return record


def compensate_side_effect(db: Session, record: WorkflowCompensation) -> WorkflowCompensation:
    """Mark a reversible side effect as compensated (audited)."""
    if not record.reversible:
        raise WorkflowReliabilityError(
            f"Side effect '{record.side_effect_type}' is irreversible and cannot be compensated"
        )
    record.status = "COMPENSATED"
    db.flush()
    return record


# ---------------------------------------------------------------------------
# Failure recovery
# ---------------------------------------------------------------------------

def classify_failure(error: str) -> str:
    """RETRYABLE vs NON_RETRYABLE (deterministic heuristics)."""
    lowered = error.lower()
    if any(marker in lowered for marker in RETRYABLE_ERROR_MARKERS):
        return "RETRYABLE"
    return "NON_RETRYABLE"


def recovery_plan(
    db: Session,
    workspace_id: int,
    workflow_execution_id: str,
    error: str,
    node_id: str,
) -> dict:
    """Plan recovery for a failed node (never loops infinitely)."""
    records = (
        db.query(WorkflowNodeExecution)
        .filter(
            WorkflowNodeExecution.workflow_execution_id == workflow_execution_id,
            WorkflowNodeExecution.node_id == node_id,
        )
        .order_by(WorkflowNodeExecution.attempt.desc())
        .all()
    )
    attempts = len(records)
    failure_class = classify_failure(error)

    if failure_class == "RETRYABLE" and attempts < MAX_AUTO_RETRIES:
        action = "AUTO_RETRY"
        note = f"Automatic retry (attempt {attempts + 1}/{MAX_AUTO_RETRIES})"
    elif attempts >= MAX_AUTO_RETRIES:
        action = "DEAD_LETTER"
        note = "Retry limit reached — manual intervention required"
    else:
        action = "MANUAL_RETRY"
        note = "Non-retryable failure — review and retry manually"

    log_audit_event(
        db, event_type="workflow", event_action="recovery_plan",
        user_id=0, resource_type="workflow_execution", resource_id=workflow_execution_id,
        details=f"Recovery plan for node {node_id}: {action} — {note}",
    )
    return {
        "workflow_execution_id": workflow_execution_id,
        "node_id": node_id,
        "attempts": attempts,
        "failure_class": failure_class,
        "action": action,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Workflow simulator (enhanced dry run — no side effects)
# ---------------------------------------------------------------------------

def simulate_workflow(definition: dict) -> dict:
    """Full simulation plan: trigger, conditions, steps, data flow, AI calls,
    token/cost estimates, permissions, approval points, side effects, and
    failure points. Never performs side effects."""
    from ..services.workflow_intel import validate_definition, dry_run
    definition = validate_definition(definition)
    base = dry_run(definition)

    nodes = definition.get("nodes", [])
    ai_nodes = [n for n in nodes if n.get("type") in ("extract", "summarize", "compare", "classify", "research")]
    notify_nodes = [n for n in nodes if n.get("type") == "notify"]
    approval_nodes = [n for n in nodes if n.get("type") == "approval"]

    estimated_tokens = len(ai_nodes) * 2000  # heuristic, clearly labeled
    risk = workflow_risk(definition)

    return {
        **base,
        "mode": "SIMULATION",
        "data_dependencies": [
            {"node": n.get("id"), "inputs": n.get("inputs", [])} for n in nodes
        ],
        "estimated_tokens": estimated_tokens,
        "estimated_cost": round(estimated_tokens / 1_000_000 * 0.002, 6),
        "cost_estimate_is_exact": False,
        "side_effects": [
            {"node": n.get("id"), "effect": "notification"} for n in notify_nodes
        ],
        "approval_points": [n.get("id") for n in approval_nodes],
        "failure_points": [n.get("id") for n in ai_nodes],
        "risk": risk,
        "permissions": ["workflow:execute"],
    }


# ---------------------------------------------------------------------------
# Automation risk engine
# ---------------------------------------------------------------------------

def workflow_risk(definition: dict) -> dict:
    """Classify workflow risk: LOW/MEDIUM/HIGH/CRITICAL with explainable factors."""
    nodes = definition.get("nodes", [])
    notify_count = sum(1 for n in nodes if n.get("type") == "notify")
    approval_count = sum(1 for n in nodes if n.get("type") == "approval")
    ai_count = sum(1 for n in nodes if n.get("type") in ("extract", "summarize", "compare", "classify", "research"))

    score = 1
    factors = []
    if notify_count:
        score += 2
        factors.append(f"{notify_count} external notification(s) (irreversible)")
    if approval_count:
        score -= 1
        factors.append(f"{approval_count} human approval gate(s)")
    if ai_count > 5:
        score += 2
        factors.append(f"{ai_count} AI calls")
    if any(n.get("type") == "notify" for n in nodes) and not approval_count:
        score += 3
        factors.append("notification without approval gate")

    if score >= 7:
        level = "CRITICAL"
    elif score >= 5:
        level = "HIGH"
    elif score >= 3:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "score": score,
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# Workflow version diff
# ---------------------------------------------------------------------------

def version_diff(v_old: WorkflowVersion, v_new: WorkflowVersion) -> dict:
    """Structural diff between two workflow versions."""
    old_def = json.loads(v_old.definition_json or "{}")
    new_def = json.loads(v_new.definition_json or "{}")
    old_nodes = {n["id"]: n for n in old_def.get("nodes", [])}
    new_nodes = {n["id"]: n for n in new_def.get("nodes", [])}

    added = [{"id": nid, "name": n.get("name", nid)} for nid, n in new_nodes.items() if nid not in old_nodes]
    removed = [{"id": nid, "name": n.get("name", nid)} for nid, n in old_nodes.items() if nid not in new_nodes]
    changed = []
    for nid in old_nodes.keys() & new_nodes.keys():
        if old_nodes[nid] != new_nodes[nid]:
            changed.append({"id": nid, "name": new_nodes[nid].get("name", nid)})

    old_trigger = old_def.get("trigger")
    new_trigger = new_def.get("trigger")
    return {
        "from_version": v_old.version,
        "to_version": v_new.version,
        "trigger_changed": old_trigger != new_trigger,
        "trigger_from": old_trigger,
        "trigger_to": new_trigger,
        "added_nodes": added,
        "removed_nodes": removed,
        "changed_nodes": changed,
        "approval_requirement_changed": (
            (old_trigger != new_trigger)
            or any(n.get("type") == "approval" for n in new_nodes.values())
            != any(n.get("type") == "approval" for n in old_nodes.values())
        ),
    }


# ---------------------------------------------------------------------------
# Provider health (recorded in DB for the quality dashboard)
# ---------------------------------------------------------------------------

def record_provider_call(
    db: Session,
    provider: str,
    model: str,
    success: bool,
    latency_ms: Optional[float] = None,
    error: Optional[str] = None,
    circuit_threshold: int = 5,
) -> ProviderHealth:
    """Record a provider call outcome and manage the circuit breaker."""
    health = (
        db.query(ProviderHealth)
        .filter(ProviderHealth.provider == provider, ProviderHealth.model == model)
        .first()
    )
    if not health:
        health = ProviderHealth(provider=provider, model=model)
        db.add(health)
        # Column defaults are applied at flush time; flush before mutating so
        # the counters exist (never None + int).
        db.flush()

    now = datetime.now(timezone.utc)
    if success:
        health.success_count += 1
        health.consecutive_failures = 0
        if latency_ms is not None:
            health.avg_latency_ms = (
                latency_ms if health.avg_latency_ms is None
                else (health.avg_latency_ms * 0.9 + latency_ms * 0.1)
            )
    else:
        health.failure_count += 1
        health.consecutive_failures += 1
        health.last_error = (error or "unknown error")[:500]

    if health.consecutive_failures >= circuit_threshold:
        health.circuit_state = "OPEN"
        health.status = "DOWN"
    elif health.consecutive_failures > 0:
        health.circuit_state = "HALF_OPEN"
        health.status = "DEGRADED"
    else:
        health.circuit_state = "CLOSED"
        health.status = "UP"

    health.last_checked_at = now
    db.flush()
    return health


def allow_provider_call(health: Optional[ProviderHealth]) -> bool:
    """Circuit breaker gate: never hammer an OPEN provider."""
    if health is None or health.circuit_state != "OPEN":
        return True
    # Half-open probe after a cooldown window
    cooldown = timedelta(minutes=5)
    if health.last_checked_at and datetime.now(timezone.utc) - health.last_checked_at > cooldown:
        return True
    return False