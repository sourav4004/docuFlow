"""Phase 21 — Agent autonomy 2.0 + workflow autonomy.

Every agent operation resolves an autonomy policy; plans are risk-classified
by tools/data/side effects/sensitivity/external systems, simulated before
execution, optionally auto-optimized within pre-approved bounds, and handed
to human review when high risk. Failed runs get durable recovery (retry/
checkpoint/resume/rollback/handoff) and dead-letter capture. Workflows get a
risk engine (external effects, sensitive data, destructive ops, financial
impact, communication, third-party integrations), governed autonomy, high-risk
gates, recurring-failure diagnosis, optimization recommendations, and safe
simulation.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..models.phase21 import (
    AgentPlanRisk, AgentRecoveryEvent, WorkflowRiskAssessment,
)
from ..models.phase19 import AgentDeadLetter
from . import autonomy

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Tools that escalate plan risk by their nature.
_HIGH_RISK_TOOLS = {"send_email", "delete_data", "run_sql", "external_api",
                    "publish_webhook", "transfer_funds", "modify_permissions"}
_MEDIUM_RISK_TOOLS = {"search_web", "write_document", "update_record"}
_SENSITIVE_KINDS = {"restricted", "confidential"}

_RECOVERY_KINDS = ("RETRY", "CHECKPOINT", "RESUME", "ROLLBACK", "HANDOFF")


def _bounded_json(value: Any, limit: int = 6000) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


# ---------------------------------------------------------------------------
# Agent autonomy 2.0
# ---------------------------------------------------------------------------

def classify_plan_risk(plan: dict) -> dict:
    """Deterministic plan risk classification.

    plan: {steps: [{tool?, reads_sensitive?, writes_external?,
    destructive?, financial_impact?, communication?}]}
    """
    steps = plan.get("steps") or []
    factors: list[str] = []
    tools = set()
    for step in steps:
        tool = step.get("tool")
        if tool:
            tools.add(tool)
        if step.get("reads_sensitive") or step.get("sensitive_data"):
            factors.append("reads_sensitive_data")
        if step.get("writes_external") or step.get("external_systems"):
            factors.append("writes_to_external_system")
        if step.get("destructive"):
            factors.append("destructive_operation")
        if step.get("financial_impact"):
            factors.append("financial_impact")
        if step.get("communication"):
            factors.append("external_communication")
    if tools & _HIGH_RISK_TOOLS:
        factors.append("high_risk_tool")
    elif tools & _MEDIUM_RISK_TOOLS:
        factors.append("medium_risk_tool")

    if "destructive_operation" in factors or "financial_impact" in factors:
        level = "CRITICAL"
    elif ("writes_to_external_system" in factors
          or "external_communication" in factors
          or "high_risk_tool" in factors):
        level = "HIGH"
    elif factors:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {"risk_level": level, "risk_factors": sorted(set(factors)),
            "tools": sorted(tools)}


def assess_plan(db: Session, workspace_id: int, plan: dict,
                agent_run_id: Optional[int] = None) -> AgentPlanRisk:
    """Classify + persist plan risk with the resolved autonomy decision."""
    classification = classify_plan_risk(plan)
    op = autonomy.guard_operation(
        db, workspace_id, "agent.plan.execute",
        risk_level=classification["risk_level"], actor="agent",
        source="AI",
        input_payload={"plan_summary": plan.get("summary"),
                       "steps": len(plan.get("steps") or [])},
        idempotency_key=f"plan:{workspace_id}:{agent_run_id or 'adhoc'}")
    row = AgentPlanRisk(
        workspace_id=workspace_id, agent_run_id=agent_run_id,
        plan_summary=(plan.get("summary") or "")[:200],
        risk_level=classification["risk_level"],
        risk_factors=_bounded_json(classification["risk_factors"]),
        decision=op.decision, handed_off=op.decision != "ALLOWED")
    db.add(row)
    db.commit()
    return row


def simulate_plan(plan: dict) -> dict:
    """Dry-run a plan: which steps would execute, what they touch.

    Zero side effects — used both for operator preview and pre-execution.
    """
    steps = plan.get("steps") or []
    executed, skipped = [], []
    for i, step in enumerate(steps):
        if "condition" in step and not step["condition"]:
            skipped.append({"index": i, "tool": step.get("tool"),
                            "reason": "condition false"})
        else:
            executed.append({"index": i, "tool": step.get("tool"),
                             "touches_external": bool(
                                 step.get("writes_external")
                                 or step.get("communication")),
                             "destructive": bool(step.get("destructive"))})
    return {"would_execute": executed, "skipped": skipped,
            "external_steps": sum(1 for e in executed
                                  if e["touches_external"]),
            "destructive_steps": sum(1 for e in executed
                                     if e["destructive"])}


def optimize_plan(plan: dict) -> dict:
    """Identify redundant steps, unnecessary retrieval, expensive calls.

    Pure analysis; nothing is removed without the autonomy-approved path.
    """
    steps = plan.get("steps") or []
    optimizations: list[dict] = []
    seen_tools: dict[str, int] = {}
    for i, step in enumerate(steps):
        tool = step.get("tool") or ""
        if tool in seen_tools and not step.get("unique"):
            optimizations.append({"index": i, "kind": "redundant_step",
                                  "tool": tool,
                                  "first_at": seen_tools[tool]})
        else:
            seen_tools[tool] = i
        if tool == "retrieve" and step.get("top_k", 0) > 20:
            optimizations.append({"index": i, "kind": "expensive_call",
                                  "detail": f"top_k={step['top_k']} > 20"})
        if tool == "retrieve" and step.get("repeat_of") is not None:
            optimizations.append({"index": i, "kind": "unnecessary_retrieval",
                                  "detail": "duplicate retrieval"})
    return {"optimizations": optimizations,
            "count": len(optimizations)}


def apply_plan_optimizations(db: Session, risk_row: AgentPlanRisk,
                             plan: dict, actor: str = "agent") -> dict:
    """Apply only policy-approved low-risk optimizations (dedup/removal)."""
    analysis = optimize_plan(plan)
    op = autonomy.guard_operation(
        db, risk_row.workspace_id, "agent.plan.optimize", risk_level="LOW",
        actor=actor, source="AI",
        input_payload={"plan_risk_id": risk_row.id,
                       "count": analysis["count"]},
        idempotency_key=f"plan-opt:{risk_row.id}")
    applied = None
    if op.decision == "ALLOWED" and analysis["count"] > 0:
        applied = _bounded_json(analysis["optimizations"])
    risk_row.applied_optimizations = applied
    db.commit()
    return {"decision": op.decision, "analysis": analysis,
            "applied": bool(applied)}


def handoff_to_human(db: Session, workspace_id: int, plan_risk: AgentPlanRisk,
                     reason: str, actor: str = "agent") -> AgentRecoveryEvent:
    """High-risk operations must enter human review."""
    plan_risk.handed_off = True
    db.flush()
    event = AgentRecoveryEvent(
        workspace_id=workspace_id, agent_run_id=plan_risk.agent_run_id,
        recovery_kind="HANDOFF", status="RUNNING",
        detail=reason[:1000],
        idempotency_key=f"handoff:{plan_risk.id}")
    db.add(event)
    db.commit()
    return event


def recover_agent_run(db: Session, workspace_id: int, agent_run_id: int,
                      recovery_kind: str, detail: Optional[str] = None,
                      idempotency_key: Optional[str] = None) \
        -> AgentRecoveryEvent:
    """Durable, idempotent agent recovery (retry/checkpoint/resume/rollback)."""
    if recovery_kind not in _RECOVERY_KINDS:
        raise ValueError(f"invalid recovery kind: {recovery_kind}")
    key = idempotency_key or f"agent-recovery:{agent_run_id}:{recovery_kind}"
    existing = (db.query(AgentRecoveryEvent)
                .filter_by(workspace_id=workspace_id, idempotency_key=key)
                .first())
    if existing is not None:
        return existing
    event = AgentRecoveryEvent(
        workspace_id=workspace_id, agent_run_id=agent_run_id,
        recovery_kind=recovery_kind, status="SUCCEEDED",
        detail=(detail or "")[:1000], idempotency_key=key)
    db.add(event)
    db.commit()
    return event


def dead_letter_run(db: Session, workspace_id: int, execution_id: str,
                    reason: str, plan: Optional[dict] = None) -> AgentDeadLetter:
    """Persist a failed execution for operator intervention."""
    letter = AgentDeadLetter(
        workspace_id=workspace_id, execution_id=execution_id,
        reason=(reason or "")[:1000],
        last_checkpoint_json=_bounded_json(plan),
        safe_replay=False)
    db.add(letter)
    db.commit()
    return letter


# ---------------------------------------------------------------------------
# Workflow autonomy
# ---------------------------------------------------------------------------

def assess_workflow_risk(db: Session, workspace_id: int,
                         workflow: dict,
                         workflow_id: Optional[int] = None,
                         workflow_version_id: Optional[int] = None) \
        -> WorkflowRiskAssessment:
    """Workflow risk engine + governed autonomy decision.

    workflow: {steps: [...], has_external_side_effects?, handles_sensitive_data?,
    destructive?, financial_impact?, sends_communication?, third_party?}
    """
    factors: list[str] = []
    if workflow.get("has_external_side_effects"):
        factors.append("external_side_effects")
    if workflow.get("handles_sensitive_data"):
        factors.append("sensitive_data")
    if workflow.get("destructive"):
        factors.append("destructive_operations")
    if workflow.get("financial_impact"):
        factors.append("financial_impact")
    if workflow.get("sends_communication"):
        factors.append("communication")
    if workflow.get("third_party"):
        factors.append("third_party_integration")
    if "destructive_operations" in factors or "financial_impact" in factors:
        risk = "CRITICAL"
    elif factors:
        risk = "HIGH" if ("external_side_effects" in factors
                          or "communication" in factors) else "MEDIUM"
    else:
        risk = "LOW"

    op = autonomy.guard_operation(
        db, workspace_id, "workflow.execute", risk_level=risk,
        actor="system", source="SYSTEM",
        input_payload={"workflow_id": workflow_id},
        idempotency_key=f"wf:{workspace_id}:{workflow_id or 'adhoc'}")
    requires_approval = op.decision != "ALLOWED"
    row = WorkflowRiskAssessment(
        workspace_id=workspace_id, workflow_id=workflow_id,
        workflow_version_id=workflow_version_id, risk_level=risk,
        risk_factors=_bounded_json(sorted(set(factors))),
        autonomy_decision=op.decision, requires_approval=requires_approval)
    db.add(row)
    db.commit()
    return row


def detect_recurring_workflow_failures(db: Session, workspace_id: int,
                                       workflow_id: int,
                                       failure_events: list[dict],
                                       min_count: int = 3) -> dict:
    """Diagnose recurring failures deterministically."""
    by_step: dict[str, int] = {}
    for ev in failure_events:
        step = str(ev.get("step") or "unknown")
        by_step[step] = by_step.get(step, 0) + 1
    hotspots = [{"step": s, "count": c} for s, c in
                sorted(by_step.items(), key=lambda kv: -kv[1])
                if c >= min_count]
    return {"workflow_id": workflow_id, "hotspots": hotspots,
            "total_failures": len(failure_events)}


def recommend_workflow_optimizations(failure_stats: dict,
                                     timings: Optional[dict] = None) -> list[dict]:
    """Recommend parallelization/caching/retries/timeouts/model selection."""
    timings = timings or {}
    recs: list[dict] = []
    for hotspot in failure_stats.get("hotspots", []):
        recs.append({"step": hotspot["step"], "kind": "add_retry",
                     "detail": f"{hotspot['count']} failures"})
    slow = timings.get("slowest_step_ms")
    if slow:
        recs.append({"step": timings.get("slowest_step"), "kind": "parallelize",
                     "detail": f"slowest step {slow}ms"})
    if timings.get("repeated_computation"):
        recs.append({"kind": "add_caching",
                     "detail": "repeated computation detected"})
    if timings.get("timeout_ms") and timings["timeout_ms"] < slow or 0:
        recs.append({"kind": "adjust_timeout",
                     "detail": "timeout below observed step duration"})
    if timings.get("model_latency_ms", 0) > 3000:
        recs.append({"kind": "model_selection",
                     "detail": "consider faster model for this step"})
    return recs


def simulate_workflow(db: Session, workspace_id: int, workflow: dict) -> dict:
    """Side-effect-free workflow simulation."""
    steps = workflow.get("steps") or []
    executed = [{"index": i, "step": (s.get("name") or s.get("tool") or "?"),
                 "external": bool(s.get("external") or
                                  s.get("writes_external"))}
                for i, s in enumerate(steps) if not s.get("condition") is False]
    return {"steps_executed": len(executed), "steps": executed,
            "external_steps": sum(1 for e in executed if e["external"]),
            "simulated": True}
