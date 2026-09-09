"""Phase 20 — Agent intelligence.

Agent success metrics, deterministic failure classification, plan-quality
scoring (unnecessary steps, redundant tools, excessive cost/latency, failed
dependencies), candidate plan optimizations (evaluation required), and an
agent safety score.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import AgentIntelligence

FAILURE_CLASSES = {
    "planning", "authorization", "tool", "provider", "knowledge",
    "timeout", "policy", "human_approval",
}


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def record_metrics(db: Session, *, execution_id: str, workspace_id: int,
                   metrics: dict,
                   failure_class: Optional[str] = None) -> AgentIntelligence:
    if failure_class is not None and failure_class not in FAILURE_CLASSES:
        raise ValueError(f"Unknown failure class: {failure_class}")
    row = AgentIntelligence(execution_id=execution_id,
                            workspace_id=workspace_id,
                            metrics_json=_dumps(metrics),
                            failure_class=failure_class)
    db.add(row)
    db.flush()
    return row


def success_metrics(db: Session, *, workspace_id: int,
                    limit: int = 500) -> dict:
    rows = db.query(AgentIntelligence).filter_by(workspace_id=workspace_id)\
        .order_by(AgentIntelligence.created_at.desc()).limit(limit).all()
    total = len(rows)
    ok = sum(1 for r in rows if r.metrics_json and
             json.loads(r.metrics_json).get("success", True))
    cancelled = sum(1 for r in rows if r.metrics_json and
                    json.loads(r.metrics_json).get("cancelled"))
    handoff = sum(1 for r in rows if r.metrics_json and
                  json.loads(r.metrics_json).get("human_handoff"))
    cost = sum(float(json.loads(r.metrics_json).get("cost", 0.0))
               for r in rows if r.metrics_json)
    duration = [float(json.loads(r.metrics_json).get("duration_s", 0.0))
                for r in rows if r.metrics_json]
    return {
        "total": total,
        "success_rate": (ok / total) if total else 0.0,
        "cancelled": cancelled, "human_handoffs": handoff,
        "total_cost": round(cost, 4),
        "avg_duration_s": (sum(duration) / len(duration))
        if duration else 0.0,
    }


def failure_analysis(db: Session, *, workspace_id: int,
                     limit: int = 500) -> dict:
    rows = db.query(AgentIntelligence).filter_by(workspace_id=workspace_id)\
        .order_by(AgentIntelligence.created_at.desc()).limit(limit).all()
    counts: dict[str, int] = {}
    for r in rows:
        if r.failure_class:
            counts[r.failure_class] = counts.get(r.failure_class, 0) + 1
    return {"total_failures": sum(counts.values()),
            "by_class": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}


def plan_quality(plan: dict) -> dict:
    """Deterministic plan-quality evaluation. `plan` is a dict with
    'steps', 'tool_calls', 'cost', 'duration_s', 'failed_dependencies'."""
    steps = plan.get("steps") or []
    tool_calls = plan.get("tool_calls") or []
    cost = float(plan.get("cost", 0.0))
    duration = float(plan.get("duration_s", 0.0))
    failed_deps = int(plan.get("failed_dependencies", 0))
    signals = {}
    signals["unnecessary_steps"] = len(steps) > 12
    signals["redundant_tools"] = len(tool_calls) > len(set(
        t.get("name") for t in tool_calls)) and len(tool_calls) > 3
    signals["excessive_cost"] = cost > float(plan.get("cost_budget", 1.0))
    signals["excessive_latency"] = duration > float(
        plan.get("latency_budget_s", 300))
    signals["failed_dependencies"] = failed_deps > 0
    score = 1.0
    penalties = {
        "unnecessary_steps": 0.2, "redundant_tools": 0.15,
        "excessive_cost": 0.25, "excessive_latency": 0.2,
        "failed_dependencies": 0.3,
    }
    for k, bad in signals.items():
        if bad:
            score -= penalties[k]
    return {"score": round(max(score, 0.0), 4), "signals": signals}


def plan_improvements(plan: dict) -> list[dict]:
    """Candidate plan optimizations — all require evaluation before use."""
    quality = plan_quality(plan)
    out = []
    if quality["signals"]["unnecessary_steps"]:
        out.append({"kind": "step_reduction",
                    "suggestion": "Plan exceeds 12 steps — split into "
                                  "sub-plans or parallelize independent steps"})
    if quality["signals"]["redundant_tools"]:
        out.append({"kind": "tool_dedup",
                    "suggestion": "Repeated tool calls — cache tool results "
                                  "within the plan"})
    if quality["signals"]["excessive_cost"]:
        out.append({"kind": "model_downgrade",
                    "suggestion": "Cost exceeds budget — evaluate cheaper "
                                  "model for non-critical steps"})
    if quality["signals"]["excessive_latency"]:
        out.append({"kind": "parallelize",
                    "suggestion": "Duration exceeds budget — parallelize "
                                  "independent tool calls"})
    return out


def safety_score(db: Session, *, workspace_id: int,
                 limit: int = 500) -> dict:
    """Safety score from recorded policy violations, rejected tools, unsafe
    attempts, and approval rate."""
    rows = db.query(AgentIntelligence).filter_by(workspace_id=workspace_id)\
        .order_by(AgentIntelligence.created_at.desc()).limit(limit).all()
    violations = sum(1 for r in rows if r.metrics_json and
                     json.loads(r.metrics_json).get("policy_violations", 0) > 0)
    rejected_tools = sum(1 for r in rows if r.metrics_json and
                         json.loads(r.metrics_json).get("rejected_tools", 0) > 0)
    unsafe = sum(1 for r in rows if r.metrics_json and
                 json.loads(r.metrics_json).get("unsafe_attempts", 0) > 0)
    approvals = sum(1 for r in rows if r.metrics_json and
                    json.loads(r.metrics_json).get("approvals_required", 0) > 0)
    total = len(rows)
    violations_rate = (violations / total) if total else 0.0
    rejected_rate = (rejected_tools / total) if total else 0.0
    unsafe_rate = (unsafe / total) if total else 0.0
    score = max(0.0, 1.0 - (violations_rate * 0.5 + rejected_rate * 0.25 +
                            unsafe_rate * 0.25))
    return {"safety_score": round(score, 4), "policy_violations": violations,
            "rejected_tools": rejected_tools, "unsafe_attempts": unsafe,
            "approval_rate": (approvals / total) if total else 0.0}