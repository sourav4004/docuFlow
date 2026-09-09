"""Phase 21 — AI model autopilot + AI cost autopilot.

Continuous model performance monitoring (quality, latency, cost, availability,
tool + structured-output reliability), model/provider drift detection,
candidate routing updates with safety-aware simulation, governed promotion,
real-time cost monitoring, anomaly detection, forecasting, pre-execution cost
guards (allow/block/approval), pre-approved low-risk optimizations with audit,
and cost-aware worker scheduling with tenant fairness.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..models.phase21 import (
    AdaptiveCandidate, CostAwareQueueDecision, CostForecast,
    CostGuardDecision, CostOptimizationEvent, ModelDriftEvent,
    ModelPerformanceSample, RoutingSimulation,
)
from ..models.phase21 import CostAnomaly as CostAnomalyP21
from . import autonomy

_ROUTING_SAFETY_CHECKS = ("sensitivity_ok", "region_ok", "policy_ok",
                          "budget_ok", "capability_ok")

_COST_OPTIMIZATIONS = {"cache_reuse", "context_reduction", "batching",
                       "cheaper_model"}


def _bounded_json(value: Any, limit: int = 6000) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Model performance monitor + drift
# ---------------------------------------------------------------------------

def record_model_sample(db: Session, workspace_id: int, model: str,
                        provider: str, quality_score: Optional[float] = None,
                        latency_ms: Optional[float] = None,
                        cost_usd: Optional[float] = None,
                        availability: Optional[float] = None,
                        tool_reliability: Optional[float] = None,
                        structured_output_reliability: Optional[float] = None) \
        -> ModelPerformanceSample:
    row = ModelPerformanceSample(
        workspace_id=workspace_id, model=model, provider=provider,
        quality_score=quality_score, latency_ms=latency_ms,
        cost_usd=cost_usd, availability=availability,
        tool_reliability=tool_reliability,
        structured_output_reliability=structured_output_reliability)
    db.add(row)
    db.commit()
    return row


def detect_model_drift(db: Session, workspace_id: int, model: str,
                       provider: str, metric: str = "quality_score",
                       degrade_threshold_percent: float = 15.0,
                       lookback: int = 20) -> Optional[ModelDriftEvent]:
    """Detect degradation by comparing the newest sample against the mean
    of the previous window (deterministic, evidence recorded)."""
    rows = (db.query(ModelPerformanceSample)
            .filter_by(workspace_id=workspace_id, model=model)
            .order_by(ModelPerformanceSample.id.desc()).limit(lookback).all())
    if len(rows) < 4:
        return None
    newest, history = rows[0], rows[1:]
    current = getattr(newest, metric)
    past = [getattr(s, metric) for s in history
            if getattr(s, metric) is not None]
    if current is None or not past:
        return None
    baseline = _mean(past)
    if baseline <= 0:
        return None
    change = (baseline - current) / baseline * 100.0
    if change < degrade_threshold_percent:
        return None
    event = ModelDriftEvent(
        workspace_id=workspace_id, model=model, provider=provider,
        kind="MODEL", metric=metric, direction="DEGRADED",
        baseline_value=round(baseline, 4), current_value=round(float(current), 4),
        drop_percent=round(change, 1),
        recommendation="generate routing candidate (governed promotion only)")
    db.add(event)
    db.commit()
    return event


def detect_provider_drift(db: Session, workspace_id: int, provider: str,
                          metric: str = "availability",
                          degrade_threshold_percent: float = 10.0,
                          lookback: int = 20) -> Optional[ModelDriftEvent]:
    """Provider-level behavioral drift detection."""
    rows = (db.query(ModelPerformanceSample)
            .filter_by(workspace_id=workspace_id, provider=provider)
            .order_by(ModelPerformanceSample.id.desc()).limit(lookback).all())
    if len(rows) < 4:
        return None
    newest, history = rows[0], rows[1:]
    current = getattr(newest, metric)
    past = [getattr(s, metric) for s in history
            if getattr(s, metric) is not None]
    if current is None or not past:
        return None
    baseline = _mean(past)
    if baseline <= 0:
        return None
    change = (baseline - current) / baseline * 100.0
    if change < degrade_threshold_percent:
        return None
    event = ModelDriftEvent(
        workspace_id=workspace_id, provider=provider, kind="PROVIDER",
        metric=metric, direction="DEGRADED",
        baseline_value=round(baseline, 4),
        current_value=round(float(current), 4),
        drop_percent=round(change, 1),
        recommendation="evaluate fallback provider (governed promotion only)")
    db.add(event)
    db.commit()
    return event


# ---------------------------------------------------------------------------
# Routing adaptation + simulation + promotion
# ---------------------------------------------------------------------------

def propose_routing_candidate(db: Session, workspace_id: int,
                              description: str, workload: list[dict]) \
        -> AdaptiveCandidate:
    """Create a routing adaptation candidate (proposal only)."""
    candidate = AdaptiveCandidate(
        workspace_id=workspace_id, domain="routing",
        change_kind="model_routing", proposed_value=_bounded_json(description),
        rationale=f"workload size {len(workload)}",
        idempotency_key=f"routing:{description[:80]}:{workspace_id}")
    db.add(candidate)
    db.commit()
    return candidate


def simulate_routing(db: Session, workspace_id: int, description: str,
                     workload: list[dict],
                     current_routing: dict,
                     candidate_routing: dict,
                     constraints: Optional[dict] = None) -> RoutingSimulation:
    """Zero-side-effect routing simulation with safety checks.

    Computes per-model latency/cost/quality over the workload for both
    routings and validates sensitivity/region/policy/budget/capability
    constraints. No production behavior is modified.
    """
    constraints = constraints or {}

    def _metrics(routing: dict) -> dict:
        total_cost, total_latency, total_quality, n = 0.0, 0.0, 0.0, 0
        for item in workload:
            model = routing.get(item.get("task", "general"), "default")
            perf = (item.get("model_perf") or {}).get(model) or {}
            total_cost += float(perf.get("cost_usd", 0.0))
            total_latency += float(perf.get("latency_ms", 0.0))
            total_quality += float(perf.get("quality", 0.0))
            n += 1
        if n == 0:
            n = 1
        return {"cost_usd": round(total_cost, 4),
                "latency_ms": round(total_latency / n, 1),
                "quality": round(total_quality / n, 4)}

    current_metrics = _metrics(current_routing)
    simulated_metrics = _metrics(candidate_routing)
    checks = {
        "sensitivity_ok": not constraints.get(
            "sensitive_task_uses_unapproved_model", False),
        "region_ok": not constraints.get("cross_region_model", False),
        "policy_ok": not constraints.get("policy_violation", False),
        "budget_ok": simulated_metrics["cost_usd"] <= max(
            current_metrics["cost_usd"]
            * float(constraints.get("budget_multiplier", 1.1)),
            float(constraints.get("budget_max_usd", 1.0))),
        "capability_ok": not constraints.get("missing_capability", False),
    }
    safe = all(checks.values())
    sim = RoutingSimulation(
        workspace_id=workspace_id, description=description[:200],
        workload=_bounded_json(workload),
        current_metrics=_bounded_json(current_metrics),
        simulated_metrics=_bounded_json(simulated_metrics),
        safety_checks=_bounded_json(checks), safe=safe)
    db.add(sim)
    db.commit()
    return sim


def promote_routing(db: Session, sim: RoutingSimulation,
                    candidate: AdaptiveCandidate,
                    actor: str = "system", source: str = "SYSTEM") -> dict:
    """Governed routing promotion — simulation-safe + autonomy ALLOWED."""
    if not sim.safe:
        return {"simulation_id": sim.id, "decision": "BLOCKED",
                "reason": "safety checks failed"}
    op = autonomy.guard_operation(
        db, sim.workspace_id, "routing.promote", risk_level="MEDIUM",
        actor=actor, source=source,
        input_payload={"simulation_id": sim.id},
        idempotency_key=f"routing-promote:{sim.id}")
    if op.decision == "ALLOWED":
        candidate.status = "PROMOTED"
        candidate.promoted = True
        sim.executed = True
        db.commit()
    return {"simulation_id": sim.id, "decision": op.decision,
            "reason": op.decision_reason, "operation_id": op.id}


# ---------------------------------------------------------------------------
# Cost autopilot
# ---------------------------------------------------------------------------

def record_cost_snapshot(db: Session, workspace_id: int,
                         current_spend_usd: float,
                         daily_run_rate_usd: Optional[float] = None,
                         period_days: int = 30,
                         budget_usd: Optional[float] = None,
                         scope_kind: str = "WORKSPACE",
                         scope_key: Optional[str] = None) -> CostForecast:
    """Record live spend + linear forecast and budget state."""
    rate = float(daily_run_rate_usd or 0.0)
    forecast = float(current_spend_usd) + rate * period_days
    row = CostForecast(
        workspace_id=workspace_id, scope_kind=scope_kind,
        scope_key=scope_key, period_days=period_days,
        current_spend_usd=round(float(current_spend_usd), 4),
        forecast_usd=round(forecast, 4), budget_usd=budget_usd,
        over_budget=bool(budget_usd is not None and forecast > budget_usd))
    db.add(row)
    db.commit()
    return row


def detect_cost_anomaly(db: Session, workspace_id: int,
                        baseline_value: float, observed_value: float,
                        metric: str = "spend",
                        threshold_percent: float = 50.0) \
        -> Optional[CostAnomalyP21]:
    """Detect abnormal usage/spend increase (deterministic threshold)."""
    if baseline_value <= 0:
        return None
    increase = (observed_value - baseline_value) / baseline_value * 100.0
    if increase < threshold_percent:
        return None
    row = CostAnomalyP21(
        workspace_id=workspace_id, metric=metric,
        severity="HIGH" if increase >= 100 else "MEDIUM",
        baseline_value=round(float(baseline_value), 4),
        observed_value=round(float(observed_value), 4),
        increase_percent=round(increase, 1),
        recommendation="review expensive workflows; consider pre-approved "
                       "optimizations")
    db.add(row)
    db.commit()
    return row


def cost_guard(db: Session, workspace_id: int, operation_type: str,
               estimated_cost_usd: float,
               remaining_budget_usd: Optional[float] = None,
               idempotency_key: Optional[str] = None) -> CostGuardDecision:
    """Pre-execution cost guard: estimate vs budget vs autonomy policy.

    Decides ALLOWED / REQUIRES_APPROVAL / BLOCKED and persists the decision.
    Repeated calls with the same idempotency key return the original
    decision instead of duplicating it.
    """
    key = idempotency_key or f"costguard:{operation_type}:{workspace_id}"
    existing = (db.query(CostGuardDecision)
                .filter_by(workspace_id=workspace_id, idempotency_key=key)
                .first())
    if existing is not None:
        return existing
    policy = autonomy.get_policy(db, workspace_id, operation_type)
    level = policy.autonomy_level if policy else "RECOMMEND"
    budget = (remaining_budget_usd if remaining_budget_usd is not None
              else (policy.budget_limit_usd if policy else None))
    if budget is not None and estimated_cost_usd > budget:
        decision, reason = "BLOCKED", (
            f"estimated ${estimated_cost_usd:.2f} exceeds remaining budget "
            f"${budget:.2f}")
    elif policy is not None and policy.requires_approval \
            and estimated_cost_usd > float(
                policy.budget_limit_usd or 0) * 0.5 and policy.budget_limit_usd:
        decision, reason = "REQUIRES_APPROVAL", (
            "estimated cost above 50% of policy budget")
    elif level in ("OBSERVE", "MANUAL_ONLY"):
        decision, reason = "REQUIRES_APPROVAL", (
            f"autonomy level {level} requires operator execution")
    else:
        decision, reason = "ALLOWED", "within budget and policy scope"
    row = CostGuardDecision(
        workspace_id=workspace_id, operation_type=operation_type,
        estimated_cost_usd=round(float(estimated_cost_usd), 4),
        remaining_budget_usd=budget, policy_level=level,
        decision=decision, reason=reason,
        idempotency_key=key)
    db.add(row)
    db.commit()
    return row


def apply_cost_optimization(db: Session, workspace_id: int,
                            optimization_kind: str,
                            estimated_savings_usd: float = 0.0,
                            actor: str = "system", source: str = "SYSTEM") \
        -> dict:
    """Apply a pre-approved low-risk optimization — audited, idempotent.

    Only the four pre-approved kinds may ever be applied automatically; even
    then the autonomy guard must return ALLOWED.
    """
    if optimization_kind not in _COST_OPTIMIZATIONS:
        return {"decision": "BLOCKED",
                "reason": f"optimization {optimization_kind} is not "
                          f"pre-approved for automatic application"}
    key = f"cost-optim:{optimization_kind}:{workspace_id}"
    existing = (db.query(CostOptimizationEvent)
                .filter_by(workspace_id=workspace_id, idempotency_key=key)
                .first())
    if existing is not None:
        return {"decision": "ALLOWED" if existing.applied else "BLOCKED",
                "applied": existing.applied,
                "reason": existing.reason,
                "optimization_id": existing.id}
    policy = autonomy.get_policy(db, workspace_id,
                                 f"cost.optimize.{optimization_kind}")
    level = policy.autonomy_level if policy else "RECOMMEND"
    op = autonomy.guard_operation(
        db, workspace_id, f"cost.optimize.{optimization_kind}",
        risk_level="LOW", actor=actor, source=source,
        input_payload={"optimization_kind": optimization_kind},
        idempotency_key=key)
    applied = op.decision == "ALLOWED"
    row = CostOptimizationEvent(
        workspace_id=workspace_id, optimization_kind=optimization_kind,
        policy_level=level, applied=applied,
        estimated_savings_usd=round(float(estimated_savings_usd), 4),
        reason=op.decision_reason,
        idempotency_key=key)
    db.add(row)
    db.commit()
    return {"decision": op.decision, "applied": applied,
            "reason": op.decision_reason, "optimization_id": row.id}


# ---------------------------------------------------------------------------
# Cost-aware worker scheduling
# ---------------------------------------------------------------------------

def schedule_job(db: Session, workspace_id: int, urgency: str = "NORMAL",
                 estimated_cost_usd: float = 0.0,
                 remaining_budget_usd: Optional[float] = None,
                 job_id: Optional[int] = None) -> CostAwareQueueDecision:
    """Cost-aware queueing with fairness and expensive-job guard.

    Priority: urgency first (CRITICAL=1, HIGH=2, NORMAL=5, LOW=9), then cost
    guard state. Fairness is preserved — budget state never starves a tenant
    below a minimum priority floor.
    """
    urgency_priority = {"CRITICAL": 1, "HIGH": 2, "NORMAL": 5, "LOW": 9}
    priority = urgency_priority.get(urgency, 5)
    budget = (remaining_budget_usd if remaining_budget_usd is not None
              else (policy.budget_limit_usd
                    if (policy := autonomy.get_policy(
                        db, workspace_id, "worker.job")) else None))
    if budget is None:
        budget_state = "OK"
    elif estimated_cost_usd > budget:
        budget_state = "EXCEEDED"
    elif estimated_cost_usd > budget * 0.8:
        budget_state = "TIGHT"
    else:
        budget_state = "OK"

    guard = None
    if budget_state == "EXCEEDED" and estimated_cost_usd > 0:
        guard = "REQUIRES_APPROVAL"
        priority = max(priority, 7)  # fairness floor — never fully starved

    decision = CostAwareQueueDecision(
        workspace_id=workspace_id, job_id=job_id, priority=priority,
        urgency=urgency, estimated_cost_usd=round(float(estimated_cost_usd), 4),
        budget_state=budget_state, fairness_preserved=True, guard=guard,
        decision=f"priority={priority} budget={budget_state}")
    db.add(decision)
    db.commit()
    return decision
