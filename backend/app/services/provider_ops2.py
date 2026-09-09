"""Phase 19 — AI provider platform 2.0.

Capability matrix 2.0 (task→capability mapping incl. long context and
reasoning metadata), deterministic model routing 3.0 with quality-aware
modes (cheapest/fastest/highest_quality/balanced), pre-call admission
control, load shedding with structured errors, fallback chain 2.0 with a
degraded deterministic mode, usage accounting, and shadow-testing harness
that defaults to synthetic/redacted inputs only.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

ROUTING_MODES = ("cheapest", "fastest", "highest_quality", "balanced")

TASK_CAPABILITIES = {
    "text": ["text"],
    "chat": ["text"],
    "stream": ["text", "streaming"],
    "tools": ["text", "tools"],
    "structured": ["text", "structured"],
    "vision": ["text", "vision"],
    "embeddings": ["embedding"],
    "long_context": ["text", "long_context"],
    "reasoning": ["text", "reasoning"],
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def capability_matrix() -> dict:
    """Static task→capability matrix used for routing decisions."""
    return {task: sorted(caps) for task, caps in TASK_CAPABILITIES.items()}


def register_capability(db: Session, *, provider: str, model: str,
                        supports_text: bool = True,
                        supports_vision: bool = False,
                        supports_tools: bool = False,
                        supports_structured: bool = False,
                        supports_streaming: bool = False,
                        context_window: Optional[int] = None,
                        max_output: Optional[int] = None,
                        embedding_dimensions: Optional[int] = None,
                        cost_per_1k_input: Optional[float] = None,
                        cost_per_1k_output: Optional[float] = None,
                        latency_class: Optional[str] = "medium",
                        notes: Optional[str] = None):
    from ..models.phase16 import ProviderCapability
    row = (db.query(ProviderCapability)
           .filter(ProviderCapability.provider == provider,
                   ProviderCapability.model == model).first())
    if row is None:
        row = ProviderCapability(provider=provider, model=model)
        db.add(row)
    for attr, value in (("supports_text", supports_text),
                        ("supports_vision", supports_vision),
                        ("supports_tools", supports_tools),
                        ("supports_structured", supports_structured),
                        ("supports_streaming", supports_streaming),
                        ("context_window", context_window),
                        ("max_output", max_output),
                        ("embedding_dimensions", embedding_dimensions),
                        ("cost_per_1k_input", cost_per_1k_input),
                        ("cost_per_1k_output", cost_per_1k_output),
                        ("latency_class", latency_class),
                        ("notes", notes)):
        if value is not None:
            setattr(row, attr, value)
    db.flush()
    return row


def _model_capability_dict(row) -> dict:
    return {
        "provider": row.provider, "model": row.model,
        "capabilities": _available(row),
        "context_window": row.context_window,
        "max_output": row.max_output,
        "embedding_dimensions": row.embedding_dimensions,
        "cost_per_1k_input": row.cost_per_1k_input,
        "cost_per_1k_output": row.cost_per_1k_output,
        "latency_class": row.latency_class,
    }


def _available(row) -> list[str]:
    caps = []
    if row.supports_text:
        caps.append("text")
    if row.supports_streaming:
        caps.append("streaming")
    if row.supports_tools:
        caps.append("tools")
    if row.supports_structured:
        caps.append("structured")
    if row.supports_vision:
        caps.append("vision")
    if row.embedding_dimensions:
        caps.append("embedding")
    if row.context_window and row.context_window >= 64000:
        caps.append("long_context")
    if _reasoning_like(row):
        caps.append("reasoning")
    return caps


def _reasoning_like(row) -> bool:
    name = f"{row.provider} {row.model}".lower()
    return bool(re.search(r"(reason|o1|o3|thinking)", name))


def list_capabilities(db: Session, *, provider: Optional[str] = None,
                      limit: int = 200) -> list[dict]:
    from ..models.phase16 import ProviderCapability
    q = db.query(ProviderCapability)
    if provider:
        q = q.filter(ProviderCapability.provider == provider)
    rows = q.limit(min(limit, 500)).all()
    return [_model_capability_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Routing 3.0
# ---------------------------------------------------------------------------

def route_model(db: Session, *, task: str, candidates: list[dict],
                mode: str = "balanced",
                sensitivity: str = "INTERNAL",
                required_capabilities: Optional[list] = None,
                organization_id: Optional[int] = None,
                workspace_id: Optional[int] = None,
                region_id: Optional[str] = None) -> dict:
    """Deterministic quality-aware routing over capability candidates.

    Respects capability (task + explicit), policy allowlists/deny lists and
    sensitivity routing (see governance4), residency (region rules) and the
    requested mode. Never silently violates budget or governance: blocked
    candidates are excluded with reasons recorded.
    """
    if mode not in ROUTING_MODES:
        raise ValueError("unknown routing mode")
    needed = set(TASK_CAPABILITIES.get(task, ["text"]))
    if required_capabilities:
        needed |= set(required_capabilities)
    eligible, excluded = [], []
    for cand in candidates:
        caps = set(cand.get("capabilities") or [])
        if not needed.issubset(caps):
            excluded.append({"candidate": cand,
                             "reason": "missing capability"})
            continue
        decision = _policy_check(db, cand, organization_id=organization_id,
                                 workspace_id=workspace_id,
                                 sensitivity=sensitivity,
                                 region_id=region_id)
        if not decision["allowed"]:
            excluded.append({"candidate": cand,
                             "reason": decision["reason"]})
            continue
        eligible.append({**cand, "_score": _mode_score(cand, mode)})
    if not eligible:
        return {"selected": None, "mode": mode,
                "reason": "no candidate satisfies capability and policy",
                "excluded": excluded[:10]}
    eligible.sort(key=lambda c: (-c["_score"], c["provider"], c["model"]))
    chosen = eligible[0]
    chosen.pop("_score", None)
    return {"selected": chosen, "mode": mode, "eligible": len(eligible),
            "excluded": excluded[:10]}


def _mode_score(cand: dict, mode: str) -> float:
    """Higher is better. Quality index is a documented heuristic only."""
    cost_in = float(cand.get("cost_per_1k_input") or 0.0)
    cost_out = float(cand.get("cost_per_1k_output") or 0.0)
    total_cost = cost_in + cost_out
    latency = {"fast": 3, "medium": 2, "slow": 1}.get(
        cand.get("latency_class"), 2)
    caps = set(cand.get("capabilities") or [])
    quality = 1.0
    if "reasoning" in caps:
        quality += 1.0
    if "long_context" in caps:
        quality += 0.5
    quality += 1.0 / (1.0 + total_cost * 10)
    if mode == "cheapest":
        return 1.0 / (1.0 + total_cost * 100)
    if mode == "fastest":
        return float(latency)
    if mode == "highest_quality":
        return quality
    # balanced
    return 0.5 * quality + 0.3 * latency + 0.2 * (1.0 / (1.0 + total_cost))


def _policy_check(db: Session, cand: dict, *, organization_id, workspace_id,
                  sensitivity: str, region_id: Optional[str]) -> dict:
    try:
        from .governance4 import check_model_routing
        return check_model_routing(
            db, provider=cand["provider"], model=cand["model"],
            organization_id=organization_id, workspace_id=workspace_id,
            sensitivity=sensitivity, region_id=region_id)
    except Exception as exc:  # noqa: BLE001 — fail closed on governance
        logger.error("governance check failed: %s", exc)
        return {"allowed": False, "reason": "governance check error"}


# ---------------------------------------------------------------------------
# Admission control + load shedding
# ---------------------------------------------------------------------------

def admission_control(db: Session, *, provider: str, model: str,
                      task: str, organization_id: Optional[int],
                      workspace_id: Optional[int],
                      sensitivity: str = "INTERNAL",
                      estimated_cost: Optional[float] = None,
                      budget_remaining: Optional[float] = None,
                      region_id: Optional[str] = None) -> dict:
    """Pre-call admission: health, capability, policy, budget, region."""
    from ..models.phase15 import ProviderHealth
    from ..models.phase16 import ProviderCapability
    row = (db.query(ProviderCapability)
           .filter(ProviderCapability.provider == provider,
                   ProviderCapability.model == model).first())
    if row is None:
        return {"allowed": False, "code": "PROVIDER_UNKNOWN",
                "reason": "capability not registered"}
    needed = set(TASK_CAPABILITIES.get(task, ["text"]))
    available = set(_available(row))
    if not needed.issubset(available):
        return {"allowed": False, "code": "CAPABILITY_MISSING",
                "reason": f"task {task} needs {sorted(needed)}"}
    policy = _policy_check(db, {"provider": provider, "model": model,
                                "capabilities": list(available)},
                           organization_id=organization_id,
                           workspace_id=workspace_id,
                           sensitivity=sensitivity, region_id=region_id)
    if not policy["allowed"]:
        return {"allowed": False, "code": "GOVERNANCE_BLOCKED",
                "reason": policy["reason"]}
    health = (db.query(ProviderHealth)
              .filter(ProviderHealth.provider == provider)
              .order_by(ProviderHealth.created_at.desc()).first())
    if health is not None and (getattr(health, "status", "UNKNOWN") == "DOWN"
                               or getattr(health, "circuit_state",
                                          "CLOSED") == "OPEN"):
        return {"allowed": False, "code": "PROVIDER_UNHEALTHY",
                "reason": "provider circuit/health is open or down"}
    if budget_remaining is not None and estimated_cost is not None \
            and estimated_cost > budget_remaining:
        return {"allowed": False, "code": "BUDGET_EXCEEDED",
                "reason": "estimated cost exceeds remaining budget"}
    return {"allowed": True, "code": None, "reason": "admission granted"}


def load_shedding(db: Session, *, queue_depth: int, provider_load: float,
                  priority: str, active_budget_ratio: float = 0.9) -> dict:
    """Provider capacity pressure: shed low-priority work first."""
    shedding = provider_load >= active_budget_ratio \
        or queue_depth >= int(active_budget_ratio * 10000)
    if not shedding:
        return {"shed": False, "reason": "capacity available"}
    if priority == "HIGH":
        return {"shed": False,
                "reason": "high priority admitted under pressure"}
    if priority == "NORMAL" and queue_depth < 5000:
        return {"shed": False, "reason": "moderate pressure, normal admitted"}
    return {"shed": True, "code": "PROVIDER_LOAD_SHEDDING",
            "reason": "provider capacity exhausted; retry later",
            "retryable": True}


# ---------------------------------------------------------------------------
# Fallback chain 2.0
# ---------------------------------------------------------------------------

def fallback_plan(db: Session, *, task: str,
                  primary: Optional[dict] = None,
                  secondary: Optional[dict] = None,
                  candidates: Optional[list] = None,
                  circuit_open: bool = False,
                  sensitivity: str = "INTERNAL",
                  organization_id: Optional[int] = None,
                  workspace_id: Optional[int] = None,
                  degraded_allowed: bool = True) -> dict:
    """Ordered fallback chain: primary → bounded retry → circuit breaker →
    secondary → degraded deterministic mode → final failure."""
    chain = [{"step": "primary",
              "decision": "SKIP" if circuit_open else "TRY",
              "provider": (primary or {}).get("provider"),
              "model": (primary or {}).get("model")}]
    if circuit_open:
        chain.append({"step": "retry", "decision": "SKIP",
                      "reason": "circuit open — no bounded retries on open "
                                "circuit"})
    else:
        chain.append({"step": "retry", "decision": "TRY",
                      "reason": "bounded exponential retry (max 3)"})
    if circuit_open:
        chain.append({"step": "circuit_breaker", "decision": "HALF_OPEN",
                      "reason": "probing with a single request"})
    if secondary:
        chain.append({"step": "secondary",
                      "decision": "TRY" if circuit_open else "STANDBY",
                      "provider": secondary.get("provider"),
                      "model": secondary.get("model")})
    if degraded_allowed:
        chain.append({"step": "degraded", "decision": "TRY",
                      "reason": "deterministic local mode (no external "
                                "provider)"})
    chain.append({"step": "final_failure", "decision": "FAIL",
                  "reason": "all fallback steps exhausted"})
    return {"chain": chain,
            "note": "degraded mode is deterministic and never sends data "
                    "to a provider"}


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------

def account_usage(db: Session, *, execution_id: str,
                  provider: str, model: str,
                  input_tokens: int = 0, output_tokens: int = 0,
                  embedding_tokens: int = 0,
                  estimated_cost: Optional[float] = None,
                  actual_cost: Optional[float] = None) -> dict:
    """Record token usage + cost against a durable AIExecution (absolute)."""
    from ..models.ai_execution import AIExecution
    execution = db.query(AIExecution).filter(
        AIExecution.id == execution_id).first()
    if execution is None:
        return {"recorded": False, "reason": "execution not found"}
    execution.input_tokens = int(input_tokens)
    execution.output_tokens = int(output_tokens)
    execution.total_tokens = int(input_tokens) + int(output_tokens)
    execution.model = model
    execution.provider = provider
    if estimated_cost is not None:
        execution.estimated_cost = round(estimated_cost, 6)
    if actual_cost is not None:
        execution.actual_cost = round(actual_cost, 6)
    db.flush()
    return {"recorded": True, "execution_id": execution_id,
            "tokens": execution.total_tokens,
            "cost": execution.estimated_cost}


# ---------------------------------------------------------------------------
# Shadow testing (synthetic/redacted by default)
# ---------------------------------------------------------------------------

def build_shadow_sample(*, sensitivity: str = "PUBLIC",
                        query: Optional[str] = None) -> dict:
    """Synthetic shadow input only — never production data by default."""
    if sensitivity != "PUBLIC":
        return {"kind": "shadow",
                "note": "non-public sensitivities use REDACTED synthetic "
                        "inputs only",
                "text": "[redacted synthetic shadow sample]",
                "sensitivity": sensitivity}
    return {"kind": "shadow",
            "text": query or "shadow query: what is the refund policy?",
            "sensitivity": "PUBLIC"}


def compare_shadow_results(primary: dict, secondary: dict) -> dict:
    """Offline shadow comparison: latency delta + exact-text agreement."""
    p_text = str(primary.get("text") or "")
    s_text = str(secondary.get("text") or "")
    latency_delta_ms = (float(secondary.get("latency_ms") or 0.0)
                        - float(primary.get("latency_ms") or 0.0))
    agreement = (p_text.strip().lower() == s_text.strip().lower()) if \
        (p_text and s_text) else None
    return {"agreement": agreement,
            "latency_delta_ms": round(latency_delta_ms, 1),
            "note": "offline shadow comparison on synthetic inputs only"}
