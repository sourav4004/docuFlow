"""Provider platform 2.0 — capability registry, health windows, routing 3.0.

- Capability registry: every (provider, model) declares what it can do.
  The router checks capabilities BEFORE selection — a vision task never
  reaches a text-only model.
- Health 2.0: success/error/latency/timeout/rate-limit counters with
  deterministic circuit breaker transitions (never hammer a failing provider).
- Routing 3.0: deterministic, explainable model choice considering task
  capability, sensitivity policy, cost budget, latency class and health.
  RESTRICTED data only ever routes to explicitly allowed providers.

Real provider credentials are optional and config-only (``LLM_API_KEY``,
``LLM_BASE_URL``, ``LLM_MODEL``). Without them the platform reports that the
gateway is unconfigured and callers fall back to the deterministic fake
provider — tests never require real credentials.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase16 import ProviderCapability
from ..models.phase15 import ProviderHealth

logger = logging.getLogger(__name__)

CAPABILITY_KEYS = (
    "text", "vision", "tools", "structured", "streaming",
)

SENSITIVITY_LEVELS = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")

CIRCUIT_FAILURE_THRESHOLD = 5
CIRCUIT_COOLDOWN_SECONDS = 300


class ProviderRoutingError(Exception):
    """Raised when no provider/model satisfies the routing constraints."""


class CapabilityValidationError(Exception):
    """Raised when a capability record is invalid."""


# ---------------------------------------------------------------------------
# Capability registry
# ---------------------------------------------------------------------------

def upsert_capability(
    db: Session,
    provider: str,
    model: str,
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
    latency_class: Optional[str] = None,
    notes: Optional[str] = None,
) -> ProviderCapability:
    cap = (
        db.query(ProviderCapability)
        .filter(ProviderCapability.provider == provider,
                ProviderCapability.model == model)
        .first()
    )
    if cap is None:
        cap = ProviderCapability(provider=provider, model=model)
        db.add(cap)
    cap.supports_text = supports_text
    cap.supports_vision = supports_vision
    cap.supports_tools = supports_tools
    cap.supports_structured = supports_structured
    cap.supports_streaming = supports_streaming
    cap.context_window = context_window
    cap.max_output = max_output
    cap.embedding_dimensions = embedding_dimensions
    cap.cost_per_1k_input = cost_per_1k_input
    cap.cost_per_1k_output = cost_per_1k_output
    cap.latency_class = latency_class if latency_class in (
        "fast", "medium", "slow", None) else "medium"
    cap.notes = notes
    db.flush()
    return cap


def get_capability(db: Session, provider: str, model: str) -> Optional[ProviderCapability]:
    return (
        db.query(ProviderCapability)
        .filter(ProviderCapability.provider == provider,
                ProviderCapability.model == model)
        .first()
    )


def list_capabilities(db: Session, provider: Optional[str] = None,
                      limit: int = 200) -> list[ProviderCapability]:
    query = db.query(ProviderCapability)
    if provider:
        query = query.filter(ProviderCapability.provider == provider)
    return query.order_by(ProviderCapability.provider,
                          ProviderCapability.model).limit(min(limit, 500)).all()


def capabilities_meet(cap: ProviderCapability, requires: dict) -> tuple[bool, str]:
    """Check that ``requires`` (e.g. {'tools': True, 'vision': True}) holds."""
    for key in CAPABILITY_KEYS:
        want = requires.get(key)
        if want is None:
            continue
        attr = {"text": "supports_text", "vision": "supports_vision",
                "tools": "supports_tools", "structured": "supports_structured",
                "streaming": "supports_streaming"}[key]
        if want and not getattr(cap, attr):
            return False, f"model lacks {key} capability"
    if requires.get("min_context") and cap.context_window:
        if cap.context_window < requires["min_context"]:
            return False, "context window too small"
    return True, "ok"


# ---------------------------------------------------------------------------
# Health 2.0 — rolling counters + deterministic circuit breaker
# ---------------------------------------------------------------------------

def _scale_down(health: ProviderHealth) -> None:
    # Keep counters bounded so old history decays (rolling window effect).
    total = (health.success_count or 0) + (health.failure_count or 0)
    if total > 5000:
        health.success_count = int((health.success_count or 0) / 10)
        health.failure_count = int((health.failure_count or 0) / 10)
        health.consecutive_failures = int((health.consecutive_failures or 0) / 10)


def circuit_state_for(
    health: Optional[ProviderHealth],
    now: Optional[datetime] = None,
    failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
    cooldown_seconds: int = CIRCUIT_COOLDOWN_SECONDS,
) -> str:
    """Deterministic effective circuit state for a provider/model.

    OPEN after ``failure_threshold`` consecutive failures; HALF_OPEN once the
    cooldown has passed (a successful probe closes the circuit); CLOSED
    otherwise.
    """
    now = now or datetime.now(timezone.utc)
    if health is None or health.circuit_state == "CLOSED":
        if health and health.consecutive_failures >= failure_threshold:
            return "OPEN"
        return "CLOSED"
    if health.circuit_state == "OPEN":
        last = health.last_checked_at or health.updated_at
        if last is not None:
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if (now - last).total_seconds() >= cooldown_seconds:
                return "HALF_OPEN"
        return "OPEN"
    return "HALF_OPEN"


def record_provider_result(
    db: Session,
    provider: str,
    model: str,
    ok: bool,
    latency_ms: Optional[float] = None,
    error_class: Optional[str] = None,
    error_text: Optional[str] = None,
    now: Optional[datetime] = None,
    failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
) -> ProviderHealth:
    """Record one provider call result and update the circuit state."""
    now = now or datetime.now(timezone.utc)
    health = (
        db.query(ProviderHealth)
        .filter(ProviderHealth.provider == provider,
                ProviderHealth.model == model)
        .first()
    )
    if health is None:
        health = ProviderHealth(provider=provider, model=model)
        db.add(health)
    _scale_down(health)

    if ok:
        health.success_count = (health.success_count or 0) + 1
        health.consecutive_failures = 0
        # A successful probe after the cooldown closes an open circuit.
        effective = circuit_state_for(health, now=now,
                                      failure_threshold=failure_threshold)
        if effective == "HALF_OPEN":
            health.circuit_state = "CLOSED"
    else:
        health.failure_count = (health.failure_count or 0) + 1
        health.consecutive_failures = (health.consecutive_failures or 0) + 1
        health.last_error = (error_text or error_class or "unknown")[:2000]
        if health.consecutive_failures >= failure_threshold:
            health.circuit_state = "OPEN"
    if latency_ms is not None:
        old = health.avg_latency_ms
        total = (health.success_count or 0) + (health.failure_count or 0)
        # Exponential moving average — stable and cheap.
        health.avg_latency_ms = (
            latency_ms if old is None else round(old * 0.9 + latency_ms * 0.1, 1)
        )
    health.last_checked_at = now
    health.updated_at = now
    health.status = "UP" if health.circuit_state == "CLOSED" else (
        "DEGRADED" if health.circuit_state == "HALF_OPEN" else "DOWN")
    db.flush()
    return health


def provider_status(db: Session, provider: Optional[str] = None,
                    limit: int = 200) -> list[ProviderHealth]:
    query = db.query(ProviderHealth)
    if provider:
        query = query.filter(ProviderHealth.provider == provider)
    return query.order_by(ProviderHealth.updated_at.desc()).limit(
        min(limit, 500)).all()


# ---------------------------------------------------------------------------
# Sensitivity policy
# ---------------------------------------------------------------------------

def is_allowed_for_sensitivity(
    provider: str,
    sensitivity: str,
    allowed_providers: Optional[list[str]] = None,
) -> bool:
    """RESTRICTED/CONFIDENTIAL data routes only to explicitly allowed
    providers — a deny-by-default rule, never a frontend decision."""
    if sensitivity not in SENSITIVITY_LEVELS:
        sensitivity = "INTERNAL"
    if sensitivity in ("RESTRICTED", "CONFIDENTIAL"):
        if not allowed_providers:
            return False
        return provider in allowed_providers
    if allowed_providers:
        return provider in allowed_providers
    return True


# ---------------------------------------------------------------------------
# Routing 3.0 — deterministic + explainable
# ---------------------------------------------------------------------------

def recommend_model(
    db: Session,
    requires: Optional[dict] = None,
    sensitivity: str = "INTERNAL",
    allowed_providers: Optional[list[str]] = None,
    max_cost_per_1k: Optional[float] = None,
    prefer: str = "quality",  # quality / latency / cost
    now: Optional[datetime] = None,
) -> dict:
    """Choose the best (provider, model) deterministically.

    Selection factors (in order): capability match → sensitivity policy →
    health (OPEN circuits excluded) → cost ceiling → weighted score by
    ``prefer`` → stable name tie-break.

    Raises ProviderRoutingError when nothing satisfies the constraints —
    routing never silently downgrades a required capability.
    """
    requires = requires or {}
    caps = list_capabilities(db, limit=500)
    now = now or datetime.now(timezone.utc)
    candidates = []
    for cap in caps:
        ok, reason = capabilities_meet(cap, requires)
        if not ok:
            continue
        if not is_allowed_for_sensitivity(cap.provider, sensitivity,
                                          allowed_providers):
            continue
        health = (
            db.query(ProviderHealth)
            .filter(ProviderHealth.provider == cap.provider,
                    ProviderHealth.model == cap.model)
            .first()
        )
        circuit = circuit_state_for(health, now=now)
        if circuit == "OPEN":
            continue
        est_in = cap.cost_per_1k_input or 0.0
        est_out = cap.cost_per_1k_output or 0.0
        if max_cost_per_1k is not None and est_in > max_cost_per_1k:
            continue
        # Weighted quality/latency/cost score (0..100 scale, additive).
        quality = 60.0
        if requires.get("structured"):
            quality += 15.0
        if requires.get("tools"):
            quality += 10.0
        if requires.get("vision"):
            quality += 10.0
        latency_score = {"fast": 90.0, "medium": 60.0, "slow": 30.0}.get(
            cap.latency_class or "medium", 60.0)
        if requires.get("min_context") and cap.context_window:
            ratio = min(1.0, cap.context_window / requires["min_context"])
            quality += 5.0 * ratio
        cost_score = 100.0
        if est_in:
            cost_score = max(0.0, 100.0 - est_in * 5.0)
        weight = {"quality": (1.0, 0.2, 0.3),
                  "latency": (0.3, 1.0, 0.3),
                  "cost": (0.3, 0.2, 1.0)}[prefer]
        score = (quality * weight[0] + latency_score * weight[1]
                 + cost_score * weight[2])
        degraded = -10.0 if circuit == "HALF_OPEN" else 0.0
        candidates.append({
            "provider": cap.provider,
            "model": cap.model,
            "score": round(score + degraded, 2),
            "circuit": circuit,
            "latency_class": cap.latency_class,
            "context_window": cap.context_window,
            "est_cost_per_1k_input": est_in,
            "factors": {
                "capability_match": True,
                "sensitivity_allowed": True,
                "quality_component": round(quality, 1),
                "latency_component": round(latency_score, 1),
                "cost_component": round(cost_score, 1),
            },
        })
    if not candidates:
        raise ProviderRoutingError(
            "No model satisfies the routing constraints "
            "(capability/sensitivity/health/cost)"
        )
    candidates.sort(key=lambda c: (-c["score"], c["provider"], c["model"]))
    best = candidates[0]
    return {
        "provider": best["provider"],
        "model": best["model"],
        "score": best["score"],
        "circuit_state": best["circuit"],
        "factors": best["factors"],
        "routed_from": len(candidates),
    }


# ---------------------------------------------------------------------------
# Config-driven gateway status (real providers stay optional)
# ---------------------------------------------------------------------------

def gateway_status() -> dict:
    """Report whether a real OpenAI-compatible gateway is configured.

    Exposes only booleans and safe names — never keys or URLs with secrets.
    """
    from ..core.config import settings
    provider = getattr(settings, "llm_provider", "fake").lower()
    api_key_set = bool(getattr(settings, "llm_api_key", None))
    base_url = getattr(settings, "llm_base_url", None)
    return {
        "configured_provider": provider,
        "real_gateway_configured": api_key_set,
        "model": getattr(settings, "llm_model", None),
        "timeout_s": getattr(settings, "llm_timeout", None),
        "default_is_fake": not api_key_set,
        "base_url_safe": (base_url or "").split("//")[-1] if base_url else None,
        "note": "Real provider use requires LLM_API_KEY/LLM_BASE_URL config; "
                "the deterministic fake provider is the default.",
    }
