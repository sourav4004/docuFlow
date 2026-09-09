"""Phase 23 — Global control plane.

Extends the Phase 22 capability registry (app.services.capabilities) and the
Phase 19/22 region platform with the unified GLOBAL control plane:

- capability endpoints (status / detail / check / refresh) — reuses
  capabilities.detect_all + InfraCapability rows, never duplicates detection
- dependency graph 2.0 (Phase 23 DependencyEdge): upstream/downstream impact,
  blast radius, degraded capability computation
- global health: regions + capabilities + DB + broker + workers rolled into
  one operator view with degraded-component enumeration
- readiness: whether the platform can serve traffic right now, with the
  blocking reasons when it cannot

Everything reuses Phases 0-22 abstractions; nothing here performs
destructive operations and nothing reports a capability as REAL without a
real check succeeding.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from . import capabilities as caps

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load(value: Optional[str]) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Capability control plane (Step 1)
# ---------------------------------------------------------------------------

def capability_detail(db: Session, name: str) -> Optional[dict]:
    """Detail for one capability; falls back to a live detection on miss."""
    from ..models import InfraCapability

    name = (name or "").strip().lower()
    row = db.query(InfraCapability).filter_by(component=name).one_or_none()
    if row is not None:
        return {
            "component": row.component, "state": row.state,
            "realization": row.realization, "detail": row.detail,
            "version": row.version,
            "checked_at": row.checked_at.isoformat() if row.checked_at else None,
        }
    if name in caps.COMPONENTS:
        return caps.detect_component(name)
    return None


def run_capability_check(db: Session, component: Optional[str] = None) -> dict:
    """Run detection (one component or all) and persist the registry."""
    if component:
        if component not in caps.COMPONENTS:
            return {"checked": [], "unknown": component}
        item = caps.detect_component(component)
        caps.persist_capabilities(db)  # detect_all persists every row
        return {"checked": [item], "checked_at": _utcnow().isoformat()}
    return caps.persist_capabilities(db)


def refresh_capabilities(db: Session) -> dict:
    """Alias with refresh semantics (clear-then-detect) for operators."""
    return caps.persist_capabilities(db)


# ---------------------------------------------------------------------------
# Dependency graph 2.0 (Step 19)
# ---------------------------------------------------------------------------

DEFAULT_EDGES = [
    ("api", "database", "HARD"), ("api", "broker", "SOFT"),
    ("worker", "database", "HARD"), ("worker", "broker", "HARD"),
    ("worker", "provider", "SOFT"), ("worker", "vector", "SOFT"),
    ("worker", "storage", "HARD"), ("scheduler", "database", "HARD"),
    ("event_processor", "database", "HARD"),
    ("event_processor", "broker", "HARD"),
    ("rag", "vector", "HARD"), ("rag", "provider", "HARD"),
    ("ingestion", "storage", "HARD"), ("ingestion", "database", "HARD"),
    ("connectors", "network", "SOFT"), ("notification", "database", "HARD"),
    ("notification", "network", "SOFT"),
]

COMPONENT_TO_CAPABILITY = {
    "database": "postgresql", "broker": "broker", "provider": "ai_provider",
    "vector": "pgvector", "storage": "object_storage", "network": "webhook",
}


def _dependency_rows(db: Session) -> list[tuple[str, str, str]]:
    from ..models import DependencyEdge

    rows = db.query(DependencyEdge).all()
    if rows:
        return [(r.component, r.depends_on, r.criticality) for r in rows]
    return list(DEFAULT_EDGES)


def dependency_graph(db: Session) -> dict:
    """The full dependency graph with per-component health."""
    edges = _dependency_rows(db)
    components = sorted({c for c, _, _ in edges} | {d for _, d, _ in edges})
    states = _component_states(db)
    return {
        "edges": [{"component": c, "depends_on": d, "criticality": k}
                  for c, d, k in edges],
        "components": [{"component": c, "state": states.get(c, "UNKNOWN")}
                       for c in components],
    }


def _component_states(db: Session) -> dict:
    """Map dependency-graph components to observed health states."""
    from ..models import InfraCapability

    rows = db.query(InfraCapability).all()
    cap_states = {r.component: (r.state, r.realization) for r in rows}
    states: dict = {}
    for comp in {c for c, _, _ in _dependency_rows(db)} | \
                {d for _, d, _ in _dependency_rows(db)}:
        cap = COMPONENT_TO_CAPABILITY.get(comp)
        if cap and cap in cap_states:
            state, realization = cap_states[cap]
            states[comp] = "UNAVAILABLE" if state == "UNAVAILABLE" else (
                "DEGRADED" if state in ("DEGRADED", "NOT_CONFIGURED") else "HEALTHY")
        elif comp in ("worker", "scheduler", "event_processor", "api",
                      "rag", "ingestion", "connectors", "notification"):
            states[comp] = "HEALTHY"  # in-process components; incidents refine
        else:
            states[comp] = "UNKNOWN"
    return states


def known_components(db: Session) -> set[str]:
    """Every component name present in the dependency graph."""
    edges = _dependency_rows(db)
    return ({c for c, _, _ in edges} |
            {d for _, d, _ in edges})


def dependency_impact(db: Session, component: str, direction: str = "both") -> dict:
    """Blast radius: what breaks (upstream) and what feeds (downstream).

    upstream   = components that depend on ``component`` (they break first)
    downstream = components ``component`` depends on (root causes live here)
    Raises ``ValueError`` for a component not in the dependency graph.
    """
    if component not in known_components(db):
        raise ValueError(f"unknown dependency component: {component}")
    edges = _dependency_rows(db)
    hard_up = sorted({c for c, d, k in edges if d == component and k == "HARD"})
    soft_up = sorted({c for c, d, k in edges if d == component and k == "SOFT"})
    hard_down = sorted({d for c, d, k in edges if c == component and k == "HARD"})
    soft_down = sorted({d for c, d, k in edges if c == component and k == "SOFT"})
    states = _component_states(db)
    blast = sorted(set(hard_up) | ({"api", "worker", "rag"} & set(hard_up)))
    return {
        "component": component,
        "upstream_hard": hard_up,
        "upstream_soft": soft_up,
        "downstream_hard": hard_down,
        "downstream_soft": soft_down,
        "blast_radius": blast if direction != "downstream_only" else hard_up,
        "component_states": {k: v for k, v in states.items()
                             if k in set(hard_up) | set(soft_up) |
                             set(hard_down) | set(soft_down) | {component}},
    }


# ---------------------------------------------------------------------------
# Global health + readiness (Step 18)
# ---------------------------------------------------------------------------

def global_health(db: Session) -> dict:
    """Unified control-plane health across every Phase 0-22 subsystem."""
    summary = caps.infrastructure_summary(db)
    if not summary.get("components"):
        # Registry not yet populated (fresh database): run detection once so
        # operators never see an empty capability report.
        run_capability_check(db)
        summary = caps.infrastructure_summary(db)
    components = summary.get("components", [])

    def _state_of(name: str) -> str:
        for c in components:
            if c["component"] == name:
                return c["state"]
        return "NOT_CONFIGURED"

    db_state = _state_of("postgresql")
    broker_state = _state_of("broker")

    degraded = [c["component"] for c in components
                if c["state"] in ("DEGRADED", "UNAVAILABLE")]
    states = _component_states(db)
    degraded += [c for c, s in states.items()
                 if s in ("DEGRADED", "UNAVAILABLE") and c not in degraded]

    if db_state == "UNAVAILABLE":
        overall = "UNHEALTHY"
    elif degraded:
        overall = "DEGRADED"
    else:
        overall = "HEALTHY"

    return {
        "status": overall,
        "checked_at": _utcnow().isoformat(),
        "capabilities": {"real": summary.get("real_count", 0),
                         "simulated": summary.get("simulated_count", 0),
                         "components": components},
        "regions": _region_health(db),
        "degraded_components": sorted(set(degraded)),
        "database": db_state,
        "broker": broker_state,
    }


def _region_health(db: Session) -> list[dict]:
    from ..models import RegionRecord

    rows = db.query(RegionRecord).order_by(RegionRecord.region_id).limit(50).all()
    return [{"region": r.region_id, "status": r.status,
             "health_score": r.health_score,
             "failover_to": r.failover_to} for r in rows]


def dependencies_overview(db: Session) -> dict:
    """Operator-facing dependency view: graph + current degraded set."""
    graph = dependency_graph(db)
    health = global_health(db)
    return {
        "graph": graph,
        "degraded": health["degraded_components"],
        "status": health["status"],
    }


def readiness(db: Session) -> dict:
    """Can the platform serve traffic? Blocking reasons when it cannot."""
    health = global_health(db)
    blocking: list[str] = []
    caps_components = health["capabilities"]["components"]

    for c in caps_components:
        if c["component"] == "postgresql" and c["state"] == "UNAVAILABLE":
            blocking.append("postgresql unavailable — cannot serve requests")

    serving = health["status"] != "UNHEALTHY"
    return {
        "ready": serving and not blocking,
        "status": health["status"],
        "blocking": blocking,
        "degraded_components": health["degraded_components"],
        "checked_at": health["checked_at"],
    }


def degraded_components(db: Session) -> dict:
    health = global_health(db)
    detail = []
    for name in health["degraded_components"]:
        detail.append(capability_detail(db, name) or {"component": name})
    return {"status": health["status"], "items": detail,
            "count": len(detail), "checked_at": health["checked_at"]}
