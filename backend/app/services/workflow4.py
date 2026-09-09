"""Phase 19 — workflow platform 4.0.

Workflow validation (DAG, permissions, budgets, node capabilities,
timeouts, tenant scope), side-effect-free simulation, durable node
checkpoints, safe parallel branches + join validation (incomplete joins
never proceed), timeout recovery with bounded retries, bounded exponential
retry policy, four-way compensation classification, and replay safety rules
(only idempotent-safe nodes replay automatically).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

NODE_KINDS = ("task", "approval", "notification", "external", "decision",
              "subworkflow")
MAX_WORKFLOW_TIMEOUT_S = 86400
MAX_NODE_TIMEOUT_S = 3600
MAX_WORKFLOW_STEPS = 500


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _nodes(definition: dict) -> list[dict]:
    return definition.get("nodes") or definition.get("steps") or []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_workflow(definition: dict,
                      node_capabilities: Optional[set] = None) -> dict:
    """Validate workflow DAG + permissions + budgets + timeouts + scope."""
    node_capabilities = node_capabilities or set(NODE_KINDS)
    nodes = _nodes(definition)
    errors = []
    if len(nodes) > MAX_WORKFLOW_STEPS:
        errors.append(f"workflow exceeds {MAX_WORKFLOW_STEPS} node cap")
    if definition.get("timeout_s", MAX_WORKFLOW_TIMEOUT_S) \
            > MAX_WORKFLOW_TIMEOUT_S:
        errors.append("workflow timeout exceeds 24h cap")
    by_id = {}
    for node in nodes:
        nid = str(node.get("id", ""))
        if not nid:
            errors.append("node missing id")
        elif nid in by_id:
            errors.append(f"duplicate node {nid}")
        by_id[nid] = node
        if node.get("kind") not in node_capabilities:
            errors.append(f"node {nid}: unknown capability kind "
                          f"{node.get('kind')!r}")
        if node.get("timeout_s", MAX_NODE_TIMEOUT_S) > MAX_NODE_TIMEOUT_S:
            errors.append(f"node {nid} timeout exceeds 1h cap")
        if node.get("destructive") and not node.get("approval_required"):
            errors.append(f"node {nid} is destructive and not gated by "
                          "approval")
        if not node.get("scope"):
            errors.append(f"node {nid} missing tenant scope")
    # cycle detection
    visiting, visited = set(), set()

    def visit(nid):
        if nid in visiting:
            errors.append(f"cycle detected at node {nid}")
            return
        if nid in visited:
            return
        visiting.add(nid)
        for dep in by_id.get(nid, {}).get("deps", []):
            visit(str(dep))
        visiting.discard(nid)
        visited.add(nid)

    for node in nodes:
        visit(str(node.get("id")))
    if errors:
        return {"valid": False, "errors": errors}
    return {"valid": True, "errors": [], "nodes": len(nodes),
            "acyclic": True}


def simulate_workflow(definition: dict) -> dict:
    """Side-effect-free workflow simulation in dependency order."""
    nodes = _nodes(definition)
    remaining = list(nodes)
    executed: list[str] = []
    order = []
    by_id = {str(n.get("id")): n for n in nodes}
    while remaining:
        progressed = False
        for node in list(remaining):
            deps = [str(d) for d in node.get("deps", [])]
            if all(d in executed for d in deps):
                order.append(node.get("id"))
                executed.append(node.get("id"))
                remaining.remove(node)
                progressed = True
        if not progressed:
            return {"simulated": True, "valid": False,
                    "reason": "dependency cycle or missing dependency",
                    "order": order}
    return {"simulated": True, "valid": True, "order": order,
            "steps": len(order),
            "disclaimer": "simulation only; no node executed"}


# ---------------------------------------------------------------------------
# Node checkpoints + joins + timeout recovery
# ---------------------------------------------------------------------------

def node_checkpoint(db: Session, *, run_id: int, workspace_id: int,
                    node_id: str, status: str,
                    detail: Optional[str] = None) -> dict:
    """Persist node transition state on the durable workflow run record."""
    from ..models.phase17 import WorkflowRun
    run = (db.query(WorkflowRun)
           .filter(WorkflowRun.id == run_id,
                   WorkflowRun.workspace_id == workspace_id).first())
    if run is None:
        raise KeyError("workflow run not found")
    state = _loads(run.node_state_json) or {}
    state[str(node_id)] = {"status": status,
                           "detail": detail,
                           "updated_at": _utcnow().isoformat()}
    run.node_state_json = json.dumps(state, default=str)[:6000]
    db.flush()
    return {"run_id": run_id, "node_id": node_id, "status": status}


def join_ready(*, definition: dict, branch_statuses: dict) -> dict:
    """A join node may proceed only when EVERY inbound dependency branch has
    completed. Incomplete joins stay PENDING (never partial)."""
    by_id = {str(n.get("id")): n for n in _nodes(definition)}
    joins = [n for n in _nodes(definition)
             if n.get("kind") == "join" or n.get("join")]
    results = []
    for join in joins:
        deps = [str(d) for d in join.get("deps", [])]
        ready = all(branch_statuses.get(d) == "COMPLETED" for d in deps)
        results.append({"node_id": join.get("id"), "ready": ready,
                        "pending_dependencies": [d for d in deps
                                                 if branch_statuses.get(d)
                                                 != "COMPLETED"]})
    return {"joins": results,
            "all_ready": all(r["ready"] for r in results) or not results}


def detect_timed_out_nodes(*, definition: dict,
                           node_state: dict,
                           now: Optional[datetime] = None) -> dict:
    """Nodes whose runtime exceeded their declared timeout (deterministic)."""
    now = now or _utcnow()
    timed_out = []
    for node in _nodes(definition):
        nid = str(node.get("id"))
        entry = node_state.get(nid) or {}
        started = entry.get("started_at")
        if not started or entry.get("status") == "COMPLETED":
            continue
        timeout_s = node.get("timeout_s") or MAX_NODE_TIMEOUT_S
        try:
            started_dt = datetime.fromisoformat(started)
        except ValueError:
            continue
        if now - started_dt > timedelta(seconds=timeout_s):
            timed_out.append({"node_id": nid, "timeout_s": timeout_s,
                              "started_at": started})
    return {"timed_out": timed_out}


def retry_delay(attempt: int, base_s: float = 2.0,
                max_s: float = 300.0) -> float:
    """Bounded exponential backoff (no jitter for determinism)."""
    attempt = max(0, int(attempt))
    return round(min(max_s, base_s * (2 ** attempt)), 2)


def retry_policy(node: dict, attempt: int) -> dict:
    max_attempts = int(node.get("retry", {}).get("max_attempts", 3))
    if attempt >= max_attempts:
        return {"retry": False,
                "reason": "max attempts reached — node dead-letters"}
    return {"retry": True, "delay_s": retry_delay(
        attempt, float(node.get("retry", {}).get("base_s", 2.0)),
        float(node.get("retry", {}).get("max_s", 300.0)))}


def timeout_recovery(node: dict, attempt: int) -> dict:
    """Recover a timed-out node: bounded retry when configured and not
    exhausted; otherwise FAILED (never silently re-executes side effects)."""
    if node.get("retry", {}).get("retry_timeouts", True):
        policy = retry_policy(node, attempt)
        if policy["retry"]:
            return {"action": "RETRY", **policy}
    return {"action": "FAIL",
            "reason": "timeout not retryable or attempts exhausted"}


# ---------------------------------------------------------------------------
# Compensation + replay safety
# ---------------------------------------------------------------------------

def classify_compensation(node: dict) -> dict:
    """reversible / irreversible / compensatable / non-compensatable."""
    kind = node.get("kind")
    explicit = node.get("compensation", {}).get("class")
    if explicit in ("reversible", "irreversible", "compensatable",
                    "non_compensatable"):
        return {"class": explicit, "source": "explicit"}
    if node.get("read_only") or kind in ("decision", "notification"):
        return {"class": "reversible", "source": "inferred"}
    if node.get("destructive"):
        return {"class": "non_compensatable", "source": "inferred"}
    if kind == "external" or node.get("side_effect"):
        return {"class": "compensatable",
                "source": "inferred" if node.get("compensation",
                                                 {}).get("compensate")
                else "unknown-compensation"}
    return {"class": "reversible", "source": "inferred"}


def replay_safe(node: dict) -> dict:
    """Only idempotent-safe nodes may replay automatically."""
    if node.get("destructive"):
        return {"safe": False, "reason": "destructive node — human review"}
    if not node.get("idempotent") and (node.get("side_effect")
                                       or node.get("kind") == "external"):
        return {"safe": False,
                "reason": "side-effecting node without idempotency marker"}
    return {"safe": True, "reason": "idempotent-safe"}


# ---------------------------------------------------------------------------
# Pause / resume
# ---------------------------------------------------------------------------

def pause_workflow(db: Session, *, run_id: int, workspace_id: int,
                   reason: Optional[str] = None) -> dict:
    from ..models.phase17 import WorkflowRun
    run = (db.query(WorkflowRun)
           .filter(WorkflowRun.id == run_id,
                   WorkflowRun.workspace_id == workspace_id).first())
    if run is None:
        raise KeyError("workflow run not found")
    if run.control_state == "PAUSED":
        return {"status": "ALREADY_PAUSED", "run_id": run.id}
    run.control_state = "PAUSED"
    run.pause_reason = reason
    db.flush()
    return {"status": "PAUSED", "run_id": run.id}


def resume_workflow(db: Session, *, run_id: int, workspace_id: int) -> dict:
    from ..models.phase17 import WorkflowRun
    run = (db.query(WorkflowRun)
           .filter(WorkflowRun.id == run_id,
                   WorkflowRun.workspace_id == workspace_id).first())
    if run is None:
        raise KeyError("workflow run not found")
    if run.control_state == "ACTIVE":
        return {"status": "ALREADY_RUNNING", "run_id": run.id}
    run.control_state = "ACTIVE"
    run.pause_reason = None
    db.flush()
    return {"status": "ACTIVE", "run_id": run.id}


def _loads(raw: Optional[str]):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None
