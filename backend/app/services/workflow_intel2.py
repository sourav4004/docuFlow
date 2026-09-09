"""Phase 20 — Workflow intelligence (Phase 20 analytics).

Workflow success analytics, bottleneck detection (slow nodes), failure
hotspots (repeatedly failing nodes), deterministic optimization
recommendations, and side-effect-free workflow simulation to validate
proposed changes before activation.

Named workflow_intel2 to avoid clashing with the pre-existing Phase 14
workflow_intel module (DAG validation + versioned workflow definitions).
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import WorkflowIntelligence


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def record_node(db: Session, *, run_id: int, node_id: str,
                workspace_id: int, metrics: dict) -> WorkflowIntelligence:
    row = WorkflowIntelligence(run_id=run_id, node_id=node_id,
                               workspace_id=workspace_id,
                               metrics_json=_dumps(metrics))
    db.add(row)
    db.flush()
    return row


def success_analytics(db: Session, *, workspace_id: int,
                      limit: int = 1000) -> dict:
    rows = db.query(WorkflowIntelligence).filter_by(
        workspace_id=workspace_id)\
        .order_by(WorkflowIntelligence.created_at.desc()).limit(limit).all()
    runs: dict[int, list] = {}
    for r in rows:
        runs.setdefault(r.run_id, []).append(r)
    completed = sum(1 for rs in runs.values()
                    if _any_metric(rs, "completed"))
    failed = sum(1 for rs in runs.values() if _any_metric(rs, "failed"))
    retried = sum(1 for r in rows if _any_metric([r], "retries", 0) > 0)
    timed_out = sum(1 for r in rows if _any_metric([r], "timed_out"))
    compensated = sum(1 for r in rows if _any_metric([r], "compensated"))
    return {"runs": len(runs), "completed": completed, "failed": failed,
            "completion_rate": (completed / len(runs)) if runs else 0.0,
            "retried_nodes": retried, "timed_out_nodes": timed_out,
            "compensated_nodes": compensated}


def _any_metric(rows: list, key: str, default: Optional[float] = None):
    for r in rows:
        v = json.loads(r.metrics_json or "{}").get(key)
        if v is not None:
            return v
    return default


def bottlenecks(db: Session, *, workspace_id: int,
                limit: int = 1000) -> list[dict]:
    rows = db.query(WorkflowIntelligence).filter_by(
        workspace_id=workspace_id)\
        .order_by(WorkflowIntelligence.created_at.desc()).limit(limit).all()
    per_node: dict[str, list] = {}
    for r in rows:
        per_node.setdefault(r.node_id, []).append(r)
    out = []
    for node, rs in per_node.items():
        durations = [float(json.loads(r.metrics_json).get("duration_s", 0.0))
                     for r in rs]
        failures = sum(1 for r in rs if
                       json.loads(r.metrics_json).get("status") == "failed")
        avg = sum(durations) / len(durations) if durations else 0.0
        out.append({
            "node_id": node, "runs": len(rs),
            "avg_duration_s": round(avg, 2),
            "failures": failures,
            "failure_rate": round(failures / len(rs), 4) if rs else 0.0,
        })
    out.sort(key=lambda x: -x["avg_duration_s"])
    return out[:20]


def failure_hotspots(db: Session, *, workspace_id: int,
                     limit: int = 1000) -> list[dict]:
    nodes = bottlenecks(db, workspace_id=workspace_id, limit=limit)
    return [n for n in nodes if n["failure_rate"] > 0.2]


def optimization_recommendations(db: Session, *, workspace_id: int,
                                 limit: int = 1000) -> list[dict]:
    recs = []
    for b in bottlenecks(db, workspace_id=workspace_id, limit=limit):
        if b["avg_duration_s"] > 60:
            recs.append({
                "node": b["node_id"], "kind": "parallelize",
                "suggestion": f"Node avg {b['avg_duration_s']}s — evaluate "
                              "parallelization for independent upstream "
                              "branches"})
        if b["failure_rate"] > 0.2:
            recs.append({
                "node": b["node_id"], "kind": "retry_policy",
                "suggestion": f"Node failure rate {b['failure_rate']:.0%} — "
                              "evaluate retry policy or timeout adjustment"})
    if not recs:
        recs.append({"node": None, "kind": "none",
                     "suggestion": "No workflow optimization candidates"})
    return recs[:limit]


def simulate(workflow: dict, node_metrics: dict) -> dict:
    """Side-effect-free workflow simulation: predict duration/cost/quality
    from provided per-node metrics."""
    nodes = workflow.get("nodes") or []
    total_duration = 0.0
    total_cost = 0.0
    critical_path = []
    for node in nodes:
        nid = node.get("id") or node.get("name")
        m = node_metrics.get(nid, {})
        duration = float(m.get("avg_duration_s", 0.0))
        cost = float(m.get("avg_cost", 0.0))
        total_duration += duration
        total_cost += cost
        critical_path.append({"id": nid, "duration_s": duration})
    parallel_gain = 0.0
    if workflow.get("parallelizable"):
        # upper bound: 50% ceiling on savings from parallel branches
        parallel_gain = min(total_duration * 0.5,
                            total_duration * 0.3)
    return {
        "serial_duration_s": round(total_duration, 2),
        "parallel_duration_s": round(
            max(total_duration - parallel_gain, 0.0), 2),
        "estimated_cost": round(total_cost, 4),
        "critical_path": critical_path,
    }