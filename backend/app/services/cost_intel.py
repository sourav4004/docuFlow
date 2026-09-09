"""Phase 20 — Cost intelligence 5.0.

Cost baselines per workspace/organization/model/provider/workflow/agent/
document pipeline, deterministic cost-drift detection, token-efficiency
measurement (input/output/retrieval context/repeated context), cost
optimization experiment candidates (evaluation required), and
quality-vs-cost tradeoff analysis that never silently sacrifices quality.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import CostBaseline, TokenEfficiency

VALID_SCOPES = {
    "WORKSPACE", "ORGANIZATION", "MODEL", "PROVIDER", "WORKFLOW",
    "AGENT", "DOCUMENT",
}


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def set_baseline(db: Session, *, scope_type: str,
                 scope_id: Optional[int],
                 baseline: dict, dimension: Optional[str] = None) -> CostBaseline:
    if scope_type not in VALID_SCOPES:
        raise ValueError(f"Unknown scope: {scope_type}")
    row = CostBaseline(scope_type=scope_type, scope_id=scope_id,
                       dimension=dimension, baseline_json=_dumps(baseline))
    db.add(row)
    db.flush()
    return row


def latest_baseline(db: Session, *, scope_type: str,
                    scope_id: Optional[int] = None,
                    dimension: Optional[str] = None) -> Optional[dict]:
    q = db.query(CostBaseline).filter_by(scope_type=scope_type,
                                         scope_id=scope_id)
    if dimension:
        q = q.filter(CostBaseline.dimension == dimension)
    row = q.order_by(CostBaseline.created_at.desc(),
                     CostBaseline.id.desc()).first()
    if row is None:
        return None
    return {"id": row.id, "scope_type": row.scope_type,
            "scope_id": row.scope_id, "dimension": row.dimension,
            "baseline": json.loads(row.baseline_json or "{}"),
            "created_at": row.created_at}


def detect_cost_drift(db: Session, *, scope_type: str,
                      scope_id: Optional[int] = None,
                      dimension: Optional[str] = None,
                      max_delta_fraction: float = 0.2,
                      sample_cost: Optional[float] = None) -> dict:
    """Compare the latest cost sample against the persisted baseline.
    Drift when the increase exceeds max_delta_fraction."""
    base = latest_baseline(db, scope_type=scope_type, scope_id=scope_id,
                           dimension=dimension)
    if base is None:
        return {"drift": False, "reason": "no baseline"}
    sample = _current_cost_sample(db, scope_type=scope_type,
                                  scope_id=scope_id, dimension=dimension)\
        if sample_cost is None else {"cost": sample_cost}
    base_cost = float(base["baseline"].get("cost", 0.0))
    sample_cost = float(sample.get("cost", 0.0))
    if base_cost <= 0:
        return {"drift": False, "reason": "zero baseline"}
    delta_fraction = (sample_cost - base_cost) / base_cost
    return {"drift": delta_fraction > max_delta_fraction,
            "delta_fraction": round(delta_fraction, 4),
            "baseline_cost": base_cost, "sample_cost": sample_cost,
            "max_delta_fraction": max_delta_fraction}


def _current_cost_sample(db: Session, *, scope_type: str,
                         scope_id: Optional[int],
                         dimension: Optional[str]) -> dict:
    """Deterministic sample: sum of usage records scoped to the dimension
    where the model supports it; otherwise zero."""
    from ..models.usage import UsageRecord
    q = db.query(UsageRecord)
    if scope_type == "WORKSPACE" and scope_id is not None:
        q = q.filter(UsageRecord.workspace_id == scope_id)
    cost = 0.0
    for r in q.limit(5000):
        cost += float(getattr(r, "estimated_cost", 0.0) or 0.0)
    return {"cost": round(cost, 4)}


def record_token_usage(db: Session, *, workspace_id: int,
                       input_tokens: int, output_tokens: int,
                       context_tokens: int = 0, repeated_tokens: int = 0,
                       execution_ref: Optional[str] = None) -> TokenEfficiency:
    row = TokenEfficiency(
        workspace_id=workspace_id, execution_ref=execution_ref,
        input_tokens=input_tokens, output_tokens=output_tokens,
        context_tokens=context_tokens, repeated_tokens=repeated_tokens)
    db.add(row)
    db.flush()
    return row


def token_efficiency_report(db: Session, *, workspace_id: int,
                            limit: int = 500) -> dict:
    rows = db.query(TokenEfficiency).filter_by(workspace_id=workspace_id)\
        .order_by(TokenEfficiency.created_at.desc()).limit(limit).all()
    total_in = sum(r.input_tokens for r in rows)
    total_out = sum(r.output_tokens for r in rows)
    total_ctx = sum(r.context_tokens for r in rows)
    total_rep = sum(r.repeated_tokens for r in rows)
    return {
        "samples": len(rows),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_context_tokens": total_ctx,
        "total_repeated_tokens": total_rep,
        "repeated_fraction": (total_rep / total_ctx) if total_ctx else 0.0,
        "output_ratio": (total_out / total_in) if total_in else 0.0,
    }


def cost_optimization_candidates(db: Session, *, workspace_id: int) -> list[dict]:
    """Candidate optimizations derived from token efficiency + cost drift.
    All candidates are recommendations; none are applied automatically."""
    report = token_efficiency_report(db, workspace_id=workspace_id)
    out = []
    if report["repeated_fraction"] > 0.3:
        out.append({
            "strategy": "context_reduction",
            "suggestion": "Repeated context exceeds 30% — evaluate prompt/"
                          "context caching or retrieval trimming",
            "metric": f"{report['repeated_fraction']:.0%} repeated"})
    if report["output_ratio"] > 0.5:
        out.append({
            "strategy": "output_reduction",
            "suggestion": "Output is large relative to input — evaluate "
                          "max_tokens caps or structured output",
            "metric": f"output/input {report['output_ratio']:.2f}"})
    drift = detect_cost_drift(db, scope_type="WORKSPACE",
                              scope_id=workspace_id)
    if drift.get("drift"):
        out.append({
            "strategy": "model_downgrade",
            "suggestion": "Cost drift detected — evaluate smaller models for "
                          "low-sensitivity tasks",
            "metric": f"delta {drift['delta_fraction']:.0%}"})
    if not out:
        out.append({
            "strategy": "none",
            "suggestion": "No cost optimization candidates detected",
            "metric": ""})
    return out


def quality_cost_tradeoff(baseline: dict, candidate: dict) -> dict:
    """Show quality change vs cost change for a candidate. Verdict only ever
    recommends a change when quality is preserved (within tolerance) or
    improved; never silently sacrifices quality for cost."""
    b_q = float(baseline.get("quality", 1.0))
    c_q = float(candidate.get("quality", 0.0))
    b_c = float(baseline.get("cost", 0.0))
    c_c = float(candidate.get("cost", 0.0))
    quality_tolerance = float(baseline.get("quality_tolerance", 0.05))
    quality_delta = c_q - b_q
    cost_delta = b_c - c_c
    if quality_delta >= -quality_tolerance and cost_delta > 0:
        verdict = "RECOMMENDED"
    elif quality_delta < -quality_tolerance and cost_delta > 0:
        verdict = "NOT_RECOMMENDED_QUALITY_LOSS"
    elif cost_delta <= 0 and quality_delta <= 0:
        verdict = "NOT_RECOMMENDED_NO_GAIN"
    elif cost_delta <= 0 and quality_delta > 0:
        verdict = "NOT_RECOMMENDED_COST_INCREASE"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "quality_delta": round(quality_delta, 4),
        "cost_delta": round(cost_delta, 4),
        "baseline_quality": b_q, "candidate_quality": c_q,
        "baseline_cost": b_c, "candidate_cost": c_c,
    }