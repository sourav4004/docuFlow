"""Phase 20 — Model / provider optimization.

Per-model and per-provider performance profiles, candidate routing
recommendations (never auto-applied), deterministic routing simulation
against historical/synthetic workloads, promotion gates requiring policy +
quality + cost validation, and provider anomaly detection (latency spikes,
error spikes, unexpected cost, quality degradation).
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import (
    ProviderProfile, RoutingRecommendation, ProviderAnomaly,
)


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def record_profile(db: Session, *, provider: str, model: str,
                   metrics: dict, period: str = "daily") -> ProviderProfile:
    row = ProviderProfile(provider=provider, model=model, period=period,
                          metrics_json=_dumps(metrics))
    db.add(row)
    db.flush()
    return row


def profile_summary(db: Session, *, provider: Optional[str] = None,
                    limit: int = 100) -> list[dict]:
    q = db.query(ProviderProfile)
    if provider:
        q = q.filter(ProviderProfile.provider == provider)
    rows = q.order_by(ProviderProfile.created_at.desc()).limit(limit).all()
    return [{"id": r.id, "provider": r.provider, "model": r.model,
             "period": r.period,
             "metrics": json.loads(r.metrics_json or "{}"),
             "created_at": r.created_at} for r in rows]


def routing_recommendations(db: Session, *, profiles: Optional[list[dict]] = None) -> list[dict]:
    """Generate candidate routing improvements from performance profiles.
    Uses the provided profiles (or the persisted ones) and records each
    candidate in PROPOSED state."""
    src = profiles if profiles is not None else profile_summary(db)
    out = []
    by_model: dict[str, list[dict]] = {}
    for p in src:
        by_model.setdefault(p["model"], []).append(p)
    for model, rows in by_model.items():
        latest = rows[0]
        m = latest["metrics"]
        latency = float(m.get("latency_p95_ms") or m.get("latency_ms", 0.0))
        err = float(m.get("error_rate", 0.0))
        cost = float(m.get("cost_per_1k", 0.0))
        if err > 0.05:
            out.append(_recommend(
                db, "provider_routing", model,
                f"Error rate {err:.2%} exceeds 5% for {model} — evaluate "
                "failover/secondary", "error_rate"))
        if latency > 5000:
            out.append(_recommend(
                db, "model_routing", model,
                f"p95 latency {latency:.0f}ms for {model} — evaluate "
                "faster tier", "latency"))
        if cost > 0 and cost > 0.02:
            out.append(_recommend(
                db, "cost", model,
                f"Cost {cost:.4f}/1k tokens for {model} — evaluate cheaper "
                "equivalent for low-sensitivity tasks", "cost"))
    return out


def _recommend(db: Session, domain: str, candidate: str, rationale: str,
               gain: str) -> dict:
    rec = RoutingRecommendation(domain=domain, candidate=candidate,
                                rationale=rationale, expected_gain=gain)
    db.add(rec)
    db.flush()
    return {"id": rec.id, "domain": domain, "candidate": candidate,
            "rationale": rationale, "status": rec.status}


def list_routing_recommendations(db: Session, *,
                                 status: Optional[str] = None,
                                 limit: int = 100) -> list[dict]:
    q = db.query(RoutingRecommendation)
    if status:
        q = q.filter(RoutingRecommendation.status == status)
    rows = q.order_by(RoutingRecommendation.created_at.desc())\
        .limit(limit).all()
    return [{"id": r.id, "domain": r.domain, "candidate": r.candidate,
             "rationale": r.rationale, "expected_gain": r.expected_gain,
             "status": r.status, "created_at": r.created_at} for r in rows]


def simulate_routing(db: Session, *, workload: list[dict],
                     candidates: list[dict], metric: str = "cost") -> dict:
    """Simulate a proposed routing policy against a workload.

    workload: [{task, model, provider, cost, latency_ms, quality}]
    candidates: [{task_pattern, model, provider, cost_per_call,
                  latency_ms, quality}]
    Returns the aggregated cost/latency/quality delta per candidate policy.
    """
    results = []
    for cand in candidates:
        total_cost = 0.0
        total_latency = 0.0
        total_quality = 0.0
        n = 0
        matched = 0
        for w in workload:
            n += 1
            if _task_matches(cand, w):
                matched += 1
                total_cost += float(cand.get("cost_per_call", 0.0))
                total_latency += float(cand.get("latency_ms", 0.0))
                total_quality += float(cand.get("quality", 1.0))
            else:
                total_cost += float(w.get("cost", 0.0))
                total_latency += float(w.get("latency_ms", 0.0))
                total_quality += float(w.get("quality", 1.0))
        results.append({
            "candidate": cand.get("name", cand.get("model")),
            "matched": matched, "total_cost": round(total_cost, 4),
            "total_latency_ms": round(total_latency, 1),
            "avg_quality": round(total_quality / n if n else 0.0, 4),
        })
    baseline = {
        "total_cost": round(sum(float(w.get("cost", 0.0)) for w in workload), 4),
        "total_latency_ms": round(sum(float(w.get("latency_ms", 0.0))
                                      for w in workload), 1),
        "avg_quality": round(sum(float(w.get("quality", 1.0))
                                 for w in workload) / len(workload)
                             if workload else 0.0, 4),
    }
    return {"baseline": baseline, "candidates": results,
            "workload_size": len(workload)}


def _task_matches(cand: dict, work: dict) -> bool:
    pattern = cand.get("task_pattern", "")
    task = work.get("task", "")
    return pattern in task


def promotion_validation(candidate: dict, policy: dict) -> dict:
    """Routing promotion gate: policy + quality + cost validation."""
    reasons = []
    allowed_models = set(policy.get("allowed_models") or [])
    allowed_providers = set(policy.get("allowed_providers") or [])
    if allowed_models and candidate.get("model") not in allowed_models:
        reasons.append("model not in policy allowlist")
    if allowed_providers and candidate.get("provider") not in allowed_providers:
        reasons.append("provider not in policy allowlist")
    min_quality = float(policy.get("min_quality", 0.0))
    if float(candidate.get("quality", 0.0)) < min_quality:
        reasons.append("quality below policy minimum")
    max_cost = policy.get("max_cost_per_call")
    if max_cost is not None and float(candidate.get("cost_per_call", 0.0)) \
            > float(max_cost):
        reasons.append("cost above policy maximum")
    return {"valid": not reasons, "reasons": reasons}


def detect_anomalies(db: Session, *, provider: Optional[str] = None,
                     threshold: float = 3.0) -> list[dict]:
    """Detect provider anomalies from persisted profiles: latency spikes,
    error spikes, unexpected cost, quality degradation (deterministic z-score
    vs the provider's own history)."""
    rows = db.query(ProviderProfile)
    if provider:
        rows = rows.filter(ProviderProfile.provider == provider)
    rows = rows.order_by(ProviderProfile.created_at.asc(),
                         ProviderProfile.id.asc()).limit(500).all()
    anomalies = []
    by_provider_model: dict[tuple, list] = {}
    for r in rows:
        key = (r.provider, r.model)
        by_provider_model.setdefault(key, []).append(r)
    for (prov, model), items in by_provider_model.items():
        if len(items) < 4:
            continue
        def _metric(i, key, alt=None):
            m = json.loads(i.metrics_json or "{}")
            value = m.get(key)
            if value is None and alt:
                value = m.get(alt)
            return float(value or 0.0)
        metrics = [_metric(i, "latency_p95_ms", "latency_ms")
                   for i in items]
        errs = [_metric(i, "error_rate") for i in items]
        costs = [_metric(i, "cost_per_1k") for i in items]
        # items are ordered oldest -> newest; the last is the latest sample
        for label, series in (("latency", metrics), ("error_rate", errs),
                              ("cost_per_1k", costs)):
            history = series[:-1]
            latest = series[-1]
            mean = sum(history) / len(history)
            variance = sum((x - mean) ** 2 for x in history) / len(history)
            std = variance ** 0.5
            if std < 1e-9:
                continue
            z = (latest - mean) / std
            if z > threshold:
                anom = ProviderAnomaly(
                    provider=prov, model=model, anomaly_type=label,
                    detail=(f"{label} z-score {z:.2f} "
                            f"(latest {series[-1]:.4f} vs mean {mean:.4f})"))
                db.add(anom)
                db.flush()
                anomalies.append({"provider": prov, "model": model,
                                  "anomaly_type": label, "z_score": round(z, 2),
                                  "severity": anom.severity})
    return anomalies


def list_anomalies(db: Session, *, resolved: Optional[bool] = None,
                   limit: int = 50) -> list[dict]:
    q = db.query(ProviderAnomaly)
    if resolved is not None:
        q = q.filter(ProviderAnomaly.resolved == resolved)
    rows = q.order_by(ProviderAnomaly.created_at.desc()).limit(limit).all()
    return [{"id": a.id, "provider": a.provider, "model": a.model,
             "anomaly_type": a.anomaly_type, "detail": a.detail,
             "severity": a.severity, "resolved": a.resolved,
             "created_at": a.created_at} for a in rows]