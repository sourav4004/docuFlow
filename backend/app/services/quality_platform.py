"""Phase 19 — AI quality platform.

Golden datasets (retrieval/citations/temporal/conflicts/multi-document),
repeatable offline evaluation runs, provider/model comparison on synthetic
datasets, quality-regression detection between versions, configurable
quality gates, and feedback aggregation (thumbs/citations/corrections).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _dumps(value) -> str:
    return json.dumps(value, default=str, sort_keys=True)


# ---------------------------------------------------------------------------
# Golden datasets (deterministic, in-code)
# ---------------------------------------------------------------------------

GOLDEN_RETRIEVAL = {
    "name": "retrieval_basic",
    "kind": "retrieval",
    "queries": [
        {"query": "refund policy deadline", "relevant": {"d1", "d3"}},
        {"query": "employee benefits eligibility", "relevant": {"d2"}},
        {"query": "security incident reporting", "relevant": {"d4", "d5"}},
    ],
}

GOLDEN_CITATION = {
    "name": "citation_basic",
    "kind": "citations",
    "claims": [
        "Refund requests must arrive within 30 days.",
        "Benefits eligibility requires 90 days of service.",
    ],
    "evidence": [
        {"id": "e1", "content": "Refund requests must arrive within 30 "
                                "days of purchase."},
        {"id": "e2", "content": "Eligibility begins after 90 days of "
                                "service."},
    ],
}

GOLDEN_TEMPORAL = {
    "name": "temporal_basic",
    "kind": "temporal",
    "as_of": "2025-06-01T00:00:00+00:00",
    "evidence": [
        {"id": "t1", "content": "Policy A applies as of 2025-01-01.",
         "valid_from": "2025-01-01T00:00:00+00:00"},
        {"id": "t2", "content": "Policy A expires 2024-12-31.",
         "valid_to": "2024-12-31T23:59:59+00:00"},
    ],
}


def golden_datasets() -> dict:
    return {"retrieval": GOLDEN_RETRIEVAL, "citations": GOLDEN_CITATION,
            "temporal": GOLDEN_TEMPORAL}


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_retrieval(retrieved: list[list], dataset: Optional[dict] =
                       None) -> dict:
    """Offline retrieval evaluation against the golden dataset."""
    from .vector_ops2 import retrieval_benchmark as benchmark
    dataset = dataset or GOLDEN_RETRIEVAL
    relevant = [set(q["relevant"]) for q in dataset["queries"]]
    if len(retrieved) != len(relevant):
        raise ValueError("retrieved rows must match dataset query count")
    return benchmark(mode="offline", retrieved=retrieved,
                     relevant=relevant)


def evaluate_citations(claims: Optional[list] = None,
                       evidence: Optional[list] = None) -> dict:
    from .rag7 import (claim_matrix2, citation_coverage2,
                       citation_correctness2)
    claims = claims if claims is not None else GOLDEN_CITATION["claims"]
    evidence = evidence if evidence is not None else \
        GOLDEN_CITATION["evidence"]
    matrix = claim_matrix2(claims, evidence)
    coverage = citation_coverage2(claims, evidence)
    correct = [citation_correctness2(c, evidence)["citation_ok"]
               for c in claims]
    return {"claims": matrix["total"], "supported": matrix["supported"],
            "coverage": coverage["coverage"],
            "correctness": round(sum(correct) / len(correct), 4)
            if correct else 1.0}


def run_evaluation(db: Session, *, dataset_name: str,
                   metrics: dict) -> dict:
    """Persist one repeatable offline evaluation run."""
    from ..models.phase19 import EvaluationRun
    passed = _within_gates(metrics)
    row = EvaluationRun(dataset_name=dataset_name, mode="offline",
                        sample_count=int(metrics.get("sample_count")
                                         or metrics.get("queries") or 0),
                        metric_json=_dumps(metrics), passed=passed)
    db.add(row)
    db.flush()
    return {"evaluation_id": row.id, "dataset": dataset_name,
            "passed": passed, "metrics": metrics}


def _within_gates(metrics: dict) -> bool:
    for key, minimum in (("recall", 0.3), ("mrr", 0.2),
                         ("coverage", 0.3)):
        value = metrics.get(key)
        if value is not None and value < minimum:
            return False
    return True


def list_evaluations(db: Session, *, dataset_name: Optional[str] = None,
                     limit: int = 50) -> list:
    from ..models.phase19 import EvaluationRun
    q = db.query(EvaluationRun)
    if dataset_name:
        q = q.filter(EvaluationRun.dataset_name == dataset_name)
    return q.order_by(EvaluationRun.created_at.desc()
                      ).limit(min(limit, 200)).all()


# ---------------------------------------------------------------------------
# Model comparison + quality regression + gates + feedback
# ---------------------------------------------------------------------------

def compare_models(results_a: dict, results_b: dict) -> dict:
    """Compare provider/model outputs on synthetic datasets (offline)."""
    metrics = sorted(set(results_a) & set(results_b))
    wins = {"a": 0, "b": 0, "tie": 0}
    deltas = {}
    for metric in metrics:
        va, vb = results_a.get(metric), results_b.get(metric)
        if not isinstance(va, (int, float)) or \
                not isinstance(vb, (int, float)):
            continue
        deltas[metric] = round(va - vb, 4)
        if va > vb:
            wins["a"] += 1
        elif vb > va:
            wins["b"] += 1
        else:
            wins["tie"] += 1
    return {"wins": wins, "deltas": deltas,
            "note": "comparison uses synthetic datasets; production "
                    "behavior requires real evaluation"}


def quality_regression(current: dict, previous: dict,
                       threshold: float = 0.05) -> dict:
    """Detect degradation between versions on shared metrics."""
    regressions = []
    for metric, value in current.items():
        if metric not in previous or not isinstance(value, (int, float)) \
                or not isinstance(previous[metric], (int, float)):
            continue
        delta = value - previous[metric]
        if delta < -threshold:
            regressions.append({"metric": metric,
                                "previous": previous[metric],
                                "current": value,
                                "delta": round(delta, 4)})
    return {"regressed": bool(regressions), "regressions": regressions,
            "note": "quality gates use these deltas for CI gating"}


def add_gate(db: Session, *, name: str, metric: str, operator: str,
             threshold: float,
             organization_id: Optional[int] = None) -> dict:
    from ..models.phase19 import QualityGate
    row = QualityGate(organization_id=organization_id, name=name,
                      metric=metric, operator=operator, threshold=threshold)
    db.add(row)
    db.flush()
    return {"gate_id": row.id, "name": name, "metric": metric}


def check_gate(db: Session, *, metric: str, value: float) -> dict:
    from ..models.phase19 import QualityGate
    gates = db.query(QualityGate).filter(
        QualityGate.enabled.is_(True),
        QualityGate.metric == metric).all()
    results = []
    for gate in gates:
        ok = {"<=": value <= gate.threshold,
              ">=": value >= gate.threshold,
              "<": value < gate.threshold,
              ">": value > gate.threshold}.get(gate.operator, False)
        results.append({"gate": gate.name, "metric": metric,
                        "operator": gate.operator,
                        "threshold": gate.threshold, "value": value,
                        "passed": ok})
    return {"gates_evaluated": len(results),
            "all_passed": all(r["passed"] for r in results),
            "results": results}


def aggregate_feedback(db: Session, *, workspace_id: int,
                       days: int = 30) -> dict:
    from ..models.search_intel import AIFeedback
    from datetime import datetime, timezone, timedelta
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = (db.query(AIFeedback)
            .filter(AIFeedback.workspace_id == workspace_id,
                    AIFeedback.created_at >= since)
            .limit(5000).all())
    thumbs_up = sum(1 for r in rows if r.rating == "thumbs_up")
    categories = {}
    for r in rows:
        if r.category:
            categories[r.category] = categories.get(r.category, 0) + 1
    return {"feedback_count": len(rows),
            "thumbs_up_rate": round(thumbs_up / len(rows), 4)
            if rows else None,
            "categories": categories,
            "note": "feedback is aggregated, never used to auto-train "
                    "models"}
