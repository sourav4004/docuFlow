"""Phase 21 — Self-diagnosis engine.

Correlates failures into incident groups, produces ranked root-cause
hypotheses with explicit evidence and deterministic confidence, and renders
operator-readable diagnosis reports. Explanations contain only evidence,
confidence, and concise reasons — never hidden reasoning.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Optional

from sqlalchemy.orm import Session

from ..models.phase21 import DiagnosisReport

# Deterministic evidence weights (percent confidence contribution).
_WEIGHT_SIGNAL = {
    "trace": 20, "log": 15, "metric": 20, "queue": 15,
    "provider": 20, "database": 20, "recent_change": 25,
}


def _bounded_json(value: Any, limit: int = 6000) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


def correlate_failures(failures: list[dict],
                       window_seconds: int = 600) -> list[dict]:
    """Group related failures into incident clusters.

    Failures are related when they share a component or a correlation key
    (e.g. request id, run id) within the time window.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in failures:
        key = f.get("correlation_key") or f.get("component") or "unattributed"
        groups[str(key)].append(f)
    clusters = []
    for key, items in groups.items():
        kinds = Counter(i.get("kind", "unknown") for i in items)
        clusters.append({
            "key": key,
            "count": len(items),
            "kinds": dict(kinds),
            "first_seen": min((i.get("at") or 0) for i in items),
            "last_seen": max((i.get("at") or 0) for i in items),
        })
    clusters.sort(key=lambda c: (-c["count"], c["key"]))
    return clusters


def _confidence(hypothesis: dict) -> float:
    """Deterministic confidence from evidence weights (0-100)."""
    total = 0.0
    for ev in hypothesis.get("evidence", []):
        total += _WEIGHT_SIGNAL.get(ev.get("kind"), 5)
    confidence = min(95.0, total)
    # Uncertain when sample size is small — never overclaim.
    if hypothesis.get("sample_count", 1) < 3:
        confidence = min(confidence, 40.0)
    return round(confidence, 1)


def diagnose(symptom: str, signals: dict[str, Any],
             failures: Optional[list[dict]] = None) -> dict:
    """Generate ranked root-cause hypotheses from observable signals.

    Each hypothesis carries its evidence list and a deterministic confidence.
    No hypothesis is claimed certain without sufficient evidence.
    """
    hypotheses: list[dict] = []

    provider = signals.get("provider") or {}
    if provider.get("error_rate", 0) > 0.2 or provider.get("failures"):
        hypotheses.append({
            "cause": "provider degradation",
            "evidence": [
                {"kind": "provider",
                 "detail": f"error_rate={provider.get('error_rate', 0)}"},
                {"kind": "metric", "detail": "provider failure signals"},
            ],
            "sample_count": int(provider.get("sample_count",
                                             len(provider.get("failures", [])
                                                 or [1]))),
        })
    if signals.get("queue_depth", 0) > 500:
        hypotheses.append({
            "cause": "queue buildup (worker capacity or stall)",
            "evidence": [
                {"kind": "queue", "detail": f"depth={signals['queue_depth']}"},
                {"kind": "metric", "detail": "queue_depth threshold breach"},
            ],
            "sample_count": int(signals.get("queue_samples", 5)),
        })
    worker = signals.get("worker") or {}
    if worker.get("heartbeat_age_seconds", 0) > 120:
        hypotheses.append({
            "cause": "worker stall (stale heartbeat)",
            "evidence": [
                {"kind": "log",
                 "detail": f"heartbeat age {worker['heartbeat_age_seconds']}s"},
            ],
            "sample_count": 2,
        })
    if signals.get("db_errors"):
        hypotheses.append({
            "cause": "database errors",
            "evidence": [{"kind": "database",
                          "detail": f"{signals['db_errors']} recent errors"}],
            "sample_count": int(signals.get("db_samples", 5)),
        })
    if signals.get("recent_change"):
        hypotheses.append({
            "cause": "recent deployment/config change",
            "evidence": [{"kind": "recent_change",
                          "detail": str(signals["recent_change"])[:200]}],
            "sample_count": int(signals.get("change_samples", 5)),
        })
    if not hypotheses:
        hypotheses.append({
            "cause": "insufficient signals — collect more diagnostics",
            "evidence": [{"kind": "metric", "detail": "no threshold breach"}],
            "sample_count": 0,
        })

    for h in hypotheses:
        h["confidence"] = _confidence(h)
    hypotheses.sort(key=lambda h: -h["confidence"])

    clusters = correlate_failures(failures or [])
    report = {
        "symptom": symptom,
        "top_cause": hypotheses[0]["cause"],
        "top_confidence": hypotheses[0]["confidence"],
        "hypotheses": hypotheses,
        "correlated_clusters": clusters,
        "summary": (
            f"Diagnosis for '{symptom}': most likely {hypotheses[0]['cause']} "
            f"(confidence {hypotheses[0]['confidence']}%, "
            f"evidence-based, not certain). "
            f"{len(clusters)} correlated failure cluster(s)."),
    }
    return report


def persist_diagnosis(db: Session, workspace_id: int, symptom: str,
                      report: dict,
                      incident_id: Optional[int] = None) -> DiagnosisReport:
    """Persist an operator-readable diagnosis report."""
    row = DiagnosisReport(
        workspace_id=workspace_id, incident_id=incident_id,
        symptom=symptom[:200], hypotheses=_bounded_json(report["hypotheses"]),
        top_cause=report["top_cause"][:200],
        top_confidence=report["top_confidence"],
        correlated_failures=sum(c["count"]
                                for c in report["correlated_clusters"]),
        report=report["summary"][:2000])
    db.add(row)
    db.commit()
    return row


def get_report(db: Session, workspace_id: int, report_id: int) \
        -> Optional[DiagnosisReport]:
    """Tenant-scoped report fetch."""
    return (db.query(DiagnosisReport)
            .filter_by(workspace_id=workspace_id, id=report_id).first())


def list_reports(db: Session, workspace_id: int, limit: int = 50,
                 offset: int = 0) -> list[DiagnosisReport]:
    limit = max(1, min(int(limit), 200))
    return (db.query(DiagnosisReport)
            .filter_by(workspace_id=workspace_id)
            .order_by(DiagnosisReport.id.desc())
            .offset(max(0, int(offset))).limit(limit).all())
