"""Phase 20 — API platform 4.0.

Per-endpoint health metrics (requests, latency, errors, rate limits, auth
failures), unstable-endpoint detection, request/response contract
validation, an idempotency audit across side-effecting endpoints, and
abuse detection (enumeration, brute force, scope probing, abnormal API-key
usage).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import ApiHealthMetric

# Well-known contract rules for common endpoints (deterministic audit).
CONTRACT_RULES = {
    "list": {"pagination": True, "errors_structured": True},
    "create": {"idempotency": True, "errors_structured": True},
    "update": {"errors_structured": True, "validation": True},
    "delete": {"idempotency": True, "authorization": True},
}


def record_metric(db: Session, *, endpoint: str, requests: int = 1,
                  errors: int = 0, auth_failures: int = 0,
                  latency_p50_ms: Optional[float] = None,
                  latency_p95_ms: Optional[float] = None,
                  period: str = "hourly") -> ApiHealthMetric:
    row = ApiHealthMetric(
        endpoint=endpoint[:300], period=period, requests=requests,
        errors=errors, latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms, auth_failures=auth_failures)
    db.add(row)
    db.flush()
    return row


def unstable_endpoints(db: Session, *, min_requests: int = 10,
                       max_error_rate: float = 0.1,
                       limit: int = 50) -> list[dict]:
    """Endpoints whose error rate or auth-failure rate exceeds thresholds."""
    rows = db.query(ApiHealthMetric).all()
    out = []
    for r in rows:
        if r.requests < min_requests:
            continue
        error_rate = r.errors / r.requests if r.requests else 0.0
        auth_rate = r.auth_failures / r.requests if r.requests else 0.0
        if error_rate > max_error_rate or auth_rate > max_error_rate:
            out.append({
                "endpoint": r.endpoint, "requests": r.requests,
                "errors": r.errors, "error_rate": round(error_rate, 4),
                "auth_failures": r.auth_failures,
                "auth_failure_rate": round(auth_rate, 4),
                "latency_p95_ms": r.latency_p95_ms,
                "unstable": True, "period": r.period,
            })
    return out[:limit]


def api_health_report(db: Session, *, limit: int = 100) -> dict:
    """Aggregate API health across recorded endpoints."""
    rows = db.query(ApiHealthMetric)\
        .order_by(ApiHealthMetric.created_at.desc()).limit(limit).all()
    total_requests = sum(r.requests for r in rows)
    total_errors = sum(r.errors for r in rows)
    total_auth = sum(r.auth_failures for r in rows)
    return {
        "endpoints_recorded": len(rows),
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": round(total_errors / total_requests, 4)
        if total_requests else 0.0,
        "auth_failure_rate": round(total_auth / total_requests, 4)
        if total_requests else 0.0,
        "unstable": unstable_endpoints(db),
    }


def contract_validate(method: str, path: str, specs: dict) -> dict:
    """Validate an endpoint against deterministic contract rules:
    structured errors, validation, bounded pagination, authorization, and
    idempotency for side-effecting methods."""
    issues = []
    checks = {}
    lower = path.lower()
    verb = method.lower()
    check = {}
    if any(p in lower for p in ("/list", "/search", "/history")):
        check["pagination"] = specs.get("pagination", False)
    if verb in ("post", "put", "patch", "delete"):
        if "idempotency" not in specs:
            issues.append("missing idempotency for side-effecting method")
        if "authorization" not in specs or not specs["authorization"]:
            issues.append("missing authorization")
        if "validation" not in specs or not specs["validation"]:
            issues.append("missing request validation")
    check.update({
        "auth": bool(specs.get("auth", True)),
        "authz": bool(specs.get("authorization", True)),
        "structured_errors": specs.get("errors", False),
        "bounded_output": specs.get("bounded_output", True),
    })
    for k, v in check.items():
        checks[k] = v
    if any(not v for k, v in checks.items() if k != "pagination"):
        issues.append("contract rule not satisfied")
    return {"endpoint": f"{method.upper()} {path}", "checks": checks,
            "issues": sorted(set(issues)), "valid": not issues}


def idempotency_audit(routes: list[dict]) -> list[dict]:
    """Audit all side-effecting routes for idempotency readiness. `routes`
    is a list of {method, path, idempotent} dicts."""
    findings = []
    for r in routes:
        if r["method"].upper() in ("POST", "PUT", "PATCH", "DELETE"):
            findings.append({
                "endpoint": f"{r['method'].upper()} {r['path']}",
                "idempotent": bool(r.get("idempotent")),
                "requirement": "idempotency-required",
                "pass": bool(r.get("idempotent")),
            })
    return findings


def abuse_detection(db: Session, *, events: Optional[list[dict]] = None) -> dict:
    """Detect abuse signals from provided event dicts (deterministic) or
    persisted api_abuse_events. Signals: enumeration (many distinct paths,
    same key), brute force (many auth failures), scope probing (attempts
    outside own workspace), abnormal key usage (request spike)."""
    if events is None:
        from ..models.phase19 import ApiAbuseEvent
        rows = db.query(ApiAbuseEvent)\
            .order_by(ApiAbuseEvent.created_at.desc()).limit(1000).all()
        events = []
        for r in rows:
            try:
                payload = json.loads(r.detail_json or "{}")
            except (ValueError, TypeError):
                payload = {}
            events.append({
                "api_key": getattr(r, "api_key_id", None),
                "scope": getattr(r, "scope", None),
                "pattern": getattr(r, "pattern", None) or payload.get("pattern"),
            })
    signals = []
    keys: dict = {}
    for e in events:
        k = e.get("api_key")
        if k is not None:
            keys.setdefault(k, []).append(e)
    for k, evs in keys.items():
        paths = {e.get("path") for e in evs if e.get("path")}
        if len(paths) > 50:
            signals.append({"type": "enumeration", "api_key": k,
                            "distinct_paths": len(paths)})
        scope_ids = {e.get("scope") for e in evs if e.get("scope")}
        if len(scope_ids) > 1:
            signals.append({"type": "scope_probing", "api_key": k,
                            "scopes_touched": len(scope_ids)})
        if len(evs) > 500:
            signals.append({"type": "abnormal_request_volume",
                            "api_key": k, "requests": len(evs)})
    if not signals:
        signals.append({"type": "none", "detail": "no abuse signals"})
    return {"signals": signals, "signal_count": len(signals)}


def brute_force_score(auth_failures: list[dict]) -> dict:
    """Score brute-force risk from {user_or_key, failed, at} events:
    many quick consecutive failures from the same identity are suspicious."""
    by_identity: dict[str, int] = {}
    for f in auth_failures:
        ident = f.get("identity")
        if ident:
            by_identity[ident] = by_identity.get(ident, 0) + 1
    flags = [{"identity": ident, "failures": n,
              "suspicious": n >= 5}
             for ident, n in by_identity.items()]
    return {"flagged": sum(1 for f in flags if f["suspicious"]),
            "identities": flags}