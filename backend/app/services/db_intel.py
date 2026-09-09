"""Phase 20 — Database platform.

Database health tracking (connections, slow queries, locks, table growth,
migration head), query-regression detection against baselines, index
effectiveness review (where environment permits), and high-growth table
monitoring for jobs/traces/events/usage/notifications/analytics/embeddings.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from ..models.phase20 import DbHealthMetric

HIGH_GROWTH_TABLES = [
    "worker_jobs", "trace_spans", "events", "usage_records",
    "notifications", "search_events", "embeddings", "api_health_metrics",
    "quality_trends", "experiment_runs", "feedback_events",
]


def record_metric(db: Session, *, metric: str, value: Optional[float] = None,
                  detail: Optional[dict] = None) -> DbHealthMetric:
    row = DbHealthMetric(metric=metric, value=value,
                         detail_json=json.dumps(detail) if detail else None)
    db.add(row)
    db.flush()
    return row


def health_report(db: Session) -> dict:
    """Latest recorded database health metrics."""
    rows = db.query(DbHealthMetric)\
        .order_by(DbHealthMetric.created_at.desc(),
                  DbHealthMetric.id.desc()).limit(200).all()
    latest: dict[str, Optional[float]] = {}
    for r in rows:
        if r.metric not in latest:
            latest[r.metric] = r.value
    return {"metrics": latest, "recorded": len(rows)}


def query_regression(baseline: dict, current: dict,
                     max_ratio: float = 1.5) -> dict:
    """Compare important query latency against baseline. Regression when the
    current latency exceeds max_ratio * baseline."""
    regressions = []
    for query, base_ms in baseline.items():
        cur = current.get(query)
        if cur is None:
            continue
        if base_ms <= 0:
            continue
        ratio = cur / base_ms
        if ratio > max_ratio:
            regressions.append({
                "query": query, "baseline_ms": base_ms,
                "current_ms": cur, "ratio": round(ratio, 2),
            })
    return {"regressed": bool(regressions),
            "regressions": regressions,
            "max_ratio": max_ratio}


def index_effectiveness(indexes: list[dict]) -> list[dict]:
    """Review index usage. Each index: {name, table, scans, uses}.
    Indexes scanned frequently but rarely used are candidates for review;
    indexes never used are reported as potentially unused."""
    findings = []
    for idx in indexes:
        scans = int(idx.get("scans", 0))
        uses = int(idx.get("uses", 0))
        if uses == 0 and scans > 0:
            findings.append({**idx, "finding": "unused_index",
                             "recommendation": "review/remove"})
        elif scans > 0 and uses / max(scans, 1) < 0.1:
            findings.append({**idx, "finding": "low_use_index",
                             "recommendation": "review query usage"})
    return findings


def table_growth(db: Session, counts: Optional[dict] = None) -> dict:
    """High-growth table monitoring. `counts` may be provided for
    deterministic tests, otherwise read from the live DB."""
    total = 0
    per_table = []
    if counts is not None:
        for t in HIGH_GROWTH_TABLES:
            n = int(counts.get(t, 0))
            total += n
            per_table.append({"table": t, "rows": n})
    else:
        from sqlalchemy import text
        for t in HIGH_GROWTH_TABLES:
            try:
                n = db.execute(text(
                    f"SELECT count(*) FROM {t}")).scalar() or 0
            except Exception:
                n = None
            total += n or 0
            per_table.append({"table": t, "rows": n})
    per_table.sort(key=lambda x: -(x["rows"] or 0))
    return {"total_rows": total, "tables": per_table,
            "fastest_growing": per_table[0] if per_table else None}


def migration_head(db: Session) -> dict:
    """Read the current migration head from the alembic_version table."""
    from sqlalchemy import text
    try:
        row = db.execute(text(
            "SELECT version_num FROM alembic_version ORDER BY "
            "version_num DESC LIMIT 1")).first()
        return {"migration_head": row[0] if row else None,
                "in_sync": bool(row)}
    except Exception:
        return {"migration_head": None, "in_sync": False}