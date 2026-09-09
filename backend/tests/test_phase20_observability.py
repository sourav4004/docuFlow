"""Phase 20 tests — observability 5.0, SLO 2.0, notification intelligence,
versioned reporting, API platform 4.0, and database platform intelligence.

Incident correlation + lifecycle + timelines; unified AI system health and
the service dependency graph; SLO history, error budgets and policy, and
reliability scores; alert prioritization/deduplication/escalation and
notification preferences; immutable versioned reports; API health metrics,
unstable-endpoint detection, contract validation, idempotency audit and
abuse/brute-force detection; DB health metrics, query-regression detection,
index effectiveness, table growth and migration-head reporting.
"""

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace, WorkspaceMember  # noqa: E402
from app.models.phase20 import (  # noqa: E402
    Incident, IncidentEvent, SloHistory, ErrorBudget, ReliabilityScore,
    OpsAlertEvent, NotificationPreference, ReportVersion,
    ApiHealthMetric, DbHealthMetric,
)
from app.services import observability5 as ob5  # noqa: E402
from app.services import slo2  # noqa: E402
from app.services import notifications_intel as ni  # noqa: E402
from app.services import reporting3 as rep  # noqa: E402
from app.services import api_intel as apii  # noqa: E402
from app.services import db_intel as dbi  # noqa: E402

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


PH20_TABLES = [
    Incident, IncidentEvent, SloHistory, ErrorBudget, ReliabilityScore,
    OpsAlertEvent, NotificationPreference, ReportVersion, ApiHealthMetric,
    DbHealthMetric,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in PH20_TABLES:
        db_session.query(model).delete()
    db_session.query(WorkspaceMember).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p20ob"):
    _counter[0] += 1
    user = User(name=f"P20OB {_counter[0]}",
                email=f"{tag}{_counter[0]}@p20-ob.example",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p20 ob ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


# ===========================================================================
# Observability 5.0 — system health + incidents
# ===========================================================================

class TestSystemHealth:
    def test_dependency_graph_structure(self):
        out = ob5.dependency_graph()
        assert "api" in out["services"]
        assert any(e["from"] == "api" and "broker" in e["to"]
                   for e in out["edges"])

    def test_dependency_graph_covers_components(self):
        out = ob5.dependency_graph()
        for svc in ("api", "broker", "worker", "scheduler",
                    "event_processor", "provider", "vector", "connectors"):
            assert svc in out["services"]

    def test_ai_system_health_healthy(self):
        out = ob5.ai_system_health({
            "availability": 0.99, "latency": 0.9, "quality": 0.95,
            "cost": 0.8, "error_rate": 0.95, "queue_health": 0.9,
            "provider_health": 0.9,
        })
        assert out["health_score"] >= 0.8
        assert out["status"] == "healthy"

    def test_ai_system_health_critical(self):
        out = ob5.ai_system_health({
            "availability": 0.3, "latency": 0.2, "error_rate": 0.1,
        })
        assert out["status"] == "critical"
        assert out["health_score"] < 0.5

    def test_ai_system_health_degraded(self):
        out = ob5.ai_system_health({
            "availability": 0.6, "latency": 0.6, "error_rate": 0.5,
        })
        assert out["status"] == "degraded"
        assert 0.5 <= out["health_score"] < 0.8

    def test_ai_system_health_empty(self):
        out = ob5.ai_system_health({})
        assert out["health_score"] == 0.0


class TestIncidents:
    def test_correlate_creates_incident(self, db_session):
        inc = ob5.correlate_incident(
            db_session, title="DB slow", severity="HIGH",
            affected_systems=["database"], summary="queries timing out")
        assert inc.status == "OPEN"
        assert inc.severity == "HIGH"
        timeline = ob5.incident_timeline(db_session, inc.id)
        assert timeline[0]["event_type"] == "detected"

    def test_correlate_dedupes_by_fingerprint(self, db_session):
        inc1 = ob5.correlate_incident(
            db_session, title="A", severity="MEDIUM",
            affected_systems=["broker"], summary="delivery delay")
        inc2 = ob5.correlate_incident(
            db_session, title="A again", severity="MEDIUM",
            affected_systems=["broker"], summary="delivery delay")
        assert inc1.id == inc2.id
        assert len(ob5.incident_timeline(db_session, inc1.id)) == 2

    def test_correlate_distinct_summary_new_incident(self, db_session):
        inc1 = ob5.correlate_incident(
            db_session, title="A", severity="LOW",
            affected_systems=["api"], summary="latency 1")
        inc2 = ob5.correlate_incident(
            db_session, title="B", severity="LOW",
            affected_systems=["api"], summary="latency 2")
        assert inc1.id != inc2.id

    def test_transition_valid_path(self, db_session):
        inc = ob5.correlate_incident(
            db_session, title="T", severity="MEDIUM",
            affected_systems=["worker"], summary="crash loop")
        out = ob5.transition_incident(db_session, incident_id=inc.id,
                                      to_status="INVESTIGATING",
                                      actor_user_id=1)
        assert out["previous"] == "OPEN"
        assert out["new_status"] == "INVESTIGATING"

    def test_transition_invalid(self, db_session):
        inc = ob5.correlate_incident(
            db_session, title="T", severity="MEDIUM",
            affected_systems=["api"], summary="x")
        with pytest.raises(ValueError):
            ob5.transition_incident(db_session, incident_id=inc.id,
                                    to_status="POSTMORTEM")  # skip states

    def test_transition_unknown_status(self, db_session):
        inc = ob5.correlate_incident(
            db_session, title="T", severity="LOW",
            affected_systems=["api"], summary="x")
        with pytest.raises(ValueError):
            ob5.transition_incident(db_session, incident_id=inc.id,
                                    to_status="TELEPORT")

    def test_transition_unknown_incident(self, db_session):
        with pytest.raises(KeyError):
            ob5.transition_incident(db_session, incident_id=999999,
                                    to_status="RESOLVED")

    def test_transition_records_timeline(self, db_session):
        inc = ob5.correlate_incident(
            db_session, title="T", severity="HIGH",
            affected_systems=["provider"], summary="outage")
        ob5.transition_incident(db_session, incident_id=inc.id,
                                to_status="MITIGATED", actor_user_id=5,
                                detail="switched fallback")
        events = ob5.incident_timeline(db_session, inc.id)
        assert events[-1]["event_type"] == "mitigated"
        assert events[-1]["actor_user_id"] == 5

    def test_full_lifecycle(self, db_session):
        inc = ob5.correlate_incident(
            db_session, title="T", severity="CRITICAL",
            affected_systems=["database"], summary="downtime")
        for status in ("INVESTIGATING", "MITIGATED", "RESOLVED",
                       "POSTMORTEM"):
            ob5.transition_incident(db_session, incident_id=inc.id,
                                    to_status=status)
        assert db_session.get(Incident, inc.id).status == "POSTMORTEM"
        # 1 detected event + 4 transition events
        assert len(ob5.incident_timeline(db_session, inc.id)) == 5

    def test_list_incidents_filter(self, db_session):
        a = ob5.correlate_incident(
            db_session, title="A", severity="HIGH",
            affected_systems=["api"], summary="s1")
        b = ob5.correlate_incident(
            db_session, title="B", severity="LOW",
            affected_systems=["api"], summary="s2")
        ob5.transition_incident(db_session, incident_id=b.id,
                                to_status="INVESTIGATING")
        ob5.transition_incident(db_session, incident_id=b.id,
                                to_status="RESOLVED")
        high = ob5.list_incidents(db_session, severity="HIGH")
        assert [i["id"] for i in high] == [a.id]
        resolved = ob5.list_incidents(db_session, status="RESOLVED")
        assert [i["id"] for i in resolved] == [b.id]

    def test_list_incidents_bounded(self, db_session):
        for i in range(5):
            ob5.correlate_incident(
                db_session, title=f"I{i}", severity="LOW",
                affected_systems=["api"], summary=f"s{i}")
        out = ob5.list_incidents(db_session, limit=3)
        assert len(out) == 3


# ===========================================================================
# SLO 2.0
# ===========================================================================

class TestSlo:
    def test_record_history_met(self, db_session):
        row = slo2.record_history(db_session, name="api.availability",
                                  measured_value=99.9, target=99.5)
        assert row.met is True
        assert row.window == "daily"

    def test_record_history_missed(self, db_session):
        row = slo2.record_history(db_session, name="api.availability",
                                  measured_value=90.0, target=99.5)
        assert row.met is False

    def test_slo_recent_filtered(self, db_session):
        slo2.record_history(db_session, name="api.latency",
                            measured_value=200, target=300)
        slo2.record_history(db_session, name="worker.completion",
                            measured_value=99.0, target=98.0)
        rows = slo2.slo_recent(db_session, name="api.latency")
        assert len(rows) == 1
        assert rows[0]["name"] == "api.latency"

    def test_error_budget_ok(self, db_session):
        row = slo2.error_budget(db_session, name="api.availability",
                                budget=5.0, consumed_override=10.0)
        assert row.status == "OK"

    def test_error_budget_warning(self, db_session):
        row = slo2.error_budget(db_session, name="api.availability",
                                budget=5.0, consumed_override=85.0)
        assert row.status == "WARNING"

    def test_error_budget_exhausted(self, db_session):
        row = slo2.error_budget(db_session, name="api.availability",
                                budget=5.0, consumed_override=100.0)
        assert row.status == "EXHAUSTED"

    def test_error_budget_computed_from_history(self, db_session):
        slo2.record_history(db_session, name="api.availability",
                            measured_value=90.0, target=99.5)
        slo2.record_history(db_session, name="api.availability",
                            measured_value=90.0, target=99.5)
        slo2.record_history(db_session, name="api.availability",
                            measured_value=99.9, target=99.5)
        row = slo2.error_budget(db_session, name="api.availability",
                                budget=5.0)
        assert row.consumed == pytest.approx(66.67, abs=0.1)

    def test_error_budget_policy_ok(self, db_session):
        row = slo2.error_budget(db_session, name="api.availability",
                                budget=5.0, consumed_override=10.0)
        policy = slo2.error_budget_policy(row)
        assert policy["action"] == "none"
        assert policy["recommend_freeze"] is False

    def test_error_budget_policy_warning(self, db_session):
        row = slo2.error_budget(db_session, name="api.availability",
                                budget=5.0, consumed_override=85.0)
        policy = slo2.error_budget_policy(row)
        assert policy["level"] == "WARNING"
        assert policy["action"] == "surface_warning"

    def test_error_budget_policy_exhausted_no_auto_block(self, db_session):
        row = slo2.error_budget(db_session, name="api.availability",
                                budget=5.0, consumed_override=100.0)
        policy = slo2.error_budget_policy(row)
        assert policy["level"] == "CRITICAL"
        assert policy["action"] == "require_operator_review"
        assert policy["recommend_freeze"] is True
        # must NOT automatically block unrelated operations
        assert "block" not in policy["action"]

    def test_reliability_score(self, db_session):
        row = slo2.reliability_score(
            db_session, service="api",
            factors={"availability": 99.9, "latency": 90.0,
                     "error_rate": 95.0, "recovery": 80.0})
        assert row.score > 90.0

    def test_reliability_score_clamped(self, db_session):
        row = slo2.reliability_score(
            db_session, service="api",
            factors={"availability": 150.0, "latency": -5.0,
                     "error_rate": 100.0, "recovery": 100.0})
        assert 0.0 <= row.score <= 100.0

    def test_reliability_report(self, db_session):
        slo2.reliability_score(db_session, service="api", factors={
            "availability": 99.0, "latency": 80.0, "error_rate": 90.0,
            "recovery": 70.0})
        slo2.reliability_score(db_session, service="worker", factors={
            "availability": 98.0, "latency": 70.0, "error_rate": 80.0,
            "recovery": 60.0})
        rows = slo2.reliability_report(db_session, service="api")
        assert len(rows) == 1
        assert rows[0]["service"] == "api"
        assert "detail" in rows[0]


# ===========================================================================
# Notification intelligence
# ===========================================================================

class TestNotifications:
    def test_raise_alert_valid(self, db_session):
        alert = ni.raise_alert(db_session, category="quality",
                               severity="medium",
                               message="quality dropped")
        assert alert.status == "OPEN"
        assert alert.occurrence_count == 1

    def test_raise_alert_invalid_severity(self, db_session):
        with pytest.raises(ValueError):
            ni.raise_alert(db_session, category="quality",
                           severity="extreme", message="x")

    def test_alert_deduplication(self, db_session):
        a1 = ni.raise_alert(db_session, category="cost", severity="low",
                            message="budget approaching")
        a2 = ni.raise_alert(db_session, category="cost", severity="low",
                            message="budget approaching")
        assert a1.id == a2.id
        assert a2.occurrence_count == 2

    def test_alert_escalation_low_to_medium(self, db_session):
        alert = None
        for _ in range(3):
            alert = ni.raise_alert(db_session, category="provider",
                                   severity="low",
                                   message="provider latency rising")
        assert alert.severity == "medium"
        assert alert.occurrence_count == 3

    def test_alert_escalation_medium_to_high(self, db_session):
        alert = None
        for _ in range(5):
            alert = ni.raise_alert(db_session, category="provider",
                                   severity="low",
                                   message="provider errors repeated")
        assert alert.severity == "high"

    def test_alert_escalation_high_to_critical(self, db_session):
        alert = None
        for _ in range(8):
            alert = ni.raise_alert(db_session, category="incident",
                                   severity="high",
                                   message="incident recurring")
        assert alert.severity == "critical"

    def test_list_alerts_filters(self, db_session):
        ni.raise_alert(db_session, category="quality", severity="high",
                       message="q1")
        ni.raise_alert(db_session, category="cost", severity="low",
                       message="c1")
        alerts = ni.list_alerts(db_session, category="quality")
        assert len(alerts) == 1
        assert alerts[0]["category"] == "quality"

    def test_acknowledge(self, db_session):
        alert = ni.raise_alert(db_session, category="security",
                               severity="critical", message="auth spike")
        out = ni.acknowledge(db_session, alert.id)
        assert out["acknowledged"] is True
        assert db_session.get(OpsAlertEvent, alert.id).status == "ACKNOWLEDGED"

    def test_acknowledge_unknown(self, db_session):
        with pytest.raises(KeyError):
            ni.acknowledge(db_session, 999999)

    def test_set_preference_valid(self, db_session):
        pref = ni.set_preference(db_session, user_id=1,
                                 category="incident", enabled=False)
        assert pref.enabled is False

    def test_set_preference_invalid_category(self, db_session):
        with pytest.raises(ValueError):
            ni.set_preference(db_session, user_id=1,
                              category="spam", enabled=True)

    def test_preferences_upsert(self, db_session):
        ni.set_preference(db_session, user_id=7, category="cost",
                          enabled=True)
        ni.set_preference(db_session, user_id=7, category="cost",
                          enabled=False)
        prefs = ni.preferences(db_session, user_id=7)
        assert len(prefs) == 1
        assert prefs[0]["enabled"] is False

    def test_alert_summary(self, db_session):
        ni.raise_alert(db_session, category="quality", severity="high",
                       message="a")
        ni.raise_alert(db_session, category="cost", severity="low",
                       message="b")
        summary = ni.alert_summary(db_session)
        assert summary["total_open"] == 2
        assert summary["by_severity"]["high"] == 1


# ===========================================================================
# Reporting 3.0
# ===========================================================================

class TestReports:
    def test_create_report_versioning(self, db_session):
        r1 = rep.create_report(db_session, kind="cost", scope_type="ORG",
                               scope_id=1, content={"total": 10})
        r2 = rep.create_report(db_session, kind="cost", scope_type="ORG",
                               scope_id=1, content={"total": 20})
        assert r1.version == 1
        assert r2.version == 2

    def test_create_report_invalid_kind(self, db_session):
        with pytest.raises(ValueError):
            rep.create_report(db_session, kind="telemetry",
                              scope_type="ORG", content={})

    def test_latest_report(self, db_session):
        rep.create_report(db_session, kind="ai_quality", scope_type="WS",
                          scope_id=3, content={"quality": 0.9})
        rep.create_report(db_session, kind="ai_quality", scope_type="WS",
                          scope_id=3, content={"quality": 0.95})
        latest = rep.latest_report(db_session, kind="ai_quality",
                                   scope_type="WS", scope_id=3)
        assert latest["version"] == 2
        assert latest["content"]["quality"] == 0.95

    def test_latest_report_none(self, db_session):
        assert rep.latest_report(db_session, kind="ai_quality",
                                 scope_type="WS", scope_id=99) is None

    def test_ai_quality_report(self, db_session):
        row = rep.ai_quality_report(
            db_session, workspace_id=4, generated_by=1,
            quality={"score": 0.9}, cost={"total": 5.0},
            safety={"score": 1.0}, reliability={"score": 95.0})
        latest = rep.latest_report(db_session, kind="ai_quality",
                                   scope_type="WORKSPACE", scope_id=4)
        assert latest["content"]["sections"]["quality"]["score"] == 0.9
        assert row.version == 1

    def test_knowledge_health_report(self, db_session):
        rep.knowledge_health_report(
            db_session, workspace_id=5,
            freshness={"stale_docs": 2}, gaps=[{"q": "who owns X"}],
            conflicts=[{"memory_id": 1}], graph={"score": 0.8})
        latest = rep.latest_report(db_session, kind="knowledge_health",
                                   scope_type="WORKSPACE", scope_id=5)
        assert latest["content"]["sections"]["gaps"][0]["q"] == "who owns X"

    def test_governance_report(self, db_session):
        rep.governance_report(
            db_session, scope_type="ORGANIZATION", scope_id=6,
            policies=[{"allowed_models": ["a"]}],
            changes=[{"version": 1}], violations=[], approvals=[])
        latest = rep.latest_report(db_session, kind="governance",
                                   scope_type="ORGANIZATION", scope_id=6)
        assert latest["content"]["sections"]["changes"][0]["version"] == 1

    def test_reports_immutable(self, db_session):
        row = rep.create_report(db_session, kind="cost", scope_type="ORG",
                                scope_id=2, content={"total": 10})
        db_session.expire_all()
        got = db_session.get(ReportVersion, row.id)
        assert got.content_json is not None  # stored, not regenerated


# ===========================================================================
# API platform 4.0
# ===========================================================================

class TestApiIntel:
    def test_record_metric(self, db_session):
        row = apii.record_metric(db_session, endpoint="/ops/health",
                                 requests=100, errors=2,
                                 latency_p95_ms=250.0)
        assert row.id is not None
        assert row.requests == 100

    def test_unstable_endpoints_error_rate(self, db_session):
        apii.record_metric(db_session, endpoint="/api/stable",
                           requests=100, errors=2)
        apii.record_metric(db_session, endpoint="/api/flaky",
                           requests=100, errors=50)
        unstable = apii.unstable_endpoints(db_session)
        assert [u["endpoint"] for u in unstable] == ["/api/flaky"]

    def test_unstable_endpoints_auth_rate(self, db_session):
        apii.record_metric(db_session, endpoint="/api/bruteforced",
                           requests=100, auth_failures=40)
        unstable = apii.unstable_endpoints(db_session)
        assert any(u["auth_failure_rate"] > 0.1 for u in unstable)

    def test_unstable_endpoints_min_requests(self, db_session):
        apii.record_metric(db_session, endpoint="/api/tiny",
                           requests=2, errors=2)
        unstable = apii.unstable_endpoints(db_session)
        assert unstable == []

    def test_api_health_report(self, db_session):
        apii.record_metric(db_session, endpoint="/a", requests=100,
                           errors=10)
        apii.record_metric(db_session, endpoint="/b", requests=100,
                           errors=0)
        report = apii.api_health_report(db_session)
        assert report["total_requests"] == 200
        assert report["error_rate"] == 0.05
        assert report["endpoints_recorded"] == 2

    def test_contract_validate_side_effecting_complete(self):
        out = apii.contract_validate(
            "POST", "/api/documents", {
                "auth": True, "authorization": True, "validation": True,
                "errors": True, "idempotency": True,
                "bounded_output": True})
        assert out["valid"] is True
        assert out["issues"] == []

    def test_contract_validate_missing_idempotency(self):
        out = apii.contract_validate(
            "POST", "/api/documents", {
                "auth": True, "authorization": True, "validation": True,
                "errors": True})
        assert out["valid"] is False
        assert any("idempotency" in i for i in out["issues"])

    def test_contract_validate_list_requires_pagination(self):
        out = apii.contract_validate(
            "GET", "/api/documents/list",
            {"auth": True, "authorization": True, "errors": True,
             "pagination": False})
        assert out["checks"]["pagination"] is False

    def test_idempotency_audit(self):
        routes = [
            {"method": "GET", "path": "/api/docs", "idempotent": True},
            {"method": "POST", "path": "/api/docs", "idempotent": True},
            {"method": "DELETE", "path": "/api/docs/1", "idempotent": False},
        ]
        findings = apii.idempotency_audit(routes)
        assert len(findings) == 2  # GET excluded
        assert findings[0]["pass"] is True
        assert findings[1]["pass"] is False

    def test_abuse_detection_enumeration(self, db_session):
        events = [{"api_key": "k1", "path": f"/api/res{i}"}
                  for i in range(60)]
        out = apii.abuse_detection(db_session, events=events)
        assert any(s["type"] == "enumeration" for s in out["signals"])

    def test_abuse_detection_scope_probing(self, db_session):
        events = [{"api_key": "k2", "scope": "ws-1"},
                  {"api_key": "k2", "scope": "ws-2"},
                  {"api_key": "k2", "scope": "org-3"}]
        out = apii.abuse_detection(db_session, events=events)
        assert any(s["type"] == "scope_probing" for s in out["signals"])

    def test_abuse_detection_clean(self, db_session):
        out = apii.abuse_detection(db_session, events=[
            {"api_key": "k3", "path": "/api/a"},
            {"api_key": "k3", "path": "/api/b"},
        ])
        assert any(s["type"] == "none" for s in out["signals"])

    def test_brute_force_score(self):
        out = apii.brute_force_score([
            {"identity": "user1", "failed": True},
            {"identity": "user1", "failed": True},
            {"identity": "user1", "failed": True},
            {"identity": "user1", "failed": True},
            {"identity": "user1", "failed": True},
            {"identity": "user2", "failed": True},
        ])
        assert out["flagged"] == 1
        assert [f for f in out["identities"]
                if f["identity"] == "user1"][0]["suspicious"] is True


# ===========================================================================
# Database platform
# ===========================================================================

class TestDbIntel:
    def test_record_metric(self, db_session):
        row = dbi.record_metric(db_session, metric="connections",
                                value=12.0, detail={"pool": 20})
        assert row.id is not None

    def test_health_report_latest_per_metric(self, db_session):
        dbi.record_metric(db_session, metric="connections", value=10.0)
        dbi.record_metric(db_session, metric="connections", value=15.0)
        dbi.record_metric(db_session, metric="slow_queries", value=3.0)
        report = dbi.health_report(db_session)
        assert report["metrics"]["connections"] == 15.0
        assert report["metrics"]["slow_queries"] == 3.0

    def test_query_regression_detected(self):
        out = dbi.query_regression({"search": 100.0}, {"search": 300.0},
                                   max_ratio=1.5)
        assert out["regressed"] is True
        assert out["regressions"][0]["ratio"] == 3.0

    def test_query_regression_ok(self):
        out = dbi.query_regression({"search": 100.0}, {"search": 120.0},
                                   max_ratio=1.5)
        assert out["regressed"] is False

    def test_query_regression_missing_query_skipped(self):
        out = dbi.query_regression({"a": 100.0}, {"b": 500.0})
        assert out["regressed"] is False

    def test_index_effectiveness_unused(self):
        out = dbi.index_effectiveness([
            {"name": "ix_jobs_status", "table": "worker_jobs",
             "scans": 10, "uses": 0}])
        assert out[0]["finding"] == "unused_index"

    def test_index_effectiveness_low_use(self):
        out = dbi.index_effectiveness([
            {"name": "ix_traces_ws", "table": "trace_spans",
             "scans": 100, "uses": 5}])
        assert out[0]["finding"] == "low_use_index"

    def test_index_effectiveness_healthy(self):
        out = dbi.index_effectiveness([
            {"name": "ix_events_ws", "table": "events",
             "scans": 100, "uses": 80}])
        assert out == []

    def test_table_growth_deterministic(self, db_session):
        out = dbi.table_growth(db_session, counts={
            "worker_jobs": 1000, "trace_spans": 500, "events": 50,
        })
        assert out["total_rows"] == 1550
        assert out["fastest_growing"]["table"] == "worker_jobs"

    def test_table_growth_sorted(self, db_session):
        out = dbi.table_growth(db_session, counts={
            "events": 5, "worker_jobs": 200, "usage_records": 50,
        })
        assert out["tables"][0]["table"] == "worker_jobs"

    def test_migration_head_contract(self, db_session):
        out = dbi.migration_head(db_session)
        assert "migration_head" in out
        assert "in_sync" in out