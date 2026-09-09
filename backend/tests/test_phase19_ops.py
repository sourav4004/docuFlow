"""Phase 19 tests — observability 4.0 + search 5.0 + enterprise analytics +
data consistency + DR 2.0 + API platform 3.0 + broker 2.0 + perf + quality.

Observability: spans, trace correlation, latency breakdown, SLO engine with
burn rate, alert dedupe. Search: planning, explainability, quality events
(hash-only), personalization scoping. Analytics: aggregate-only, no private
data. Consistency: orphan/cross-tenant/integrity checks + safe repair
planner. DR: backup inventory, validation, readiness. API: errors,
pagination, idempotency, abuse detection. Broker: delivery guarantees,
recovery accounting. Perf: single-flight + stampede protection. Quality:
golden datasets, evaluations, gates, regression, feedback.
"""

import json
from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.organization import Organization  # noqa: E402
from app.models.phase15 import AIMemory  # noqa: E402
from app.models.phase16 import TraceSpan, WorkerJob  # noqa: E402
from app.models.phase17 import JobLease  # noqa: E402
from app.models.phase18 import SearchAnalyticsEvent  # noqa: E402
from app.models.phase19 import (  # noqa: E402
    SloDefinition, SloBudgetWindow, ApiAbuseEvent, BackupRecord,
    EvaluationRun, QualityGate, ConsistencyReport,
    SearchPersonalizationProfile,
)
from app.services import observability4 as obs4  # noqa: E402
from app.services import search5  # noqa: E402
from app.services import analytics2  # noqa: E402
from app.services import data_consistency as dc  # noqa: E402
from app.services import dr2  # noqa: E402
from app.services import api3  # noqa: E402
from app.services import broker2  # noqa: E402
from app.services import perf2  # noqa: E402
from app.services import quality_platform as qp  # noqa: E402

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(SloBudgetWindow).delete()
    db_session.query(SloDefinition).delete()
    db_session.query(ApiAbuseEvent).delete()
    db_session.query(EvaluationRun).delete()
    db_session.query(QualityGate).delete()
    db_session.query(BackupRecord).delete()
    db_session.query(ConsistencyReport).delete()
    db_session.query(SearchPersonalizationProfile).delete()
    db_session.query(SearchAnalyticsEvent).delete()
    db_session.query(TraceSpan).delete()
    db_session.query(JobLease).delete()
    db_session.query(WorkerJob).delete()
    db_session.query(AIMemory).delete()
    # Global consistency checks (execution_integrity, cross_tenant_edges,
    # orphan_chunks) scan the whole DB, so delete child rows before their
    # workspaces to avoid manufacturing orphans from earlier suites.
    from app.models.ai_execution import AIExecution as _AIExecution
    from app.models.document import Document as _Document
    from app.models.document_chunk import DocumentChunk as _Chunk
    from app.models.knowledge_graph import (Entity as _Entity,
                                            EntityRelationship as _ER)
    from app.models.search_intel import AIFeedback as _Feedback
    db_session.query(_Feedback).delete()
    db_session.query(_Chunk).delete()
    db_session.query(_ER).delete()
    db_session.query(_Entity).delete()
    db_session.query(_AIExecution).delete()
    db_session.query(_Document).delete()
    db_session.query(Workspace).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p19ops"):
    _counter[0] += 1
    user = User(name=f"P19 OPS {_counter[0]}",
                email=f"{tag}{_counter[0]}@p19-ops.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_org(db, user):
    _counter[0] += 1
    org = Organization(name=f"p19 ops org {_counter[0]}",
                       slug=f"p19ops-{_counter[0]}", owner_id=user.id)
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


def fresh_workspace(db, user, org=None):
    _counter[0] += 1
    ws = Workspace(name=f"p19 ops ws {_counter[0]}", owner_id=user.id,
                   organization_id=org.id if org else None)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


# ===========================================================================
# Observability 4.0
# ===========================================================================

class TestSpans:
    def test_start_and_end_span(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        started = obs4.start_span(db_session, workspace_id=ws.id,
                                  span_type="provider", provider="openai",
                                  model="gpt-4o")
        ended = obs4.end_span(db_session, span_id=started["span_id"],
                              status="OK", input_tokens=10,
                              output_tokens=5)
        assert ended["status"] == "OK"
        assert ended["latency_ms"] >= 0

    def test_unknown_span_type_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        with pytest.raises(ValueError):
            obs4.start_span(db_session, workspace_id=ws.id,
                            span_type="quantum")

    def test_end_twice_idempotent(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        started = obs4.start_span(db_session, workspace_id=ws.id,
                                  span_type="rag")
        obs4.end_span(db_session, span_id=started["span_id"])
        second = obs4.end_span(db_session, span_id=started["span_id"])
        assert second["status"] == "ALREADY_COMPLETED"

    def test_invalid_end_status_rejected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        started = obs4.start_span(db_session, workspace_id=ws.id,
                                  span_type="agent")
        with pytest.raises(ValueError):
            obs4.end_span(db_session, span_id=started["span_id"],
                          status="MAYBE")

    def test_input_summary_redacted(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        started = obs4.start_span(
            db_session, workspace_id=ws.id, span_type="provider",
            input_summary="call with api key sk-abcdef1234567890 inside")
        row = db_session.query(TraceSpan).filter(
            TraceSpan.span_id == started["span_id"]).first()
        assert "sk-abcdef" not in (row.input_summary or "")

    def test_trace_summary_correlates(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        a = obs4.start_span(db_session, workspace_id=ws.id,
                            span_type="request")
        b = obs4.start_span(db_session, workspace_id=ws.id,
                            span_type="provider", trace_id=a["trace_id"],
                            parent_span_id=a["span_id"])
        summary = obs4.trace_summary(db_session, a["trace_id"])
        assert summary["span_count"] == 2
        assert {s["span_type"] for s in summary["spans"]} == \
            {"request", "provider"}

    def test_latency_breakdown(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        span = obs4.start_span(db_session, workspace_id=ws.id,
                               span_type="provider")
        obs4.end_span(db_session, span_id=span["span_id"])
        breakdown = obs4.latency_breakdown(db_session, since_minutes=60)
        types = {b["span_type"] for b in breakdown["breakdown"]}
        assert "provider" in types

    def test_error_correlation(self, db_session):
        import uuid as _uuid
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        exec_id = f"exec-{_uuid.uuid4().hex[:16]}"
        span = obs4.start_span(db_session, workspace_id=ws.id,
                               span_type="provider",
                               execution_id=exec_id)
        obs4.end_span(db_session, span_id=span["span_id"],
                      status="ERROR", error_class="RateLimitError")
        result = obs4.correlate_errors(db_session, execution_id=exec_id)
        assert result["failures"] == 1
        assert result["by_error_class"]["RateLimitError"] == 1


class TestSloEngine:
    def test_create_and_evaluate_compliant(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        slo = obs4.create_slo(db_session, name="api-latency",
                              metric="latency_p95_ms", operator="<=",
                              target_value=2000.0)
        result = obs4.evaluate_slo(db_session,
                                   definition_id=slo["definition_id"],
                                   metric_value=1200.0)
        assert result["status"] == "COMPLIANT"
        assert result["burn_rate"] == 0.0

    def test_slo_breach_detected(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        slo = obs4.create_slo(db_session, name="api-availability",
                              metric="error_rate", operator="<=",
                              target_value=0.01,
                              burn_rate_threshold=200.0)
        result = obs4.evaluate_slo(db_session,
                                   definition_id=slo["definition_id"],
                                   metric_value=0.05)
        assert result["status"] == "BREACHED"

    def test_slo_burn_rate_high(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        slo = obs4.create_slo(db_session, name="ingestion-completion",
                              metric="success_rate", operator=">=",
                              target_value=0.99, burn_rate_threshold=1.5)
        result = obs4.evaluate_slo(db_session,
                                   definition_id=slo["definition_id"],
                                   metric_value=0.90)
        assert result["burn_rate"] >= 1.5
        assert result["status"] == "BURNING"

    def test_slo_health_reports(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        slo = obs4.create_slo(db_session, name="x-slo",
                              metric="latency_p95_ms", operator="<=",
                              target_value=1000.0)
        obs4.evaluate_slo(db_session, definition_id=slo["definition_id"],
                          metric_value=100.0)
        health = obs4.slo_health(db_session)
        assert health["by_name"]["x-slo"] == "COMPLIANT"


class TestAlertDedupe:
    def test_duplicates_suppressed(self):
        now = datetime.now(timezone.utc)
        events = [
            {"fingerprint": "f1", "fired_at": now},
            {"fingerprint": "f1", "fired_at": now + timedelta(minutes=1)},
            {"fingerprint": "f1", "fired_at": now + timedelta(minutes=45)},
            {"fingerprint": "f2", "fired_at": now},
        ]
        result = obs4.dedupe_alerts(events, cooldown_minutes=30)
        assert len(result["kept"]) == 3
        assert result["suppressed"] == 1

    def test_different_fingerprints_all_kept(self):
        events = [{"fingerprint": "a"}, {"fingerprint": "b"},
                  {"fingerprint": "c"}]
        result = obs4.dedupe_alerts(events)
        assert result["suppressed"] == 0


# ===========================================================================
# Search 5.0
# ===========================================================================

class TestSearchPlanner:
    def test_short_query_keyword(self):
        plan = search5.planner2("refunds")
        assert plan["strategy"] == "keyword"

    def test_long_query_hybrid(self):
        plan = search5.planner2("how do refunds work for enterprise plans")
        assert "vector" in plan["sources"]
        assert plan["strategy"] == "hybrid"

    def test_entity_query_uses_graph(self):
        plan = search5.planner2("who manages the Acme Corp")
        assert "graph" in plan["sources"]

    def test_memory_trigger(self):
        plan = search5.planner2("what do we know about this vendor")
        assert "memory" in plan["sources"]

    def test_freshness_boost(self):
        plan = search5.planner2("current refund policy", filters={"fresh": True})
        assert plan["freshness_boost"] is True

    def test_explain_never_reveals_cot(self):
        plan = search5.planner2("refund policy")
        explanation = search5.explain_search(
            plan, [{"id": "d1", "source_category": "keyword"}])
        assert "chain-of-thought" not in json.dumps(explanation)
        assert explanation["why_top_result"]["top_result_id"] == "d1"

    def test_scope_propagates(self):
        plan = search5.planner2("policy", scope="organization")
        assert plan["scope"] == "organization"


class TestSearchQuality:
    def test_record_zero_result(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = search5.record_quality(
            db_session, workspace_id=ws.id, query="totally unique",
            retrieval_mode="keyword", result_count=0)
        assert result["zero_results"] is True

    def test_quality_summary_aggregates(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        search5.record_quality(db_session, workspace_id=ws.id,
                               query="refund", result_count=5,
                               latency_ms=50, useful_signal=True)
        search5.record_quality(db_session, workspace_id=ws.id,
                               query="xyz-zzz", result_count=0,
                               latency_ms=200, useful_signal=False)
        summary = search5.quality_summary(db_session, workspace_id=ws.id)
        assert summary["events"] == 2
        assert summary["zero_result_rate"] == 0.5
        assert summary["useful_rate"] == 0.5

    def test_query_text_never_stored(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        search5.record_quality(db_session, workspace_id=ws.id,
                               query="super secret query phrase")
        rows = (db_session.query(SearchAnalyticsEvent)
                .filter(SearchAnalyticsEvent.workspace_id == ws.id).all())
        assert all("super secret" not in str(getattr(r, "query_hash", ""))
                   for r in rows)


class TestPersonalization:
    def test_profile_scope_is_per_workspace(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        search5.update_profile(db_session, user_id=user.id,
                               workspace_id=ws1.id, intent="refund policy")
        profile2 = search5.get_profile(db_session, user_id=user.id,
                                       workspace_id=ws2.id)
        assert profile2["recent_intents"] == []

    def test_personalize_uses_own_intents(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        search5.update_profile(db_session, user_id=user.id,
                               workspace_id=ws.id, intent="refund policy")
        profile = search5.get_profile(db_session, user_id=user.id,
                                      workspace_id=ws.id)
        result = search5.personalize_query(profile, "refund policy details")
        assert result["personalized"] is True
        assert result["boosted_terms"]

    def test_no_personalization_without_history(self):
        result = search5.personalize_query(
            {"recent_intents": []}, "refund policy")
        assert result["personalized"] is False


# ===========================================================================
# Analytics (aggregate-only)
# ===========================================================================

class TestAnalytics:
    def test_privacy_guard_strips_private_fields(self):
        payload = {"total_cost_usd": 12.5,
                   "private_user_email": "a@b.c",
                   "by_workspace": {"w1": {"count": 3}},
                   "raw_query": "secret"}
        guarded = analytics2.privacy_guard(payload)
        assert guarded["total_cost_usd"] == 12.5
        assert "private_user_email" not in guarded
        assert "raw_query" not in guarded

    def test_user_analytics_scoped(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = analytics2.user_analytics(db_session, workspace_id=ws.id,
                                           user_id=user.id)
        assert result["my_ai_requests"] == 0
        assert "privacy_note" in result

    def test_workspace_analytics_shape(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = analytics2.workspace_analytics(db_session,
                                                workspace_id=ws.id)
        assert result["document_count"] == 0
        assert result["workflow_count"] == 0

    def test_org_analytics_aggregate_only(self, db_session):
        user = fresh_user(db_session)
        org = fresh_org(db_session, user)
        ws = fresh_workspace(db_session, user, org)
        result = analytics2.org_analytics(db_session,
                                          organization_id=org.id)
        assert result["workspaces"] == 1
        assert result["document_count"] == 0
        assert "ai_requests" in result


# ===========================================================================
# Data consistency
# ===========================================================================

class TestConsistency:
    def test_clean_workspace(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        baseline = dc.run_consistency_checks(db_session)
        result = dc.run_consistency_checks(db_session,
                                           workspace_id=ws.id)
        # A fresh workspace adds no new issues over the global baseline.
        assert result["dry_run"] is True
        assert result["issue_count"] == baseline["issue_count"]
        assert result["by_check"]["execution_integrity"] == 0

    def test_orphan_chunk_detected(self, db_session):
        # Delta-based: earlier suites may leave chunk rows behind, so the
        # check must prove OUR orphan is surfaced, not an absolute count.
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        from app.models.document_chunk import DocumentChunk
        before = len(dc.orphan_chunks(db_session, workspace_id=ws.id))
        chunk = DocumentChunk(document_id=999999, chunk_index=0,
                              text="x", char_start=0, char_end=1)
        db_session.add(chunk)
        db_session.commit()
        orphan = dc.orphan_chunks(db_session, workspace_id=ws.id)
        assert len(orphan) == before + 1
        assert any(o["kind"] == "orphan_chunk"
                   and o["chunk_id"] == chunk.id for o in orphan)

    def test_cross_tenant_edge_detected(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        from app.models.knowledge_graph import Entity, EntityRelationship
        e1 = Entity(workspace_id=ws1.id, name="a", entity_type="org",
                    normalized_name="a")
        e2 = Entity(workspace_id=ws2.id, name="b", entity_type="org",
                    normalized_name="b")
        db_session.add_all([e1, e2])
        db_session.commit()
        db_session.add(EntityRelationship(workspace_id=ws1.id,
                                          source_id=e1.id,
                                          target_id=e2.id,
                                          relationship_type="x"))
        db_session.commit()
        found = dc.cross_tenant_edges(db_session)
        assert any(edge["kind"] == "cross_tenant_edge" for edge in found)

    def test_repair_planner_never_destructive(self):
        issues = [{"kind": "orphan_chunk", "chunk_id": 1},
                  {"kind": "cross_tenant_edge", "relationship_id": 2}]
        plan = dc.repair_planner(issues)
        assert all(not p["destructive"] for p in plan["plans"])
        assert any(p["needs_approval"] for p in plan["plans"])
        assert plan["dry_run"] is True

    def test_repair_planner_bounded(self):
        issues = [{"kind": "x", "i": i} for i in range(2000)]
        plan = dc.repair_planner(issues)
        assert len(plan["plans"]) <= 200


# ===========================================================================
# DR 2.0
# ===========================================================================

class TestDr:
    def test_record_backup(self, db_session):
        result = dr2.record_backup(db_session, scope="DATABASE",
                                   backup_ref="bkp-001",
                                   checksum="c" * 64)
        assert result["scope"] == "DATABASE"
        assert result["migration_head"]

    def test_invalid_backup_scope(self, db_session):
        with pytest.raises(ValueError):
            dr2.record_backup(db_session, scope="CASSETTE_TAPE",
                              backup_ref="x")

    def test_validate_backups_marks_records(self, db_session):
        # The SQLite test DB has no alembic_version rows, so backups can
        # never be confirmed in-sync here — they must be marked FAILED
        # (never left PENDING), which is the honest deterministic outcome.
        dr2.record_backup(db_session, scope="DATABASE",
                          backup_ref="bkp-valid")
        validation = dr2.validate_backups(db_session)
        assert validation["failed"] == 1
        assert validation["validated"] == 0
        from app.models.phase19 import BackupRecord
        row = db_session.query(BackupRecord).first()
        assert row.status == "FAILED"
        assert row.validated_at is not None

    def test_readiness_score_deterministic(self, db_session):
        score = dr2.readiness_score(db_session)
        assert 0.0 <= score["score"] <= 1.0
        assert score["grade"] in ("READY", "PARTIAL", "NOT_READY")

    def test_restore_validation_non_destructive(self, db_session):
        result = dr2.restore_validation(db_session)
        assert "restore_valid" in result
        assert "tenant_isolation" in result
        assert result["current_migration_head"]


# ===========================================================================
# API platform 3.0
# ===========================================================================

class TestApiErrorsPagination:
    def test_error_payload_shape(self):
        payload = api3.error_payload("rate_limited", "slow down",
                                     request_id="req-1", retryable=True,
                                     status=429)
        error = payload["error"]
        assert error["code"] == "rate_limited"
        assert error["retryable"] is True
        assert error["request_id"] == "req-1"

    def test_page_params_bounded(self):
        assert api3.page_params(None, None)["limit"] == 50
        assert api3.page_params(99999, -5, max_limit=200)["limit"] == 200
        assert api3.page_params(99999, -5)["offset"] == 0

    def test_request_hash_deterministic(self):
        assert api3.request_hash("POST", "/docs", {"a": 1}) == \
            api3.request_hash("POST", "/docs", {"a": 1})
        assert api3.request_hash("POST", "/docs", {"a": 1}) != \
            api3.request_hash("POST", "/docs", {"a": 2})


class TestIdempotency:
    def test_new_then_replay(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        first = api3.idempotency_check(db_session, workspace_id=ws.id,
                                       key="k1", method="POST", path="/x",
                                       body={"a": 1}, user_id=user.id)
        assert first["decision"] == "NEW"
        api3.idempotency_complete(db_session, workspace_id=ws.id, key="k1",
                                  response_json='{"ok": true}',
                                  status_code=200)
        replay = api3.idempotency_check(db_session, workspace_id=ws.id,
                                        key="k1", method="POST", path="/x",
                                        body={"a": 1}, user_id=user.id)
        assert replay["decision"] == "REPLAY"
        assert replay["status"] == 200

    def test_conflict_on_different_body(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        api3.idempotency_check(db_session, workspace_id=ws.id, key="k2",
                               method="POST", path="/x", body={"a": 1},
                               user_id=user.id)
        conflict = api3.idempotency_check(db_session, workspace_id=ws.id,
                                          key="k2", method="POST", path="/x",
                                          body={"a": 2}, user_id=user.id)
        assert conflict["decision"] == "CONFLICT"

    def test_keys_isolated_per_workspace(self, db_session):
        user = fresh_user(db_session)
        ws1 = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        api3.idempotency_check(db_session, workspace_id=ws1.id, key="k",
                               method="POST", path="/x", user_id=user.id)
        other = api3.idempotency_check(db_session, workspace_id=ws2.id,
                                       key="k", method="POST", path="/x",
                                       user_id=user.id)
        assert other["decision"] == "NEW"


class TestAbuse:
    def test_repeated_failures_flagged(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        result = api3.detect_abuse(db_session, workspace_id=ws.id,
                                   user_id=user.id,
                                   failures_in_window=25)
        assert result["flagged"] is True
        assert "REPEATED_FAILURES" in result["signals"]
        assert result["blocked"] is True

    def test_scope_violation_flagged(self, db_session):
        result = api3.detect_abuse(None, scope_violation_hint=True,
                                   persist=False)
        assert result["flagged"] is True

    def test_benign_not_flagged(self, db_session):
        result = api3.detect_abuse(None, persist=False,
                                   failures_in_window=2,
                                   requests_in_window=10)
        assert result["flagged"] is False

    def test_abuse_event_persisted(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        api3.detect_abuse(db_session, user_id=user.id,
                          workspace_id=ws.id, failures_in_window=50)
        events = api3.recent_abuse(db_session, user_id=user.id)
        assert len(events) >= 1


# ===========================================================================
# Broker 2.0
# ===========================================================================

class TestBroker:
    def test_delivery_guarantees(self):
        guarantees = broker2.delivery_guarantees()
        assert guarantees["semantics"] == "at-least-once"
        assert guarantees["acknowledgement"].startswith("explicit")

    def test_handler_idempotent_default(self):
        result = broker2.handler_idempotent("ingest_document")
        assert result["idempotent"] is True

    def test_non_idempotent_opt_out(self):
        result = broker2.handler_idempotent("pay_invoice", declared=False)
        assert result["idempotent"] is False
        assert "single-owner" in result["policy"]

    def test_redis_never_required(self):
        assert broker2.redis_available() in (True, False)
        assert broker2.broker_config()["redis_optional"] is True

    def test_outage_policy_no_silent_loss(self):
        policy = broker2.outage_policy()
        assert "never loses queued jobs" in policy["durability"]

    def test_recovery_accounts_expired_leases(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        job = WorkerJob(workspace_id=ws.id, queue_name="default",
                        job_type="test", status="RUNNING",
                        lease_expires_at=datetime.now(timezone.utc)
                        - timedelta(minutes=5))
        db_session.add(job)
        db_session.flush()
        db_session.add(JobLease(job_id=job.id, worker_id="w1",
                                lease_token="tok",
                                expires_at=datetime.now(timezone.utc)
                                - timedelta(minutes=5),
                                heartbeat_at=datetime.now(timezone.utc)
                                - timedelta(minutes=5)))
        db_session.commit()
        result = broker2.broker_recovery_check(db_session)
        assert result["recoverable"] >= 1

    def test_pool_config_sane(self):
        config = broker2.pool_config()
        assert config["pool_pre_ping"] is True
        assert config["pool_size"] > 0


# ===========================================================================
# Perf 2.0
# ===========================================================================

class TestPerf:
    def test_single_flight_leader_then_joined(self):
        registry = perf2.SingleFlightRegistry()
        now = 1000.0
        leader = registry.try_claim("key1", now=now)
        joined = registry.try_claim("key1", now=now + 1.0)
        assert leader["decision"] == "LEADER"
        assert joined["decision"] == "RUNNING"
        assert joined["joined"] is True

    def test_single_flight_release_after_complete(self):
        registry = perf2.SingleFlightRegistry()
        now = 1000.0
        registry.try_claim("key1", now=now)
        registry.complete("key1")
        second = registry.try_claim("key1", now=now + 1.0)
        assert second["decision"] == "LEADER"
        assert registry.active_count() == 1

    def test_single_flight_ttl_expiry(self):
        registry = perf2.SingleFlightRegistry()
        now = 1000.0
        registry.try_claim("k", now=now, ttl_s=5.0)
        again = registry.try_claim("k", now=now + 10.0, ttl_s=5.0)
        assert again["decision"] == "LEADER"

    def test_dedupe_policy_idempotent_only(self):
        assert perf2.dedupe_policy(idempotent=True)["dedupe_allowed"] is True
        assert perf2.dedupe_policy(idempotent=False)["dedupe_allowed"] is False

    def test_cache_audit_flags_risks(self):
        audit = perf2.cache_audit(ttl_s=None, tenant_isolated=False,
                                  invalidated_on_write=False,
                                  max_entries=None)
        assert audit["healthy"] is False
        assert len(audit["issues"]) >= 3

    def test_cache_audit_healthy(self):
        audit = perf2.cache_audit(ttl_s=60, tenant_isolated=True,
                                  invalidated_on_write=True, max_entries=100)
        assert audit["healthy"] is True

    def test_stampede_protection(self):
        cache = {}
        now = 1000.0
        decision = perf2.stampede_protect(cache, "k", ttl_s=60.0, now=now)
        assert decision["decision"] == "REGENERATE"
        # first request completes the regeneration
        perf2.complete_regeneration(cache, "k", "value", now=now)
        hit = perf2.stampede_protect(cache, "k", ttl_s=60.0, now=now + 1.0)
        assert hit["decision"] == "HIT"
        assert hit["value"] == "value"

    def test_stampede_stale_served_during_regen(self):
        cache = {}
        now = 1000.0
        perf2.complete_regeneration(cache, "k", "old", now=now)
        stale = perf2.stampede_protect(cache, "k", ttl_s=5.0,
                                       now=now + 10.0)
        assert stale["decision"] == "REGENERATE_WITH_STALE"
        assert stale["value"] == "old"
        waiter = perf2.stampede_protect(cache, "k", ttl_s=5.0,
                                        now=now + 10.1)
        assert waiter["decision"] == "WAIT_FOR_REGEN"

    def test_cache_stats(self):
        cache = {"a": (1, 1000.0, False)}
        stats = perf2.cache_stats(cache)
        assert stats["entries"] == 1
        assert stats["regenerating"] == 0


# ===========================================================================
# Quality platform
# ===========================================================================

class TestGoldenDatasets:
    def test_datasets_defined(self):
        datasets = qp.golden_datasets()
        assert set(datasets) == {"retrieval", "citations", "temporal"}

    def test_evaluate_retrieval_perfect(self):
        dataset = qp.GOLDEN_RETRIEVAL
        perfect = [sorted(q["relevant"]) for q in dataset["queries"]]
        result = qp.evaluate_retrieval(perfect, dataset=dataset)
        assert result["recall"] == 1.0

    def test_evaluate_citations(self):
        result = qp.evaluate_citations()
        assert result["coverage"] == 1.0
        assert result["correctness"] == 1.0

    def test_run_evaluation_persisted(self, db_session):
        result = qp.run_evaluation(db_session, dataset_name="retrieval_basic",
                                   metrics={"recall": 0.8, "mrr": 0.9,
                                            "sample_count": 3})
        assert result["passed"] is True
        runs = qp.list_evaluations(db_session,
                                   dataset_name="retrieval_basic")
        assert len(runs) == 1

    def test_gate_failure_recorded(self, db_session):
        result = qp.run_evaluation(db_session, dataset_name="citations",
                                   metrics={"coverage": 0.1,
                                            "sample_count": 2})
        assert result["passed"] is False

    def test_compare_models(self):
        comparison = qp.compare_models({"recall": 0.9, "mrr": 0.8},
                                       {"recall": 0.7, "mrr": 0.8})
        assert comparison["wins"]["a"] == 1
        assert comparison["wins"]["b"] == 0

    def test_quality_regression_detected(self):
        result = qp.quality_regression({"recall": 0.6, "mrr": 0.9},
                                       {"recall": 0.8, "mrr": 0.9})
        assert result["regressed"] is True
        assert result["regressions"][0]["metric"] == "recall"

    def test_quality_gate_enforced(self, db_session):
        qp.add_gate(db_session, name="min-recall", metric="recall",
                    operator=">=", threshold=0.5)
        passing = qp.check_gate(db_session, metric="recall", value=0.8)
        failing = qp.check_gate(db_session, metric="recall", value=0.2)
        assert passing["all_passed"] is True
        assert failing["all_passed"] is False

    def test_aggregate_feedback(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        from app.models.search_intel import AIFeedback
        db_session.add(AIFeedback(workspace_id=ws.id, user_id=user.id,
                                  rating="thumbs_up", category="accuracy"))
        db_session.commit()
        result = qp.aggregate_feedback(db_session, workspace_id=ws.id)
        assert result["feedback_count"] == 1
        assert result["thumbs_up_rate"] == 1.0