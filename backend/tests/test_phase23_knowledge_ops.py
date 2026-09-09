"""Phase 23 tests — knowledge maintenance 2.0 + freshness + drift,
webhook reliability, scheduler dedup, review center, security ops 4.0,
continuous evaluation 2.0, search 6.0, real-time streams.
"""

import pytest

from app.main import app  # noqa: F401  (register routes/models)
from tests.shared_db import TestingSessionLocal

from app.models.phase23 import (
    ReviewDecision, SchedulerTaskRun, WebhookDeliveryAttempt,
    WebhookEndpointHealth,
)
from app.models.phase21 import SecurityIncidentP21
from app.models.phase15 import ReviewItem
from app.models.phase22 import EvalExecution, OpsStreamEvent
from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.services import knowledge_maintenance2 as km2
from app.services import webhook_reliability as wr
from app.services import security_eval3 as se3

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


P23_TABLES = [
    ReviewDecision, SchedulerTaskRun, WebhookDeliveryAttempt,
    WebhookEndpointHealth, SecurityIncidentP21, ReviewItem,
    EvalExecution, OpsStreamEvent,
]


@pytest.fixture(autouse=True)
def _clean(db_session):
    for model in P23_TABLES:
        db_session.query(model).delete()
    # Documents from other suites' fixtures must not leak into freshness
    # scans of a newly created workspace (shared SQLite DB).
    from app.models.document_chunk import DocumentChunk
    from app.models.document import Document
    from app.models.phase23 import KnowledgeFreshnessState
    db_session.query(KnowledgeFreshnessState).delete()
    db_session.query(DocumentChunk).delete()
    db_session.query(Document).delete()
    db_session.query(Workspace).delete()
    db_session.query(User).delete()
    db_session.commit()
    yield


def _mkws(db, tag="p23k"):
    _counter[0] += 1
    n = _counter[0]
    user = User(email=f"{tag}{n}@p23k.example", name=tag, password_hash="x")
    db.add(user)
    db.commit()
    ws = Workspace(name=f"ws-{tag}-{n}", owner_id=user.id)
    db.add(ws)
    db.commit()
    return ws


# ===========================================================================
# Freshness engine (Step 24)
# ===========================================================================

class TestFreshness:
    def test_classification_unknown(self):
        result = km2.classify_freshness(updated_at=None)
        assert result["state"] == "UNKNOWN"

    def test_classification_fresh(self):
        from datetime import datetime, timedelta, timezone
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        assert km2.classify_freshness(updated_at=recent)["state"] == "FRESH"

    def test_classification_aging(self):
        from datetime import datetime, timedelta, timezone
        old = datetime.now(timezone.utc) - timedelta(days=20)
        assert km2.classify_freshness(updated_at=old)["state"] == "AGING"

    def test_classification_stale(self):
        from datetime import datetime, timedelta, timezone
        older = datetime.now(timezone.utc) - timedelta(days=60)
        assert km2.classify_freshness(updated_at=older)["state"] == "STALE"

    def test_classification_expired_by_policy(self):
        from datetime import datetime, timedelta, timezone
        doc = datetime.now(timezone.utc) - timedelta(days=10)
        result = km2.classify_freshness(updated_at=doc, expiration_days=7)
        assert result["state"] == "EXPIRED"

    def test_low_reliability_noted(self):
        from datetime import datetime, timedelta, timezone
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        result = km2.classify_freshness(updated_at=recent,
                                        source_reliability=0.3)
        assert any("reliability" in r for r in result["reasons"])

    def test_future_timestamp_not_negative(self):
        from datetime import datetime, timedelta, timezone
        future = datetime.now(timezone.utc) + timedelta(days=5)
        result = km2.classify_freshness(updated_at=future)
        assert result["age_days"] == 0.0

    def test_refresh_freshness_persists(self, db_session):
        ws = _mkws(db_session)
        result = km2.refresh_freshness(db_session, workspace_id=ws.id)
        assert result["workspace_id"] == ws.id
        overview = km2.freshness_overview(db_session, workspace_id=ws.id)
        assert overview["tracked"] == result["classified"]

    def test_states_valid(self, db_session):
        from app.models import KnowledgeFreshnessState
        ws = _mkws(db_session)
        km2.refresh_freshness(db_session, workspace_id=ws.id)
        rows = db_session.query(KnowledgeFreshnessState).all()
        for row in rows:
            assert row.state in km2.FRESHNESS_STATES


# ===========================================================================
# Drift detection (Step 25)
# ===========================================================================

class TestDrift:
    def test_detect_structure(self, db_session):
        ws = _mkws(db_session)
        result = km2.detect_drift(db_session, workspace_id=ws.id)
        assert "findings" in result and "evaluated" in result

    def test_embedding_drift_detected(self, db_session):
        from app.models.document import Document
        from app.models.document_chunk import DocumentChunk
        ws = _mkws(db_session)
        doc = Document(user_id=ws.owner_id, workspace_id=ws.id,
                       original_filename="d.txt", storage_key="k-drift",
                       mime_type="text/plain", file_size=1, status="READY")
        db_session.add(doc)
        db_session.flush()
        for i in range(5):
            db_session.add(DocumentChunk(document_id=doc.id, chunk_index=i,
                                         text=f"t{i}", char_start=i,
                                         char_end=i + 1))  # no embedding
        db_session.commit()
        result = km2.detect_drift(db_session, workspace_id=ws.id)
        kinds = {f["kind"] for f in result["findings"]}
        assert "embedding_drift" in kinds


# ===========================================================================
# Continuous maintenance 2.0 (Step 23)
# ===========================================================================

class TestMaintenance2:
    def test_scan_produces_plan(self, db_session):
        ws = _mkws(db_session)
        result = km2.run_maintenance_scan(db_session, workspace_id=ws.id)
        assert "plan" in result
        assert result["run_id"]

    def test_all_dimensions_covered(self, db_session):
        ws = _mkws(db_session)
        result = km2.run_maintenance_scan(db_session, workspace_id=ws.id)
        assert set(result["dimensions_scanned"]) <= set(
            km2.MAINTENANCE_DIMENSIONS)

    def test_unknown_dimension_ignored(self, db_session):
        ws = _mkws(db_session)
        result = km2.run_maintenance_scan(db_session, workspace_id=ws.id,
                                          dimensions=["bogus"])
        assert result["dimensions_scanned"] == []

    def test_maintenance_history(self, db_session):
        ws = _mkws(db_session)
        km2.run_maintenance_scan(db_session, workspace_id=ws.id)
        history = km2.maintenance_history(db_session, workspace_id=ws.id)
        assert history["count"] >= 1

    def test_no_destructive_actions(self, db_session):
        ws = _mkws(db_session)
        result = km2.run_maintenance_scan(db_session, workspace_id=ws.id)
        blob = str(result["plan"]).lower()
        assert "delete" not in blob and "drop" not in blob


# ===========================================================================
# Webhook reliability (Steps 50-51)
# ===========================================================================

class TestWebhookReliability:
    def test_sign_verify_roundtrip(self):
        sig = wr.sign_payload("secret-abc", {"a": 1})
        result = wr.verify_signature("secret-abc", sig["signature"],
                                     sig["body"], now_ts=sig["timestamp"])
        assert result["valid"] is True

    def test_signature_mismatch(self):
        sig = wr.sign_payload("secret-abc", {"a": 1})
        result = wr.verify_signature("other-secret", sig["signature"],
                                     sig["body"], now_ts=sig["timestamp"])
        assert result["valid"] is False

    def test_replay_protection(self):
        sig = wr.sign_payload("secret-abc", {"a": 1})
        old_ts = sig["timestamp"] - wr.SIGNATURE_TOLERANCE_S - 10
        sig_old = wr.sign_payload("secret-abc", {"a": 1}, timestamp=old_ts)
        result = wr.verify_signature("secret-abc", sig_old["signature"],
                                     sig_old["body"],
                                     now_ts=sig["timestamp"])
        assert result["valid"] is False
        assert "replay" in result["reason"] or "tolerance" in result["reason"]

    def test_malformed_signature(self):
        result = wr.verify_signature("s", "garbage", "body")
        assert result["valid"] is False

    def test_backoff_exponential(self):
        assert wr.backoff_delay_s(1) == 30
        assert wr.backoff_delay_s(2) == 60
        assert wr.backoff_delay_s(3) == 120
        assert wr.backoff_delay_s(20) <= 24 * 3600

    def test_delivery_success_resets_failures(self, db_session):
        ws = _mkws(db_session)
        wr.record_delivery_attempt(db_session, workspace_id=ws.id,
                                   endpoint_id=501, event_type="e",
                                   ok=False, error="fail")
        result = wr.record_delivery_attempt(db_session, workspace_id=ws.id,
                                            endpoint_id=501, event_type="e2",
                                            ok=True)
        assert result["state"] == "DELIVERED"
        from app.models.phase23 import WebhookEndpointHealth
        health = db_session.query(WebhookEndpointHealth).filter_by(
            endpoint_id=501).one()
        assert health.consecutive_failures == 0

    def test_repeated_failures_disable_endpoint(self, db_session):
        ws = _mkws(db_session)
        for i in range(wr.MAX_ATTEMPTS + 1):
            result = wr.record_delivery_attempt(
                db_session, workspace_id=ws.id, endpoint_id=502,
                event_type=f"e{i}", ok=False, error="down")
        assert result["endpoint_disabled"] is True
        # further deliveries are rejected
        blocked = wr.record_delivery_attempt(db_session, workspace_id=ws.id,
                                             endpoint_id=502,
                                             event_type="more", ok=True)
        assert blocked["state"] == "DISABLED_ENDPOINT"

    def test_reenable(self, db_session):
        ws = _mkws(db_session)
        for i in range(wr.MAX_ATTEMPTS + 1):
            wr.record_delivery_attempt(db_session, workspace_id=ws.id,
                                       endpoint_id=503, event_type=f"e{i}",
                                       ok=False)
        result = wr.reenable_endpoint(db_session, endpoint_id=503)
        assert result["reenabled"] is True

    def test_dead_letter_recorded(self, db_session):
        ws = _mkws(db_session)
        for i in range(wr.MAX_ATTEMPTS):
            wr.record_delivery_attempt(db_session, workspace_id=ws.id,
                                       endpoint_id=504, event_type="evt",
                                       ok=False, delivery_id="dl-1",
                                       error=f"err {i}")
        dead = wr.dead_letter_deliveries(db_session, workspace_id=ws.id)
        assert dead["count"] >= 1


# ===========================================================================
# Scheduler task dedup (Step 48)
# ===========================================================================

class TestSchedulerDedup:
    def test_first_claim_wins(self, db_session):
        r1 = wr.claim_scheduled_task(db_session, task_key="t1",
                                     leader_id="L1")
        assert r1["claimed"] is True

    def test_duplicate_claim_skipped(self, db_session):
        wr.claim_scheduled_task(db_session, task_key="t2", leader_id="L1")
        r2 = wr.claim_scheduled_task(db_session, task_key="t2",
                                     leader_id="L2")
        assert r2["claimed"] is False
        assert r2["status"] == "SKIPPED_DUPLICATE"

    def test_different_buckets_independent(self, db_session):
        wr.claim_scheduled_task(db_session, task_key="t3", leader_id="L1",
                                bucket="b1")
        r2 = wr.claim_scheduled_task(db_session, task_key="t3",
                                     leader_id="L1", bucket="b2")
        assert r2["claimed"] is True

    def test_finish_task(self, db_session):
        claim = wr.claim_scheduled_task(db_session, task_key="t4",
                                        leader_id="L1")
        result = wr.finish_scheduled_task(db_session, claim["run_id"],
                                          ok=True)
        assert result["finished"] is True

    def test_due_bucket_deterministic(self):
        assert wr.due_bucket(granularity_minutes=5) == \
               wr.due_bucket(granularity_minutes=5)


# ===========================================================================
# Review center (Step 35)
# ===========================================================================

class TestReviewCenter:
    def test_create_review_item(self, db_session):
        ws = _mkws(db_session)
        result = wr.create_unified_review(
            db_session, workspace_id=ws.id, item_type="ai_action",
            title="Approve deployment", user_id=ws.owner_id)
        assert result["item_id"]
        assert result["status"] == "PENDING"

    def test_invalid_type_rejected(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            wr.create_unified_review(db_session, workspace_id=ws.id,
                                     item_type="bogus", title="t",
                                     user_id=ws.owner_id)

    def test_decide_approve(self, db_session):
        ws = _mkws(db_session)
        item = wr.create_unified_review(db_session, workspace_id=ws.id,
                                        item_type="workflow_approval",
                                        title="wf", user_id=ws.owner_id)
        result = wr.decide_review(db_session, workspace_id=ws.id,
                                  item_id=item["item_id"],
                                  decision="APPROVE",
                                  actor_user_id=ws.owner_id)
        assert result["status"] == "APPROVED"
        assert result["audited"] is True

    def test_decision_audited_row(self, db_session):
        ws = _mkws(db_session)
        item = wr.create_unified_review(db_session, workspace_id=ws.id,
                                        item_type="security_finding",
                                        title="sec", user_id=ws.owner_id)
        wr.decide_review(db_session, workspace_id=ws.id,
                         item_id=item["item_id"], decision="REJECT",
                         actor_user_id=ws.owner_id, reason="no")
        rows = db_session.query(ReviewDecision).all()
        assert len(rows) == 1
        assert rows[0].decision == "REJECT"

    def test_unknown_decision(self, db_session):
        ws = _mkws(db_session)
        item = wr.create_unified_review(db_session, workspace_id=ws.id,
                                        item_type="ai_action", title="t",
                                        user_id=ws.owner_id)
        with pytest.raises(ValueError):
            wr.decide_review(db_session, workspace_id=ws.id,
                             item_id=item["item_id"], decision="MAYBE",
                             actor_user_id=ws.owner_id)

    def test_delegate_requires_assignee(self, db_session):
        ws = _mkws(db_session)
        item = wr.create_unified_review(db_session, workspace_id=ws.id,
                                        item_type="agent_handoff",
                                        title="t", user_id=ws.owner_id)
        with pytest.raises(Exception):
            wr.decide_review(db_session, workspace_id=ws.id,
                             item_id=item["item_id"], decision="DELEGATE",
                             actor_user_id=ws.owner_id)  # no delegate_to


# ===========================================================================
# Security operations 4.0 (Step 36)
# ===========================================================================

class TestSecurityOps:
    def test_create_finding(self, db_session):
        ws = _mkws(db_session)
        result = se3.create_security_finding(
            db_session, workspace_id=ws.id, category="prompt_injection",
            severity="HIGH", title="t", evidence={"src": "test"})
        assert result["finding_id"]
        assert result["severity"] == "SEV1"

    def test_invalid_category(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            se3.create_security_finding(db_session, workspace_id=ws.id,
                                        category="bogus", severity="HIGH",
                                        title="t")

    def test_invalid_severity(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            se3.create_security_finding(db_session, workspace_id=ws.id,
                                        category="ssrf", severity="ULTRA",
                                        title="t")

    def test_dedup_bumps_occurrences(self, db_session):
        ws = _mkws(db_session)
        r1 = se3.create_security_finding(
            db_session, workspace_id=ws.id, category="exfiltration",
            severity="CRITICAL", title="t", dedup_key="k1")
        r2 = se3.create_security_finding(
            db_session, workspace_id=ws.id, category="exfiltration",
            severity="CRITICAL", title="t", dedup_key="k1")
        assert r1["deduplicated"] is False
        assert r2["deduplicated"] is True
        assert r2["occurrences"] == 2

    def test_severity_mapping(self, db_session):
        ws = _mkws(db_session)
        result = se3.create_security_finding(
            db_session, workspace_id=ws.id, category="cross_tenant_access",
            severity="CRITICAL", title="t")
        assert result["severity"] == "SEV0"

    def test_list_and_update(self, db_session):
        ws = _mkws(db_session)
        finding = se3.create_security_finding(
            db_session, workspace_id=ws.id, category="xss",
            severity="MEDIUM", title="t")
        updated = se3.update_security_finding(
            db_session, workspace_id=ws.id,
            finding_id=finding["finding_id"], new_status="MITIGATED")
        assert updated["status"] == "MITIGATED"
        listed = se3.list_security_findings(db_session, workspace_id=ws.id,
                                            status="MITIGATED")
        assert listed["count"] == 1

    def test_severity_matrix_critical(self):
        matrix = se3.security_severity_matrix()
        assert matrix["cross_tenant_access"] == "CRITICAL"

    def test_continuous_scan_synthetic_only(self, db_session):
        ws = _mkws(db_session)
        result = se3.run_continuous_security_scan(db_session,
                                                  workspace_id=ws.id)
        assert result["probes"] >= 3
        assert "no tenant data" in result["note"]


# ===========================================================================
# Continuous evaluation 2.0 (Steps 56-57)
# ===========================================================================

class TestEvaluation2:
    def test_record_run(self, db_session):
        ws = _mkws(db_session)
        result = se3.record_evaluation_run(
            db_session, workspace_id=ws.id, domain="rag",
            dataset_version="v1", model="fake", provider="fake",
            config={"k": 5}, environment="test", metrics={"score": 0.9},
            latency_p95_ms=100.0)
        assert result["run_id"]
        assert result["status"] == "COMPLETED"

    def test_invalid_domain(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            se3.record_evaluation_run(db_session, workspace_id=ws.id,
                                      domain="bogus", dataset_version="v1",
                                      model="m", provider="p", config={},
                                      environment="e", metrics={},
                                      latency_p95_ms=0)

    def test_promotion_gates_all_pass(self, db_session):
        ws = _mkws(db_session)
        result = se3.evaluate_promotion_gates(
            db_session, workspace_id=ws.id, quality=0.9,
            security_findings=0, latency_p95_ms=500, cost=0.1,
            regression=0.0)
        assert result["eligible"] is True
        assert result["requires_explicit_authorization"] is True

    def test_promotion_gate_quality_fail(self, db_session):
        ws = _mkws(db_session)
        result = se3.evaluate_promotion_gates(
            db_session, workspace_id=ws.id, quality=0.5,
            security_findings=0, latency_p95_ms=500, cost=0.1)
        assert result["eligible"] is False
        assert result["gates"]["quality_threshold"]["passed"] is False

    def test_promotion_gate_security_fail(self, db_session):
        ws = _mkws(db_session)
        result = se3.evaluate_promotion_gates(
            db_session, workspace_id=ws.id, quality=0.9,
            security_findings=1, latency_p95_ms=500, cost=0.1)
        assert result["eligible"] is False

    def test_promotion_gate_latency_fail(self, db_session):
        ws = _mkws(db_session)
        result = se3.evaluate_promotion_gates(
            db_session, workspace_id=ws.id, quality=0.9,
            security_findings=0, latency_p95_ms=9000, cost=0.1)
        assert result["gates"]["latency_threshold"]["passed"] is False

    def test_promotion_gate_regression_fail(self, db_session):
        ws = _mkws(db_session)
        result = se3.evaluate_promotion_gates(
            db_session, workspace_id=ws.id, quality=0.9,
            security_findings=0, latency_p95_ms=500, cost=0.1,
            regression=-0.5)
        assert result["gates"]["regression_threshold"]["passed"] is False

    def test_rollback_supported(self, db_session):
        ws = _mkws(db_session)
        result = se3.evaluate_promotion_gates(
            db_session, workspace_id=ws.id, quality=0.9,
            security_findings=0, latency_p95_ms=500, cost=0.1)
        assert result["rollback_supported"] is True


# ===========================================================================
# Search 6.0 (Steps 52-53)
# ===========================================================================

class TestSearch6:
    def test_explain_ranking_order(self):
        candidates = [
            {"id": "a", "keyword_score": 0.9, "vector_score": 0.9,
             "freshness_score": 0.9, "authority_score": 0.9},
            {"id": "b", "keyword_score": 0.1, "vector_score": 0.1,
             "freshness_score": 0.1, "authority_score": 0.1},
        ]
        result = se3.explain_ranking(None, candidates=candidates)
        assert result["ranking"][0]["id"] == "a"
        assert result["ranking"][0]["fused"] > result["ranking"][1]["fused"]

    def test_explainability_breakdown(self):
        candidates = [{"id": "x", "keyword_score": 1.0, "vector_score": 0.0,
                       "freshness_score": 0.0, "authority_score": 0.0}]
        result = se3.explain_ranking(None, candidates=candidates)
        breakdown = result["ranking"][0]["breakdown"]
        assert breakdown["keyword"] == pytest.approx(
            se3.SEARCH_WEIGHTS["keyword"], abs=1e-3)

    def test_custom_weights(self):
        candidates = [{"id": "x", "keyword_score": 1.0, "vector_score": 0.0,
                       "freshness_score": 0.0, "authority_score": 0.0}]
        result = se3.explain_ranking(None, candidates=candidates,
                                     weights={"keyword": 1.0, "vector": 0.0,
                                              "freshness": 0.0,
                                              "authority": 0.0})
        assert result["ranking"][0]["fused"] == pytest.approx(1.0, abs=1e-3)

    def test_diversity_limits_dominance(self):
        ranking = [{"id": f"r{i}", "document_id": 1} for i in range(5)]
        result = se3.apply_diversity(ranking)
        top_docs = [r["document_id"] for r in result[:3]]
        assert top_docs.count(1) == 2   # capped at 2 per document

    def test_self_evaluation_healthy(self, db_session):
        ws = _mkws(db_session)
        result = se3.search_self_evaluation(
            db_session, workspace_id=ws.id, zero_result_queries=1,
            total_queries=100, reformulations=5, precision_estimate=0.9)
        assert result["healthy"] is True
        assert result["proposal_created"] is False

    def test_self_evaluation_creates_proposal_only(self, db_session):
        ws = _mkws(db_session)
        result = se3.search_self_evaluation(
            db_session, workspace_id=ws.id, zero_result_queries=30,
            total_queries=100, reformulations=60, precision_estimate=0.4)
        assert result["healthy"] is False
        assert result["proposal_created"] is True
        assert result["requires_human_approval"] is True


# ===========================================================================
# Real-time streams (Step 62)
# ===========================================================================

class TestStreams:
    def test_emit_and_read(self, db_session):
        ws = _mkws(db_session)
        emitted = se3.emit_stream_event(db_session, workspace_id=ws.id,
                                        stream="ops", kind="worker_event",
                                        payload={"a": 1})
        assert emitted["seq"] >= 1
        read = se3.read_stream(db_session, workspace_id=ws.id, stream="ops")
        assert read["count"] == 1
        assert read["items"][0]["seq"] == emitted["seq"]

    def test_sequence_monotonic(self, db_session):
        ws = _mkws(db_session)
        e1 = se3.emit_stream_event(db_session, workspace_id=ws.id,
                                   stream="slo", kind="k", payload={})
        e2 = se3.emit_stream_event(db_session, workspace_id=ws.id,
                                   stream="slo", kind="k", payload={})
        assert e2["seq"] == e1["seq"] + 1

    def test_after_seq_cursor(self, db_session):
        ws = _mkws(db_session)
        e1 = se3.emit_stream_event(db_session, workspace_id=ws.id,
                                   stream="security", kind="k", payload={})
        e2 = se3.emit_stream_event(db_session, workspace_id=ws.id,
                                   stream="security", kind="k", payload={})
        read = se3.read_stream(db_session, workspace_id=ws.id,
                               stream="security", after_seq=e1["seq"])
        assert read["count"] == 1
        assert read["items"][0]["seq"] == e2["seq"]

    def test_tenant_scoped(self, db_session):
        ws1 = _mkws(db_session)
        ws2 = _mkws(db_session)
        se3.emit_stream_event(db_session, workspace_id=ws1.id,
                              stream="incident", kind="k", payload={})
        read = se3.read_stream(db_session, workspace_id=ws2.id,
                               stream="incident")
        assert read["count"] == 0

    def test_unknown_stream_rejected(self, db_session):
        ws = _mkws(db_session)
        with pytest.raises(ValueError):
            se3.emit_stream_event(db_session, workspace_id=ws.id,
                                  stream="bogus", kind="k", payload={})

    def test_all_stream_kinds_listed(self):
        assert len(se3.STREAM_KINDS) >= 7
        assert "worker" in se3.STREAM_KINDS
        assert "security" in se3.STREAM_KINDS
