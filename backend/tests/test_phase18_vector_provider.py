"""Phase 18 tests — vector registry + provider production layer.

Vector: model registry (register/deactivate/validate), dimension
enforcement, index operations, vector health, embedding-version isolation.
Provider: timeout policy, bounded retry backoff, 429 Retry-After, error
classification, latency percentile metrics (p50/p95/p99).
"""

import time
from datetime import datetime, timezone, timedelta

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.phase18 import (  # noqa: E402
    EmbeddingModel, VectorIndexOp, ProviderCallMetric,
)
from app.services import vector_registry as vr  # noqa: E402
from app.services import provider_ops as po  # noqa: E402

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
    from app.models.document_chunk import DocumentChunk
    db_session.query(DocumentChunk).delete()
    db_session.query(VectorIndexOp).delete()
    db_session.query(EmbeddingModel).delete()
    db_session.query(ProviderCallMetric).delete()
    db_session.commit()
    yield


def fresh_user(db, tag="p18vp"):
    _counter[0] += 1
    user = User(name=f"P18 VP {_counter[0]}",
                email=f"{tag}{_counter[0]}@p18-vp.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"p18 vp ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


# ============================================================
# Vector model registry
# ============================================================

class TestVectorRegistry:
    def test_register_model(self, db_session):
        model = vr.register_model(db_session, provider="openai",
                                  model="text-embed-3-small",
                                  dimensions=1536)
        db_session.commit()
        assert model.id is not None
        assert model.active is True

    def test_register_model_idempotent(self, db_session):
        vr.register_model(db_session, provider="p", model="m",
                          dimensions=384, version="v1")
        db_session.commit()
        again = vr.register_model(db_session, provider="p", model="m",
                                  dimensions=384, version="v1")
        db_session.commit()
        count = db_session.query(EmbeddingModel).count()
        assert count == 1
        assert again.id is not None

    def test_register_invalid_dimensions(self, db_session):
        with pytest.raises(ValueError):
            vr.register_model(db_session, provider="p", model="m",
                              dimensions=0)
        with pytest.raises(ValueError):
            vr.register_model(db_session, provider="p", model="m",
                              dimensions=100000)

    def test_active_model_returns_most_recent(self, db_session):
        vr.register_model(db_session, provider="p", model="m1",
                          dimensions=384, version="v1")
        vr.register_model(db_session, provider="p", model="m2",
                          dimensions=768, version="v1")
        db_session.commit()
        model = vr.active_model(db_session)
        assert model.model == "m2"

    def test_active_model_filtered(self, db_session):
        vr.register_model(db_session, provider="openai", model="a",
                          dimensions=384)
        db_session.commit()
        model = vr.active_model(db_session, provider="openai", model="a")
        assert model is not None
        assert vr.active_model(db_session, provider="nope") is None

    def test_deactivate_model(self, db_session):
        model = vr.register_model(db_session, provider="p", model="m",
                                  dimensions=384)
        db_session.commit()
        row = vr.deactivate_model(db_session, model.id)
        db_session.commit()
        assert row.active is False
        assert vr.active_model(db_session) is None

    def test_deactivate_missing_raises(self, db_session):
        with pytest.raises(KeyError):
            vr.deactivate_model(db_session, 99999)

    def test_list_models_bounded(self, db_session):
        for i in range(5):
            vr.register_model(db_session, provider="p", model=f"m{i}",
                              dimensions=384)
        db_session.commit()
        rows = vr.list_models(db_session, limit=3)
        assert len(rows) == 3

    def test_validate_dimensions_match(self, db_session):
        vr.register_model(db_session, provider="openai", model="e3",
                          dimensions=1536, version="v1")
        db_session.commit()
        result = vr.validate_embedding_dimensions(
            db_session, 1536, provider="openai", model="e3")
        assert result["valid"] is True

    def test_validate_dimensions_mismatch_rejected(self, db_session):
        vr.register_model(db_session, provider="openai", model="e3",
                          dimensions=1536, version="v1")
        db_session.commit()
        result = vr.validate_embedding_dimensions(
            db_session, 384, provider="openai", model="e3")
        assert result["valid"] is False
        assert "mismatch" in result["note"]

    def test_validate_no_declared_model_allowed(self, db_session):
        result = vr.validate_embedding_dimensions(db_session, 384)
        assert result["valid"] is True


# ============================================================
# Vector index operations
# ============================================================

class TestVectorIndexOps:
    def test_start_index_op(self, db_session):
        op = vr.start_index_op(db_session, op_type="create",
                               index_name="idx_test")
        db_session.commit()
        assert op.status == "RUNNING"

    def test_invalid_op_type_rejected(self, db_session):
        with pytest.raises(ValueError):
            vr.start_index_op(db_session, op_type="drop_all")

    def test_finish_index_op(self, db_session):
        op = vr.start_index_op(db_session, op_type="rebuild")
        db_session.commit()
        done = vr.finish_index_op(db_session, op.id, ok=True,
                                  detail="built in 3s")
        db_session.commit()
        assert done.status == "COMPLETED"
        assert done.completed_at is not None

    def test_finish_index_op_failure(self, db_session):
        op = vr.start_index_op(db_session, op_type="validate")
        db_session.commit()
        done = vr.finish_index_op(db_session, op.id, ok=False,
                                  detail="dimension mismatch")
        db_session.commit()
        assert done.status == "FAILED"

    def test_finish_missing_op_raises(self, db_session):
        with pytest.raises(KeyError):
            vr.finish_index_op(db_session, 99999, ok=True)

    def test_index_plan_deterministic(self, db_session):
        model = vr.register_model(db_session, provider="openai",
                                  model="text-embed-3-small",
                                  dimensions=1536)
        db_session.commit()
        plan = vr.index_plan(model)
        assert plan["index_type"] == "hnsw"
        assert plan["dimensions"] == 1536
        assert plan["requires_pgvector"] is True
        assert "idx_vec_openai" in plan["index_name"]


# ============================================================
# Vector health
# ============================================================

class TestVectorHealth:
    def test_vector_health_shape(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        vr.register_model(db_session, provider="openai", model="e3",
                          dimensions=1536)
        db_session.commit()
        health = vr.vector_health(db_session)
        assert "pgvector" in health
        assert "backend" in health
        assert "active_model" in health
        assert "vector_coverage_pct" in health
        assert health["active_model"]["dimensions"] == 1536

    def test_vector_health_no_chunks_full_coverage(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        health = vr.vector_health(db_session)
        assert health["vector_coverage_pct"] == 100.0

    def test_embedding_version_isolation(self, db_session):
        """Two versions of the same model coexist; only the active one is
        used for dimension validation."""
        vr.register_model(db_session, provider="p", model="m",
                          dimensions=384, version="v1")
        v2 = vr.register_model(db_session, provider="p", model="m",
                               dimensions=768, version="v2")
        db_session.commit()
        assert vr.active_model(db_session).version == "v2"
        result = vr.validate_embedding_dimensions(db_session, 384,
                                                  provider="p", model="m")
        assert result["valid"] is False
        vr.deactivate_model(db_session, v2.id)
        db_session.commit()
        result = vr.validate_embedding_dimensions(db_session, 384,
                                                  provider="p", model="m")
        assert result["valid"] is True


# ============================================================
# Provider timeouts + retry policy
# ============================================================

class TestProviderPolicy:
    def test_timeout_policy_defaults(self):
        policy = po.timeout_policy()
        assert policy["connect_seconds"] > 0
        assert policy["read_seconds"] > 0
        assert policy["total_seconds"] > 0

    def test_timeout_policy_configurable(self, monkeypatch):
        monkeypatch.setenv("PROVIDER_CONNECT_TIMEOUT", "3.5")
        policy = po.timeout_policy()
        assert policy["connect_seconds"] == 3.5

    def test_retry_policy_max_attempts_bounded(self):
        result = po.retry_policy(attempt=po.MAX_RETRIES)
        assert result["retryable"] is False
        assert result["reason"] == "max attempts reached"

    def test_retry_policy_non_retryable_4xx(self):
        for status in (400, 401, 403, 404, 422):
            result = po.retry_policy(status_code=status)
            assert result["retryable"] is False

    def test_retry_policy_retryable_5xx(self):
        for status in (500, 502, 503, 504):
            result = po.retry_policy(status_code=status, attempt=0)
            assert result["retryable"] is True
            assert 0 < result["delay_seconds"] <= po.MAX_BACKOFF_SECONDS

    def test_retry_policy_exponential_backoff(self):
        d1 = po.retry_policy(status_code=500, attempt=0)["delay_seconds"]
        d2 = po.retry_policy(status_code=500, attempt=1)["delay_seconds"]
        assert d2 > d1

    def test_retry_policy_429_honors_retry_after(self):
        result = po.retry_policy(status_code=429, attempt=0,
                                 retry_after=30.0)
        assert result["retryable"] is True
        assert result["delay_seconds"] >= 30.0

    def test_retry_policy_408_retryable(self):
        assert po.retry_policy(status_code=408, attempt=0)["retryable"]

    def test_is_retryable_status(self):
        assert po.is_retryable_status(429)
        assert po.is_retryable_status(503)
        assert not po.is_retryable_status(403)

    def test_classify_error_rate_limit(self):
        assert po.classify_error(ValueError("rate limit"), 429) == "rate_limit"
        assert po.classify_error(TimeoutError()) == "timeout"
        assert po.classify_error(RuntimeError("x"), 503) == "provider"
        assert po.classify_error(RuntimeError("x"), 422) == "validation"
        assert po.classify_error(RuntimeError("x"), 401) == "authorization"
        assert po.classify_error(RuntimeError("x")) == "unknown"


# ============================================================
# Provider call metrics
# ============================================================

class TestProviderMetrics:
    def test_record_call(self, db_session):
        row = po.record_call(db_session, provider="openai", model="gpt-4o",
                             ok=True, latency_ms=250.0)
        db_session.commit()
        assert row.id is not None
        assert row.ok is True

    def test_latency_percentiles(self, db_session):
        for i, ms in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
            po.record_call(db_session, provider="openai", model="m",
                           ok=True, latency_ms=float(ms))
        db_session.commit()
        stats = po.latency_percentiles(db_session, provider="openai")
        assert stats["samples"] == 10
        assert stats["success_rate"] == 1.0
        assert stats["p50_ms"] is not None
        assert stats["p95_ms"] >= stats["p50_ms"]
        assert stats["p99_ms"] >= stats["p95_ms"]

    def test_percentiles_with_failures(self, db_session):
        for ms in [10, 20, 30, 40, 50]:
            po.record_call(db_session, provider="p", model="m",
                           ok=True, latency_ms=float(ms))
        for ms in [2000, 3000]:
            po.record_call(db_session, provider="p", model="m",
                           ok=False, latency_ms=float(ms),
                           error_class="timeout")
        db_session.commit()
        stats = po.latency_percentiles(db_session, provider="p")
        assert stats["success_rate"] == pytest.approx(5 / 7, abs=0.001)
        assert stats["error_rate"] == pytest.approx(2 / 7, abs=0.001)
        assert stats["error_classes"] == {"timeout": 2}

    def test_percentiles_empty(self, db_session):
        stats = po.latency_percentiles(db_session, provider="nope")
        assert stats["samples"] == 0
        assert stats["p50_ms"] is None

    def test_provider_health_dashboard(self, db_session):
        from app.services.provider_platform import record_provider_result
        for i in range(20):
            po.record_call(db_session, provider="openai", model="gpt-4o",
                           ok=True, latency_ms=float(100 + i))
            record_provider_result(db_session, "openai", "gpt-4o",
                                   ok=True, latency_ms=float(100 + i))
        db_session.commit()
        dashboard = po.provider_health_dashboard(db_session)
        assert dashboard["total"] >= 1
        entry = dashboard["providers"][0]
        assert entry["provider"] == "openai"
        assert "p95_ms" in entry and "success_rate" in entry