"""Phase 16 test suite — provider platform + vector platform.

Provider contract tests exercise the OpenAI-compatible gateway against
mocked HTTP transports (normal, streaming, malformed, timeout, 429, 5xx,
invalid JSON, usage metadata) — no network and no credentials required.
Capability registry, routing determinism, sensitivity policy, circuit
breaker behavior, embedding cache and dimension validation are covered too.
"""

import json
import logging
import time
import uuid

import httpx

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.phase16 import EmbeddingCache, ProviderCapability  # noqa: E402
from app.models.phase15 import ProviderHealth  # noqa: E402
from app.services.provider_platform import (  # noqa: E402
    upsert_capability, get_capability, list_capabilities, capabilities_meet,
    record_provider_result, provider_status, circuit_state_for,
    is_allowed_for_sensitivity, recommend_model, gateway_status,
    ProviderRoutingError,
)
from app.services.vector_platform import (  # noqa: E402
    detect_pgvector, validate_dimensions, DimensionMismatchError,
    embedding_cache_get, embedding_cache_set, embedding_cache_stats,
    content_hash,
)

_counter = [0]


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean(db_session):
    db_session.query(ProviderHealth).delete()
    db_session.query(ProviderCapability).delete()
    db_session.query(EmbeddingCache).delete()
    db_session.commit()


def fresh_user(db, tag="pu"):
    _counter[0] += 1
    user = User(name=f"Provider User {_counter[0]}",
                email=f"{tag}{_counter[0]}@p16-provider.test",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user, tag="psw"):
    _counter[0] += 1
    ws = Workspace(name=f"{tag} {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


# ============================================================
# Capability registry
# ============================================================

class TestCapabilityRegistry:
    def test_upsert_and_get(self, db_session):
        upsert_capability(db_session, "acme", "pro-1", supports_tools=True,
                          supports_structured=True, context_window=16000,
                          cost_per_1k_input=0.01, latency_class="fast")
        db_session.commit()
        cap = get_capability(db_session, "acme", "pro-1")
        assert cap is not None
        assert cap.supports_tools is True
        assert cap.context_window == 16000

    def test_upsert_is_idempotent(self, db_session):
        upsert_capability(db_session, "acme", "pro-1")
        upsert_capability(db_session, "acme", "pro-1",
                          supports_vision=True)
        db_session.commit()
        caps = list_capabilities(db_session, provider="acme")
        assert len(caps) == 1
        assert caps[0].supports_vision is True

    def test_capabilities_meet(self, db_session):
        cap = upsert_capability(db_session, "acme", "pro-1",
                                supports_tools=True)
        ok, reason = capabilities_meet(cap, {"tools": True})
        assert ok
        ok, reason = capabilities_meet(cap, {"vision": True})
        assert not ok
        assert "vision" in reason

    def test_min_context_check(self, db_session):
        cap = upsert_capability(db_session, "acme", "small",
                                context_window=4000)
        ok, _ = capabilities_meet(cap, {"min_context": 8000})
        assert not ok

    def test_latency_class_validation(self, db_session):
        cap = upsert_capability(db_session, "acme", "m", latency_class="turbo")
        db_session.commit()
        assert cap.latency_class == "medium"  # unknown → normalized


# ============================================================
# Sensitivity policy
# ============================================================

class TestSensitivityPolicy:
    def test_restricted_requires_allowlist(self):
        assert is_allowed_for_sensitivity("openai", "RESTRICTED",
                                          ["approved-llm"]) is False
        assert is_allowed_for_sensitivity("approved-llm", "RESTRICTED",
                                          ["approved-llm"]) is True

    def test_internal_allows_any_unless_allowlist(self):
        assert is_allowed_for_sensitivity("anyone", "INTERNAL") is True
        assert is_allowed_for_sensitivity("x", "PUBLIC") is True

    def test_confidential_allowlist(self):
        assert is_allowed_for_sensitivity("x", "CONFIDENTIAL",
                                          ["x", "y"]) is True
        assert is_allowed_for_sensitivity("z", "CONFIDENTIAL",
                                          ["x", "y"]) is False

    def test_invalid_sensitivity_defaults_internal(self):
        assert is_allowed_for_sensitivity("p", "SOMETHING") is True


# ============================================================
# Health 2.0 + circuit breaker
# ============================================================

class TestProviderHealth:
    def test_record_success(self, db_session):
        health = record_provider_result(db_session, "acme", "m1", ok=True,
                                        latency_ms=120)
        db_session.commit()
        assert health.success_count == 1
        assert health.circuit_state == "CLOSED"
        assert health.status == "UP"

    def test_consecutive_failures_open_circuit(self, db_session):
        for i in range(5):
            record_provider_result(db_session, "acme", "m1", ok=False,
                                   error_class="provider")
        db_session.commit()
        health = provider_status(db_session, provider="acme")[0]
        assert health.circuit_state == "OPEN"
        assert health.consecutive_failures == 5

    def test_half_open_after_cooldown_then_closed(self, db_session):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        for i in range(5):
            record_provider_result(db_session, "acme", "m1", ok=False,
                                   error_class="timeout", now=now)
        state = circuit_state_for(
            provider_status(db_session)[0],
            now=now + timedelta(seconds=301))
        assert state == "HALF_OPEN"
        # Successful probe closes the circuit.
        record_provider_result(db_session, "acme", "m1", ok=True, latency_ms=50,
                               now=now + timedelta(seconds=301))
        health = provider_status(db_session)[0]
        assert health.circuit_state == "CLOSED"

    def test_rolling_latency(self, db_session):
        record_provider_result(db_session, "acme", "m1", ok=True,
                               latency_ms=100)
        record_provider_result(db_session, "acme", "m1", ok=True,
                               latency_ms=200)
        health = provider_status(db_session)[0]
        assert health.avg_latency_ms is not None
        assert health.avg_latency_ms < 200  # EMA pulls toward the recent value

    def test_circuit_state_closed_below_threshold(self, db_session):
        for i in range(3):
            record_provider_result(db_session, "acme", "m1", ok=False)
        health = provider_status(db_session)[0]
        assert circuit_state_for(health) == "CLOSED"


# ============================================================
# Routing 3.0
# ============================================================

class TestRouting:
    def test_route_respects_required_capability(self, db_session):
        upsert_capability(db_session, "textco", "txt", supports_text=True)
        upsert_capability(db_session, "visco", "see", supports_text=True,
                          supports_vision=True)
        result = recommend_model(db_session, requires={"vision": True})
        assert result["provider"] == "visco"
        assert result["model"] == "see"

    def test_route_skips_restricted_disallowed(self, db_session):
        upsert_capability(db_session, "openai", "gpt", supports_text=True)
        upsert_capability(db_session, "onprem", "local", supports_text=True)
        result = recommend_model(db_session, sensitivity="RESTRICTED",
                                 allowed_providers=["onprem"])
        assert result["provider"] == "onprem"
        with pytest.raises(ProviderRoutingError):
            recommend_model(db_session, sensitivity="RESTRICTED",
                            allowed_providers=["other-only"])

    def test_route_skips_open_circuit_provider(self, db_session):
        upsert_capability(db_session, "broken", "b1", supports_text=True)
        upsert_capability(db_session, "healthy", "h1", supports_text=True)
        for i in range(6):
            record_provider_result(db_session, "broken", "b1", ok=False,
                                   error_class="provider")
        result = recommend_model(db_session)
        assert result["provider"] == "healthy"

    def test_route_cost_ceiling(self, db_session):
        upsert_capability(db_session, "cheap", "c1", cost_per_1k_input=0.001)
        upsert_capability(db_session, "pricey", "p1", cost_per_1k_input=9.99)
        result = recommend_model(db_session, max_cost_per_1k=1.0)
        assert result["provider"] == "cheap"

    def test_route_deterministic(self, db_session):
        upsert_capability(db_session, "a", "m", supports_text=True)
        upsert_capability(db_session, "b", "m", supports_text=True)
        first = recommend_model(db_session)
        second = recommend_model(db_session)
        assert first["provider"] == second["provider"]
        assert first["model"] == second["model"]

    def test_route_explainable_factors(self, db_session):
        upsert_capability(db_session, "a", "m", supports_text=True,
                          latency_class="fast")
        result = recommend_model(db_session, prefer="latency")
        assert "factors" in result
        assert result["factors"]["sensitivity_allowed"] is True

    def test_no_candidates_raises(self, db_session):
        with pytest.raises(ProviderRoutingError):
            recommend_model(db_session, requires={"vision": True})

    def test_gateway_status_safe_default(self):
        status = gateway_status()
        assert "api_key" not in status  # never exposes the key field
        assert "real_gateway_configured" in status
        assert status["configured_provider"] in ("fake", "openai_compatible")


# ============================================================
# Vector platform
# ============================================================

class TestVectorPlatform:
    def test_sqlite_reports_json_fallback(self, db_session):
        status = detect_pgvector(db_session)
        assert status["active_backend"] == "json_fallback"
        assert status["pgvector_available"] is False
        assert "reason" in status  # never silent

    def test_dimension_validation_ok(self):
        dims = validate_dimensions("p", "m", 384, [0.1] * 384)
        assert dims == 384

    def test_dimension_mismatch_raises(self):
        with pytest.raises(DimensionMismatchError):
            validate_dimensions("p", "m", 384, [0.1] * 128)

    def test_empty_embedding_raises(self):
        with pytest.raises(DimensionMismatchError):
            validate_dimensions("p", "m", 384, [])

    def test_embedding_cache_roundtrip(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        emb = [0.5] * 384
        embedding_cache_set(db_session, "p", "m", 384, "hello world", emb,
                            workspace_id=ws.id)
        db_session.commit()
        got = embedding_cache_get(db_session, "p", "m", 384, "hello world",
                                  workspace_id=ws.id)
        assert got == emb
        stats = embedding_cache_stats(db_session)
        assert stats["entries"] == 1

    def test_embedding_cache_tenant_bound_keys(self, db_session):
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        ws2 = fresh_workspace(db_session, user)
        embedding_cache_set(db_session, "p", "m", 8, "same content",
                            [1.0] * 8, workspace_id=ws.id)
        embedding_cache_set(db_session, "p", "m", 8, "same content",
                            [2.0] * 8, workspace_id=ws2.id)
        db_session.commit()
        stats = embedding_cache_stats(db_session)
        assert stats["entries"] == 2  # tenant-scoped keys never collide
        got = embedding_cache_get(db_session, "p", "m", 8, "same content",
                                  workspace_id=ws.id)
        assert got == [1.0] * 8

    def test_embedding_cache_miss(self, db_session):
        assert embedding_cache_get(db_session, "p", "m", 8, "never stored") is None

    def test_content_hash_deterministic(self):
        assert content_hash("abc") == content_hash("abc")
        assert content_hash("abc") != content_hash("abd")


# ============================================================
# OpenAI-compatible provider contract tests (mocked HTTP)
# ============================================================

class TestProviderContract:
    """Provider behavior against mocked transports — no network."""

    def _provider(self, handler, monkeypatch, **kwargs):
        from app.services.llm.openai_compatible import OpenAICompatibleProvider
        from app.services.llm import openai_compatible as oc
        import httpx as _httpx
        original_client = _httpx.Client

        def _client_factory(*args, **kw):
            kw["transport"] = _httpx.MockTransport(handler)
            return original_client(**kw)

        monkeypatch.setattr(oc.httpx, "Client", _client_factory)
        return OpenAICompatibleProvider(
            api_key="sk-test-secret-key",
            model=kwargs.pop("model", "gpt-test"),
            base_url="https://api.example.test/v1",
        )

    def _json_response(self, status, text, usage=None):
        body = {"id": "chatcmpl-1", "object": "chat.completion",
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": text},
                             "finish_reason": "stop"}],
                "usage": usage or {"prompt_tokens": 10,
                                   "completion_tokens": 5,
                                   "total_tokens": 15}}
        return httpx.Response(status, text=json.dumps(body),
                              headers={"content-type": "application/json"})

    def test_normal_completion(self, monkeypatch):
        import httpx
        captured = {}

        def handler(request):
            payload = json.loads(request.content)
            captured["payload"] = payload
            captured["auth"] = request.headers.get("Authorization")
            return self._json_response(200, "Hello from the model",
                                       usage={"prompt_tokens": 12,
                                              "completion_tokens": 3,
                                              "total_tokens": 15})

        provider = self._provider(handler, monkeypatch)
        response = provider.generate("sys", "hi")
        assert response.text == "Hello from the model"
        assert response.input_tokens == 12
        assert response.output_tokens == 3
        assert captured["auth"] == "Bearer sk-test-secret-key"
        assert captured["payload"]["model"] == "gpt-test"
        assert captured["payload"]["messages"][0]["role"] == "system"

    def test_timeout(self, monkeypatch):
        import httpx

        def handler(request):
            raise httpx.ConnectTimeout("slow")

        provider = self._provider(handler, monkeypatch)
        from app.services.llm.base import LLMTimeoutError
        with pytest.raises(LLMTimeoutError):
            provider.generate("sys", "hi")

    def test_http_500(self, monkeypatch):
        import httpx
        provider = self._provider(lambda request: httpx.Response(500, text="boom"),
                                  monkeypatch)
        from app.services.llm.base import LLMProviderError
        with pytest.raises(LLMProviderError):
            provider.generate("sys", "hi")

    def test_http_429(self, monkeypatch):
        import httpx
        provider = self._provider(lambda request: httpx.Response(429, text="ratelimited"),
                                  monkeypatch)
        from app.services.llm.base import LLMProviderError
        with pytest.raises(LLMProviderError):
            provider.generate("sys", "hi")

    def test_invalid_json_body(self, monkeypatch):
        import httpx
        provider = self._provider(
            lambda request: httpx.Response(200, text="not-json{{"), monkeypatch)
        from app.services.llm.base import LLMProviderError
        with pytest.raises(LLMProviderError):
            provider.generate("sys", "hi")

    def test_malformed_structure(self, monkeypatch):
        import httpx
        provider = self._provider(
            lambda request: httpx.Response(200, text=json.dumps({"nope": 1})),
            monkeypatch)
        from app.services.llm.base import LLMProviderError
        with pytest.raises(LLMProviderError):
            provider.generate("sys", "hi")

    def test_empty_text_rejected(self, monkeypatch):
        import httpx
        provider = self._provider(
            lambda request: self._json_response(200, ""), monkeypatch)
        from app.services.llm.base import LLMProviderError
        with pytest.raises(LLMProviderError):
            provider.generate("sys", "hi")

    def test_streaming_chunks(self, monkeypatch):
        import httpx
        sse = (
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hel"}}]}) + "\n\n"
            "data: " + json.dumps({"choices": [{"delta": {"content": "lo"}}]}) + "\n\n"
            "data: [DONE]\n\n"
        )

        def handler(request):
            return httpx.Response(200, content=sse)

        provider = self._provider(handler, monkeypatch)
        out = "".join(provider.stream_generate("sys", "hi"))
        assert out == "Hello"

    def test_secret_not_logged_on_error(self, monkeypatch, caplog):
        import httpx
        provider = self._provider(
            lambda request: httpx.Response(500, text="server exploded"),
            monkeypatch)
        from app.services.llm.base import LLMProviderError
        with caplog.at_level(logging.ERROR):
            with pytest.raises(LLMProviderError):
                provider.generate("sys", "hi")
        assert "sk-test-secret-key" not in caplog.text
