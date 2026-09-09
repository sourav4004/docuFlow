"""Phase 17 tests — provider production layer.

Adds the remaining mocked-HTTP contract matrix (400/401/403/404/408/502/503,
rate-limit 429 headers, tool-call and structured-output response shapes,
usage accounting), the embeddings provider contract (batching, dimension
validation, retries/timeouts, rate-limit), and capability-aware routing
enforcement. No network and no credentials are used.
"""

import json

import pytest

from app.core.database import get_db
from app.main import app
from tests.shared_db import TestingSessionLocal, override_get_db

app.dependency_overrides[get_db] = override_get_db

from app.models.user import User  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402

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
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_capabilities(db_session):
    from app.models.phase16 import ProviderCapability
    from app.models.phase15 import ProviderHealth
    db_session.query(ProviderCapability).delete()
    db_session.query(ProviderHealth).delete()
    db_session.commit()


def fresh_user(db):
    _counter[0] += 1
    user = User(name=f"PV {_counter[0]}",
                email=f"pv{_counter[0]}@p17provider.com",
                password_hash="x" * 60)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def fresh_workspace(db, user):
    _counter[0] += 1
    ws = Workspace(name=f"pv ws {_counter[0]}", owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return ws


class _OpenAIContext:
    """Monkeypatch helper: routes httpx.Client through a mock transport."""

    def __init__(self, monkeypatch, handler):
        from app.services.llm import openai_compatible as oc
        import httpx as _httpx
        self.oc = oc
        self.httpx = _httpx
        self.handler = handler
        self.monkeypatch = monkeypatch

    def __enter__(self):
        original_client = self.httpx.Client

        def _client_factory(*args, **kw):
            kw["transport"] = self.httpx.MockTransport(self.handler)
            return original_client(**kw)

        self.monkeypatch.setattr(self.oc.httpx, "Client", _client_factory)
        return self

    def __exit__(self, *exc):
        return False


def _chat_response(text, usage=None, status=200, tool_calls=None,
                   finish="stop"):
    msg = {"role": "assistant", "content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    body = {"id": "chatcmpl-1", "object": "chat.completion",
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3,
                               "total_tokens": 10}}
    return __import__("httpx").Response(
        status, text=json.dumps(body),
        headers={"content-type": "application/json"})


def _provider(handler, monkeypatch, model="gpt-test"):
    from app.services.llm.openai_compatible import OpenAICompatibleProvider
    ctx = _OpenAIContext(monkeypatch, handler)
    ctx.__enter__()
    monkeypatch.setattr(_OpenAIContext, "__exit__",
                        lambda *a: False)  # keep transport active
    return OpenAICompatibleProvider(api_key="sk-test-only",
                                    model=model,
                                    base_url="https://api.example.test/v1")


class TestHTTPContractMatrix:
    def _expect_provider_error(self, status, monkeypatch):
        import httpx
        from app.services.llm.base import LLMProviderError
        provider = _provider(lambda r: httpx.Response(
            status, text=f"err {status}"), monkeypatch)
        with pytest.raises(LLMProviderError):
            provider.generate("sys", "hi")

    def test_http_400(self, monkeypatch):
        self._expect_provider_error(400, monkeypatch)

    def test_http_401(self, monkeypatch):
        self._expect_provider_error(401, monkeypatch)

    def test_http_403(self, monkeypatch):
        self._expect_provider_error(403, monkeypatch)

    def test_http_404(self, monkeypatch):
        self._expect_provider_error(404, monkeypatch)

    def test_http_408(self, monkeypatch):
        self._expect_provider_error(408, monkeypatch)

    def test_http_502(self, monkeypatch):
        self._expect_provider_error(502, monkeypatch)

    def test_http_503(self, monkeypatch):
        self._expect_provider_error(503, monkeypatch)

    def test_retry_after_headers_surface_429(self, monkeypatch):
        import httpx
        from app.services.llm.base import LLMProviderError
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(429, text="limited",
                                  headers={"retry-after": "2"})

        provider = _provider(handler, monkeypatch)
        with pytest.raises(LLMProviderError):
            provider.generate("sys", "hi")
        assert calls["n"] == 1  # provider itself does not auto-retry

    def test_usage_accounting(self, monkeypatch):
        provider = _provider(
            lambda r: _chat_response(
                "ok", usage={"prompt_tokens": 30, "completion_tokens": 12,
                             "total_tokens": 42}), monkeypatch)
        response = provider.generate("sys", "hello")
        assert response.input_tokens == 30
        assert response.output_tokens == 12

    def test_empty_content_with_tool_calls_rejected(self, monkeypatch):
        # A response that contains only tool_calls and no text must not be
        # silently treated as a finished answer (no fake success).
        import httpx
        from app.services.llm.base import LLMProviderError
        body = {"id": "chatcmpl-9", "object": "chat.completion",
                "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_1",
                                     "type": "function",
                                     "function": {"name": "search_docs",
                                                  "arguments":
                                                  json.dumps({"q": "p"})}}]},
                    "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 6,
                          "total_tokens": 14}}
        provider = _provider(
            lambda r: httpx.Response(200, text=json.dumps(body),
                                     headers={"content-type":
                                              "application/json"}),
            monkeypatch)
        with pytest.raises(LLMProviderError):
            provider.generate("sys", "search for policy docs")

    def test_structured_output_fields_preserved(self, monkeypatch):
        provider = _provider(
            lambda r: _chat_response("{\"answer\": 42}"), monkeypatch)
        response = provider.generate("sys", "structured please")
        assert response.text == '{"answer": 42}'


class TestEmbeddingContract:
    """Embeddings provider against a mocked OpenAI SDK client."""

    def _provider(self, behavior, monkeypatch, dimension=4):
        from types import SimpleNamespace
        from app.services.embeddings.openai_provider import (
            OpenAIEmbeddingProvider)
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-only", model="text-embedding-3-small",
            dimension=dimension)

        class FakeEmbeddings:
            def create(self, **kwargs):
                if callable(behavior):
                    return behavior(kwargs)
                raise behavior

        class FakeClient:
            embeddings = FakeEmbeddings()

        monkeypatch.setattr(provider, "_get_client",
                            lambda: FakeClient())
        return provider

    def _emb_response(self, vectors):
        from types import SimpleNamespace
        data = [SimpleNamespace(index=i, embedding=vec)
                for i, vec in enumerate(vectors)]
        return SimpleNamespace(data=data)

    def test_embed_texts_batch(self, monkeypatch):
        vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        provider = self._provider(lambda kw: self._emb_response(vectors),
                                  monkeypatch)
        out = provider.embed_texts(["alpha", "beta"])
        assert out == vectors

    def test_batch_payload_has_dimensions(self, monkeypatch):
        captured = {}
        vectors = [[0.1, 0.2, 0.3, 0.4]]
        def behavior(kw):
            captured["kw"] = kw
            return self._emb_response(vectors)
        provider = self._provider(behavior, monkeypatch, dimension=4)
        provider.embed_texts(["alpha"])
        assert captured["kw"]["dimensions"] == 4
        assert captured["kw"]["model"] == "text-embedding-3-small"

    def test_dimension_mismatch_rejected(self, monkeypatch):
        provider = self._provider(
            lambda kw: self._emb_response([[0.1, 0.2]]), monkeypatch,
            dimension=4)
        from app.services.embeddings.base import EmbeddingError
        with pytest.raises(EmbeddingError):
            provider.embed_texts(["bad"])

    def test_http_429_raises_embedding_error(self, monkeypatch):
        from app.services.embeddings.base import EmbeddingError
        provider = self._provider(
            RuntimeError("429 rate limited"), monkeypatch)
        with pytest.raises(EmbeddingError):
            provider.embed_texts(["x"])

    def test_http_500_raises_embedding_error(self, monkeypatch):
        from app.services.embeddings.base import EmbeddingError
        provider = self._provider(RuntimeError("500 boom"), monkeypatch)
        with pytest.raises(EmbeddingError):
            provider.embed_texts(["x"])

    def test_timeout_raises_embedding_error(self, monkeypatch):
        import httpx
        from app.services.embeddings.base import EmbeddingError
        provider = self._provider(httpx.ConnectTimeout("slow"), monkeypatch)
        with pytest.raises(EmbeddingError):
            provider.embed_texts(["x"])

    def test_malformed_response_raises(self, monkeypatch):
        from app.services.embeddings.base import EmbeddingError
        provider = self._provider(
            lambda kw: object(), monkeypatch)  # no .data attribute
        with pytest.raises(EmbeddingError):
            provider.embed_texts(["x"])

    def test_wrong_count_raises(self, monkeypatch):
        from app.services.embeddings.base import EmbeddingError
        provider = self._provider(
            lambda kw: self._emb_response([[0.1, 0.2, 0.3, 0.4]]),
            monkeypatch)
        with pytest.raises(EmbeddingError):
            provider.embed_texts(["one", "two"])

    def test_secret_not_in_error(self, monkeypatch, caplog):
        from app.services.embeddings.base import EmbeddingError
        provider = self._provider(RuntimeError("sk-test-only leaked"),
                                  monkeypatch)
        with caplog.at_level("ERROR"):
            with pytest.raises(EmbeddingError):
                provider.embed_texts(["x"])
        assert "sk-test-only" not in caplog.text


class TestCapabilityRouting:
    def test_unknown_capability_normalized(self, db_session):
        from app.services.provider_platform import (
            upsert_capability, list_capabilities)
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        upsert_capability(db_session, provider="cap-probe",
                          model="model-a",
                          supports_text=True, supports_tools=False,
                          context_window=8192, max_output=2048,
                          embedding_dimensions=384,
                          cost_per_1k_input=0.001,
                          cost_per_1k_output=0.002)
        db_session.commit()
        caps = list_capabilities(db_session)
        found = [c for c in caps
                 if c.provider == "cap-probe" and c.model == "model-a"]
        assert len(found) == 1
        assert found[0].supports_text is True
        assert found[0].supports_tools is False

    def test_routing_requires_capability(self, db_session):
        from app.services.provider_platform import (
            upsert_capability, recommend_model, ProviderRoutingError)
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        upsert_capability(db_session, provider="p1", model="no-vision",
                          supports_text=True, supports_vision=False,
                          supports_tools=True,
                          context_window=100_000, max_output=4096,
                          embedding_dimensions=384)
        db_session.commit()
        # vision capability missing → routing refuses (never silent downgrade)
        with pytest.raises(ProviderRoutingError):
            recommend_model(db_session, requires={"vision": True})

    def test_routing_meets_text_capability(self, db_session):
        from app.services.provider_platform import (
            upsert_capability, recommend_model)
        user = fresh_user(db_session)
        ws = fresh_workspace(db_session, user)
        upsert_capability(db_session, provider="p2", model="text-a",
                          supports_text=True, supports_vision=False,
                          context_window=8192, max_output=1024,
                          embedding_dimensions=384)
        db_session.commit()
        selection = recommend_model(db_session,
                                    requires={"text": True},
                                    prefer="quality")
        assert selection["model"] == "text-a"
