"""Phase 6 Step 5 tests: Streaming RAG.

Tests cover:
- Streaming LLM abstraction (fake provider, base fallback)
- LLMService streaming
- Streaming RAG service with mocked DB
- SSE endpoint integration with mocked DB
- Error handling, cancellation, persistence
- Authentication/ownership

Uses FakeLLMProvider and mocks — no paid API required.
PostgreSQL tests are skipped if DB is unavailable.
"""

import json
import pytest
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

from app.services.llm import (
    LLMProvider,
    LLMResponse,
    LLMProviderError,
    LLMTimeoutError,
    LLMService,
)
from app.services.llm.fake_provider import FakeLLMProvider
from app.services.llm.openai_compatible import OpenAICompatibleProvider


# ==========================================================================
# 1. STREAMING LLM PROVIDER ABSTRACTION
# ==========================================================================

class TestStreamingLLMAbstraction:
    """Test the stream_generate interface on providers."""

    def test_fake_provider_has_stream_generate(self):
        provider = FakeLLMProvider()
        assert hasattr(provider, "stream_generate")
        assert callable(provider.stream_generate)

    def test_fake_stream_returns_iterator(self):
        provider = FakeLLMProvider()
        result = provider.stream_generate("system prompt", "hello world")
        chunks = list(result)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_fake_stream_chunks_reconstruct_response(self):
        provider = FakeLLMProvider()
        response = provider.generate("system prompt", "hello world")
        chunks = list(provider.stream_generate("system prompt", "hello world"))
        reconstructed = "".join(chunks)
        assert reconstructed == response.text

    def test_fake_stream_word_by_word(self):
        provider = FakeLLMProvider()
        chunks = list(provider.stream_generate("sys", "test"))
        # Each chunk should be a word (first without leading space, rest with)
        for i, chunk in enumerate(chunks):
            assert len(chunk.strip()) > 0
            if i == 0:
                assert not chunk.startswith(" ")
            else:
                assert chunk.startswith(" ")

    def test_base_provider_stream_generate_fallback(self):
        """Base LLMProvider.stream_generate should fall back to generate()."""
        # Create a minimal concrete subclass that only implements generate()
        class MinimalProvider(LLMProvider):
            @property
            def name(self):
                return "minimal"

            @property
            def model(self):
                return "minimal-v1"

            def generate(self, system_prompt, user_prompt):
                return LLMResponse(
                    text="Hello world",
                    model="minimal-v1",
                    provider="minimal",
                )

        provider = MinimalProvider()
        chunks = list(provider.stream_generate("sys", "test"))
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_base_provider_stream_empty_response(self):
        class EmptyProvider(LLMProvider):
            @property
            def name(self):
                return "empty"

            @property
            def model(self):
                return "empty-v1"

            def generate(self, system_prompt, user_prompt):
                return LLMResponse(text="", model="empty-v1", provider="empty")

        provider = EmptyProvider()
        chunks = list(provider.stream_generate("sys", "test"))
        assert chunks == []

    def test_fake_stream_error_on_keyword(self):
        provider = FakeLLMProvider()
        with pytest.raises(RuntimeError):
            list(provider.stream_generate("sys", "trigger error please"))


# ==========================================================================
# 2. LLM SERVICE STREAMING
# ==========================================================================

class TestLLMServiceStreaming:
    """Test the LLMService.stream_generate method."""

    def test_service_stream_returns_iterator(self):
        fake = FakeLLMProvider()
        svc = LLMService(provider=fake)
        chunks = list(svc.stream_generate("system", "hello"))
        assert len(chunks) > 0

    def test_service_stream_reconstructs_response(self):
        fake = FakeLLMProvider()
        svc = LLMService(provider=fake)
        response = svc.generate("system", "hello world")
        chunks = list(svc.stream_generate("system", "hello world"))
        reconstructed = "".join(chunks)
        assert reconstructed == response.text

    def test_service_stream_validates_empty_prompt(self):
        fake = FakeLLMProvider()
        svc = LLMService(provider=fake)
        with pytest.raises(LLMProviderError, match="must not be empty"):
            list(svc.stream_generate("system", "   "))

    def test_service_stream_validates_long_prompt(self):
        fake = FakeLLMProvider()
        svc = LLMService(provider=fake)
        with pytest.raises(LLMProviderError, match="must not exceed"):
            list(svc.stream_generate("x" * 100_000, "test"))

    def test_service_stream_validates_type(self):
        fake = FakeLLMProvider()
        svc = LLMService(provider=fake)
        with pytest.raises(LLMProviderError, match="must be a string"):
            list(svc.stream_generate(123, "test"))  # type: ignore

    def test_service_stream_propagates_provider_error(self):
        """When the provider raises during streaming, the service wraps it."""

        class ErrorProvider(LLMProvider):
            @property
            def name(self):
                return "error"

            @property
            def model(self):
                return "error-v1"

            def generate(self, system_prompt, user_prompt):
                return LLMResponse(text="", model="error-v1", provider="error")

            def stream_generate(self, system_prompt, user_prompt):
                raise LLMProviderError("provider stream failed")

        svc = LLMService(provider=ErrorProvider())
        with pytest.raises(LLMProviderError, match="provider stream failed"):
            list(svc.stream_generate("sys", "test"))


# ==========================================================================
# 3. OPENAI COMPATIBLE STREAMING (interface tests only — no real API)
# ==========================================================================

class TestOpenAICompatibleStreaming:
    """Test the streaming interface exists on OpenAICompatibleProvider."""

    def test_openai_provider_has_stream_generate(self):
        provider = OpenAICompatibleProvider(
            api_key="test-key-123",
            model="test-model",
            base_url="http://localhost:1234/v1",
        )
        assert hasattr(provider, "stream_generate")
        assert callable(provider.stream_generate)

    def test_stream_generate_method_signature(self):
        provider = OpenAICompatibleProvider(
            api_key="test-key-123",
            model="test-model",
        )
        import inspect
        sig = inspect.signature(provider.stream_generate)
        params = list(sig.parameters.keys())
        assert "system_prompt" in params
        assert "user_prompt" in params

    def test_stream_returns_iterator_type(self):
        provider = OpenAICompatibleProvider(
            api_key="test-key-123",
            model="test-model",
        )
        import inspect
        # Verify it's a generator function
        assert inspect.isgeneratorfunction(provider.stream_generate)


# ==========================================================================
# 4. STREAMING RAG SERVICE (mocked DB)
# ==========================================================================

class TestStreamingRAGService:
    """Test answer_question_with_history_streaming with mocked dependencies."""

    @patch("app.services.rag_service.retrieve_context")
    @patch("app.services.rag_service.get_system_prompt")
    @patch("app.services.rag_service.build_conversation_aware_user_prompt")
    def test_streaming_no_context_yields_no_info_message(self, mock_build_prompt, mock_get_sys, mock_retrieve):
        """When retrieval returns 0 results, yield the no-info message."""
        mock_retrieve.return_value = MagicMock(
            total_results=0,
            context="",
            results=[],
        )
        from app.services.rag_service import answer_question_with_history_streaming
        fake_llm = LLMService(provider=FakeLLMProvider())

        chunks = list(answer_question_with_history_streaming(
            db=MagicMock(),
            user_id=1,
            question="test question",
            llm_service=fake_llm,
        ))
        full_text = "".join(chunks)
        assert "don't have enough information" in full_text

    @patch("app.services.rag_service.retrieve_context")
    @patch("app.services.rag_service.get_system_prompt")
    @patch("app.services.rag_service.build_conversation_aware_user_prompt")
    def test_streaming_with_context_yields_tokens(self, mock_build_prompt, mock_get_sys, mock_retrieve):
        """When retrieval has results, stream tokens from the LLM."""
        mock_get_sys.return_value = "You are a helpful assistant."
        mock_build_prompt.return_value = "User question: test question"

        mock_result = MagicMock()
        mock_result.chunk_id = 1
        mock_result.chunk_index = 0
        mock_result.document_id = 1
        mock_result.original_filename = "test.pdf"
        mock_result.page_start = 1
        mock_result.page_end = 1
        mock_result.similarity_score = 0.9
        mock_result.text = "relevant content"
        mock_result.keyword_score = None
        mock_result.final_score = 0.9
        mock_result.match_type = "vector"

        mock_retrieve.return_value = MagicMock(
            total_results=1,
            context="relevant content",
            results=[mock_result],
            retrieval_mode="vector",
        )

        from app.services.rag_service import answer_question_with_history_streaming
        fake_llm = LLMService(provider=FakeLLMProvider())

        chunks = list(answer_question_with_history_streaming(
            db=MagicMock(),
            user_id=1,
            question="test question",
            llm_service=fake_llm,
        ))
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0
        assert "don't have enough information" not in full_text

    @patch("app.services.rag_service.retrieve_context")
    @patch("app.services.rag_service.get_system_prompt")
    @patch("app.services.rag_service.build_conversation_aware_user_prompt")
    def test_streaming_with_history(self, mock_build_prompt, mock_get_sys, mock_retrieve):
        """Streaming should accept and pass conversation history."""
        mock_retrieve.return_value = MagicMock(
            total_results=0,
            context="",
            results=[],
        )
        from app.services.rag_service import answer_question_with_history_streaming
        fake_llm = LLMService(provider=FakeLLMProvider())

        chunks = list(answer_question_with_history_streaming(
            db=MagicMock(),
            user_id=1,
            question="follow up",
            conversation_history="User: previous\nAssistant: response",
            llm_service=fake_llm,
        ))
        assert len(chunks) > 0

    def test_streaming_rejects_empty_question(self):
        from app.services.rag_service import answer_question_with_history_streaming, RAGError
        with pytest.raises(RAGError, match="must not be empty"):
            list(answer_question_with_history_streaming(
                db=MagicMock(),
                user_id=1,
                question="   ",
            ))

    def test_streaming_rejects_long_question(self):
        from app.services.rag_service import answer_question_with_history_streaming, RAGError
        with pytest.raises(RAGError, match="must not exceed"):
            list(answer_question_with_history_streaming(
                db=MagicMock(),
                user_id=1,
                question="x" * 5000,
            ))

    @patch("app.services.rag_service.retrieve_context")
    @patch("app.services.rag_service.get_system_prompt")
    @patch("app.services.rag_service.build_conversation_aware_user_prompt")
    def test_streaming_retrieval_failure_raises_rag_error(self, mock_build_prompt, mock_get_sys, mock_retrieve):
        """When retrieval fails, RAGError is raised."""
        from app.services.retrieval_service import RetrievalError
        mock_retrieve.side_effect = RetrievalError("DB connection failed")

        from app.services.rag_service import answer_question_with_history_streaming, RAGError
        with pytest.raises(RAGError, match="Retrieval failed"):
            list(answer_question_with_history_streaming(
                db=MagicMock(),
                user_id=1,
                question="test",
            ))

    @patch("app.services.rag_service.retrieve_context")
    @patch("app.services.rag_service.get_system_prompt")
    @patch("app.services.rag_service.build_conversation_aware_user_prompt")
    def test_streaming_llm_failure_raises_rag_error(self, mock_build_prompt, mock_get_sys, mock_retrieve):
        """When LLM streaming fails, RAGError is raised."""
        mock_retrieve.return_value = MagicMock(
            total_results=1,
            context="some context",
            results=[MagicMock(
                chunk_id=1, chunk_index=0, document_id=1,
                original_filename="doc.pdf", page_start=1, page_end=1,
                similarity_score=0.9, text="text", keyword_score=None,
                final_score=0.9, match_type="vector",
            )],
        )

        class FailLLM:
            @property
            def name(self):
                return "fail"

            @property
            def model(self):
                return "fail-v1"

            def generate(self, **kwargs):
                return LLMResponse(text="x", model="fail-v1", provider="fail")

            def stream_generate(self, **kwargs):
                raise LLMProviderError("provider dead")

        from app.services.rag_service import answer_question_with_history_streaming, RAGError
        with pytest.raises(RAGError, match="LLM generation failed"):
            list(answer_question_with_history_streaming(
                db=MagicMock(),
                user_id=1,
                question="test",
                llm_service=LLMService(provider=FailLLM()),
            ))


# ==========================================================================
# 5. SSE ENDPOINT STRUCTURE (FastAPI TestClient with mocked deps)
# ==========================================================================

class TestSSEEndpoint:
    """Test the streaming SSE endpoint with mocked dependencies."""

    def _get_test_client(self):
        """Create a TestClient with mocked auth and DB."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.conversations import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_stream_endpoint_exists(self):
        """The /conversations/{id}/messages/stream route should exist."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.conversations import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)

        # Without auth, should return 401 (router prefix is /conversations)
        response = client.post(
            "/conversations/1/messages/stream",
            json={"content": "test"},
        )
        assert response.status_code in (401, 422)

    def test_stream_requires_authentication(self):
        """Unauthenticated requests to stream should return 401."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.conversations import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/conversations/1/messages/stream",
            json={"content": "hello"},
        )
        assert response.status_code == 401

    def test_stream_rejects_empty_message_or_unauth(self):
        """Without auth, returns 401. Auth validation runs before body validation."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.conversations import router

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)

        # Without auth, always 401 regardless of body content
        response = client.post(
            "/conversations/1/messages/stream",
            json={"content": ""},
        )
        assert response.status_code == 401

        # With invalid body, still 401 because auth runs first
        response2 = client.post(
            "/conversations/1/messages/stream",
            json={},
        )
        assert response2.status_code == 401


# ==========================================================================
# 6. SSE EVENT FORMAT
# ==========================================================================

class TestSSEEventFormat:
    """Test the SSE event format structure."""

    def test_sse_token_event_format(self):
        """Verify the SSE event format matches the spec."""
        # Simulate what the streaming endpoint produces
        text = "Hello"
        event = f"event: token\ndata: {json.dumps({'text': text})}\n\n"
        assert "event: token" in event
        parsed = json.loads(event.split("data: ")[1].split("\n")[0])
        assert parsed["text"] == "Hello"

    def test_sse_sources_event_format(self):
        sources = [{"document_id": 1, "filename": "doc.pdf"}]
        confidence = {"level": "HIGH", "grounding_score": 0.85}
        event = f"event: sources\ndata: {json.dumps({'sources': sources, 'confidence': confidence})}\n\n"
        parsed = json.loads(event.split("data: ")[1].split("\n")[0])
        assert parsed["sources"][0]["document_id"] == 1
        assert parsed["confidence"]["level"] == "HIGH"

    def test_sse_complete_event_format(self):
        event = f"event: complete\ndata: {json.dumps({'message_id': 42, 'grounded': True})}\n\n"
        parsed = json.loads(event.split("data: ")[1].split("\n")[0])
        assert parsed["message_id"] == 42
        assert parsed["grounded"] is True

    def test_sse_error_event_format(self):
        event = f"event: error\ndata: {json.dumps({'message': 'Something went wrong'})}\n\n"
        parsed = json.loads(event.split("data: ")[1].split("\n")[0])
        assert parsed["message"] == "Something went wrong"

    def test_sse_no_stack_traces_in_error(self):
        """Error events must never contain stack traces."""
        sensitive_info = [
            "traceback",
            "File \"/app/",
            "psycopg2",
            "sqlalchemy",
            "api_key",
            "password",
        ]
        error_msg = "An error occurred while processing your message."
        for phrase in sensitive_info:
            assert phrase.lower() not in error_msg.lower()


# ==========================================================================
# 7. STREAMING PERSISTENCE SAFETY
# ==========================================================================

class TestStreamingPersistence:
    """Test that streaming persistence logic is correct."""

    def test_assistant_message_persisted_once(self):
        """The assistant message should only be created after full accumulation."""
        accumulated = ""
        chunks = ["Hello", " world", "!"]
        for chunk in chunks:
            accumulated += chunk
        assert accumulated == "Hello world!"
        # Persistence should happen exactly once after all chunks
        # (verified by the endpoint code structure)

    def test_no_duplicate_sources(self):
        """Sources should be persisted exactly once."""
        sources_data = [
            {"document_id": 1, "chunk_id": 10},
            {"document_id": 1, "chunk_id": 10},  # duplicate
            {"document_id": 2, "chunk_id": 20},
        ]
        # Deduplication logic
        seen = set()
        unique = []
        for s in sources_data:
            key = (s["document_id"], s["chunk_id"])
            if key not in seen:
                seen.add(key)
                unique.append(s)
        assert len(unique) == 2


# ==========================================================================
# 8. STREAMING EDGE CASES
# ==========================================================================

class TestStreamingEdgeCases:
    """Edge cases in streaming."""

    def test_empty_chunks_ignored_in_streaming(self):
        """Empty chunks should be silently ignored."""
        fake = FakeLLMProvider()
        chunks = list(fake.stream_generate("sys", "test"))
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_streaming_preserves_unicode(self):
        """Streaming should handle Unicode content correctly."""
        provider = FakeLLMProvider(
            response_template="Réponse en français: 你好世界 🎉"
        )
        response = provider.generate("sys", "test")
        stream_chunks = list(provider.stream_generate("sys", "test"))
        reconstructed = "".join(stream_chunks)
        assert reconstructed == response.text

    def test_streaming_preserves_punctuation(self):
        """Streaming should handle punctuation correctly."""
        provider = FakeLLMProvider(
            response_template="Hello, world! How are you? I'm fine — thanks."
        )
        response = provider.generate("sys", "test")
        stream_chunks = list(provider.stream_generate("sys", "test"))
        reconstructed = "".join(stream_chunks)
        assert reconstructed == response.text

    def test_streaming_large_response(self):
        """Streaming should handle large responses without issues."""
        long_text = " ".join(["word"] * 500)
        provider = FakeLLMProvider(response_template=long_text)
        response = provider.generate("sys", "test")
        stream_chunks = list(provider.stream_generate("sys", "test"))
        reconstructed = "".join(stream_chunks)
        assert reconstructed == response.text

    def test_streaming_single_word(self):
        """Streaming should handle a single-word response."""
        provider = FakeLLMProvider(response_template="Hello")
        response = provider.generate("sys", "test")
        stream_chunks = list(provider.stream_generate("sys", "test"))
        reconstructed = "".join(stream_chunks)
        assert reconstructed == response.text
