"""Phase 4.7 tests: LLM provider abstraction.

All tests use FakeLLMProvider or mocks — no paid API required.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from app.services.llm import (
    LLMProvider,
    LLMResponse,
    LLMProviderError,
    LLMTimeoutError,
    LLMConfigurationError,
    LLMService,
)
from app.services.llm.fake_provider import FakeLLMProvider
from app.services.llm.openai_compatible import OpenAICompatibleProvider
from app.services.llm.service import get_llm_provider


# ==========================================================================
# 1. PROVIDER INTERFACE
# ==========================================================================

class TestLLMProviderInterface:
    def test_fake_provider_implements_interface(self):
        provider = FakeLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_provider_has_name(self):
        provider = FakeLLMProvider()
        assert isinstance(provider.name, str)
        assert len(provider.name) > 0

    def test_provider_has_model(self):
        provider = FakeLLMProvider()
        assert isinstance(provider.model, str)
        assert len(provider.model) > 0

    def test_provider_has_generate(self):
        provider = FakeLLMProvider()
        assert callable(getattr(provider, "generate", None))


# ==========================================================================
# 2. RESPONSE MODEL
# ==========================================================================

class TestLLMResponse:
    def test_response_fields(self):
        resp = LLMResponse(
            text="Hello world",
            model="test-model",
            provider="test-provider",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
        assert resp.text == "Hello world"
        assert resp.model == "test-model"
        assert resp.provider == "test-provider"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.total_tokens == 15

    def test_response_optional_metadata(self):
        resp = LLMResponse(
            text="Hello",
            model="m",
            provider="p",
        )
        assert resp.input_tokens is None
        assert resp.output_tokens is None
        assert resp.total_tokens is None

    def test_response_is_frozen(self):
        resp = LLMResponse(text="x", model="m", provider="p")
        with pytest.raises(AttributeError):
            resp.text = "changed"


# ==========================================================================
# 3. FAKE PROVIDER
# ==========================================================================

class TestFakeLLMProvider:
    def test_basic_generation(self):
        provider = FakeLLMProvider()
        resp = provider.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 2+2?",
        )
        assert isinstance(resp, LLMResponse)
        assert "2+2" in resp.text
        assert resp.model == "fake-llm"
        assert resp.provider == "fake"

    def test_usage_metadata_populated(self):
        provider = FakeLLMProvider()
        resp = provider.generate(
            system_prompt="System instructions",
            user_prompt="User question",
        )
        assert resp.input_tokens is not None
        assert resp.output_tokens is not None
        assert resp.total_tokens is not None
        assert resp.total_tokens == resp.input_tokens + resp.output_tokens

    def test_error_simulation(self):
        provider = FakeLLMProvider()
        with pytest.raises(RuntimeError, match="Simulated LLM provider error"):
            provider.generate(
                system_prompt="System",
                user_prompt="This will cause an error",
            )

    def test_timeout_simulation(self):
        provider = FakeLLMProvider()
        with pytest.raises(TimeoutError, match="Simulated LLM timeout"):
            provider.generate(
                system_prompt="System",
                user_prompt="This will cause a timeout",
            )

    def test_custom_response_template(self):
        provider = FakeLLMProvider(
            response_template="Custom: {user}",
        )
        resp = provider.generate(
            system_prompt="sys",
            user_prompt="my question",
        )
        assert resp.text == "Custom: my question"

    def test_deterministic_output(self):
        provider = FakeLLMProvider()
        r1 = provider.generate("sys", "question")
        r2 = provider.generate("sys", "question")
        assert r1.text == r2.text
        assert r1.model == r2.model

    def test_custom_model_name(self):
        provider = FakeLLMProvider(model="custom-model")
        resp = provider.generate("sys", "q")
        assert resp.model == "custom-model"

    def test_custom_provider_name(self):
        provider = FakeLLMProvider(provider_name="custom-provider")
        resp = provider.generate("sys", "q")
        assert resp.provider == "custom-provider"


# ==========================================================================
# 4. OPENAI-COMPATIBLE PROVIDER
# ==========================================================================

class TestOpenAICompatibleProvider:
    def test_empty_api_key_raises(self):
        with pytest.raises(LLMConfigurationError, match="api_key must not be empty"):
            OpenAICompatibleProvider(api_key="")

    def test_default_values(self):
        provider = OpenAICompatibleProvider(api_key="test-key")
        assert provider.name == "openai_compatible"
        assert provider.model == "gpt-4o"

    def test_custom_model(self):
        provider = OpenAICompatibleProvider(api_key="key", model="qwen-plus")
        assert provider.model == "qwen-plus"

    def test_custom_base_url(self):
        provider = OpenAICompatibleProvider(
            api_key="key", base_url="http://localhost:11434/v1"
        )
        assert provider._base_url == "http://localhost:11434/v1"

    def test_base_url_trailing_slash_stripped(self):
        provider = OpenAICompatibleProvider(
            api_key="key", base_url="http://localhost:11434/v1/"
        )
        assert provider._base_url == "http://localhost:11434/v1"

    @patch("app.services.llm.openai_compatible.httpx.Client")
    def test_successful_generation(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "The answer is 42."}}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        provider = OpenAICompatibleProvider(api_key="test-key")
        resp = provider.generate("System", "What is the answer?")

        assert resp.text == "The answer is 42."
        assert resp.model == "gpt-4o"
        assert resp.provider == "openai_compatible"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.total_tokens == 15

    @patch("app.services.llm.openai_compatible.httpx.Client")
    def test_http_401_error(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        provider = OpenAICompatibleProvider(api_key="bad-key")
        with pytest.raises(LLMProviderError, match="status 401"):
            provider.generate("System", "Question")

    @patch("app.services.llm.openai_compatible.httpx.Client")
    def test_http_500_error(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        provider = OpenAICompatibleProvider(api_key="key")
        with pytest.raises(LLMProviderError, match="status 500"):
            provider.generate("System", "Question")

    @patch("app.services.llm.openai_compatible.httpx.Client")
    def test_timeout_error(self, mock_client_cls):
        import httpx as _httpx
        mock_client = MagicMock()
        mock_client.post.side_effect = _httpx.TimeoutException("timed out")
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        provider = OpenAICompatibleProvider(api_key="key")
        with pytest.raises(LLMTimeoutError, match="timed out"):
            provider.generate("System", "Question")

    @patch("app.services.llm.openai_compatible.httpx.Client")
    def test_connection_error(self, mock_client_cls):
        import httpx as _httpx
        mock_client = MagicMock()
        mock_client.post.side_effect = _httpx.ConnectError("Connection refused")
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        provider = OpenAICompatibleProvider(api_key="key")
        with pytest.raises(LLMProviderError, match="failed"):
            provider.generate("System", "Question")

    @patch("app.services.llm.openai_compatible.httpx.Client")
    def test_invalid_json_response(self, mock_client_cls):
        import json as _json
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = _json.JSONDecodeError("Expecting value", "", 0)
        mock_response.text = "not json"
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        provider = OpenAICompatibleProvider(api_key="key")
        with pytest.raises(LLMProviderError, match="not valid JSON"):
            provider.generate("System", "Question")

    @patch("app.services.llm.openai_compatible.httpx.Client")
    def test_empty_choices(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        provider = OpenAICompatibleProvider(api_key="key")
        with pytest.raises(LLMProviderError, match="no choices"):
            provider.generate("System", "Question")

    @patch("app.services.llm.openai_compatible.httpx.Client")
    def test_empty_response_text(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        provider = OpenAICompatibleProvider(api_key="key")
        with pytest.raises(LLMProviderError, match="empty"):
            provider.generate("System", "Question")


# ==========================================================================
# 5. PROVIDER FACTORY
# ==========================================================================

class TestProviderFactory:
    def test_fake_provider_from_config(self):
        with patch("app.services.llm.service.settings") as mock_settings:
            mock_settings.llm_provider = "fake"
            mock_settings.llm_model = "fake-llm"
            provider = get_llm_provider()
            assert isinstance(provider, FakeLLMProvider)

    def test_openai_compatible_from_config(self):
        with patch("app.services.llm.service.settings") as mock_settings:
            mock_settings.llm_provider = "openai_compatible"
            mock_settings.llm_model = "gpt-4o"
            mock_settings.llm_api_key = "test-key"
            mock_settings.llm_base_url = "https://api.openai.com/v1"
            mock_settings.llm_timeout = 30.0
            provider = get_llm_provider()
            assert isinstance(provider, OpenAICompatibleProvider)

    def test_openai_compatible_no_api_key_raises(self):
        with patch("app.services.llm.service.settings") as mock_settings:
            mock_settings.llm_provider = "openai_compatible"
            mock_settings.llm_api_key = ""
            with pytest.raises(LLMConfigurationError, match="llm_api_key"):
                get_llm_provider()

    def test_invalid_provider_raises(self):
        with patch("app.services.llm.service.settings") as mock_settings:
            mock_settings.llm_provider = "nonexistent_provider"
            with pytest.raises(LLMConfigurationError, match="Unknown LLM provider"):
                get_llm_provider()

    def test_factory_is_case_insensitive(self):
        with patch("app.services.llm.service.settings") as mock_settings:
            mock_settings.llm_provider = "FAKE"
            mock_settings.llm_model = "fake-llm"
            provider = get_llm_provider()
            assert isinstance(provider, FakeLLMProvider)


# ==========================================================================
# 6. LLM SERVICE
# ==========================================================================

class TestLLMService:
    def test_generate_success(self):
        provider = FakeLLMProvider()
        service = LLMService(provider=provider)
        resp = service.generate(
            system_prompt="You are helpful.",
            user_prompt="What is 2+2?",
        )
        assert isinstance(resp, LLMResponse)
        assert resp.text

    def test_generate_with_empty_user_prompt(self):
        provider = FakeLLMProvider()
        service = LLMService(provider=provider)
        with pytest.raises(LLMProviderError, match="must not be empty"):
            service.generate(system_prompt="sys", user_prompt="")

    def test_generate_with_whitespace_user_prompt(self):
        provider = FakeLLMProvider()
        service = LLMService(provider=provider)
        with pytest.raises(LLMProviderError, match="must not be empty"):
            service.generate(system_prompt="sys", user_prompt="   ")

    def test_generate_with_too_long_system_prompt(self):
        provider = FakeLLMProvider()
        service = LLMService(provider=provider)
        with pytest.raises(LLMProviderError, match="must not exceed"):
            service.generate(
                system_prompt="x" * 100_001,
                user_prompt="question",
            )

    def test_generate_with_too_long_user_prompt(self):
        provider = FakeLLMProvider()
        service = LLMService(provider=provider)
        with pytest.raises(LLMProviderError, match="must not exceed"):
            service.generate(
                system_prompt="sys",
                user_prompt="x" * 500_001,
            )

    def test_generate_with_non_string_system_prompt(self):
        provider = FakeLLMProvider()
        service = LLMService(provider=provider)
        with pytest.raises(LLMProviderError, match="must be a string"):
            service.generate(system_prompt=123, user_prompt="q")

    def test_generate_with_non_string_user_prompt(self):
        provider = FakeLLMProvider()
        service = LLMService(provider=provider)
        with pytest.raises(LLMProviderError, match="must be a string"):
            service.generate(system_prompt="sys", user_prompt=42)

    def test_unicode_prompts(self):
        provider = FakeLLMProvider()
        service = LLMService(provider=provider)
        resp = service.generate(
            system_prompt="你是一个有帮助的助手",
            user_prompt="什么是远程工作政策？",
        )
        assert "远程工作" in resp.text

    def test_provider_error_wrapped(self):
        """RuntimeError from provider is wrapped in LLMProviderError."""
        provider = FakeLLMProvider()
        service = LLMService(provider=provider)
        with pytest.raises(LLMProviderError, match="LLM provider failed.*RuntimeError"):
            service.generate(
                system_prompt="sys",
                user_prompt="This will cause an error",
            )

    def test_unexpected_exception_wrapped(self):
        bad_provider = MagicMock(spec=LLMProvider)
        bad_provider.name = "bad"
        bad_provider.model = "bad"
        bad_provider.generate.side_effect = ValueError("unexpected")

        service = LLMService(provider=bad_provider)
        with pytest.raises(LLMProviderError, match="LLM provider failed"):
            service.generate(system_prompt="sys", user_prompt="question")


# ==========================================================================
# 7. PROVIDER SWITCHING
# ==========================================================================

class TestProviderSwitching:
    def test_switch_between_providers(self):
        """Prove the architecture is genuinely provider-independent."""
        provider_a = FakeLLMProvider(model="model-a", provider_name="provider-a")
        provider_b = FakeLLMProvider(model="model-b", provider_name="provider-b")

        service_a = LLMService(provider=provider_a)
        service_b = LLMService(provider=provider_b)

        resp_a = service_a.generate("sys", "question")
        resp_b = service_b.generate("sys", "question")

        assert resp_a.model == "model-a"
        assert resp_a.provider == "provider-a"
        assert resp_b.model == "model-b"
        assert resp_b.provider == "provider-b"

        # Both produce valid responses
        assert resp_a.text
        assert resp_b.text

    def test_service_uses_factory_when_no_provider(self):
        with patch("app.services.llm.service.settings") as mock_settings:
            mock_settings.llm_provider = "fake"
            mock_settings.llm_model = "factory-model"
            service = LLMService()
            resp = service.generate("sys", "question")
            assert resp.model == "factory-model"


# ==========================================================================
# 8. EXCEPTION HIERARCHY
# ==========================================================================

class TestExceptionHierarchy:
    def test_timeout_is_provider_error(self):
        assert issubclass(LLMTimeoutError, LLMProviderError)

    def test_config_is_provider_error(self):
        assert issubclass(LLMConfigurationError, LLMProviderError)

    def test_provider_error_is_exception(self):
        assert issubclass(LLMProviderError, Exception)

    def test_can_catch_timeout_as_provider_error(self):
        with pytest.raises(LLMProviderError):
            raise LLMTimeoutError("timeout")

    def test_can_catch_config_as_provider_error(self):
        with pytest.raises(LLMProviderError):
            raise LLMConfigurationError("config error")
