"""LLM service layer.

Provides a clean interface above LLM providers:
- Provider factory (selects provider from configuration)
- Prompt validation
- Error normalization
- Structured logging

Architecture:

API / RAG service
    ↓
LLMService
    ↓
LLMProvider
    ↓
Concrete provider (OpenAI-compatible, fake, etc.)
"""

import logging
from typing import Optional

from ...core.config import settings
from .base import (
    LLMProvider,
    LLMResponse,
    LLMProviderError,
    LLMTimeoutError,
    LLMConfigurationError,
)

logger = logging.getLogger(__name__)

# Prompt length limits
MAX_SYSTEM_PROMPT_LENGTH = 10_000
MAX_USER_PROMPT_LENGTH = 50_000


def get_llm_provider() -> LLMProvider:
    """Factory: return the configured LLM provider.

    Uses settings.llm_provider to select the concrete implementation.
    Defaults to FakeLLMProvider for development/testing.

    Returns:
        Configured LLMProvider instance.

    Raises:
        LLMConfigurationError: If provider name is invalid or misconfigured.
    """
    provider_name = settings.llm_provider.lower()

    if provider_name == "fake":
        from .fake_provider import FakeLLMProvider
        return FakeLLMProvider(
            model=settings.llm_model,
            provider_name="fake",
        )
    elif provider_name == "openai_compatible":
        from .openai_compatible import OpenAICompatibleProvider
        if not settings.llm_api_key:
            raise LLMConfigurationError(
                "llm_api_key must be set when using openai_compatible provider"
            )
        return OpenAICompatibleProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
        )
    else:
        raise LLMConfigurationError(
            f"Unknown LLM provider: {provider_name!r}. "
            f"Supported providers: 'fake', 'openai_compatible'"
        )


class LLMService:
    """Service layer for LLM generation.

    Wraps LLMProvider with validation, error normalization, and logging.
    Does not know provider-specific SDK details.
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        """Initialize the service.

        Args:
            provider: Optional provider override (for testing).
                Uses config-based factory if None.
        """
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        """Return the active provider, creating it if needed."""
        if self._provider is None:
            self._provider = get_llm_provider()
        return self._provider

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Generate an LLM response with validation and error handling.

        Args:
            system_prompt: System-level instructions.
            user_prompt: The user's question or request.

        Returns:
            Normalized LLMResponse.

        Raises:
            LLMProviderError: If generation fails.
            LLMTimeoutError: If the request times out.
            LLMConfigurationError: If configuration is invalid.
        """
        # Validate prompts
        self._validate_prompt(system_prompt, "system_prompt", MAX_SYSTEM_PROMPT_LENGTH)
        self._validate_prompt(user_prompt, "user_prompt", MAX_USER_PROMPT_LENGTH)

        if not user_prompt.strip():
            raise LLMProviderError("user_prompt must not be empty")

        logger.info(
            "LLM generation request (provider=%s, model=%s, "
            "system_len=%d, user_len=%d)",
            self.provider.name, self.provider.model,
            len(system_prompt), len(user_prompt),
        )

        try:
            response = self.provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except LLMProviderError:
            raise
        except LLMTimeoutError:
            raise
        except Exception as exc:
            logger.error(
                "LLM provider raised unexpected error: %s", type(exc).__name__,
            )
            raise LLMProviderError(
                f"LLM provider failed: {type(exc).__name__}: {exc}"
            ) from exc

        logger.info(
            "LLM generation completed (provider=%s, model=%s, "
            "response_len=%d)",
            response.provider, response.model, len(response.text),
        )

        return response

    def _validate_prompt(
        self, value: str, field_name: str, max_length: int
    ) -> None:
        """Validate a prompt field.

        Raises:
            LLMProviderError: If validation fails.
        """
        if not isinstance(value, str):
            raise LLMProviderError(
                f"{field_name} must be a string, got {type(value).__name__}"
            )
        if len(value) > max_length:
            raise LLMProviderError(
                f"{field_name} must not exceed {max_length} characters "
                f"(got {len(value)})"
            )
