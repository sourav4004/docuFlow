"""Abstract LLM provider interface and shared types.

All concrete LLM providers must implement LLMProvider.
The rest of DocuFlow interacts with LLMProvider, not vendor SDKs.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMProviderError(Exception):
    """Base error for LLM provider failures."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM request times out."""


class LLMConfigurationError(LLMProviderError):
    """Raised when LLM provider configuration is invalid."""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM response — vendor-agnostic.

    Attributes:
        text: The generated text content.
        model: Model identifier used for generation.
        provider: Provider name (e.g., "openai_compatible").
        input_tokens: Input token count (optional, provider-dependent).
        output_tokens: Output token count (optional, provider-dependent).
        total_tokens: Total token count (optional, provider-dependent).
    """
    text: str
    model: str
    provider: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement generate(). The provider must be
    stateless and thread-safe.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'openai_compatible', 'fake')."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model identifier being used."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user's question or request.

        Returns:
            Normalized LLMResponse.

        Raises:
            LLMProviderError: If generation fails.
            LLMTimeoutError: If the request times out.
        """
