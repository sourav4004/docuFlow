"""LLM provider abstraction layer.

Provides a provider-independent interface for LLM generation.
The rest of DocuFlow interacts with LLMProvider, not vendor SDKs.

Supported providers (via configuration):
- openai_compatible: Works with OpenAI, Qwen, Kimi, Ollama, vLLM, etc.
- fake: Deterministic responses for testing.
"""

from .base import (
    LLMProvider,
    LLMResponse,
    LLMProviderError,
    LLMTimeoutError,
    LLMConfigurationError,
)
from .service import LLMService

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMConfigurationError",
    "LLMService",
]
