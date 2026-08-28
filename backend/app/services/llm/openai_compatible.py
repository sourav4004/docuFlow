"""OpenAI-compatible LLM provider.

Works with any API that implements the OpenAI chat completions format:
- OpenAI (GPT-4, GPT-3.5)
- Qwen (DashScope OpenAI-compatible endpoint)
- Kimi (Moonshot OpenAI-compatible endpoint)
- Ollama (local, OpenAI-compatible)
- vLLM (local, OpenAI-compatible)
- Any other OpenAI-compatible server

Uses httpx (already in project requirements) — no new SDK dependencies.
"""

import json
import logging
import time
from typing import Optional

import httpx

from .base import (
    LLMProvider,
    LLMResponse,
    LLMProviderError,
    LLMTimeoutError,
    LLMConfigurationError,
)

logger = logging.getLogger(__name__)

# Default timeout for LLM requests (seconds)
DEFAULT_TIMEOUT = 60.0

# Default base URL for OpenAI API
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider that communicates via OpenAI-compatible chat completions API.

    Configuration:
        api_key: API key for authentication.
        model: Model identifier (e.g., "gpt-4o", "qwen-plus").
        base_url: API base URL. Defaults to OpenAI.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key:
            raise LLMConfigurationError("api_key must not be empty")

        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "openai_compatible"

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Generate a response via OpenAI-compatible chat completions API.

        Args:
            system_prompt: System instructions.
            user_prompt: User's question.

        Returns:
            LLMResponse with generated text and metadata.

        Raises:
            LLMTimeoutError: If request times out.
            LLMProviderError: If request fails or response is invalid.
        """
        url = f"{self._base_url}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self._model,
            "messages": messages,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.monotonic()

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            elapsed = time.monotonic() - start_time
            logger.error(
                "LLM request timed out after %.1fs (provider=%s, model=%s)",
                elapsed, self.name, self._model,
            )
            raise LLMTimeoutError(
                f"LLM request timed out after {elapsed:.1f}s"
            ) from exc
        except httpx.RequestError as exc:
            elapsed = time.monotonic() - start_time
            logger.error(
                "LLM request failed (provider=%s, model=%s): %s",
                self.name, self._model, type(exc).__name__,
            )
            raise LLMProviderError(
                f"LLM request failed: {type(exc).__name__}: {exc}"
            ) from exc

        elapsed = time.monotonic() - start_time

        # Handle HTTP errors
        if response.status_code >= 400:
            error_body = response.text[:500]  # Truncate for logging
            logger.error(
                "LLM API error %d (provider=%s, model=%s, elapsed=%.1fs): %s",
                response.status_code, self.name, self._model, elapsed, error_body,
            )
            raise LLMProviderError(
                f"LLM API returned status {response.status_code}"
            )

        # Parse response
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            logger.error(
                "LLM response is not valid JSON (provider=%s, model=%s)",
                self.name, self._model,
            )
            raise LLMProviderError(
                f"LLM response is not valid JSON"
            ) from exc

        # Extract text content
        try:
            choices = data["choices"]
            if not choices:
                raise LLMProviderError("LLM response contains no choices")
            text = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(
                "LLM response has unexpected structure (provider=%s, model=%s)",
                self.name, self._model,
            )
            raise LLMProviderError(
                f"LLM response has unexpected structure: {exc}"
            ) from exc

        if not text:
            raise LLMProviderError("LLM response text is empty")

        # Extract usage metadata (optional)
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        logger.info(
            "LLM generation completed (provider=%s, model=%s, elapsed=%.1fs, "
            "input_tokens=%s, output_tokens=%s)",
            self.name, self._model, elapsed,
            input_tokens or "?", output_tokens or "?",
        )

        return LLMResponse(
            text=text,
            model=self._model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
