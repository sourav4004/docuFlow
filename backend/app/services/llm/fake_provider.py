"""Fake LLM provider for testing.

Generates deterministic responses without calling any external API.
Used for unit tests and development without paid LLM credentials.
"""

import logging
import time
from typing import Optional, Iterator

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class FakeLLMProvider(LLMProvider):
    """Deterministic fake LLM provider for testing.

    Returns a predictable response based on the user prompt.
    No network calls. No API keys required.

    Behavior:
    - If user_prompt contains "error" → raises RuntimeError (simulates failure)
    - If user_prompt contains "timeout" → sleeps and raises (simulates timeout)
    - Otherwise → returns a structured response echoing the prompt
    """

    def __init__(
        self,
        model: str = "fake-llm",
        provider_name: str = "fake",
        response_template: Optional[str] = None,
    ):
        self._model = model
        self._provider_name = provider_name
        self._response_template = response_template

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """Generate a fake response.

        Simulates failure if prompt contains special keywords.
        """
        # Simulate provider error
        if "error" in user_prompt.lower():
            raise RuntimeError("Simulated LLM provider error")

        # Simulate timeout
        if "timeout" in user_prompt.lower():
            import time
            time.sleep(0.01)  # Brief delay to simulate latency
            raise TimeoutError("Simulated LLM timeout")

        # Generate deterministic response
        if self._response_template:
            text = self._response_template.format(
                system=system_prompt,
                user=user_prompt,
            )
        else:
            text = (
                f"Based on the provided context, here is the answer to your question: "
                f"'{user_prompt}'. This is a simulated response from the {self._model} model."
            )

        return LLMResponse(
            text=text,
            model=self._model,
            provider=self._provider_name,
            input_tokens=len(system_prompt.split()) + len(user_prompt.split()),
            output_tokens=len(text.split()),
            total_tokens=len(system_prompt.split()) + len(user_prompt.split()) + len(text.split()),
        )

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Iterator[str]:
        """Generate a fake streaming response.

        Yields words from the generated response one at a time
        to simulate token-by-token streaming.
        """
        response = self.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        words = response.text.split()
        for i, word in enumerate(words):
            # Yield space-prefixed words except the first
            if i == 0:
                yield word
            else:
                yield f" {word}"
