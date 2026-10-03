"""
Phase 4: real LLM provider (OpenAI-compatible chat completions).

Configuration is entirely via environment variables / config.yaml --
no API key is ever hardcoded:

    OPENAI_API_KEY   required to actually call generate()
    OPENAI_MODEL     optional, defaults to config.yaml's llm.model

Importing this module never requires a key (so it stays importable in
offline/test environments); the key is only checked inside generate(),
with a clear error if missing.
"""

import os
from typing import Optional

from llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None  # lazy: don't import/construct the SDK client until actually used

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Set it in your environment "
                    "(or pass api_key=... explicitly) to use OpenAIProvider. "
                    "For offline/tests, use llm.mock_provider.MockLLMProvider instead."
                )
            import openai  # imported lazily so the package need not even be installed
            # for pure mock-provider usage, matching "no API key needed for tests".
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
