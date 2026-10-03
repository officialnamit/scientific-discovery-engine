"""
Real LLM provider (Google Gemini, via the google-genai SDK).

Configuration is entirely via environment variables / config.yaml --
no API key is ever hardcoded:

    GEMINI_API_KEY   required to actually call generate()
    GEMINI_MODEL     optional; used when config.yaml's llm.model is null
                     (or DEFAULT_MODEL below)

Importing this module never requires a key (so it stays importable in
offline/test environments); the key is only checked inside generate(),
with a clear error if missing.
"""

import json
import os
import re
import time
from typing import Optional

from llm.base import LLMProvider

# gemini-1.5-* models are retired; see https://ai.google.dev/gemini-api/docs/models
DEFAULT_MODEL = "gemini-3.8-flash"
RETRYABLE_CODES = (429, 500, 503)
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 5  # 5, 10, 20, 40, 80 s


def _repair_json_escapes(text: Optional[str]) -> Optional[str]:
    """Even in JSON mode Gemini occasionally emits LaTeX ("\\partial") inside strings,
    which is an invalid JSON escape. Only if the text fails to parse, double every
    backslash that does not start a legal JSON escape; valid JSON is returned unchanged."""
    if not text:
        return text
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


class GeminiProvider(LLMProvider):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None  # lazy: don't import/construct the SDK client until actually used

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Set it in your environment "
                    "(or pass api_key=... explicitly) to use GeminiProvider. "
                    "For offline/tests, use llm.mock_provider.MockLLMProvider instead."
                )
            from google import genai  # imported lazily so google-genai need not even be installed
            # for pure mock-provider usage, matching "no API key needed for tests".
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        client = self._get_client()
        from google.genai import errors, types

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system or None,
                        temperature=temperature,
                        # every caller expects a JSON array (llm/base.py); without JSON mode Gemini
                        # sometimes writes LaTeX like "\partial" in strings -> invalid JSON escapes
                        response_mime_type="application/json",
                    ),
                )
                return _repair_json_escapes(response.text)
            except errors.APIError as e:
                # 429/500/503 ("model is currently experiencing high demand") are transient
                # server-side conditions; anything else (bad key, bad model name) fails at once.
                if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_BASE_SECONDS * 2 ** attempt)
