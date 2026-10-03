"""
Phase 4: provider factory.

hypotheses/pipeline.py (and experiments/06) call get_provider(cfg)
rather than importing a specific provider class, so swapping providers
is a config/environment change, never a code change.
"""

import os
import warnings

from llm.base import LLMProvider
from llm.mock_provider import MockLLMProvider


def get_provider(llm_cfg: dict) -> LLMProvider:
    provider_name = (llm_cfg or {}).get("provider", "gemini").lower()

    if provider_name == "mock":
        return MockLLMProvider()

    if provider_name == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            warnings.warn(
                "llm.provider='gemini' but GEMINI_API_KEY is not set in the environment; "
                "falling back to MockLLMProvider. Set GEMINI_API_KEY to use a real LLM."
            )
            return MockLLMProvider()
        from llm.gemini_provider import GeminiProvider
        return GeminiProvider(model=llm_cfg.get("model"))

    raise ValueError(f"Unknown llm.provider '{provider_name}' (expected 'gemini' or 'mock')")


__all__ = ["LLMProvider", "MockLLMProvider", "get_provider"]
