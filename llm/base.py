"""
Phase 4: provider-agnostic LLM interface.

Every provider (mock, Gemini, or anything added later) implements this
one method. Nothing in hypotheses/pipeline.py imports a specific
provider class directly -- it receives an LLMProvider instance chosen
by config/environment (see llm/__init__.py's get_provider()).
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        """Return the raw text completion for `prompt`. Expected (but not
        enforced at this layer) to be a JSON array of hypothesis objects --
        schema validation happens one layer up, in hypotheses/pipeline.py."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__
