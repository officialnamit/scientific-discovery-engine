"""
Phase 5: embedding provider interface.

LocalEmbeddingProvider is REQUIRED to be deterministic (Phase 5
constraint: "deterministic tests", "no uncontrolled randomness"). It
uses the hashing trick (token -> stable hash -> fixed-size vector
index) rather than Python's built-in hash() (which is randomized
per-process via PYTHONHASHSEED unless disabled) -- hashlib.md5 is used
instead so the same text always embeds to the same vector, in any
process, forever.

HONESTY NOTE: this is a bag-of-words hashing embedding, not a learned
semantic embedding model (no sentence-transformers, no network, no
GPU). It captures lexical/keyword overlap, not deep semantic
similarity -- adequate for retrieval over this small, keyword-distinct
fixture corpus (see rag/corpus/), not a claim of general semantic
search quality. APIEmbeddingProvider (real semantic embeddings) is
provided for when that matters more than offline reproducibility.
"""

import hashlib
import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(t) for t in texts]

    @property
    def name(self) -> str:
        return type(self).__name__


class LocalEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline, dependency-free bag-of-words hashing embedding."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _token_index(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16) % self.dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = _TOKEN_RE.findall(text.lower())
        for tok in tokens:
            vec[self._token_index(tok)] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class APIEmbeddingProvider(EmbeddingProvider):
    """Real embedding provider (OpenAI embeddings endpoint). Not used by
    any test; lazy-imports the SDK and reads the key from the
    environment, matching llm/openai_provider.py's pattern."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Use rag.embeddings.LocalEmbeddingProvider "
                    "for offline/test usage."
                )
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, text: str) -> np.ndarray:
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=text)
        return np.array(response.data[0].embedding, dtype=np.float64)
