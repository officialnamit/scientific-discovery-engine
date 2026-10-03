"""
Phase 5: vector store interface.

InMemoryVectorStore is the default -- brute-force cosine similarity,
which is exact (no approximation error to reason about) and entirely
sufficient for a corpus of this size (a handful of fixture documents).
Deterministic tie-breaking (sort by id when scores are equal) is used
throughout so retrieval order never depends on dict/set iteration
order or process-level randomness.

PersistentVectorStore adds JSON save/load on top of the same in-memory
search, satisfying "replaceable interface" / "persistent" without
pulling in a real vector database dependency (consistent with the
Phase 5 priority: "minimal dependencies").
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, item_id: str, vector: np.ndarray, payload: Dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Returns [(item_id, score, payload), ...] sorted by score descending,
        ties broken by item_id ascending for determinism."""
        raise NotImplementedError

    @abstractmethod
    def get(self, item_id: str) -> Optional[Dict]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._vectors: Dict[str, np.ndarray] = {}
        self._payloads: Dict[str, Dict] = {}

    def add(self, item_id: str, vector: np.ndarray, payload: Dict) -> None:
        self._vectors[item_id] = np.asarray(vector, dtype=np.float64)
        self._payloads[item_id] = payload

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        q = np.asarray(query_vector, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        scored = []
        for item_id, vec in self._vectors.items():
            v_norm = np.linalg.norm(vec)
            if q_norm == 0 or v_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(q, vec) / (q_norm * v_norm))
            scored.append((item_id, score))
        # sort by score desc, then item_id asc for deterministic tie-breaking
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return [(item_id, score, self._payloads[item_id]) for item_id, score in scored[:top_k]]

    def get(self, item_id: str) -> Optional[Dict]:
        return self._payloads.get(item_id)

    def all_ids(self) -> List[str]:
        return sorted(self._vectors.keys())


class PersistentVectorStore(InMemoryVectorStore):
    """Same brute-force search as InMemoryVectorStore, plus JSON
    save/load so an index can be rebuilt once and reused across runs."""

    def save(self, path: str) -> None:
        data = {
            item_id: {"vector": self._vectors[item_id].tolist(), "payload": self._payloads[item_id]}
            for item_id in self._vectors
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "PersistentVectorStore":
        store = cls()
        with open(path) as f:
            data = json.load(f)
        for item_id, entry in data.items():
            store.add(item_id, np.array(entry["vector"], dtype=np.float64), entry["payload"])
        return store
