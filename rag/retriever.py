"""
Phase 5: scientific retriever.

Returns Evidence objects (rag/evidence.py), not raw text -- per the
Phase 5 spec. Filtering (domain/section/metadata) is applied AFTER
vector search by over-fetching candidates, so filters never change
the underlying similarity ranking, only which of the ranked results
are kept.

A high similarity score here means "this chunk is lexically close to
the query under a bag-of-words hashing embedding" -- it is explicitly
NOT a claim of semantic correctness (see rag/embeddings.py and the
Evidence.confidence field docs).
"""

from typing import Dict, List, Optional

from rag.documents import Document
from rag.embeddings import EmbeddingProvider
from rag.evidence import Evidence, RetrievalResult, evidence_from_retrieval
from rag.vector_store import VectorStore


class Retriever:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore,
                 documents: Dict[str, Document]):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.documents = documents

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        section: Optional[str] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Evidence]:
        if not query or not query.strip():
            return []

        query_vec = self.embedding_provider.embed(query)
        # over-fetch so post-hoc filtering still leaves room for top_k results
        overfetch = max(top_k * 10, 50)
        raw_hits = self.vector_store.search(query_vec, top_k=overfetch)

        results: List[RetrievalResult] = []
        for item_id, score, payload in raw_hits:
            if domain is not None and payload.get("domain") != domain:
                continue
            if section is not None and payload.get("section") != section:
                continue
            if metadata_filter:
                chunk_meta = payload.get("metadata", {})
                if not all(chunk_meta.get(k) == v for k, v in metadata_filter.items()):
                    continue
            results.append(RetrievalResult(
                chunk_id=payload["chunk_id"], document_id=payload["document_id"],
                text=payload["text"], score=score, source=payload["source"],
                section=payload.get("section"), metadata=payload.get("metadata", {}),
            ))
            if len(results) >= top_k:
                break

        return [evidence_from_retrieval(r, self.documents[r.document_id]) for r in results]
