"""
Phase 5: RAG pipeline orchestration.

    Document(s) -> chunk_document() -> embed each chunk -> index in VectorStore
                                                                    |
                                                              Retriever(query)
"""

from typing import Dict, List, Optional

from rag.chunker import chunk_document
from rag.documents import Chunk, Document
from rag.embeddings import EmbeddingProvider, LocalEmbeddingProvider
from rag.retriever import Retriever
from rag.vector_store import InMemoryVectorStore, VectorStore


class RAGPipeline:
    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None,
                 vector_store: Optional[VectorStore] = None):
        self.embedding_provider = embedding_provider or LocalEmbeddingProvider()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}

    def ingest(self, documents: List[Document]) -> List[Chunk]:
        """Chunk + embed + index every document. Returns all chunks created
        (useful for inspection/tests)."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            self.documents[doc.metadata.document_id] = doc
            doc_chunks = chunk_document(doc)
            for chunk in doc_chunks:
                self.chunks[chunk.chunk_id] = chunk
                vector = self.embedding_provider.embed(chunk.text)
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "section": chunk.section,
                    "source": doc.metadata.source,
                    "domain": doc.metadata.domain,
                    "metadata": chunk.metadata,
                }
                self.vector_store.add(chunk.chunk_id, vector, payload)
            all_chunks.extend(doc_chunks)
        return all_chunks

    def get_retriever(self) -> Retriever:
        return Retriever(self.embedding_provider, self.vector_store, self.documents)
