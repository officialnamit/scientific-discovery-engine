"""
Phase 5 RAG tests. All offline/deterministic -- no network, no API key
(LocalEmbeddingProvider only).

Run:
    python tests/test_rag.py
or:
    python -m pytest tests/test_rag.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunker import chunk_document
from rag.corpus_loader import load_corpus
from rag.documents import Document, DocumentMetadata
from rag.embeddings import LocalEmbeddingProvider
from rag.evidence import evidence_from_retrieval
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.vector_store import InMemoryVectorStore, PersistentVectorStore


def _toy_document(doc_id="toy_doc", with_headers=True) -> Document:
    if with_headers:
        text = "# Title\nIntro text.\n## Section A\nContent A about diffusion.\n## Section B\nContent B about advection."
    else:
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    return Document(
        metadata=DocumentMetadata(document_id=doc_id, title="Toy", source="benchmark fixture", domain="toy"),
        text=text, declared_mechanism="toy_mechanism", declared_variables=["u", "x"],
        declared_parameters=["k"], declared_equations=["u_t = k*u"], declared_assumptions=["toy assumption"],
    )


# 1. Document ingestion
def test_corpus_ingestion():
    docs = load_corpus()
    assert len(docs) == 5
    ids = {d.metadata.document_id for d in docs}
    assert ids == {"doc_diffusion", "doc_advection", "doc_reaction", "doc_reaction_diffusion", "doc_advection_diffusion"}
    print(f"[PASS] test_corpus_ingestion: {len(docs)} documents")


# 2. Metadata preservation
def test_metadata_preserved():
    docs = load_corpus()
    diffusion_doc = next(d for d in docs if d.metadata.document_id == "doc_diffusion")
    assert diffusion_doc.metadata.source == "benchmark fixture"
    assert diffusion_doc.metadata.is_benchmark_fixture is True
    assert diffusion_doc.metadata.domain == "diffusion"
    assert diffusion_doc.declared_mechanism == "diffusion"
    assert "D" in diffusion_doc.declared_parameters
    assert diffusion_doc.metadata.year is None  # honestly unknown, not fabricated
    print("[PASS] test_metadata_preserved")


# 3. Chunking (header-based + fallback)
def test_chunking_preserves_sections():
    doc = _toy_document(with_headers=True)
    chunks = chunk_document(doc)
    sections = [c.section for c in chunks]
    assert "Section A" in sections and "Section B" in sections
    assert all(c.document_id == "toy_doc" for c in chunks)
    assert len(set(c.chunk_id for c in chunks)) == len(chunks)  # stable, unique IDs
    print(f"[PASS] test_chunking_preserves_sections: {len(chunks)} chunks, sections={sections}")


def test_chunking_fallback_no_headers():
    doc = _toy_document(with_headers=False)
    chunks = chunk_document(doc)
    assert len(chunks) == 3  # one per paragraph
    print("[PASS] test_chunking_fallback_no_headers")


# 8. Offline embedding provider + determinism
def test_local_embedding_deterministic():
    provider = LocalEmbeddingProvider(dim=64)
    v1 = provider.embed("diffusion spreads out over time")
    v2 = provider.embed("diffusion spreads out over time")
    assert np.allclose(v1, v2), "LocalEmbeddingProvider is not deterministic"
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-9 or np.linalg.norm(v1) == 0.0
    v3 = provider.embed("completely unrelated text about rockets")
    assert not np.allclose(v1, v3)
    print("[PASS] test_local_embedding_deterministic")


# Retrieval: deterministic, top-k, filtering, provenance
def _build_test_pipeline():
    docs = load_corpus()
    pipeline = RAGPipeline()
    pipeline.ingest(docs)
    return pipeline


def test_deterministic_retrieval():
    pipeline = _build_test_pipeline()
    retriever = pipeline.get_retriever()
    query = "the field decreases in amplitude and smooths out spatially"
    r1 = retriever.retrieve(query, top_k=3)
    r2 = retriever.retrieve(query, top_k=3)
    assert [e.evidence_id for e in r1] == [e.evidence_id for e in r2]
    print("[PASS] test_deterministic_retrieval")


def test_top_k_behavior():
    pipeline = _build_test_pipeline()
    retriever = pipeline.get_retriever()
    results = retriever.retrieve("diffusion advection reaction transport", top_k=3)
    assert len(results) == 3
    results_more = retriever.retrieve("diffusion advection reaction transport", top_k=10)
    assert len(results_more) <= 10 and len(results_more) >= len(results)
    print("[PASS] test_top_k_behavior")


def test_domain_filtering():
    pipeline = _build_test_pipeline()
    retriever = pipeline.get_retriever()
    results = retriever.retrieve("transport of a quantity over time", top_k=10, domain="advection")
    assert len(results) > 0
    assert all(r.document_id == "doc_advection" for r in results)
    print(f"[PASS] test_domain_filtering: {len(results)} results all from doc_advection")


def test_section_filtering():
    pipeline = _build_test_pipeline()
    retriever = pipeline.get_retriever()
    results = retriever.retrieve("equation", top_k=10, section="Governing equation")
    assert len(results) > 0
    assert all(r.section == "Governing equation" for r in results)
    print(f"[PASS] test_section_filtering: {len(results)} results")


def test_provenance_preserved():
    pipeline = _build_test_pipeline()
    retriever = pipeline.get_retriever()
    results = retriever.retrieve("diffusion coefficient spreading", top_k=1)
    assert len(results) == 1
    e = results[0]
    assert e.source == "benchmark fixture"
    assert e.document_id in pipeline.documents
    assert e.chunk_id in pipeline.chunks
    assert e.provenance_type == "documented"
    print("[PASS] test_provenance_preserved")


def test_empty_query_returns_empty():
    pipeline = _build_test_pipeline()
    retriever = pipeline.get_retriever()
    assert retriever.retrieve("", top_k=5) == []
    assert retriever.retrieve("   ", top_k=5) == []
    print("[PASS] test_empty_query_returns_empty")


def test_empty_index_returns_empty():
    retriever = Retriever(LocalEmbeddingProvider(), InMemoryVectorStore(), {})
    assert retriever.retrieve("anything", top_k=5) == []
    print("[PASS] test_empty_index_returns_empty")


# Vector store: persistence round-trip
def test_persistent_vector_store_roundtrip(tmp_path_str="/tmp/phase5_vecstore_test.json"):
    provider = LocalEmbeddingProvider(dim=32)
    store = PersistentVectorStore()
    store.add("a", provider.embed("diffusion"), {"chunk_id": "a", "document_id": "d", "text": "diffusion",
                                                   "source": "x", "section": None, "metadata": {}})
    store.save(tmp_path_str)
    loaded = PersistentVectorStore.load(tmp_path_str)
    results = loaded.search(provider.embed("diffusion"), top_k=1)
    assert results[0][0] == "a"
    os.remove(tmp_path_str)
    print("[PASS] test_persistent_vector_store_roundtrip")


# Integration: document -> chunk -> retrieve -> graph -> scientific context -> Phase 4 -> Phase 3 adapter
def test_integration_full_chain_to_phase3_adapter():
    from hypotheses.adapter import hypothesis_to_pinn_config
    from hypotheses.pipeline import generate_hypotheses
    from knowledge_graph.builder import build_graph_from_documents
    from llm.mock_provider import MockLLMProvider
    from rag.context import build_scientific_context

    docs = load_corpus()
    pipeline = RAGPipeline()
    chunks = pipeline.ingest(docs)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    graph = build_graph_from_documents(docs, chunks_by_id)
    retriever = pipeline.get_retriever()

    problem = "A field decreases in amplitude over time and its spatial profile smooths out, with no bulk translation."
    context = build_scientific_context(problem, retriever, graph, top_k=5)
    assert len(context.retrieved_evidence) > 0
    assert len(context.mechanisms) > 0

    provider = MockLLMProvider()
    result = generate_hypotheses(problem, provider, n_hypotheses=4, context=context.model_dump())
    assert result["mode"] == "evidence-grounded"
    assert len(result["valid_hypotheses"]) >= 3

    # every surviving hypothesis must still compile into a Phase 3-compatible config
    for h in result["valid_hypotheses"]:
        adapted = hypothesis_to_pinn_config(h)
        assert callable(adapted["hypothesis_cfg"]["fn"])
        assert set(adapted["param_init"].keys()) == set(p.name for p in h.parameters)

    print(f"[PASS] test_integration_full_chain_to_phase3_adapter: "
          f"{len(result['valid_hypotheses'])} hypotheses all compiled for Phase 3")


if __name__ == "__main__":
    test_corpus_ingestion()
    test_metadata_preserved()
    test_chunking_preserves_sections()
    test_chunking_fallback_no_headers()
    test_local_embedding_deterministic()
    test_deterministic_retrieval()
    test_top_k_behavior()
    test_domain_filtering()
    test_section_filtering()
    test_provenance_preserved()
    test_empty_query_returns_empty()
    test_empty_index_returns_empty()
    test_persistent_vector_store_roundtrip()
    test_integration_full_chain_to_phase3_adapter()
    print("\nAll Phase 5 RAG tests passed.")
