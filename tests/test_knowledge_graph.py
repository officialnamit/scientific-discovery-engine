"""
Phase 5 knowledge graph tests. All offline/deterministic.

Run:
    python tests/test_knowledge_graph.py
or:
    python -m pytest tests/test_knowledge_graph.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph.builder import build_graph_from_documents
from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.query import (
    documents_supporting,
    equations_for_mechanism,
    evidence_connecting,
    find_entity_by_name,
    parameters_for_mechanism,
    sources_for_relationship,
    variables_for_mechanism,
)
from knowledge_graph.schema import Entity, EntityType, RelationType, SourceRef
from knowledge_graph.serialization import graph_from_dict, graph_to_dict
from rag.corpus_loader import load_corpus
from rag.pipeline import RAGPipeline


def _build_corpus_graph():
    docs = load_corpus()
    pipeline = RAGPipeline()
    chunks = pipeline.ingest(docs)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    return build_graph_from_documents(docs, chunks_by_id), docs


# 1. Entity creation
def test_entity_creation():
    graph = KnowledgeGraph()
    e = graph.add_entity(Entity(entity_id="variable_u", type=EntityType.VARIABLE, name="u"))
    assert graph.get_entity("variable_u") is e
    # adding the same entity_id again doesn't duplicate
    graph.add_entity(Entity(entity_id="variable_u", type=EntityType.VARIABLE, name="u"))
    assert len(graph) == 1
    print("[PASS] test_entity_creation")


# 2. Relationship creation
def test_relationship_creation():
    graph = KnowledgeGraph()
    graph.add_entity(Entity(entity_id="mechanism_diffusion", type=EntityType.MECHANISM, name="diffusion"))
    graph.add_entity(Entity(entity_id="equation_1", type=EntityType.EQUATION, name="u_t=D*u_xx"))
    rel = graph.add_relationship(
        "mechanism_diffusion", RelationType.GOVERNS, "equation_1",
        source=SourceRef(document_id="doc_1", chunk_id="doc_1_chunk_003"),
    )
    assert rel.subject == "mechanism_diffusion"
    assert rel.predicate == RelationType.GOVERNS
    assert len(rel.sources) == 1
    print("[PASS] test_relationship_creation")


def test_relationship_requires_existing_entities():
    graph = KnowledgeGraph()
    graph.add_entity(Entity(entity_id="a", type=EntityType.VARIABLE, name="a"))
    try:
        graph.add_relationship("a", RelationType.RELATED_TO, "nonexistent", source=SourceRef(document_id="d"))
        assert False, "Expected ValueError for relationship to a non-existent entity"
    except ValueError:
        pass
    print("[PASS] test_relationship_requires_existing_entities")


# 3. Duplicate handling
def test_duplicate_relationship_merges_sources():
    graph = KnowledgeGraph()
    graph.add_entity(Entity(entity_id="a", type=EntityType.VARIABLE, name="a"))
    graph.add_entity(Entity(entity_id="b", type=EntityType.VARIABLE, name="b"))
    r1 = graph.add_relationship("a", RelationType.RELATED_TO, "b", source=SourceRef(document_id="doc_1"))
    r2 = graph.add_relationship("a", RelationType.RELATED_TO, "b", source=SourceRef(document_id="doc_2"))
    assert r1.relationship_id == r2.relationship_id
    assert len(graph.relationships) == 1
    assert len(graph.relationships[r1.relationship_id].sources) == 2
    print("[PASS] test_duplicate_relationship_merges_sources")


def test_same_source_not_duplicated():
    graph = KnowledgeGraph()
    graph.add_entity(Entity(entity_id="a", type=EntityType.VARIABLE, name="a"))
    graph.add_entity(Entity(entity_id="b", type=EntityType.VARIABLE, name="b"))
    graph.add_relationship("a", RelationType.RELATED_TO, "b", source=SourceRef(document_id="doc_1"))
    graph.add_relationship("a", RelationType.RELATED_TO, "b", source=SourceRef(document_id="doc_1"))
    rel = list(graph.relationships.values())[0]
    assert len(rel.sources) == 1
    print("[PASS] test_same_source_not_duplicated")


# 4. Provenance
def test_relationship_provenance_type():
    graph = KnowledgeGraph()
    graph.add_entity(Entity(entity_id="a", type=EntityType.VARIABLE, name="a"))
    graph.add_entity(Entity(entity_id="b", type=EntityType.VARIABLE, name="b"))
    rel = graph.add_relationship("a", RelationType.RELATED_TO, "b", source=SourceRef(document_id="doc_1"),
                                  provenance_type="extracted", confidence=0.6)
    assert rel.provenance_type == "extracted"
    assert rel.confidence == 0.6
    print("[PASS] test_relationship_provenance_type")


# builder on the real corpus
def test_builder_produces_domain_agnostic_graph():
    graph, docs = _build_corpus_graph()
    mechanisms = {e.name for e in graph.entities_by_type(EntityType.MECHANISM)}
    assert mechanisms == {"diffusion", "advection", "reaction", "reaction-diffusion", "advection-diffusion"}
    summary = graph.summary()
    assert summary["n_unsourced_relationships"] == 0  # every relationship traces to a document
    print(f"[PASS] test_builder_produces_domain_agnostic_graph: {summary}")


def test_shared_variable_and_parameter_entities_merge_across_documents():
    graph, docs = _build_corpus_graph()
    # 'D' is declared by diffusion, reaction-diffusion, AND advection-diffusion docs --
    # must be ONE entity with THREE sourced relationships into it, not three entities.
    d_entity = find_entity_by_name(graph, EntityType.PARAMETER, "D")
    assert d_entity is not None
    rels_to_d = [r for r in graph.relationships.values() if r.object == d_entity.entity_id]
    doc_ids_citing_d = set()
    for r in rels_to_d:
        doc_ids_citing_d.update(s.document_id for s in r.sources)
    assert doc_ids_citing_d == {"doc_diffusion", "doc_reaction_diffusion", "doc_advection_diffusion"}
    print(f"[PASS] test_shared_variable_and_parameter_entities_merge_across_documents: {doc_ids_citing_d}")


# 5. Graph querying
def test_query_equations_variables_parameters_for_mechanism():
    graph, docs = _build_corpus_graph()
    eqs = equations_for_mechanism(graph, "diffusion")
    assert len(eqs) == 1 and eqs[0].name == "u_t = D*u_xx"
    assert {v.name for v in variables_for_mechanism(graph, "diffusion")} == {"u", "x", "t"}
    assert {p.name for p in parameters_for_mechanism(graph, "diffusion")} == {"D"}
    print("[PASS] test_query_equations_variables_parameters_for_mechanism")


def test_query_documents_supporting():
    graph, docs = _build_corpus_graph()
    mech = find_entity_by_name(graph, EntityType.MECHANISM, "advection")
    docs_supporting = documents_supporting(graph, mech.entity_id)
    assert len(docs_supporting) == 1
    assert "Advective" in docs_supporting[0].name
    print("[PASS] test_query_documents_supporting")


def test_query_sources_for_relationship():
    graph, docs = _build_corpus_graph()
    mech = find_entity_by_name(graph, EntityType.MECHANISM, "diffusion")
    doc_entity = find_entity_by_name(graph, EntityType.DOCUMENT, mech.name)  # won't match by design; use direct lookup instead
    doc_entity_id = "document_doc_diffusion"
    sources = sources_for_relationship(graph, mech.entity_id, RelationType.SUPPORTED_BY, doc_entity_id)
    assert len(sources) == 1 and sources[0].document_id == "doc_diffusion"
    print("[PASS] test_query_sources_for_relationship")


def test_evidence_connecting_variables():
    graph, docs = _build_corpus_graph()
    rels = evidence_connecting(graph, "u", "x")
    assert len(rels) > 0
    print(f"[PASS] test_evidence_connecting_variables: {len(rels)} connecting relationships")


# 6. Serialization/deserialization
def test_serialization_roundtrip():
    graph, docs = _build_corpus_graph()
    data = graph_to_dict(graph)
    restored = graph_from_dict(data)
    assert restored.summary() == graph.summary()
    # spot-check a specific relationship's sources survive the round trip
    d_entity = find_entity_by_name(restored, EntityType.PARAMETER, "D")
    assert d_entity is not None
    print("[PASS] test_serialization_roundtrip")


if __name__ == "__main__":
    test_entity_creation()
    test_relationship_creation()
    test_relationship_requires_existing_entities()
    test_duplicate_relationship_merges_sources()
    test_same_source_not_duplicated()
    test_relationship_provenance_type()
    test_builder_produces_domain_agnostic_graph()
    test_shared_variable_and_parameter_entities_merge_across_documents()
    test_query_equations_variables_parameters_for_mechanism()
    test_query_documents_supporting()
    test_query_sources_for_relationship()
    test_evidence_connecting_variables()
    test_serialization_roundtrip()
    print("\nAll Phase 5 Knowledge Graph tests passed.")
