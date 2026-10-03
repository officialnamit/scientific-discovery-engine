"""
Phase 5: deterministic graph construction.

Builds entities/relationships from each Document's own declared_*
fields (rag/documents.py) -- NOT from LLM extraction. This keeps graph
construction fully deterministic and avoids the Phase 5 integrity
requirement violation of "creating unsupported scientific relationships
simply because they sound plausible": every relationship here traces
to a specific document's own stated facts, marked provenance_type=
"documented". An LLM-based extractor over raw literature text could be
added later (marked "extracted"/"inferred" per the schema), but is out
of scope for this phase's fixture corpus.

This is domain-agnostic: nothing below hardcodes "diffusion" -- it
reads whatever declared_mechanism/variables/parameters/equations a
Document provides, for any domain.
"""

import hashlib
from typing import Dict, List, Optional

from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.schema import Entity, EntityType, RelationType, SourceRef
from rag.documents import Chunk, Document


def _slug(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def _find_chunk_id(chunks: Optional[Dict[str, Chunk]], document_id: str, section_keyword: str) -> Optional[str]:
    if not chunks:
        return None
    for chunk in chunks.values():
        if chunk.document_id == document_id and chunk.section and section_keyword.lower() in chunk.section.lower():
            return chunk.chunk_id
    return None


def build_graph_from_documents(documents: List[Document], chunks: Optional[Dict[str, Chunk]] = None) -> KnowledgeGraph:
    graph = KnowledgeGraph()

    for doc in documents:
        doc_id = doc.metadata.document_id
        doc_entity = graph.add_entity(Entity(
            entity_id=f"document_{doc_id}", type=EntityType.DOCUMENT, name=doc.metadata.title,
            metadata={"source": doc.metadata.source, "domain": doc.metadata.domain,
                      "is_benchmark_fixture": doc.metadata.is_benchmark_fixture},
        ))

        if not doc.declared_mechanism:
            continue  # a real (non-fixture) document with no declared structure contributes no graph facts

        mechanism_id = f"mechanism_{doc.declared_mechanism.replace(' ', '_').replace('-', '_')}"
        mechanism_entity = graph.add_entity(Entity(
            entity_id=mechanism_id, type=EntityType.MECHANISM, name=doc.declared_mechanism,
        ))

        overview_chunk = _find_chunk_id(chunks, doc_id, "overview")
        graph.add_relationship(
            mechanism_id, RelationType.SUPPORTED_BY, doc_entity.entity_id,
            source=SourceRef(document_id=doc_id, chunk_id=overview_chunk),
        )

        variable_entities = {}
        for var_name in doc.declared_variables:
            var_id = f"variable_{var_name}"
            graph.add_entity(Entity(entity_id=var_id, type=EntityType.VARIABLE, name=var_name))
            variable_entities[var_name] = var_id

        # Domain-agnostic rule: if a dependent variable 'u' is declared, it DEPENDS_ON
        # every other declared independent variable (x, t, ...).
        if "u" in variable_entities:
            for var_name, var_id in variable_entities.items():
                if var_name != "u":
                    graph.add_relationship(
                        variable_entities["u"], RelationType.DEPENDS_ON, var_id,
                        source=SourceRef(document_id=doc_id),
                    )

        parameter_entities = {}
        for param_name in doc.declared_parameters:
            param_id = f"parameter_{param_name}"
            graph.add_entity(Entity(entity_id=param_id, type=EntityType.PARAMETER, name=param_name))
            parameter_entities[param_name] = param_id

        eq_chunk = _find_chunk_id(chunks, doc_id, "governing equation")
        for eq_str in doc.declared_equations:
            eq_id = f"equation_{_slug(eq_str)}"
            graph.add_entity(Entity(
                entity_id=eq_id, type=EntityType.EQUATION, name=eq_str, metadata={"equation": eq_str},
            ))
            graph.add_relationship(
                mechanism_id, RelationType.GOVERNS, eq_id,
                source=SourceRef(document_id=doc_id, chunk_id=eq_chunk),
            )
            graph.add_relationship(
                eq_id, RelationType.SUPPORTED_BY, doc_entity.entity_id,
                source=SourceRef(document_id=doc_id, chunk_id=eq_chunk),
            )
            for var_id in variable_entities.values():
                graph.add_relationship(
                    eq_id, RelationType.HAS_VARIABLE, var_id,
                    source=SourceRef(document_id=doc_id, chunk_id=eq_chunk),
                )
            for param_id in parameter_entities.values():
                graph.add_relationship(
                    eq_id, RelationType.HAS_PARAMETER, param_id,
                    source=SourceRef(document_id=doc_id, chunk_id=eq_chunk),
                )

    return graph
