"""
Phase 5, Part 12: the unified scientific context builder.

    ScientificProblem -> Retriever -> Evidence -> KG query -> ScientificContext -> Phase 4

This is the ONLY new thing Phase 4 needs to know about: a
ScientificContext object (or None, for evidence-free Mode A). Nothing
about hypotheses/pipeline.py's core generation logic changes -- see
llm/prompts.py's build_hypothesis_prompt_with_context() and
hypotheses/pipeline.py's `context` parameter.

ANSWER-LEAKAGE NOTE: `known_equations` here are GENERIC textbook forms
retrieved from the fixture corpus (e.g. "u_t = D*u_xx" as the general
form of Fick's law) -- never the specific fitted numeric parameter
value for this experiment's hidden D, and never an explicit claim like
"this dataset follows diffusion". The hypothesis engine still has to
decide which documented mechanism (if any) applies to the specific
observations and estimate parameters itself via PINN validation. See
docs/development_log.md (Phase 5, "preventing answer leakage") for the full
reasoning.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.query import equations_for_mechanism, find_entity_by_name, parameters_for_mechanism
from knowledge_graph.schema import EntityType, Relationship
from rag.evidence import Evidence
from rag.retriever import Retriever


class ScientificContext(BaseModel):
    problem: str
    observations: List[str] = Field(default_factory=list)
    retrieved_evidence: List[Evidence] = Field(default_factory=list)
    variables: List[str] = Field(default_factory=list)
    mechanisms: List[str] = Field(default_factory=list)
    known_equations: List[str] = Field(default_factory=list)
    parameters: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    graph_relationships: List[Relationship] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


def build_scientific_context(
    problem_description: str,
    retriever: Retriever,
    graph: Optional[KnowledgeGraph] = None,
    top_k: int = 5,
) -> ScientificContext:
    evidence_list: List[Evidence] = retriever.retrieve(problem_description, top_k=top_k)

    variables = sorted({v for e in evidence_list for v in e.variables})
    mechanisms = sorted({m for e in evidence_list for m in e.mechanisms})
    known_equations = sorted({eq for e in evidence_list for eq in e.equations})
    assumptions = sorted({a for e in evidence_list for a in e.assumptions})
    sources = sorted({e.source for e in evidence_list})

    parameters: List[str] = []
    graph_relationships: List[Relationship] = []
    if graph is not None:
        for mechanism_name in mechanisms:
            mech_entity = find_entity_by_name(graph, EntityType.MECHANISM, mechanism_name)
            if mech_entity is None:
                continue
            graph_relationships.extend(graph.relationships_for(mech_entity.entity_id))
            for eq_entity in equations_for_mechanism(graph, mechanism_name):
                graph_relationships.extend(graph.relationships_for(eq_entity.entity_id))
            parameters.extend(p.name for p in parameters_for_mechanism(graph, mechanism_name))

    # dedupe graph_relationships by id while preserving order
    seen_ids = set()
    unique_relationships = []
    for r in graph_relationships:
        if r.relationship_id not in seen_ids:
            seen_ids.add(r.relationship_id)
            unique_relationships.append(r)

    return ScientificContext(
        problem=problem_description,
        observations=[],
        retrieved_evidence=evidence_list,
        variables=variables,
        mechanisms=mechanisms,
        known_equations=known_equations,
        parameters=sorted(set(parameters)),
        assumptions=assumptions,
        graph_relationships=unique_relationships,
        sources=sources,
    )
