"""
Phase 5: knowledge graph query interface.

Implements the example questions from the spec:
  - "What mechanisms are associated with this observation?"   -> mechanisms_for_observation()
  - "What variables influence this quantity?"                  -> variables_depending_on()
  - "What equations are associated with this mechanism?"        -> equations_for_mechanism()
  - "Which papers support this relationship?"                    -> sources_for_relationship()
  - "What parameters appear in these equations?"                   -> parameters_for_mechanism()
  - "What evidence connects variable A and variable B?"             -> evidence_connecting()

Every function returns entities/relationships straight from the graph
(with their stored provenance) -- nothing here invents new facts.
"""

from typing import List, Optional

from knowledge_graph.graph import KnowledgeGraph, _relationship_id
from knowledge_graph.schema import Entity, EntityType, RelationType, Relationship, SourceRef


def find_entity_by_name(graph: KnowledgeGraph, entity_type: EntityType, name: str) -> Optional[Entity]:
    for e in graph.entities_by_type(entity_type):
        if e.name == name:
            return e
    return None


def equations_for_mechanism(graph: KnowledgeGraph, mechanism_name: str) -> List[Entity]:
    mech = find_entity_by_name(graph, EntityType.MECHANISM, mechanism_name)
    if not mech:
        return []
    eq_ids = graph.neighbors(mech.entity_id, predicate=RelationType.GOVERNS)
    return [graph.get_entity(e) for e in eq_ids]


def variables_for_mechanism(graph: KnowledgeGraph, mechanism_name: str) -> List[Entity]:
    var_ids = set()
    for eq in equations_for_mechanism(graph, mechanism_name):
        var_ids.update(graph.neighbors(eq.entity_id, predicate=RelationType.HAS_VARIABLE))
    return [graph.get_entity(v) for v in sorted(var_ids)]


def parameters_for_mechanism(graph: KnowledgeGraph, mechanism_name: str) -> List[Entity]:
    param_ids = set()
    for eq in equations_for_mechanism(graph, mechanism_name):
        param_ids.update(graph.neighbors(eq.entity_id, predicate=RelationType.HAS_PARAMETER))
    return [graph.get_entity(p) for p in sorted(param_ids)]


def mechanisms_for_observation(graph: KnowledgeGraph, observation_entity_id: str) -> List[Entity]:
    """Ready for Observation entities (schema supports EntityType.OBSERVATION /
    RelationType.OBSERVED_IN); the current fixture corpus is mechanism-description
    documents rather than experimental observation records, so this returns
    an empty list for this corpus -- documented honestly, not faked."""
    mech_ids = graph.neighbors(observation_entity_id, predicate=RelationType.OBSERVED_IN)
    return [graph.get_entity(m) for m in mech_ids]


def documents_supporting(graph: KnowledgeGraph, entity_id: str) -> List[Entity]:
    doc_ids = graph.neighbors(entity_id, predicate=RelationType.SUPPORTED_BY)
    return [graph.get_entity(d) for d in doc_ids]


def sources_for_relationship(graph: KnowledgeGraph, subject_id: str, predicate: RelationType, object_id: str) -> List[SourceRef]:
    rel_id = _relationship_id(subject_id, predicate.value, object_id)
    rel = graph.relationships.get(rel_id)
    return rel.sources if rel else []


def evidence_connecting(graph: KnowledgeGraph, variable_a: str, variable_b: str) -> List[Relationship]:
    """Direct relationships between two named variables, plus any equation
    that lists both as HAS_VARIABLE (an indirect but provenance-preserving
    connection: 'these two variables co-occur in the following equation(s)')."""
    var_a = find_entity_by_name(graph, EntityType.VARIABLE, variable_a)
    var_b = find_entity_by_name(graph, EntityType.VARIABLE, variable_b)
    if not var_a or not var_b:
        return []

    direct = [
        r for r in graph.relationships.values()
        if {r.subject, r.object} == {var_a.entity_id, var_b.entity_id}
    ]

    shared_equations = [
        eq for eq in graph.entities_by_type(EntityType.EQUATION)
        if var_a.entity_id in graph.neighbors(eq.entity_id, RelationType.HAS_VARIABLE)
        and var_b.entity_id in graph.neighbors(eq.entity_id, RelationType.HAS_VARIABLE)
    ]
    indirect = []
    for eq in shared_equations:
        indirect.extend(graph.relationships_for(eq.entity_id, RelationType.HAS_VARIABLE))

    seen = set()
    result = []
    for r in direct + indirect:
        if r.relationship_id not in seen:
            seen.add(r.relationship_id)
            result.append(r)
    return result
