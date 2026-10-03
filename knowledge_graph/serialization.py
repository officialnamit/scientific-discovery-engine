"""
Phase 5: knowledge graph serialization.

Round-trips a KnowledgeGraph to/from a plain JSON-able dict (entities +
relationships lists), and to/from a file. Rebuilding from the saved
dict goes through graph.add_entity()/add_relationship() again, so the
same dedup/merge rules apply on load as on original construction.
"""

import json
from typing import Dict

from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.schema import Entity, Relationship


def graph_to_dict(graph: KnowledgeGraph) -> Dict:
    return {
        "entities": [json.loads(e.model_dump_json()) for e in graph.entities.values()],
        "relationships": [json.loads(r.model_dump_json()) for r in graph.relationships.values()],
        "summary": graph.summary(),
    }


def graph_from_dict(data: Dict) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for e_data in data["entities"]:
        graph.add_entity(Entity(**e_data))
    for r_data in data["relationships"]:
        rel = Relationship(**r_data)
        for source in rel.sources:
            graph.add_relationship(
                rel.subject, rel.predicate, rel.object,
                source=source, provenance_type=rel.provenance_type, confidence=rel.confidence,
            )
    return graph


def save_graph(graph: KnowledgeGraph, path: str) -> None:
    with open(path, "w") as f:
        json.dump(graph_to_dict(graph), f, indent=2)


def load_graph(path: str) -> KnowledgeGraph:
    with open(path) as f:
        data = json.load(f)
    return graph_from_dict(data)
