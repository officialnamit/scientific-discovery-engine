"""
Phase 5: knowledge graph container.

Backed by networkx (already a project dependency-of-choice for small
graphs; no new heavy dependency). Relationship IDs are a stable hash of
(subject, predicate, object) -- NOT of insertion order or a counter --
so building the same corpus twice always produces the same IDs
(reproducibility requirement), and adding the same relationship twice
from two different source documents MERGES into one relationship with
two sources rather than creating a duplicate edge (duplicate handling
requirement).
"""

import hashlib
from typing import Dict, List, Optional

import networkx as nx

from knowledge_graph.schema import Entity, EntityType, RelationType, Relationship, SourceRef


def _relationship_id(subject: str, predicate: str, obj: str) -> str:
    digest = hashlib.md5(f"{subject}|{predicate}|{obj}".encode("utf-8")).hexdigest()
    return f"rel_{digest[:12]}"


class KnowledgeGraph:
    def __init__(self):
        self._graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}

    # --- entities ---
    def add_entity(self, entity: Entity) -> Entity:
        if entity.entity_id in self.entities:
            return self.entities[entity.entity_id]  # already present -- no duplicate node
        self.entities[entity.entity_id] = entity
        self._graph.add_node(entity.entity_id, type=entity.type.value, name=entity.name, **entity.metadata)
        return entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        return [e for e in self.entities.values() if e.type == entity_type]

    # --- relationships ---
    def add_relationship(
        self, subject: str, predicate: RelationType, obj: str,
        source: SourceRef, provenance_type: str = "documented", confidence: Optional[float] = None,
    ) -> Relationship:
        if subject not in self.entities or obj not in self.entities:
            raise ValueError(f"Both subject '{subject}' and object '{obj}' must be added as "
                              f"entities before a relationship between them.")
        rel_id = _relationship_id(subject, predicate.value, obj)
        if rel_id in self.relationships:
            existing = self.relationships[rel_id]
            if source not in existing.sources:  # duplicate handling: merge sources, don't duplicate edge
                existing.sources.append(source)
            return existing

        rel = Relationship(
            relationship_id=rel_id, subject=subject, predicate=predicate, object=obj,
            sources=[source], provenance_type=provenance_type, confidence=confidence,
        )
        self.relationships[rel_id] = rel
        self._graph.add_edge(subject, obj, key=rel_id, predicate=predicate.value, relationship_id=rel_id)
        return rel

    # --- traversal ---
    def neighbors(self, entity_id: str, predicate: Optional[RelationType] = None) -> List[str]:
        if entity_id not in self._graph:
            return []
        out = []
        for _, target, data in self._graph.out_edges(entity_id, data=True):
            if predicate is None or data.get("predicate") == predicate.value:
                out.append(target)
        return sorted(set(out))

    def relationships_for(self, entity_id: str, predicate: Optional[RelationType] = None) -> List[Relationship]:
        return [
            r for r in self.relationships.values()
            if r.subject == entity_id and (predicate is None or r.predicate == predicate)
        ]

    def __len__(self):
        return len(self.entities)

    def summary(self) -> Dict:
        by_type = {}
        for e in self.entities.values():
            by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
        sourced = sum(1 for r in self.relationships.values() if r.sources)
        return {
            "n_entities": len(self.entities),
            "n_relationships": len(self.relationships),
            "entities_by_type": by_type,
            "n_sourced_relationships": sourced,
            "n_unsourced_relationships": len(self.relationships) - sourced,
        }
