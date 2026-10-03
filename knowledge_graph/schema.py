"""
Phase 5: domain-agnostic knowledge graph schema.

Nothing here mentions diffusion/advection/etc. by name -- entity and
relationship TYPES are generic scientific concepts; the specific
instances (a "diffusion" Mechanism entity, a "D" Parameter entity) are
created by knowledge_graph/builder.py from whatever documents it is
given. A different corpus (a different scientific domain entirely)
would populate the same schema with different entity instances.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    VARIABLE = "Variable"
    QUANTITY = "Quantity"
    PHYSICAL_SYSTEM = "PhysicalSystem"
    MECHANISM = "Mechanism"
    MATERIAL = "Material"
    PARAMETER = "Parameter"
    EQUATION = "Equation"
    OBSERVATION = "Observation"
    EXPERIMENT = "Experiment"
    SCIENTIFIC_CONCEPT = "ScientificConcept"
    DOCUMENT = "Document"


class RelationType(str, Enum):
    CAUSES = "CAUSES"
    DEPENDS_ON = "DEPENDS_ON"
    AFFECTS = "AFFECTS"
    MEASURED_BY = "MEASURED_BY"
    GOVERNS = "GOVERNS"
    CONSTRAINS = "CONSTRAINS"
    OCCURS_IN = "OCCURS_IN"
    HAS_PARAMETER = "HAS_PARAMETER"
    HAS_VARIABLE = "HAS_VARIABLE"
    SUPPORTED_BY = "SUPPORTED_BY"
    OBSERVED_IN = "OBSERVED_IN"
    RELATED_TO = "RELATED_TO"


class Entity(BaseModel):
    entity_id: str
    type: EntityType
    name: str
    metadata: dict = Field(default_factory=dict)


class SourceRef(BaseModel):
    document_id: str
    chunk_id: Optional[str] = None


class Relationship(BaseModel):
    relationship_id: str
    subject: str  # entity_id
    predicate: RelationType
    object: str  # entity_id
    sources: List[SourceRef] = Field(default_factory=list)
    provenance_type: str = Field(
        default="documented",
        description="'documented': derived directly from a source document's own declared "
                    "structured facts (deterministic, no LLM -- what this phase uses). "
                    "'extracted'/'inferred': would mark output of an LLM-based relationship "
                    "extractor, were one added -- not implemented in this phase, per the "
                    "requirement that extracted relationships never be silently treated as "
                    "established fact.",
    )
    confidence: Optional[float] = Field(
        default=None, description="Extraction confidence only, never scientific truth.",
    )
