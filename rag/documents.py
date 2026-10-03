"""
Phase 5: document and chunk models.

Every document and chunk carries a stable ID and provenance metadata.
Unknown fields are represented as None/null -- never fabricated (see
Phase 5 constraint: "If a field is unknown, represent it as
unknown/null rather than inventing it.").
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    source: str = Field(
        ..., description="e.g. 'benchmark fixture' for synthetic corpus docs, or a real "
                          "citation string for sourced material. Never fabricated."
    )
    year: Optional[int] = None
    domain: Optional[str] = None  # e.g. "diffusion", "advection", "reaction-diffusion"
    is_benchmark_fixture: bool = False


class Document(BaseModel):
    metadata: DocumentMetadata
    text: str
    # Structured facts the fixture document author explicitly declares about itself --
    # used by knowledge_graph/builder.py to build graph entities/relationships
    # deterministically, without needing an LLM extractor. Real (non-fixture)
    # documents would leave these empty and rely on chunking + retrieval only.
    declared_mechanism: Optional[str] = None
    declared_variables: List[str] = Field(default_factory=list)
    declared_parameters: List[str] = Field(default_factory=list)
    declared_equations: List[str] = Field(default_factory=list)
    declared_assumptions: List[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    section: Optional[str] = None
    text: str
    metadata: dict = Field(default_factory=dict)
