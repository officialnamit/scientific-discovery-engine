"""
Phase 5: evidence objects.

`confidence` on an Evidence object is a RETRIEVAL/EXTRACTION confidence
only (here: the embedding similarity score) -- never scientific truth.
This is stated in the field description itself and re-stated in the
README so it can't be missed downstream.

Variables/mechanisms/equations/assumptions are pulled from the SOURCE
DOCUMENT's own declared_* fields (rag/documents.py), not inferred by
an LLM -- this keeps evidence construction fully deterministic and
avoids fabricating structured claims the fixture document didn't
actually make. A real (LLM-based) extractor could be added later; if
it were, the Phase 5 spec requires marking its output as
extracted/inferred rather than established fact -- there is a
provenance_type field reserved for exactly that distinction.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from rag.documents import Document


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str
    section: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class Evidence(BaseModel):
    evidence_id: str
    claim: str  # the retrieved chunk's text itself -- no LLM summarization/paraphrase
    source: str
    chunk_id: str
    document_id: str
    section: Optional[str] = None
    variables: List[str] = Field(default_factory=list)
    mechanisms: List[str] = Field(default_factory=list)
    equations: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(
        default=None,
        description="Retrieval/extraction confidence ONLY (here: embedding similarity "
                    "score in [-1,1]). NOT a measure of scientific truth.",
    )
    provenance_type: str = Field(
        default="documented",
        description="'documented': fields taken directly from the source document's own "
                    "declared metadata (deterministic, no LLM). 'extracted'/'inferred' would "
                    "mark output from an LLM-based extractor, were one added -- not done in "
                    "this phase.",
    )


def evidence_from_retrieval(result: RetrievalResult, document: Document) -> Evidence:
    return Evidence(
        evidence_id=f"evidence_{result.chunk_id}",
        claim=result.text,
        source=result.source,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        section=result.section,
        variables=list(document.declared_variables),
        mechanisms=[document.declared_mechanism] if document.declared_mechanism else [],
        equations=list(document.declared_equations),
        assumptions=list(document.declared_assumptions),
        confidence=result.score,
        provenance_type="documented",
    )
