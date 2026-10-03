"""
Phase 5: metadata filter helpers used by rag/retriever.py.

Kept separate from rag/documents.py (which defines the schemas) so the
filtering LOGIC has one place to live and one place to test.
"""

from typing import Dict, Optional

from rag.documents import Chunk, DocumentMetadata


def matches_filters(
    chunk: Chunk,
    doc_metadata: DocumentMetadata,
    domain: Optional[str] = None,
    section: Optional[str] = None,
    metadata_filter: Optional[Dict] = None,
) -> bool:
    if domain is not None and doc_metadata.domain != domain:
        return False
    if section is not None and chunk.section != section:
        return False
    if metadata_filter:
        for key, value in metadata_filter.items():
            if chunk.metadata.get(key) != value:
                return False
    return True
