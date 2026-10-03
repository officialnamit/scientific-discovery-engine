"""
Phase 5: scientific-aware chunking.

Splits on markdown headers ('#', '##', '###', ...) so a chunk boundary
coincides with a document's own title/section/subsection structure,
rather than cutting mid-paragraph at an arbitrary character count. A
header line becomes the chunk's `section` field; everything under it
(until the next header of any level) is that chunk's text.

LIMITATION, stated honestly: this is structural (header-based)
chunking, not semantic chunking -- it does not specifically detect or
isolate equations, definitions, or experimental-observation sentences
as separate chunk types within a section (the Phase 5 spec's "prefer
preserving ... equations, definitions, ..." is satisfied only insofar
as those appear naturally grouped within whatever section already
contains them, e.g. the corpus's "Governing equation" subsections).
A document with no headers at all falls back to one chunk per
paragraph (blank-line-separated).
"""

from typing import List

from rag.documents import Chunk, Document


def chunk_document(doc: Document) -> List[Chunk]:
    lines = doc.text.split("\n")
    doc_id = doc.metadata.document_id

    has_headers = any(line.strip().startswith("#") for line in lines)
    chunks: List[Chunk] = []
    idx = 0

    def make_chunk(section: str, text: str):
        nonlocal idx
        text = text.strip()
        if not text:
            return
        idx += 1
        chunk_id = f"{doc_id}_chunk_{idx:03d}"
        chunks.append(Chunk(
            chunk_id=chunk_id, document_id=doc_id, section=section, text=text,
            metadata={"domain": doc.metadata.domain},
        ))

    if has_headers:
        current_section = "Preamble"
        current_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                make_chunk(current_section, "\n".join(current_lines))
                current_section = stripped.lstrip("#").strip()
                current_lines = []
            else:
                current_lines.append(line)
        make_chunk(current_section, "\n".join(current_lines))
    else:
        paragraphs = [p for p in doc.text.split("\n\n")]
        for p in paragraphs:
            make_chunk(None, p)

    return chunks
