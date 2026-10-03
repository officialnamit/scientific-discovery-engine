"""
Phase 5: local reproducible scientific corpus loader.

Loads every rag/corpus/*.md fixture file into a Document. Each file is
a YAML frontmatter block (document_id, title, source, domain,
declared_mechanism/variables/parameters/equations/assumptions) followed
by a markdown body (the actual chunked text).

All corpus files are marked `is_benchmark_fixture: true` and
`source: "benchmark fixture"` -- they are original summaries of
well-known, generic textbook physics (Fick's law, the advection
equation, logistic growth, etc.) written for this project, NOT
reproductions of any specific real publication, and they do not state
or imply this project's specific hidden parameter values (see
"Preventing answer leakage" in the README's Phase 5 section).
"""

import os
from typing import List

import yaml

from rag.documents import Document, DocumentMetadata

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")


def _parse_frontmatter(raw: str):
    if not raw.startswith("---"):
        raise ValueError("Corpus fixture file missing YAML frontmatter (---...---)")
    _, fm_text, body = raw.split("---", 2)
    frontmatter = yaml.safe_load(fm_text)
    return frontmatter, body.strip()


def load_document(path: str) -> Document:
    with open(path, "r") as f:
        raw = f.read()
    frontmatter, body = _parse_frontmatter(raw)

    metadata = DocumentMetadata(
        document_id=frontmatter["document_id"],
        title=frontmatter["title"],
        authors=frontmatter.get("authors", []),
        source=frontmatter["source"],
        year=frontmatter.get("year"),
        domain=frontmatter.get("domain"),
        is_benchmark_fixture=frontmatter.get("is_benchmark_fixture", False),
    )
    return Document(
        metadata=metadata,
        text=body,
        declared_mechanism=frontmatter.get("declared_mechanism"),
        declared_variables=frontmatter.get("declared_variables", []),
        declared_parameters=frontmatter.get("declared_parameters", []),
        declared_equations=frontmatter.get("declared_equations", []),
        declared_assumptions=frontmatter.get("declared_assumptions", []),
    )


def load_corpus(corpus_dir: str = CORPUS_DIR) -> List[Document]:
    docs = []
    for fname in sorted(os.listdir(corpus_dir)):
        if fname.endswith(".md"):
            docs.append(load_document(os.path.join(corpus_dir, fname)))
    return docs
