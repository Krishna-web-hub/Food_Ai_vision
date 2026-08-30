"""Retrieval over the food-safety knowledge base.

TF-IDF + cosine similarity on markdown sections. That is a deliberate choice,
not a shortcut: the corpus is five documents, so an embedding model would add a
network dependency, a warm-up cost and a source of non-determinism to solve a
problem lexical retrieval already solves. The `Retriever` protocol is the seam —
swapping in pgvector or Qdrant means implementing `search`, and nothing above
this module changes. See docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from freshsense.domain import Citation

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Chunk:
    source_id: str
    title: str
    text: str


def chunk_markdown(path: Path) -> list[Chunk]:
    """Split a markdown file on `##` headings; the `#` title prefixes every chunk.

    Section-level chunking beats fixed-width windows here because the documents
    are already written as self-contained sections — a chunk boundary mid-rule
    would strand the qualifier that makes the rule safe.
    """
    raw = path.read_text(encoding="utf-8")
    doc_title = next((ln[2:].strip() for ln in raw.splitlines() if ln.startswith("# ")), path.stem)

    sections: list[tuple[str, list[str]]] = []
    current_title, buffer = doc_title, []
    for line in raw.splitlines():
        if line.startswith("## "):
            if any(text.strip() for text in buffer):
                sections.append((current_title, buffer))
            current_title, buffer = line[3:].strip(), []
        elif not line.startswith("# "):
            buffer.append(line)
    if any(line.strip() for line in buffer):
        sections.append((current_title, buffer))

    chunks = []
    for i, (title, lines) in enumerate(sections, start=1):
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                source_id=f"{path.stem}#{i}",
                title=f"{doc_title} — {title}" if title != doc_title else doc_title,
                text=text,
            )
        )
    return chunks


class Retriever(Protocol):
    """The seam a vector database would slot into."""

    def search(self, query: str, top_k: int) -> list[Citation]: ...


class KnowledgeBase:
    """In-memory TF-IDF index over a directory of markdown documents."""

    def __init__(self, kb_dir: Path, min_score: float = 0.02) -> None:
        self.kb_dir = Path(kb_dir)
        self.min_score = min_score
        self.chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    def load(self) -> KnowledgeBase:
        if self.ready:
            return self
        files = sorted(self.kb_dir.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"knowledge base is empty: {self.kb_dir}")

        self.chunks = [c for f in files for c in chunk_markdown(f)]
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])
        logger.info(
            "knowledge base indexed",
            extra={"documents": len(files), "chunks": len(self.chunks)},
        )
        return self

    def search(self, query: str, top_k: int = 3) -> list[Citation]:
        """Top-k chunks above `min_score`. Returns [] rather than padding with noise."""
        self.load()
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            Citation(
                source_id=self.chunks[i].source_id,
                title=self.chunks[i].title,
                text=self.chunks[i].text,
                score=float(score),
            )
            for i, score in ranked
            if score >= self.min_score
        ]

    def source_ids(self) -> set[str]:
        """Every id the guardrails will accept in an explanation."""
        self.load()
        return {c.source_id for c in self.chunks}
