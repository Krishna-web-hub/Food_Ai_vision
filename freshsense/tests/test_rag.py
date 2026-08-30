"""Chunking and retrieval."""

from __future__ import annotations

import pytest

from freshsense.genai.rag import KnowledgeBase, chunk_markdown


def test_chunks_split_on_h2_and_carry_the_document_title(kb_dir):
    chunks = chunk_markdown(kb_dir / "alpha.md")
    assert [c.source_id for c in chunks] == ["alpha#1", "alpha#2"]
    assert chunks[0].title == "Alpha doc — Mould"
    assert "Fuzzy growth" in chunks[0].text
    assert "# Alpha doc" not in chunks[0].text


def test_retrieval_ranks_the_relevant_chunk_first(kb_dir):
    kb = KnowledgeBase(kb_dir, min_score=0.0).load()
    top = kb.search("fuzzy mould growth discard", top_k=1)
    assert top and top[0].source_id == "alpha#1"


def test_retrieval_respects_top_k(kb_dir):
    kb = KnowledgeBase(kb_dir, min_score=0.0).load()
    assert len(kb.search("mould slime escalation", top_k=2)) == 2


def test_irrelevant_query_returns_nothing_rather_than_noise(kb_dir):
    """Padding results to top_k with unrelated chunks is how RAG starts lying."""
    kb = KnowledgeBase(kb_dir, min_score=0.05).load()
    assert kb.search("quarterly amortisation schedule", top_k=3) == []


def test_source_ids_covers_every_chunk(kb_dir):
    kb = KnowledgeBase(kb_dir).load()
    assert kb.source_ids() == {"alpha#1", "alpha#2", "beta#1"}


def test_empty_directory_fails_loudly(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="knowledge base is empty"):
        KnowledgeBase(tmp_path / "empty").load()


def test_shipped_knowledge_base_indexes(settings):
    """The real kb/ directory must chunk cleanly — it ships with the service."""
    from freshsense.config import Settings

    kb = KnowledgeBase(Settings().kb_dir).load()
    assert len(kb.chunks) >= 10
    assert all(c.text.strip() for c in kb.chunks)
