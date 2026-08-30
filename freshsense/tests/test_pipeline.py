"""The triage agent, wired to fakes so no weights and no network are needed."""

from __future__ import annotations

import pytest

from freshsense.agent.pipeline import TriageAgent
from freshsense.domain import Action, Verdict
from freshsense.genai.llm import LLMError, OfflineClient
from freshsense.genai.rag import KnowledgeBase

from .conftest import FakeClassifier, FakeLLM


def build_agent(settings, kb_dir, probabilities, llm=None):
    settings = settings.model_copy(update={"kb_dir": kb_dir})
    return TriageAgent(
        classifier=FakeClassifier(probabilities, settings),
        knowledge_base=KnowledgeBase(kb_dir, min_score=0.0),
        llm=llm or OfflineClient(),
        settings=settings,
    )


def test_spoiled_image_produces_a_discard_assessment(settings, kb_dir, image_bytes):
    agent = build_agent(settings, kb_dir, {"fresh": 0.1, "spoiled": 0.9})
    result = agent.assess(image_bytes, food_hint="strawberries")

    assert result.prediction.verdict is Verdict.SPOILED
    assert result.action is Action.DISCARD
    assert result.explanation.summary.strip()
    assert result.explanation.citations
    assert result.latency_ms > 0


def test_fresh_image_produces_a_release_assessment(settings, kb_dir, image_bytes):
    agent = build_agent(settings, kb_dir, {"fresh": 0.97, "spoiled": 0.03})
    result = agent.assess(image_bytes)
    assert result.prediction.verdict is Verdict.FRESH
    assert result.action is Action.RELEASE


def test_uncertain_image_routes_to_manual_review(settings, kb_dir, image_bytes):
    agent = build_agent(settings, kb_dir, {"fresh": 0.80, "spoiled": 0.20})
    result = agent.assess(image_bytes)
    assert result.prediction.verdict is Verdict.UNCERTAIN
    assert result.action is Action.MANUAL_REVIEW


def test_every_node_appears_in_the_trace(settings, kb_dir, image_bytes):
    agent = build_agent(settings, kb_dir, {"fresh": 0.1, "spoiled": 0.9})
    result = agent.assess(image_bytes)
    nodes = [t["node"] for t in result.trace]
    assert nodes == ["ingest", "classify", "retrieve", "cheap_explain", "guard", "finalise"]


def test_llm_is_used_when_configured_and_confident(settings, kb_dir, image_bytes):
    settings = settings.model_copy(update={"anthropic_api_key": "sk-ant-test"})
    llm = FakeLLM()
    agent = build_agent(settings, kb_dir, {"fresh": 0.1, "spoiled": 0.9}, llm=llm)
    result = agent.assess(image_bytes)

    assert llm.calls == 1
    assert result.explanation.llm_mode == "anthropic"
    assert [t["node"] for t in result.trace][3] == "explain"


def test_llm_is_skipped_when_the_model_is_guessing(settings, kb_dir, image_bytes):
    """A near-uniform prediction gets the fixed text, not a fluent narrative."""
    settings = settings.model_copy(update={"anthropic_api_key": "sk-ant-test"})
    llm = FakeLLM()
    agent = build_agent(settings, kb_dir, {"fresh": 0.52, "spoiled": 0.48}, llm=llm)
    result = agent.assess(image_bytes)

    assert llm.calls == 0
    assert result.explanation.llm_mode == "offline"
    assert "cheap_explain" in [t["node"] for t in result.trace]


def test_llm_failure_degrades_to_the_offline_explanation(settings, kb_dir, image_bytes):
    settings = settings.model_copy(update={"anthropic_api_key": "sk-ant-test"})
    llm = FakeLLM(raises=LLMError("503 overloaded"))
    agent = build_agent(settings, kb_dir, {"fresh": 0.1, "spoiled": 0.9}, llm=llm)
    result = agent.assess(image_bytes)

    assert llm.calls == 1
    assert result.explanation.llm_mode == "offline"
    assert result.explanation.summary.strip()


def test_hallucinated_llm_citation_is_caught_end_to_end(settings, kb_dir, image_bytes):
    settings = settings.model_copy(update={"anthropic_api_key": "sk-ant-test"})
    llm = FakeLLM(
        payload={
            "summary": "According to FDA circular 12-B this is contaminated.",
            "handling_guidance": "Discard.",
            "sources": ["fda-circular-12b#3"],
        }
    )
    agent = build_agent(settings, kb_dir, {"fresh": 0.1, "spoiled": 0.9}, llm=llm)
    result = agent.assess(image_bytes)

    assert not result.explanation.grounded
    assert "FDA circular" not in result.explanation.summary
    assert any("fda-circular-12b#3" in n for n in result.explanation.guardrail_notes)


def test_unsafe_llm_output_never_reaches_the_caller(settings, kb_dir, image_bytes):
    settings = settings.model_copy(update={"anthropic_api_key": "sk-ant-test"})
    llm = FakeLLM(
        payload={
            "summary": "This item is safe to eat.",
            "handling_guidance": "Taste-test a small piece to confirm.",
            "sources": ["alpha#1"],
        }
    )
    agent = build_agent(settings, kb_dir, {"fresh": 0.97, "spoiled": 0.03}, llm=llm)
    result = agent.assess(image_bytes)

    assert "safe to eat" not in result.explanation.summary
    assert "taste" not in result.explanation.handling_guidance.lower()
    assert not result.explanation.grounded


def test_retrieval_query_differs_by_verdict(settings, kb_dir, image_bytes):
    """A fresh clearance and a spoiled condemnation must not cite the same passage."""
    spoiled = build_agent(settings, kb_dir, {"fresh": 0.1, "spoiled": 0.9}).assess(image_bytes)
    uncertain = build_agent(settings, kb_dir, {"fresh": 0.8, "spoiled": 0.2}).assess(image_bytes)

    top_spoiled = spoiled.explanation.citations[0].source_id
    top_uncertain = uncertain.explanation.citations[0].source_id
    assert top_spoiled != top_uncertain


def test_invalid_image_propagates(settings, kb_dir):
    from freshsense.vision.predict import InvalidImageError

    agent = build_agent(settings, kb_dir, {"fresh": 0.9, "spoiled": 0.1})
    with pytest.raises(InvalidImageError):
        agent.assess(b"definitely not an image")


def test_assessment_serialises_to_a_json_safe_dict(settings, kb_dir, image_bytes):
    import json

    agent = build_agent(settings, kb_dir, {"fresh": 0.1, "spoiled": 0.9})
    result = agent.assess(image_bytes)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["verdict"] == "spoiled"
    assert payload["action"] == "discard"
    assert payload["explanation"]["citations"][0]["source_id"]


def test_request_id_is_carried_through(settings, kb_dir, image_bytes):
    result = build_agent(settings, kb_dir, {"fresh": 0.9, "spoiled": 0.1}).assess(
        image_bytes, request_id="req-123"
    )
    assert result.request_id == "req-123"
