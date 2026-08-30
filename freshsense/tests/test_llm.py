"""LLM client selection, JSON extraction and the offline explainer."""

from __future__ import annotations

import pytest

from freshsense.config import Settings
from freshsense.genai import prompts
from freshsense.genai.llm import LLMError, OfflineClient, _extract_json, build_llm_client


@pytest.mark.parametrize(
    "raw",
    [
        '{"summary": "s", "handling_guidance": "g", "sources": []}',
        '```json\n{"summary": "s", "handling_guidance": "g", "sources": []}\n```',
        'Sure:\n{"summary": "s", "handling_guidance": "g", "sources": []}\nHope that helps.',
        '  \n {"summary": "s", "handling_guidance": "g", "sources": []} \n ',
    ],
)
def test_json_is_extracted_from_realistic_completions(raw):
    assert _extract_json(raw)["summary"] == "s"


@pytest.mark.parametrize("raw", ["not json at all", "", "[1, 2, 3]", "{unclosed"])
def test_unparseable_completions_raise(raw):
    with pytest.raises(LLMError):
        _extract_json(raw)


def test_offline_client_is_deterministic(spoiled_prediction, citations):
    client = OfflineClient()
    first = client.explain(spoiled_prediction, citations)
    second = client.explain(spoiled_prediction, citations)
    assert first == second


def test_offline_client_cites_every_retrieved_source(spoiled_prediction, citations):
    payload = OfflineClient().explain(spoiled_prediction, citations)
    assert payload["sources"] == [c.source_id for c in citations]


def test_offline_client_covers_every_verdict(settings, citations):
    from freshsense.vision.predict import decide

    for probs in (
        {"fresh": 0.97, "spoiled": 0.03},
        {"fresh": 0.1, "spoiled": 0.9},
        {"fresh": 0.80, "spoiled": 0.20},
    ):
        payload = OfflineClient().explain(decide(probs, settings), citations)
        assert payload["summary"].strip() and payload["handling_guidance"].strip()


def test_offline_uncertain_text_says_escalate(settings, citations):
    from freshsense.vision.predict import decide

    payload = OfflineClient().explain(decide({"fresh": 0.8, "spoiled": 0.2}, settings), citations)
    assert "inspector" in payload["handling_guidance"]


def test_offline_output_survives_its_own_guardrails(settings, citations):
    """The fallback must never trip the rules it is the fallback for."""
    from freshsense.genai import guardrails
    from freshsense.vision.predict import decide

    for probs in (
        {"fresh": 0.97, "spoiled": 0.03},
        {"fresh": 0.1, "spoiled": 0.9},
        {"fresh": 0.80, "spoiled": 0.20},
    ):
        prediction = decide(probs, settings)
        payload = OfflineClient().explain(prediction, citations)
        result = guardrails.apply(payload, prediction.verdict, citations, payload)
        assert not any(n.startswith("blocked:") for n in result.notes)


def test_food_hint_is_prefixed(spoiled_prediction, citations):
    payload = OfflineClient().explain(spoiled_prediction, citations, food_hint="strawberries")
    assert payload["summary"].startswith("Strawberries:")


def test_client_selection_without_a_key_is_offline():
    client = build_llm_client(Settings(anthropic_api_key=None))
    assert client.mode == "offline"


def test_prompt_forbids_tasting_and_safety_claims():
    assert "taste" in prompts.SYSTEM_PROMPT.lower()
    assert "safe to eat" in prompts.SYSTEM_PROMPT.lower()
    assert prompts.PROMPT_VERSION


def test_user_prompt_carries_verdict_probabilities_and_context(spoiled_prediction, citations):
    text = prompts.build_user_prompt(spoiled_prediction, citations, "strawberries")
    assert "VERDICT: spoiled" in text
    assert "p(spoiled)=" in text
    assert "source_id=alpha#1" in text
    assert "strawberries" in text


def test_empty_context_is_stated_explicitly(spoiled_prediction):
    assert "no passages retrieved" in prompts.build_user_prompt(spoiled_prediction, ())
