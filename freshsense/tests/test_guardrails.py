"""Guardrails: what the service refuses to say, regardless of what the model said."""

from __future__ import annotations

import pytest

from freshsense.domain import Verdict
from freshsense.genai import guardrails

FALLBACK = {
    "summary": "Offline summary describing visible spoilage findings.",
    "handling_guidance": "Offline guidance: isolate, bag and sanitise.",
}


def _apply(payload, verdict=Verdict.SPOILED, retrieved=()):
    return guardrails.apply(payload, verdict, retrieved, FALLBACK)


def test_a_well_formed_grounded_explanation_passes_through(citations):
    result = _apply(
        {
            "summary": "Visible mould is present.",
            "handling_guidance": "Discard it.",
            "sources": ["alpha#1"],
        },
        retrieved=citations,
    )
    assert result.grounded
    assert result.summary == "Visible mould is present."
    assert [c.source_id for c in result.citations] == ["alpha#1"]
    assert result.notes == ()


def test_hallucinated_citation_replaces_the_whole_explanation(citations):
    """A model inventing a source is not trustworthy on the prose either."""
    result = _apply(
        {
            "summary": "Per the FDA guidance.",
            "handling_guidance": "Do this.",
            "sources": ["alpha#1", "fda-2019#7"],
        },
        retrieved=citations,
    )
    assert not result.grounded
    assert result.summary == FALLBACK["summary"]
    assert any("fda-2019#7" in n for n in result.notes)


def test_missing_fields_fall_back(citations):
    result = _apply({"sources": ["alpha#1"]}, retrieved=citations)
    assert result.summary == FALLBACK["summary"]
    assert any("missing required fields" in n for n in result.notes)


def test_no_citations_despite_context_is_flagged(citations):
    result = _apply(
        {"summary": "Some claim.", "handling_guidance": "Some step.", "sources": []},
        retrieved=citations,
    )
    assert not result.grounded
    assert any("cited no sources" in n for n in result.notes)


@pytest.mark.parametrize(
    "text",
    [
        "You could taste-test a small amount to check.",
        "Try a small bite to see whether it has turned.",
        "This item is safe to eat.",
        "It is perfectly safe and there is no risk.",
        "We guarantee this produce is fresh.",
        "This is 100% safe.",
    ],
)
def test_forbidden_phrases_are_replaced_not_edited(text, citations):
    result = _apply(
        {"summary": text, "handling_guidance": "Proceed.", "sources": ["alpha#1"]},
        retrieved=citations,
    )
    assert not result.grounded
    assert result.summary == FALLBACK["summary"]
    assert any(n.startswith("blocked:") for n in result.notes)


def test_forbidden_phrase_in_guidance_is_caught_too(citations):
    result = _apply(
        {
            "summary": "Looks acceptable.",
            "handling_guidance": "Have a taste to confirm.",
            "sources": ["alpha#1"],
        },
        retrieved=citations,
    )
    assert not result.grounded


def test_fresh_verdict_always_carries_the_scope_disclaimer(citations):
    result = _apply(
        {
            "summary": "Nothing unusual was detected.",
            "handling_guidance": "Check the date.",
            "sources": ["alpha#1"],
        },
        verdict=Verdict.FRESH,
        retrieved=citations,
    )
    assert "decision support" in result.summary
    assert "temperature history" in result.summary


def test_disclaimer_is_not_duplicated_when_already_present(citations):
    """An explanation that already carries the caveat must not receive it twice."""
    result = _apply(
        {
            "summary": (
                "Nothing unusual was detected. This is decision support for a trained "
                "operator and does not clear the item on its date."
            ),
            "handling_guidance": "Check the date.",
            "sources": ["alpha#1"],
        },
        verdict=Verdict.FRESH,
        retrieved=citations,
    )
    assert result.summary.count("decision support for a trained operator") == 1


def test_offline_fresh_text_gets_exactly_one_disclaimer(citations, settings):
    """Regression: the offline summary and the appended disclaimer used to overlap."""
    from freshsense.genai.llm import OfflineClient
    from freshsense.vision.predict import decide

    prediction = decide({"fresh": 0.99, "spoiled": 0.01}, settings)
    payload = OfflineClient().explain(prediction, citations)
    result = guardrails.apply(payload, prediction.verdict, citations, payload)

    assert result.summary.count("decision support for a trained operator") == 1
    assert result.summary.count("temperature history") == 1


def test_overlong_text_is_truncated(citations):
    result = _apply(
        {"summary": "x" * 5000, "handling_guidance": "y" * 5000, "sources": ["alpha#1"]},
        retrieved=citations,
    )
    assert len(result.summary) <= guardrails.MAX_SUMMARY_CHARS
    assert len(result.handling_guidance) <= guardrails.MAX_GUIDANCE_CHARS


def test_non_string_fields_do_not_crash(citations):
    result = _apply(
        {"summary": {"nested": "object"}, "handling_guidance": 42, "sources": "not-a-list"},
        retrieved=citations,
    )
    assert isinstance(result.summary, str) and result.summary
