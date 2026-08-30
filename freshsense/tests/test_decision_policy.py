"""The asymmetric safety policy — the most consequential logic in the service."""

from __future__ import annotations

import pytest

from freshsense.domain import Action, Verdict
from freshsense.vision.predict import action_for, decide


def test_clear_spoilage_is_condemned(settings):
    p = decide({"fresh": 0.05, "spoiled": 0.95}, settings)
    assert p.verdict is Verdict.SPOILED
    assert action_for(p.verdict) is Action.DISCARD


def test_clear_freshness_is_released(settings):
    p = decide({"fresh": 0.97, "spoiled": 0.03}, settings)
    assert p.verdict is Verdict.FRESH
    assert action_for(p.verdict) is Action.RELEASE


def test_weak_spoilage_signal_still_condemns(settings):
    """The whole point of the asymmetry: argmax would say 'fresh' at p=0.35."""
    p = decide({"fresh": 0.65, "spoiled": 0.35}, settings)
    assert p.verdict is Verdict.SPOILED, "a 0.35 spoilage signal must not be released"


def test_abstain_band_routes_to_a_human(settings):
    p = decide({"fresh": 0.80, "spoiled": 0.20}, settings)
    assert p.verdict is Verdict.UNCERTAIN
    assert p.abstained
    assert action_for(p.verdict) is Action.MANUAL_REVIEW


@pytest.mark.parametrize("p_spoiled", [0.0, 0.14, 0.15, 0.16, 0.29, 0.30, 0.31, 1.0])
def test_policy_is_monotone_in_spoilage(settings, p_spoiled):
    """Raising p(spoiled) may only make the outcome more cautious, never less."""
    order = {Verdict.FRESH: 0, Verdict.UNCERTAIN: 1, Verdict.SPOILED: 2}
    lower = decide({"fresh": 1 - p_spoiled, "spoiled": p_spoiled}, settings)
    bumped = min(p_spoiled + 0.05, 1.0)
    higher = decide({"fresh": 1 - bumped, "spoiled": bumped}, settings)
    assert order[higher.verdict] >= order[lower.verdict]


def test_boundaries_are_inclusive_on_the_cautious_side(settings):
    assert decide({"fresh": 0.70, "spoiled": 0.30}, settings).verdict is Verdict.SPOILED
    assert decide({"fresh": 0.85, "spoiled": 0.15}, settings).verdict is Verdict.FRESH


def test_policy_note_is_populated(settings):
    for probs in (
        {"fresh": 0.9, "spoiled": 0.1},
        {"fresh": 0.1, "spoiled": 0.9},
        {"fresh": 0.8, "spoiled": 0.2},
    ):
        assert decide(probs, settings).policy_note.strip()
