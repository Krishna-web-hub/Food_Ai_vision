"""Configuration validation — the checks that stop a silently-wrong deployment."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from freshsense.config import Settings


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_thresholds_must_be_in_the_unit_interval(value):
    with pytest.raises(ValidationError):
        Settings(abstain_threshold=value)


def test_empty_abstain_band_is_rejected():
    """abstain=0.65 with spoiled=0.30 leaves no room to ever return UNCERTAIN."""
    with pytest.raises(ValidationError, match="never return UNCERTAIN"):
        Settings(abstain_threshold=0.65, spoiled_threshold=0.30)


def test_valid_band_is_accepted():
    s = Settings(abstain_threshold=0.85, spoiled_threshold=0.30)
    assert s.abstain_threshold > 1 - s.spoiled_threshold


def test_llm_enabled_follows_the_api_key():
    assert Settings(anthropic_api_key=None).llm_enabled is False
    assert Settings(anthropic_api_key="sk-ant-test").llm_enabled is True
