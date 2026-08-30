"""The live-client path, with the SDK faked at the transport boundary."""

from __future__ import annotations

import sys
import types

import pytest

from freshsense.config import Settings
from freshsense.genai.llm import AnthropicClient, LLMError, build_llm_client


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.usage = types.SimpleNamespace(input_tokens=120, output_tokens=45)


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install a stand-in `anthropic` module that records the call it receives."""
    calls = {}

    class Messages:
        def create(self, **kwargs):
            calls.update(kwargs)
            if calls.get("_fail"):
                raise RuntimeError("overloaded")
            return _Response(calls["_reply"])

    class Anthropic:
        def __init__(self, api_key=None, timeout=None):
            calls["api_key"] = api_key
            calls["timeout"] = timeout
            self.messages = Messages()

    module = types.ModuleType("anthropic")
    module.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    calls["_reply"] = '{"summary": "s", "handling_guidance": "g", "sources": ["alpha#1"]}'
    return calls


def _settings() -> Settings:
    return Settings(anthropic_api_key="sk-ant-test", llm_model="claude-sonnet-5")


def test_build_llm_client_returns_the_live_client_when_a_key_is_set(fake_sdk):
    assert build_llm_client(_settings()).mode == "anthropic"


def test_the_request_carries_the_configured_model_and_system_prompt(
    fake_sdk, spoiled_prediction, citations
):
    AnthropicClient(_settings()).explain(spoiled_prediction, citations)

    assert fake_sdk["model"] == "claude-sonnet-5"
    assert "never contradict" in fake_sdk["system"]
    assert fake_sdk["messages"][0]["role"] == "user"
    assert "source_id=alpha#1" in fake_sdk["messages"][0]["content"]


def test_token_usage_is_returned_for_cost_accounting(fake_sdk, spoiled_prediction, citations):
    payload = AnthropicClient(_settings()).explain(spoiled_prediction, citations)
    assert payload["_usage"] == {"input_tokens": 120, "output_tokens": 45}


def test_a_fenced_reply_is_still_parsed(fake_sdk, spoiled_prediction, citations):
    fake_sdk["_reply"] = '```json\n{"summary": "s", "handling_guidance": "g", "sources": []}\n```'
    assert AnthropicClient(_settings()).explain(spoiled_prediction, citations)["summary"] == "s"


def test_a_non_json_reply_raises_llm_error(fake_sdk, spoiled_prediction, citations):
    fake_sdk["_reply"] = "I'm afraid I can't help with that."
    with pytest.raises(LLMError, match="parseable JSON"):
        AnthropicClient(_settings()).explain(spoiled_prediction, citations)


def test_a_transport_failure_raises_llm_error(fake_sdk, spoiled_prediction, citations):
    fake_sdk["_fail"] = True
    with pytest.raises(LLMError, match="Anthropic call failed"):
        AnthropicClient(_settings()).explain(spoiled_prediction, citations)


def test_a_missing_sdk_degrades_to_offline(monkeypatch):
    """`pip install freshsense` without the llm extra must still start."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("no module named anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert build_llm_client(_settings()).mode == "offline"
