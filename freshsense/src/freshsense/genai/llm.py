"""LLM access, behind one small interface with a deterministic offline twin.

`build_llm_client` returns the Anthropic client when a key is configured and the
offline explainer when it is not. The offline path is not a stub — it is a
supported mode that produces a correct, cited, guard-railed explanation with no
network, which is what keeps the test suite hermetic and the service usable in
an air-gapped deployment.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from freshsense.config import Settings, get_settings
from freshsense.domain import Citation, Prediction, Verdict
from freshsense.genai import prompts

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Any failure to obtain a usable completion."""


class LLMClient(Protocol):
    mode: str

    def explain(
        self,
        prediction: Prediction,
        citations: tuple[Citation, ...],
        food_hint: str | None = None,
    ) -> dict[str, Any]: ...


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a completion, tolerating fences and preamble."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"model did not return parseable JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("model returned JSON that was not an object")
    return parsed


class AnthropicClient:
    """Live Claude client. Falls back to the offline explainer on any failure."""

    mode = "anthropic"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError("anthropic SDK not installed; `pip install '.[llm]'`") from exc
        self._client = Anthropic(
            api_key=self.settings.anthropic_api_key, timeout=self.settings.llm_timeout_s
        )

    def explain(
        self,
        prediction: Prediction,
        citations: tuple[Citation, ...],
        food_hint: str | None = None,
    ) -> dict[str, Any]:
        user_prompt = prompts.build_user_prompt(prediction, citations, food_hint)
        logger.info(
            "llm request",
            extra={"model": self.settings.llm_model, "prompt_version": prompts.PROMPT_VERSION},
        )
        try:
            response = self._client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                system=prompts.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:  # network, auth, rate limit, overload
            raise LLMError(f"Anthropic call failed: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        payload = _extract_json(text)
        payload["_usage"] = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        return payload


class OfflineClient:
    """Deterministic, fully grounded explainer. No network, no key, no variance."""

    mode = "offline"

    _SUMMARY = {
        Verdict.SPOILED: (
            "The classifier found visible indicators of spoilage in this image "
            "(p(spoiled)={p_spoiled:.2f}). Under the service's safety policy, spoilage is "
            "acted on even at modest confidence, so this item is condemned on appearance."
        ),
        # No scope caveat here on purpose: the guardrail owns the disclaimer and
        # appends it to every fresh verdict, so stating it twice is the bug.
        Verdict.FRESH: (
            "The classifier found no visible indicators of spoilage "
            "(p(fresh)={p_fresh:.2f}), which clears the bar required to release an item."
        ),
        Verdict.UNCERTAIN: (
            "The classifier is not confident either way (p(spoiled)={p_spoiled:.2f}), which "
            "falls inside the abstain band. The service will not issue a verdict on this "
            "image and is routing it to a human inspector."
        ),
    }

    _GUIDANCE = {
        Verdict.SPOILED: (
            "Do not taste the item. Isolate it from surrounding stock, bag it before "
            "disposal, then clean and sanitise every surface and container it touched. "
            "Inspect adjacent items on the same shelf, since spoilage spreads by contact "
            "and shared humidity."
        ),
        Verdict.FRESH: (
            "Before use, confirm the item is within its use-by date and has held "
            "refrigeration below 5 °C, and check for off odours. A use-by date is a safety "
            "limit that a clean visual inspection does not override."
        ),
        Verdict.UNCERTAIN: (
            "Hold the item and escalate to a trained inspector. Do not release it on this "
            "assessment, and do not discard it automatically — the abstention exists so a "
            "person applies judgement."
        ),
    }

    def explain(
        self,
        prediction: Prediction,
        citations: tuple[Citation, ...],
        food_hint: str | None = None,
    ) -> dict[str, Any]:
        probs = prediction.probabilities
        fields = {"p_spoiled": probs.get("spoiled", 0.0), "p_fresh": probs.get("fresh", 0.0)}
        summary = self._SUMMARY[prediction.verdict].format(**fields)
        if food_hint:
            summary = f"{food_hint.strip().capitalize()}: {summary}"
        return {
            "summary": summary,
            "handling_guidance": self._GUIDANCE[prediction.verdict],
            "sources": [c.source_id for c in citations],
        }


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    """Pick a client from configuration; degrade to offline rather than fail."""
    settings = settings or get_settings()
    if not settings.llm_enabled:
        logger.info("no ANTHROPIC_API_KEY set — using the offline explainer")
        return OfflineClient()
    try:
        return AnthropicClient(settings)
    except LLMError as exc:
        logger.warning("falling back to offline explainer", extra={"reason": str(exc)})
        return OfflineClient()
