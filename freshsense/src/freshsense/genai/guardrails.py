"""Validation applied to every explanation before it leaves the service.

The premise: an LLM's output is an *input* to your system, not its conclusion.
Nothing here trusts the model — each check is mechanical and testable, and a
failed check downgrades or replaces the text rather than blocking the response,
because the verdict is still worth returning even when the prose is not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from freshsense.domain import Citation, Verdict

logger = logging.getLogger(__name__)

MAX_SUMMARY_CHARS = 1200
MAX_GUIDANCE_CHARS = 1600

# Phrases that must never reach an operator, whatever the model produced.
_FORBIDDEN = [
    (
        re.compile(r"\btaste[- ]?test|\btry a (?:small )?(?:bite|piece)|\bhave a taste", re.I),
        "suggested tasting the item to assess safety",
    ),
    (
        re.compile(r"\b(?:safe|fine|ok(?:ay)?) to eat\b|\bperfectly safe\b|\bno risk\b", re.I),
        "asserted the item is safe to eat",
    ),
    (
        re.compile(r"\bguarantee[ds]?\b|\bcertainly (?:safe|fresh)\b|\b100% (?:safe|sure)\b", re.I),
        "used absolute-certainty language",
    ),
    (
        re.compile(
            r"\byou (?:should )?(?:see a doctor|take|consult).{0,20}\b(?:antibiotic|medication)",
            re.I,
        ),
        "gave medical advice",
    ),
]

_DISCLAIMER_MARKER = "decision support for a trained operator"

_DISCLAIMER = (
    "This assessment reflects visible spoilage in a single photograph. It is decision "
    "support for a trained operator and cannot clear an item on temperature history, "
    "use-by date, packaging integrity or provenance."
)


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    summary: str
    handling_guidance: str
    citations: tuple[Citation, ...]
    grounded: bool
    notes: tuple[str, ...]


def _coerce_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit].rstrip()


def apply(
    payload: dict[str, Any],
    verdict: Verdict,
    retrieved: tuple[Citation, ...],
    fallback: dict[str, Any],
) -> GuardrailResult:
    """Validate a raw explanation payload against the retrieved context.

    Four checks, in order:

    1. **Shape.** Missing or non-string fields fall back to the offline text.
    2. **Grounding.** Every cited source_id must be one we actually retrieved; a
       hallucinated citation drops the whole explanation to the offline text,
       because a model inventing sources is not trustworthy on the rest either.
    3. **Forbidden content.** Tasting advice, safety assurances, absolute claims
       and medical advice are replaced wholesale, not edited around.
    4. **Disclaimer.** A `fresh` verdict always carries its scope limitation.
    """
    notes: list[str] = []
    grounded = True

    summary = _coerce_text(payload.get("summary"), MAX_SUMMARY_CHARS)
    guidance = _coerce_text(payload.get("handling_guidance"), MAX_GUIDANCE_CHARS)

    if not summary or not guidance:
        notes.append("explanation was missing required fields; used the offline explainer")
        summary = _coerce_text(fallback.get("summary"), MAX_SUMMARY_CHARS)
        guidance = _coerce_text(fallback.get("handling_guidance"), MAX_GUIDANCE_CHARS)

    # --- grounding ---
    valid_ids = {c.source_id for c in retrieved}
    claimed = payload.get("sources") or []
    if not isinstance(claimed, list):
        claimed = []
    claimed_ids = {str(s) for s in claimed}
    invented = claimed_ids - valid_ids

    if invented:
        grounded = False
        notes.append(f"cited unknown sources {sorted(invented)}; replaced with the offline text")
        logger.warning("ungrounded citation", extra={"invented": sorted(invented)})
        summary = _coerce_text(fallback.get("summary"), MAX_SUMMARY_CHARS)
        guidance = _coerce_text(fallback.get("handling_guidance"), MAX_GUIDANCE_CHARS)
        cited = retrieved
    else:
        cited = tuple(c for c in retrieved if c.source_id in claimed_ids) or retrieved
        if retrieved and not claimed_ids:
            grounded = False
            notes.append("explanation cited no sources despite retrieved context")

    # --- forbidden content ---
    for pattern, why in _FORBIDDEN:
        if pattern.search(summary) or pattern.search(guidance):
            grounded = False
            notes.append(f"blocked: {why}")
            logger.warning("guardrail block", extra={"reason": why})
            summary = _coerce_text(fallback.get("summary"), MAX_SUMMARY_CHARS)
            guidance = _coerce_text(fallback.get("handling_guidance"), MAX_GUIDANCE_CHARS)
            break

    # --- mandatory disclaimer on a clearance ---
    # Keyed on a phrase the disclaimer itself contains, so an explanation that
    # already carries it (the LLM was asked to) does not get it twice.
    if verdict is Verdict.FRESH and _DISCLAIMER_MARKER not in summary.lower():
        summary = f"{summary} {_DISCLAIMER}".strip()
        notes.append("appended the scope disclaimer to a fresh verdict")

    return GuardrailResult(
        summary=summary,
        handling_guidance=guidance,
        citations=cited,
        grounded=grounded,
        notes=tuple(notes),
    )
