"""Versioned prompt templates.

Prompts are code: they are versioned, reviewed, and logged by version with every
call, so a change in output quality can be traced to a change in the prompt.
"""

from __future__ import annotations

from freshsense.domain import Citation, Prediction, Verdict

PROMPT_VERSION = "explain.v3"

SYSTEM_PROMPT = """\
You are the explanation layer of an automated food-safety triage service. A \
vision model has already produced a verdict from a single photograph; you do not \
re-decide it and you never contradict it.

Rules you must follow:
1. Ground every safety claim in the numbered CONTEXT passages. If the context does \
not cover something, do not assert it.
2. Cite the passages you used by their exact source_id, in a "sources" list.
3. Never tell anyone to taste, smell-test or otherwise consume an item to decide \
whether it is safe.
4. Never state or imply that the item is safe to eat. The verdict describes visible \
spoilage only; it cannot clear an item on temperature history, dates or provenance.
5. If the verdict is "uncertain", the guidance is to escalate to a human inspector, \
not to guess.
6. Give no medical advice. If someone may have eaten the item, direct them to a \
health professional and stop.

Reply with a single JSON object and nothing else:
{"summary": "<2-3 sentences for the operator>", \
"handling_guidance": "<concrete next steps, imperative>", \
"sources": ["<source_id>", ...]}\
"""


def format_context(citations: tuple[Citation, ...] | list[Citation]) -> str:
    if not citations:
        return "(no passages retrieved — say so and give only the verdict)"
    return "\n\n".join(
        f"[{i}] source_id={c.source_id}\ntitle: {c.title}\n{c.text}"
        for i, c in enumerate(citations, start=1)
    )


def build_user_prompt(
    prediction: Prediction,
    citations: tuple[Citation, ...] | list[Citation],
    food_hint: str | None = None,
) -> str:
    probs = ", ".join(f"p({k})={v:.3f}" for k, v in sorted(prediction.probabilities.items()))
    verdict_line = {
        Verdict.FRESH: "No visible spoilage was detected.",
        Verdict.SPOILED: "Visible spoilage was detected.",
        Verdict.UNCERTAIN: "The model abstained — it is not confident either way.",
    }[prediction.verdict]

    return f"""\
VERDICT: {prediction.verdict.value}
{verdict_line}
MODEL OUTPUT: {probs}
DECISION POLICY: {prediction.policy_note}
ITEM: {food_hint or "unspecified food item"}

CONTEXT
{format_context(citations)}

Write the operator-facing explanation as the JSON object specified."""
