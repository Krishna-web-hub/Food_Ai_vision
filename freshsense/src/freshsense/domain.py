"""Domain types shared by every layer.

These are plain, framework-free dataclasses on purpose: the vision code, the
agent graph and the CLI all speak them without importing FastAPI, and the API
schemas in `freshsense.api.schemas` are a thin serialisation shell over them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """What the service is willing to say about an image."""

    FRESH = "fresh"
    SPOILED = "spoiled"
    UNCERTAIN = "uncertain"  # below the abstain threshold — a human decides


class Action(str, Enum):
    """The operational recommendation that follows from a verdict."""

    RELEASE = "release"
    DISCARD = "discard"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True, slots=True)
class Prediction:
    """Raw classifier output plus the policy decision applied to it."""

    verdict: Verdict
    confidence: float
    probabilities: dict[str, float]
    policy_note: str

    @property
    def abstained(self) -> bool:
        return self.verdict is Verdict.UNCERTAIN


@dataclass(frozen=True, slots=True)
class Citation:
    """One retrieved knowledge-base passage the explanation is allowed to lean on."""

    source_id: str
    title: str
    text: str
    score: float

    def short(self, limit: int = 320) -> str:
        return self.text if len(self.text) <= limit else self.text[: limit - 1] + "…"


@dataclass(frozen=True, slots=True)
class Explanation:
    """The narrated half of an assessment. `grounded` is set by the guardrails."""

    summary: str
    handling_guidance: str
    citations: tuple[Citation, ...]
    llm_mode: str  # "anthropic" | "offline"
    grounded: bool = True
    guardrail_notes: tuple[str, ...] = ()


@dataclass(slots=True)
class Assessment:
    """The unit of work the whole system exists to produce."""

    request_id: str
    prediction: Prediction
    action: Action
    explanation: Explanation
    trace: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "verdict": self.prediction.verdict.value,
            "confidence": round(self.prediction.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.prediction.probabilities.items()},
            "action": self.action.value,
            "policy_note": self.prediction.policy_note,
            "explanation": {
                "summary": self.explanation.summary,
                "handling_guidance": self.explanation.handling_guidance,
                "llm_mode": self.explanation.llm_mode,
                "grounded": self.explanation.grounded,
                "guardrail_notes": list(self.explanation.guardrail_notes),
                "citations": [
                    {
                        "source_id": c.source_id,
                        "title": c.title,
                        "score": round(c.score, 4),
                        "excerpt": c.short(),
                    }
                    for c in self.explanation.citations
                ],
            },
            "trace": self.trace,
            "latency_ms": round(self.latency_ms, 2),
        }
