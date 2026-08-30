"""Request/response models. The public contract, separate from the domain types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    source_id: str = Field(examples=["04-disposition-and-handling#1"])
    title: str
    score: float
    excerpt: str


class ExplanationOut(BaseModel):
    summary: str
    handling_guidance: str
    llm_mode: Literal["anthropic", "offline"]
    grounded: bool = Field(description="False if a guardrail rewrote or blocked the explanation")
    guardrail_notes: list[str] = []
    citations: list[CitationOut] = []


class AssessmentOut(BaseModel):
    request_id: str
    verdict: Literal["fresh", "spoiled", "uncertain"]
    confidence: float
    probabilities: dict[str, float]
    action: Literal["release", "discard", "manual_review"]
    policy_note: str
    explanation: ExplanationOut
    trace: list[dict[str, Any]] = Field(default=[], description="Per-node execution trace")
    latency_ms: float


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str


class ReadyOut(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]
    llm_mode: Literal["anthropic", "offline"]


class ErrorOut(BaseModel):
    error: str
    detail: str
    request_id: str
