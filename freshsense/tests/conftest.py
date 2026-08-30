"""Shared fixtures.

Nothing here touches the network or the 330 MB ViT backbone: the classifier is
faked at the probability boundary, which is the only thing the layers above it
actually consume. That keeps the suite runnable in CI in seconds.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from freshsense.config import Settings
from freshsense.domain import Citation, Prediction, Verdict
from freshsense.vision.predict import decide


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        checkpoint=tmp_path / "vit_head.pt",
        abstain_threshold=0.85,
        spoiled_threshold=0.30,
        llm_min_confidence=0.55,
        anthropic_api_key=None,
        log_json=False,
    )


@pytest.fixture
def kb_dir(tmp_path):
    """A two-document knowledge base with predictable vocabulary."""
    d = tmp_path / "kb"
    d.mkdir()
    (d / "alpha.md").write_text(
        "# Alpha doc\n\n## Mould\nFuzzy growth means discard the item.\n\n"
        "## Slime\nSurface slime on meat is a bacterial biofilm.\n",
        encoding="utf-8",
    )
    (d / "beta.md").write_text(
        "# Beta doc\n\n## Escalation\nUncertain assessments go to a human inspector.\n",
        encoding="utf-8",
    )
    return d


def make_image(color=(200, 40, 40), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def image_bytes() -> bytes:
    return make_image()


class FakeClassifier:
    """Stands in for the ViT: returns fixed probabilities, applies the real policy."""

    def __init__(self, probabilities: dict[str, float], settings: Settings) -> None:
        self.probabilities_map = probabilities
        self.settings = settings
        self.calls = 0

    @property
    def ready(self) -> bool:
        return True

    def predict(self, image) -> Prediction:
        self.calls += 1
        return decide(self.probabilities_map, self.settings)


class FakeLLM:
    """Returns a canned payload, or raises, so the failure path is testable."""

    mode = "anthropic"

    def __init__(self, payload=None, raises: Exception | None = None) -> None:
        self.payload = payload
        self.raises = raises
        self.calls = 0

    def explain(self, prediction, citations, food_hint=None):
        self.calls += 1
        if self.raises:
            raise self.raises
        if self.payload is not None:
            return self.payload
        return {
            "summary": "A generated summary of the visible spoilage assessment.",
            "handling_guidance": "Isolate and discard the item, then sanitise the surfaces.",
            "sources": [c.source_id for c in citations],
        }


@pytest.fixture
def fresh_prediction(settings) -> Prediction:
    return decide({"fresh": 0.97, "spoiled": 0.03}, settings)


@pytest.fixture
def spoiled_prediction(settings) -> Prediction:
    return decide({"fresh": 0.20, "spoiled": 0.80}, settings)


@pytest.fixture
def citations() -> tuple[Citation, ...]:
    return (
        Citation(source_id="alpha#1", title="Alpha doc — Mould", text="Fuzzy growth.", score=0.5),
        Citation(source_id="beta#1", title="Beta doc — Escalation", text="Escalate.", score=0.3),
    )


@pytest.fixture
def verdicts():
    return Verdict
