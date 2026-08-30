"""End-to-end tests against the real ViT and the real dataset.

Skipped automatically when the checkpoint or the dataset is absent, so the fast
suite still runs anywhere. These are the tests that would catch a checkpoint
that loads but predicts nonsense — the unit suite cannot, by construction.

    pytest -m slow          # run only these
    pytest -m "not slow"    # what CI runs by default
"""

from __future__ import annotations

from pathlib import Path

import pytest

from freshsense.agent.pipeline import TriageAgent
from freshsense.config import Settings
from freshsense.domain import Verdict
from freshsense.vision.predict import Classifier

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "artifacts" / "vit_head.pt"
DATASET = REPO_ROOT.parent / "PD-lab" / "dataset" / "test" / "spoilage_detection"

needs_model = pytest.mark.skipif(not CHECKPOINT.exists(), reason="no checkpoint; run `make model`")
needs_data = pytest.mark.skipif(not DATASET.exists(), reason="PD-lab dataset not present")


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    return Settings(checkpoint=CHECKPOINT, anthropic_api_key=None, log_json=False)


@pytest.fixture(scope="module")
def real_classifier(real_settings) -> Classifier:
    return Classifier(real_settings).load()


def _sample(label: str, n: int = 5) -> list[Path]:
    return sorted((DATASET / label).glob("*"))[:n]


@needs_model
def test_the_shipped_checkpoint_loads_with_the_expected_classes(real_classifier):
    assert real_classifier.class_names == ("fresh", "spoiled")


@needs_model
@needs_data
def test_probabilities_are_a_distribution(real_classifier):
    from PIL import Image

    probs = real_classifier.probabilities(Image.open(_sample("fresh")[0]).convert("RGB"))
    assert set(probs) == {"fresh", "spoiled"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)


@needs_model
@needs_data
def test_the_model_separates_the_two_classes_on_held_out_data(real_classifier):
    """A sanity bar, not a benchmark: mean p(spoiled) must be higher on spoiled items."""
    from PIL import Image

    def mean_p_spoiled(label: str) -> float:
        paths = _sample(label, 8)
        scores = [
            real_classifier.probabilities(Image.open(p).convert("RGB"))["spoiled"] for p in paths
        ]
        return sum(scores) / len(scores)

    assert mean_p_spoiled("spoiled") > mean_p_spoiled("fresh") + 0.2


@needs_model
@needs_data
def test_no_held_out_spoiled_item_is_released_as_fresh(real_classifier, real_settings):
    """The error the safety policy exists to prevent. Sampled, not exhaustive."""
    from PIL import Image

    released = [
        p.name
        for p in _sample("spoiled", 10)
        if real_classifier.predict(Image.open(p).convert("RGB")).verdict is Verdict.FRESH
    ]
    assert not released, f"spoiled items cleared for release: {released}"


@needs_model
@needs_data
def test_full_agent_run_on_a_real_photograph(real_settings):
    agent = TriageAgent(settings=real_settings)
    result = agent.assess(_sample("spoiled")[0].read_bytes(), food_hint="strawberries")

    assert result.prediction.verdict in {Verdict.SPOILED, Verdict.UNCERTAIN}
    assert result.explanation.summary.strip()
    assert result.explanation.citations
    assert result.explanation.grounded
    assert len(result.trace) == 6
    assert result.latency_ms > 0


@needs_model
def test_a_blank_image_does_not_crash_the_pipeline(real_settings):
    """Off-distribution input must produce a verdict, not an exception."""
    from .conftest import make_image

    result = TriageAgent(settings=real_settings).assess(make_image(color=(255, 255, 255)))
    assert result.prediction.verdict in set(Verdict)
    assert result.explanation.handling_guidance.strip()


@needs_data
def test_a_two_epoch_training_run_learns_something(tmp_path):
    """Trains on a small slice: proves the loop runs and the checkpoint is loadable."""
    from freshsense.vision.model import load_classifier_head
    from freshsense.vision.train import train

    train_dir = REPO_ROOT.parent / "PD-lab" / "dataset" / "train" / "spoilage_detection"
    val_dir = REPO_ROOT.parent / "PD-lab" / "dataset" / "val" / "spoilage_detection"
    if not train_dir.exists():
        pytest.skip("training split not present")

    out = tmp_path / "head.pt"
    history = train(
        train_dir=train_dir,
        val_dir=val_dir,
        checkpoint_out=out,
        epochs=2,
        batch_size=16,
        device="cpu",
        metrics_out=tmp_path / "metrics.json",
    )

    assert len(history) == 2
    assert out.exists() and out.stat().st_size < 100_000
    assert (tmp_path / "metrics.json").exists()

    model, classes = load_classifier_head(out)
    assert classes == ("fresh", "spoiled")
    assert history[-1].val_spoiled_recall > 0.5
