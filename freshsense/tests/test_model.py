"""Checkpointing and metric maths.

These avoid `pretrained=True` so the suite never downloads the 330 MB backbone.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freshsense.vision.model import (
    CHECKPOINT_FORMAT,
    IMAGE_SIZE,
    build_model,
    eval_transform,
    load_classifier_head,
    save_classifier_head,
)
from freshsense.vision.train import metrics_from_confusion


def test_eval_transform_produces_the_shape_vit_expects():
    from PIL import Image

    tensor = eval_transform()(Image.new("RGB", (400, 250)))
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_transform_normalises_away_from_the_unit_interval():
    """A ToTensor-only pipeline would leave everything in [0,1]; ViT needs ImageNet stats."""
    from PIL import Image

    tensor = eval_transform()(Image.new("RGB", (64, 64), (0, 0, 0)))
    assert tensor.min() < 0


@pytest.mark.slow
def test_head_checkpoint_is_small_and_self_describing(tmp_path):
    model = build_model(num_classes=2, pretrained=False)
    path = save_classifier_head(model, tmp_path / "head.pt", ("fresh", "spoiled"))

    assert path.stat().st_size < 100_000, "head checkpoints must not carry the backbone"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["format"] == CHECKPOINT_FORMAT
    assert blob["class_names"] == ["fresh", "spoiled"]
    assert blob["backbone"] == "vit_b_16"
    assert set(blob["head_state_dict"]) == {"weight", "bias"}


@pytest.mark.slow
def test_saved_head_weights_match_the_model(tmp_path):
    model = build_model(num_classes=2, pretrained=False)
    save_classifier_head(model, tmp_path / "head.pt", ("fresh", "spoiled"))
    blob = torch.load(tmp_path / "head.pt", map_location="cpu", weights_only=False)
    assert torch.equal(blob["head_state_dict"]["weight"], model.heads.head.weight)


def test_a_missing_checkpoint_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="freshsense train"):
        load_classifier_head(tmp_path / "absent.pt")


def test_metrics_penalise_the_lazy_majority_classifier():
    """456 fresh / 142 spoiled: always saying 'fresh' scores 76% accuracy and 0 recall."""
    confusion = np.array([[456, 0], [142, 0]])
    scores = metrics_from_confusion(confusion, ["fresh", "spoiled"])

    assert scores["accuracy"] == pytest.approx(456 / 598, abs=1e-3)
    assert scores["spoiled_recall"] == 0.0
    assert scores["macro_f1"] < 0.5, "macro-F1 must expose what accuracy hides"


def test_metrics_on_a_perfect_matrix():
    scores = metrics_from_confusion(np.array([[97, 0], [0, 30]]), ["fresh", "spoiled"])
    assert scores["accuracy"] == 1.0
    assert scores["macro_f1"] == 1.0


def test_metrics_reproduce_the_notebooks_reported_run():
    """The real 5-epoch run: 96/97 fresh and 29/30 spoiled on the validation split."""
    scores = metrics_from_confusion(np.array([[96, 1], [1, 29]]), ["fresh", "spoiled"])
    assert scores["accuracy"] == pytest.approx(125 / 127, abs=1e-4)
    assert scores["spoiled_recall"] == pytest.approx(29 / 30, abs=1e-4)


def test_metrics_handle_an_absent_class_without_dividing_by_zero():
    scores = metrics_from_confusion(np.array([[10, 0], [0, 0]]), ["fresh", "spoiled"])
    assert scores["spoiled_recall"] == 0.0
    assert scores["spoiled_precision"] == 0.0
