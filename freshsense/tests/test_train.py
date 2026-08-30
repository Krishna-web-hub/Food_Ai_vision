"""Training-loop helpers that do not require running a training loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freshsense.vision.train import _class_weights, set_seed


class _FakeDataset:
    def __init__(self, targets, classes):
        self.targets = targets
        self.classes = classes


def test_class_weights_favour_the_minority_class():
    """456 fresh / 142 spoiled — the split the real dataset actually has."""
    dataset = _FakeDataset([0] * 456 + [1] * 142, ["fresh", "spoiled"])
    weights = _class_weights(dataset, "cpu")

    assert weights[1] > weights[0], "spoiled must be weighted above fresh"
    assert float(weights[1] / weights[0]) == pytest.approx(456 / 142, rel=1e-3)


def test_loss_scale_is_preserved_so_the_learning_rate_still_applies():
    """sum(count_i * w_i) == N, i.e. the sample-weighted mean weight is exactly 1."""
    counts = [456, 142]
    dataset = _FakeDataset([0] * counts[0] + [1] * counts[1], ["fresh", "spoiled"])
    weights = _class_weights(dataset, "cpu")

    total = sum(c * float(w) for c, w in zip(counts, weights, strict=True))
    assert total == pytest.approx(sum(counts), rel=1e-5)


def test_balanced_data_gets_uniform_weights():
    dataset = _FakeDataset([0] * 50 + [1] * 50, ["fresh", "spoiled"])
    weights = _class_weights(dataset, "cpu")
    assert float(weights[0]) == pytest.approx(float(weights[1]))


def test_an_empty_class_does_not_produce_infinite_weight():
    dataset = _FakeDataset([0] * 20, ["fresh", "spoiled"])
    assert torch.isfinite(_class_weights(dataset, "cpu")).all()


def test_seeding_makes_runs_reproducible():
    set_seed(1337)
    first = (torch.randn(4).tolist(), np.random.rand(4).tolist())
    set_seed(1337)
    second = (torch.randn(4).tolist(), np.random.rand(4).tolist())
    assert first == second
