"""Reusable fine-tuning entrypoint — the notebook's training loop, made callable.

Differences from the notebook, and why:

* **Class weighting.** The train split is 456 fresh / 142 spoiled. Unweighted
  cross-entropy lets the model buy accuracy by leaning fresh, which is exactly
  the error this domain cannot afford.
* **Selection on recall, not accuracy.** The best epoch is the one with the best
  spoiled-recall, tie-broken on macro-F1; 98% accuracy on a 3:1 split is not the
  number that matters.
* **Head-only checkpoints.** 8 KB instead of 343 MB.
* **Seeded.** Same seed, same numbers.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from freshsense.vision.model import (
    build_model,
    eval_transform,
    save_classifier_head,
    train_transform,
)

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_spoiled_recall: float
    val_fresh_recall: float
    val_macro_f1: float


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _class_weights(dataset: datasets.ImageFolder, device: str) -> torch.Tensor:
    """Inverse-frequency ("balanced") weights: w_i = N / (K * count_i).

    The sample-weighted average of these is exactly 1 — sum(count_i * w_i) == N —
    so the loss stays on the same scale as the unweighted version and the
    learning rate does not have to be retuned when the class balance shifts.
    """
    counts = np.bincount(np.array(dataset.targets), minlength=len(dataset.classes)).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _evaluate(model, loader, criterion, device: str, num_classes: int):
    """One eval pass returning loss plus a confusion matrix."""
    model.eval()
    total_loss, seen = 0.0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            seen += labels.size(0)
            predicted = outputs.argmax(dim=1)
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy(), strict=True):
                confusion[t, p] += 1

    return total_loss / max(seen, 1), confusion


def metrics_from_confusion(confusion: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Per-class recall/precision/F1 plus macro-F1 and accuracy."""
    out: dict[str, float] = {}
    f1s = []
    for i, name in enumerate(class_names):
        tp = int(confusion[i, i])
        fn = int(confusion[i].sum() - tp)
        fp = int(confusion[:, i].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"{name}_recall"] = recall
        out[f"{name}_precision"] = precision
        out[f"{name}_f1"] = f1
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    out["accuracy"] = float(np.trace(confusion) / max(confusion.sum(), 1))
    return out


def train(
    train_dir: Path,
    val_dir: Path,
    checkpoint_out: Path,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    device: str = "cpu",
    num_workers: int = 0,
    seed: int = 1337,
    pretrained: bool = True,
    metrics_out: Path | None = None,
) -> list[EpochMetrics]:
    """Fine-tune the ViT head and save the best epoch by spoiled-recall."""
    set_seed(seed)
    t0 = time.time()

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_transform())
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=eval_transform())
    if train_ds.classes != val_ds.classes:
        raise ValueError(f"class mismatch: train={train_ds.classes} val={val_ds.classes}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_ds.classes
    num_classes = len(class_names)
    logger.info(
        "dataset loaded",
        extra={"classes": class_names, "train_n": len(train_ds), "val_n": len(val_ds)},
    )

    model = build_model(num_classes=num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_ds, device))
    optimizer = optim.AdamW(model.heads.head.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    spoiled_idx = class_names.index("spoiled") if "spoiled" in class_names else num_classes - 1
    history: list[EpochMetrics] = []
    best_key = (-1.0, -1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += int((outputs.argmax(dim=1) == labels).sum())
            seen += labels.size(0)

        scheduler.step()
        val_loss, confusion = _evaluate(model, val_loader, criterion, device, num_classes)
        scores = metrics_from_confusion(confusion, class_names)

        row = EpochMetrics(
            epoch=epoch,
            train_loss=running_loss / max(seen, 1),
            train_acc=correct / max(seen, 1),
            val_loss=val_loss,
            val_acc=scores["accuracy"],
            val_spoiled_recall=scores.get(f"{class_names[spoiled_idx]}_recall", 0.0),
            val_fresh_recall=scores.get("fresh_recall", 0.0),
            val_macro_f1=scores["macro_f1"],
        )
        history.append(row)
        logger.info("epoch complete", extra=asdict(row))

        key = (row.val_spoiled_recall, row.val_macro_f1)
        if key > best_key:
            best_key = key
            save_classifier_head(model, checkpoint_out, tuple(class_names))
            logger.info("new best checkpoint", extra={"epoch": epoch, "selected_on": key})

    if metrics_out:
        Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_out).write_text(
            json.dumps(
                {
                    "history": [asdict(r) for r in history],
                    "best_spoiled_recall": best_key[0],
                    "best_macro_f1": best_key[1],
                    "class_names": class_names,
                    "seed": seed,
                    "epochs": epochs,
                    "wall_seconds": round(time.time() - t0, 1),
                },
                indent=2,
            )
        )

    logger.info("training complete", extra={"seconds": round(time.time() - t0, 1)})
    return history
