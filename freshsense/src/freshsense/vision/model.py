"""ViT-B/16 with a swapped classification head, plus head-only checkpointing.

Why head-only: the notebook saved `model.state_dict()` — 343 MB — even though
training touched 1,538 parameters. The backbone is public ImageNet weights, so
only the head is ours; persisting just the head takes the artefact to ~8 KB and
makes the model something you can actually put in a container image or a
release asset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

CHECKPOINT_FORMAT = "freshsense.head.v1"


def eval_transform() -> transforms.Compose:
    """Inference-time preprocessing. Must match training exactly or scores drift."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = True):
    """ViT-B/16 with `model.heads.head` replaced by a `num_classes` linear layer.

    `pretrained=False` skips the 330 MB download — used by tests and by any path
    that is about to load a full state dict anyway.
    """
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def save_classifier_head(model: nn.Module, path: Path, class_names: tuple[str, ...]) -> Path:
    """Persist only the trained head, with the metadata needed to rebuild it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "class_names": list(class_names),
        "backbone": "vit_b_16",
        "head_state_dict": model.heads.head.state_dict(),
    }
    torch.save(payload, path)
    logger.info("saved head checkpoint", extra={"path": str(path), "bytes": path.stat().st_size})
    return path


def load_classifier_head(path: Path, device: str = "cpu") -> tuple[nn.Module, tuple[str, ...]]:
    """Rebuild a ready-to-infer model from a head-only checkpoint.

    Also accepts a legacy full `state_dict` (what the notebook produced) so old
    artefacts keep working; that path is logged loudly enough to notice.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train one with `freshsense train`, or export "
            f"an existing full state dict with `python scripts/export_head.py`."
        )

    blob = torch.load(path, map_location=device, weights_only=False)

    if isinstance(blob, dict) and blob.get("format") == CHECKPOINT_FORMAT:
        class_names = tuple(blob["class_names"])
        model = build_model(num_classes=len(class_names), pretrained=True)
        model.heads.head.load_state_dict(blob["head_state_dict"])
    else:
        logger.warning("loading legacy full state dict; run scripts/export_head.py to shrink it")
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        head_weight = state["heads.head.weight"]
        class_names = ("fresh", "spoiled")[: head_weight.shape[0]]
        model = build_model(num_classes=head_weight.shape[0], pretrained=False)
        model.load_state_dict(state)

    model.to(device).eval()
    return model, class_names
