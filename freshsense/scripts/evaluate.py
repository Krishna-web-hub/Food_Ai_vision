"""Evaluate a checkpoint on a held-out split and write the model card's numbers.

    python scripts/evaluate.py --data-dir ../PD-lab/dataset/test/spoilage_detection

Reports per-class recall/precision/F1 — not just accuracy, which on a 3:1 split
flatters a model that has learned to say "fresh". Also reports the numbers under
the *deployed decision policy*, including the abstention rate, because that is
what the service actually does.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from freshsense.config import get_settings
from freshsense.domain import Verdict
from freshsense.vision.model import eval_transform, load_classifier_head
from freshsense.vision.predict import decide
from freshsense.vision.train import metrics_from_confusion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="ImageFolder root with class subdirs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default="artifacts/eval_metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    settings = get_settings()
    checkpoint = Path(args.checkpoint) if args.checkpoint else settings.checkpoint
    model, class_names = load_classifier_head(checkpoint, device=settings.device)

    dataset = datasets.ImageFolder(root=args.data_dir, transform=eval_transform())
    if list(dataset.classes) != list(class_names):
        raise SystemExit(f"class mismatch: data={dataset.classes} checkpoint={list(class_names)}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    n = len(class_names)
    confusion = np.zeros((n, n), dtype=int)
    policy_counts: Counter[str] = Counter()
    missed_spoiled = 0

    with torch.inference_mode():
        for inputs, labels in loader:
            probs = F.softmax(model(inputs.to(settings.device)), dim=1)
            preds = probs.argmax(dim=1)
            for true, pred, row in zip(labels.numpy(), preds.cpu().numpy(), probs.cpu(), strict=True):
                confusion[true, pred] += 1
                decision = decide({name: float(row[i]) for i, name in enumerate(class_names)}, settings)
                policy_counts[decision.verdict.value] += 1
                if class_names[true] == "spoiled" and decision.verdict is Verdict.FRESH:
                    missed_spoiled += 1

    scores = metrics_from_confusion(confusion, list(class_names))
    total = int(confusion.sum())
    # Store a repo-relative path: an absolute one leaks the machine it ran on
    # into a file that gets committed.
    try:
        checkpoint_label = str(checkpoint.resolve().relative_to(Path.cwd()))
    except ValueError:
        checkpoint_label = checkpoint.name

    report = {
        "checkpoint": checkpoint_label,
        "data_dir": args.data_dir,
        "n": total,
        "class_counts": {c: int(confusion[i].sum()) for i, c in enumerate(class_names)},
        "confusion_matrix": confusion.tolist(),
        "argmax_metrics": {k: round(v, 4) for k, v in scores.items()},
        "policy": {
            "abstain_threshold": settings.abstain_threshold,
            "spoiled_threshold": settings.spoiled_threshold,
            "verdicts": dict(policy_counts),
            "abstention_rate": round(policy_counts["uncertain"] / max(total, 1), 4),
            "spoiled_released_as_fresh": missed_spoiled,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
