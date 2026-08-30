"""Convert a legacy full ViT state dict into a head-only FreshSense checkpoint.

    python scripts/export_head.py ../vit_food_detection_model.pth artifacts/vit_head.pt

343 MB -> ~8 KB. The backbone is public ImageNet weights, so the only thing
worth persisting is the head we actually trained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from freshsense.vision.model import build_model, save_classifier_head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="full state_dict .pth from the notebook")
    parser.add_argument("dest", nargs="?", default="artifacts/vit_head.pt")
    parser.add_argument("--classes", nargs="+", default=["fresh", "spoiled"])
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"no such file: {source}", file=sys.stderr)
        return 2

    blob = torch.load(source, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    if "heads.head.weight" not in state:
        print("not a ViT classifier state dict: no 'heads.head.weight'", file=sys.stderr)
        return 2

    num_classes = state["heads.head.weight"].shape[0]
    if num_classes != len(args.classes):
        print(
            f"checkpoint has {num_classes} classes but --classes gave {len(args.classes)}",
            file=sys.stderr,
        )
        return 2

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state)

    dest = save_classifier_head(model, Path(args.dest), tuple(args.classes))
    print(
        f"{source.name}  {source.stat().st_size / 1e6:.1f} MB"
        f"  ->  {dest}  {dest.stat().st_size / 1e3:.1f} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
