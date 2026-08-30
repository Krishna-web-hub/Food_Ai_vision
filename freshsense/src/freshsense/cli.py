"""Command line entrypoint: `freshsense {assess,train,serve,kb}`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from freshsense import __version__
from freshsense.config import get_settings
from freshsense.logging_setup import configure_logging


def _cmd_assess(args: argparse.Namespace) -> int:
    from freshsense.agent.pipeline import TriageAgent
    from freshsense.vision.predict import InvalidImageError

    path = Path(args.image)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 2
    try:
        assessment = TriageAgent().assess(path.read_bytes(), food_hint=args.hint)
    except InvalidImageError as exc:
        print(f"invalid image: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"model unavailable: {exc}", file=sys.stderr)
        return 3

    payload = assessment.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"\n  verdict    {payload['verdict'].upper()}  ({payload['confidence']:.1%} confident)")
    print(f"  action     {payload['action']}")
    print(f"  policy     {payload['policy_note']}")
    print(f"\n  {payload['explanation']['summary']}")
    print(f"\n  Next steps: {payload['explanation']['handling_guidance']}")
    if cites := payload["explanation"]["citations"]:
        print("\n  Sources:")
        for c in cites:
            print(f"    - {c['source_id']}  {c['title']}  (score {c['score']:.3f})")
    print(f"\n  [{payload['explanation']['llm_mode']} explainer, {payload['latency_ms']:.0f} ms]\n")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from freshsense.vision.train import train

    history = train(
        train_dir=Path(args.train_dir),
        val_dir=Path(args.val_dir),
        checkpoint_out=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        metrics_out=Path(args.metrics) if args.metrics else None,
    )
    best = max(history, key=lambda r: (r.val_spoiled_recall, r.val_macro_f1))
    print(
        f"\nbest epoch {best.epoch}: spoiled-recall {best.val_spoiled_recall:.3f}, "
        f"macro-F1 {best.val_macro_f1:.3f}, accuracy {best.val_acc:.3f}"
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("freshsense.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_kb(args: argparse.Namespace) -> int:
    from freshsense.genai.rag import KnowledgeBase

    kb = KnowledgeBase(get_settings().kb_dir).load()
    if args.query:
        for c in kb.search(args.query, top_k=args.top_k):
            print(f"\n[{c.score:.3f}] {c.source_id}  {c.title}\n{c.short(400)}")
    else:
        print(f"{len(kb.chunks)} chunks indexed from {kb.kb_dir}:")
        for c in kb.chunks:
            print(f"  {c.source_id:<44} {c.title}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="freshsense", description="Food spoilage triage.")
    parser.add_argument("--version", action="version", version=f"freshsense {__version__}")
    parser.add_argument("--log-level", default=None, help="override FRESHSENSE_LOG_LEVEL")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("assess", help="assess one image")
    p.add_argument("image")
    p.add_argument("--hint", default=None, help="what the item is, e.g. 'strawberries'")
    p.add_argument("--json", action="store_true", help="emit the full JSON assessment")
    p.set_defaults(func=_cmd_assess)

    p = sub.add_parser("train", help="fine-tune the classifier head")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--out", default="artifacts/vit_head.pt")
    p.add_argument("--metrics", default="artifacts/train_metrics.json")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("kb", help="inspect or query the knowledge base")
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--top-k", type=int, default=3)
    p.set_defaults(func=_cmd_kb)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(args.log_level or settings.log_level, json_output=settings.log_json)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
