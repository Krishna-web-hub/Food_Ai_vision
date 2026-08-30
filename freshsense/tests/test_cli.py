"""CLI argument parsing and command wiring."""

from __future__ import annotations

import json

import pytest

from freshsense import cli
from freshsense.domain import Action, Assessment, Explanation
from freshsense.vision.predict import decide

from .conftest import make_image


@pytest.fixture
def assessment(settings, citations) -> Assessment:
    return Assessment(
        request_id="req-1",
        prediction=decide({"fresh": 0.1, "spoiled": 0.9}, settings),
        action=Action.DISCARD,
        explanation=Explanation(
            summary="Visible spoilage detected.",
            handling_guidance="Isolate and discard.",
            citations=citations,
            llm_mode="offline",
        ),
        trace=[{"step": 1, "node": "ingest"}],
        latency_ms=42.0,
    )


@pytest.fixture
def patched_agent(monkeypatch, assessment):
    class StubAgent:
        def __init__(self, *a, **k):
            pass

        def assess(self, data, food_hint=None, request_id=None):
            return assessment

    monkeypatch.setattr("freshsense.agent.pipeline.TriageAgent", StubAgent)
    return StubAgent


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_assess_defaults():
    args = cli.build_parser().parse_args(["assess", "x.jpg"])
    assert args.image == "x.jpg" and args.hint is None and args.json is False


def test_train_parses_its_hyperparameters():
    args = cli.build_parser().parse_args(
        ["train", "--train-dir", "t", "--val-dir", "v", "--epochs", "3", "--lr", "0.01"]
    )
    assert args.epochs == 3 and args.lr == 0.01 and args.out.endswith("vit_head.pt")


def test_serve_defaults_to_all_interfaces_on_8000():
    args = cli.build_parser().parse_args(["serve"])
    assert args.host == "0.0.0.0" and args.port == 8000


def test_assess_on_a_missing_file_exits_2(capsys):
    assert cli.main(["assess", "/no/such/image.jpg"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_assess_prints_a_human_report(tmp_path, patched_agent, capsys):
    image = tmp_path / "item.jpg"
    image.write_bytes(make_image())

    assert cli.main(["assess", str(image)]) == 0
    out = capsys.readouterr().out
    assert "SPOILED" in out
    assert "discard" in out
    assert "alpha#1" in out
    assert "offline explainer" in out


def test_assess_json_mode_emits_the_full_payload(tmp_path, patched_agent, capsys):
    image = tmp_path / "item.jpg"
    image.write_bytes(make_image())

    assert cli.main(["assess", str(image), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "spoiled"
    assert payload["explanation"]["citations"]
    assert payload["trace"]


def test_assess_reports_a_missing_model_as_exit_3(tmp_path, monkeypatch, capsys):
    class Broken:
        def __init__(self, *a, **k):
            pass

        def assess(self, *a, **k):
            raise FileNotFoundError("No checkpoint at artifacts/vit_head.pt")

    monkeypatch.setattr("freshsense.agent.pipeline.TriageAgent", Broken)
    image = tmp_path / "item.jpg"
    image.write_bytes(make_image())

    assert cli.main(["assess", str(image)]) == 3
    assert "model unavailable" in capsys.readouterr().err


def test_kb_lists_the_shipped_knowledge_base(capsys):
    assert cli.main(["kb"]) == 0
    out = capsys.readouterr().out
    assert "chunks indexed" in out
    assert "01-visual-spoilage-indicators#1" in out


def test_kb_query_returns_scored_passages(capsys):
    assert cli.main(["kb", "mould discard sanitise", "--top-k", "2"]) == 0
    assert "04-disposition-and-handling" in capsys.readouterr().out
