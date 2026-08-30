"""HTTP contract tests against the real app with a faked agent dependency."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from freshsense.agent.pipeline import TriageAgent
from freshsense.api.app import app, get_agent
from freshsense.config import get_settings
from freshsense.genai.llm import OfflineClient
from freshsense.genai.rag import KnowledgeBase

from .conftest import FakeClassifier, make_image


@pytest.fixture
def client(settings, kb_dir):
    local = settings.model_copy(update={"kb_dir": kb_dir})
    agent = TriageAgent(
        classifier=FakeClassifier({"fresh": 0.1, "spoiled": 0.9}, local),
        knowledge_base=KnowledgeBase(kb_dir, min_score=0.0),
        llm=OfflineClient(),
        settings=local,
    )
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[get_settings] = lambda: local
    with TestClient(app) as c:
        c.agent = agent
        yield c
    app.dependency_overrides.clear()


def test_healthz_is_dependency_free(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["version"]


def test_readyz_reports_each_check_separately(client):
    body = client.get("/readyz").json()
    assert body["checks"]["knowledge_base"] is True
    assert "checkpoint_present" in body["checks"]
    assert body["llm_mode"] == "offline"


def test_assess_returns_the_full_contract(client):
    response = client.post(
        "/v1/assess",
        files={"image": ("item.jpg", make_image(), "image/jpeg")},
        data={"food_hint": "strawberries"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "spoiled"
    assert body["action"] == "discard"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["explanation"]["summary"]
    assert body["explanation"]["citations"]
    assert body["explanation"]["llm_mode"] == "offline"
    assert body["trace"]
    assert body["request_id"]


def test_food_hint_is_optional(client):
    response = client.post("/v1/assess", files={"image": ("i.jpg", make_image(), "image/jpeg")})
    assert response.status_code == 200


def test_a_corrupt_upload_is_a_422_not_a_500(client):
    response = client.post(
        "/v1/assess", files={"image": ("evil.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_image"
    assert response.json()["request_id"]


def test_a_missing_file_is_a_422(client):
    assert client.post("/v1/assess").status_code == 422


def test_request_id_is_echoed_back(client):
    response = client.get("/healthz", headers={"x-request-id": "trace-me-42"})
    assert response.headers["x-request-id"] == "trace-me-42"


def test_request_id_is_generated_when_absent(client):
    assert client.get("/healthz").headers["x-request-id"]


def test_the_completion_log_keeps_the_request_id(client, caplog):
    """Regression: the contextvar was reset before the summary line was logged."""
    import logging

    from freshsense.logging_setup import request_id_var

    seen = []
    handler_logger = logging.getLogger("freshsense.api.app")

    class Capture(logging.Handler):
        def emit(self, record):
            if record.getMessage() == "request complete":
                seen.append(request_id_var.get())

    handler = Capture()
    handler_logger.addHandler(handler)
    try:
        client.get("/healthz", headers={"x-request-id": "keep-me"})
    finally:
        handler_logger.removeHandler(handler)

    assert seen == ["keep-me"]


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/v1/assess" in schema["paths"]
    assert schema["info"]["title"] == "FreshSense"
