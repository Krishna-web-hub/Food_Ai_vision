"""Structured logging with a request id carried on a contextvar.

JSON by default so the logs are grep-able in a container platform; set
FRESHSENSE_LOG_JSON=false for readable lines while developing.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return (
            f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} "
            f"[{request_id_var.get()}] {record.name}: {record.getMessage()}"
        )


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Idempotent root-logger setup. Safe to call from the API, the CLI or a test."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    logging.getLogger("PIL").setLevel(logging.WARNING)
