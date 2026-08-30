"""FastAPI application.

Thin on purpose: the HTTP layer parses, authorises the size of the upload,
delegates to the agent and serialises. Everything interesting is testable
without a server.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from freshsense import __version__
from freshsense.agent.pipeline import TriageAgent
from freshsense.api.schemas import AssessmentOut, ErrorOut, HealthOut, ReadyOut
from freshsense.config import Settings, get_settings
from freshsense.logging_setup import configure_logging, new_request_id, request_id_var
from freshsense.vision.predict import InvalidImageError

logger = logging.getLogger(__name__)

_agent: TriageAgent | None = None


def get_agent() -> TriageAgent:
    """FastAPI dependency. Overridable in tests via `app.dependency_overrides`."""
    global _agent
    if _agent is None:
        _agent = TriageAgent()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    logger.info(
        "starting freshsense",
        extra={
            "version": __version__,
            "llm_mode": "anthropic" if settings.llm_enabled else "offline",
        },
    )
    # Warm the knowledge base at boot so the first request does not pay for it.
    # The classifier stays lazy: a 330 MB backbone should not block readiness.
    try:
        get_agent().kb.load()
    except Exception:
        logger.exception("knowledge base failed to warm; will retry per request")
    yield
    logger.info("shutting down")


app = FastAPI(
    title="FreshSense",
    version=__version__,
    description=(
        "Food spoilage triage. A fine-tuned ViT decides fresh vs spoiled under an "
        "asymmetric safety policy; a guard-railed agent grounds that decision in a "
        "food-safety knowledge base. Decision support for a trained operator — never "
        "a clearance to release food."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id to logs and responses, and time every call."""
    request_id = request.headers.get("x-request-id") or new_request_id()
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        # Logged before the contextvar is reset — resetting first would strip the
        # request id from the one log line that summarises the whole request.
        logger.info(
            "request complete",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round(elapsed_ms, 2),
            },
        )
        return response
    finally:
        request_id_var.reset(token)


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request: Request, exc: InvalidImageError):
    return JSONResponse(
        status_code=422,
        content=ErrorOut(
            error="invalid_image", detail=str(exc), request_id=request_id_var.get()
        ).model_dump(),
    )


@app.get("/healthz", response_model=HealthOut, tags=["ops"])
async def healthz() -> HealthOut:
    """Liveness. Cheap and dependency-free — the process is up."""
    return HealthOut(status="ok", version=__version__)


@app.get("/readyz", response_model=ReadyOut, tags=["ops"])
async def readyz(
    agent: TriageAgent = Depends(get_agent), settings: Settings = Depends(get_settings)
) -> ReadyOut:
    """Readiness. Reports each dependency separately so a failure is diagnosable."""
    checks = {
        "knowledge_base": False,
        "checkpoint_present": settings.checkpoint.exists(),
    }
    try:
        agent.kb.load()
        checks["knowledge_base"] = agent.kb.ready
    except Exception:
        logger.exception("readiness: knowledge base unavailable")

    return ReadyOut(
        status="ready" if all(checks.values()) else "not_ready",
        checks=checks,
        llm_mode="anthropic" if settings.llm_enabled else "offline",
    )


@app.post(
    "/v1/assess",
    response_model=AssessmentOut,
    responses={422: {"model": ErrorOut}, 503: {"model": ErrorOut}},
    tags=["triage"],
)
async def assess(
    image: UploadFile = File(..., description="A photograph of a single food item"),
    food_hint: str | None = Form(
        None, description="Optional: what the item is, e.g. 'strawberries'"
    ),
    agent: TriageAgent = Depends(get_agent),
) -> AssessmentOut:
    """Assess one image and return a verdict, an action and a grounded explanation."""
    data = await image.read()
    try:
        assessment = agent.assess(data, food_hint=food_hint, request_id=request_id_var.get())
    except InvalidImageError:
        raise
    except FileNotFoundError as exc:
        return JSONResponse(  # type: ignore[return-value]
            status_code=503,
            content=ErrorOut(
                error="model_unavailable", detail=str(exc), request_id=request_id_var.get()
            ).model_dump(),
        )
    return AssessmentOut(**assessment.to_dict())
