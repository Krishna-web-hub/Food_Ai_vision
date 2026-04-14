import io
import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from torchvision import models, transforms

# ───────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
MODEL_FILE = os.path.join(BASE_DIR, "vit_food_detection_model.pth")
LABELS_FILE = os.path.join(BASE_DIR, "labels.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_UPLOAD_MB = 10
MAX_HISTORY = 50

# ───────────────────────────────────────────────
# App
# ───────────────────────────────────────────────

app = FastAPI(
    title="FoodAI Vision API",
    description=(
        "## Food Detection Backend\n\n"
        "AI-powered food analysis using Vision Transformer (ViT-B/16).\n\n"
        "### Detection Classes\n"
        "- **spoilage_detection** — visible spoilage, mold, discoloration\n"
        "- **ai_detection** — AI-generated / synthetic food imagery\n\n"
        "### Usage\n"
        "`POST /predict` with a multipart image upload."
    ),
    version="1.0.0",
    contact={"name": "FoodAI Vision"},
    license_info={"name": "MIT"},
)

# CORS — allow frontend served from same origin and localhost dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ───────────────────────────────────────────────
# In-memory prediction history (ring buffer)
# ───────────────────────────────────────────────

prediction_history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
server_start_time = time.time()
total_predictions = 0

# ───────────────────────────────────────────────
# Model Helpers
# ───────────────────────────────────────────────


def load_labels() -> list[str]:
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["ai_detection", "spoilage_detection"]


def build_model(num_classes: int) -> torch.nn.Module:
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: str, label_list: list[str]) -> torch.nn.Module:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Place the .pth checkpoint in the backend folder and restart."
        )
    mdl = build_model(len(label_list))
    state_dict = torch.load(model_path, map_location=DEVICE)
    mdl.load_state_dict(state_dict)
    mdl.to(DEVICE)
    mdl.eval()
    return mdl


# Load labels & model at startup
labels = load_labels()
model = None
model_load_error: str | None = None

try:
    model = load_model(MODEL_FILE, labels)
    print(f"✅  Model loaded on {DEVICE} | Classes: {labels}")
except FileNotFoundError as exc:
    model_load_error = str(exc)
    print(f"⚠️  WARNING: {exc}")

# Inference preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ───────────────────────────────────────────────
# Image Helpers
# ───────────────────────────────────────────────


def read_imagefile(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")


def validate_upload_size(data: bytes) -> None:
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Maximum allowed is {MAX_UPLOAD_MB} MB.",
        )

# ───────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage() -> str:
    """Serve the frontend single-page application."""
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=500, detail="Frontend index.html not found.")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health", tags=["System"])
def health_check() -> JSONResponse:
    """
    System health check.
    Returns model status, device info, uptime, and prediction count.
    """
    uptime_seconds = time.time() - server_start_time
    hours, rem = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(rem, 60)

    return JSONResponse(content={
        "status": "healthy" if model is not None else "degraded",
        "model": {
            "loaded": model is not None,
            "device": str(DEVICE),
            "classes": labels,
            "architecture": "ViT-B/16",
            "error": model_load_error,
        },
        "server": {
            "uptime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "uptime_seconds": round(uptime_seconds, 1),
            "total_predictions": total_predictions,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.get("/labels", tags=["Model"])
def get_labels() -> JSONResponse:
    """Return the list of detection class labels."""
    return JSONResponse(content={
        "labels": labels,
        "count": len(labels),
    })


@app.get("/history", tags=["Predictions"])
def get_history(limit: int = 20) -> JSONResponse:
    """
    Return the most recent server-side prediction records.

    - **limit**: Number of entries to return (max 50).
    """
    limit = min(limit, MAX_HISTORY)
    records = list(prediction_history)[:limit]
    return JSONResponse(content={
        "history": records,
        "total": len(prediction_history),
    })


@app.delete("/history", tags=["Predictions"])
def clear_history() -> JSONResponse:
    """Clear all server-side prediction history."""
    prediction_history.clear()
    return JSONResponse(content={"message": "History cleared."})


@app.post("/predict", tags=["Predictions"])
async def predict(file: UploadFile = File(..., description="Food image to classify")) -> JSONResponse:
    """
    Classify a food image using the ViT-B/16 model.

    **Returns:**
    - `predicted_class` — Most likely class label.
    - `confidence` — Probability of the predicted class (0–1).
    - `probabilities` — Full softmax distribution over all classes.
    - `metadata` — Filename, file size, inference time.
    """
    global total_predictions

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. "
                "Place vit_food_detection_model.pth in the backend folder and restart the server. "
                f"Error: {model_load_error}"
            ),
        )

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Upload an image file.",
        )

    # Read & validate size
    image_data = await file.read()
    validate_upload_size(image_data)

    # Preprocess
    image = read_imagefile(image_data)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        score, predicted_idx = torch.max(probabilities, dim=0)
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    predicted_class = labels[predicted_idx.item()]
    confidence = float(score.item())
    probs_dict = {labels[i]: float(probabilities[i].item()) for i in range(len(labels))}

    total_predictions += 1

    # Build response
    result = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 2),
        "probabilities": probs_dict,
        "metadata": {
            "filename": file.filename or "unknown",
            "file_size_kb": round(len(image_data) / 1024, 1),
            "inference_ms": inference_ms,
            "device": str(DEVICE),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prediction_id": total_predictions,
        },
    }

    # Record in server-side history (no image bytes stored)
    prediction_history.appendleft({
        "id": total_predictions,
        "filename": file.filename or "unknown",
        "predicted_class": predicted_class,
        "confidence": confidence,
        "inference_ms": inference_ms,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    return JSONResponse(content=result)
