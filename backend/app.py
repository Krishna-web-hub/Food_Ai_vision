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
import gdown

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

FILE_ID = "1bZzWG16IFdQwL_OxrdmnXFsrdmXGnssb"

# ───────────────────────────────────────────────
# App
# ───────────────────────────────────────────────

app = FastAPI(title="FoodAI Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ───────────────────────────────────────────────
# History
# ───────────────────────────────────────────────

prediction_history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
server_start_time = time.time()
total_predictions = 0

# ───────────────────────────────────────────────
# Model Helpers
# ───────────────────────────────────────────────


def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    return ["ai_detection", "spoilage_detection"]


def build_model(num_classes):
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features, num_classes)
    return model


def download_model():
    if os.path.exists(MODEL_FILE):
        print("✅ Model already exists")
        return

    print("⬇️ Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    try:
        gdown.download(url, MODEL_FILE, quiet=False)
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    # verify
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1_000_000:
        raise RuntimeError("Downloaded model is invalid")

    print("✅ Model downloaded successfully")


def load_model():
    download_model()

    print("📂 Files in backend:", os.listdir(BASE_DIR))

    mdl = build_model(len(labels))

    state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
    mdl.load_state_dict(state_dict)

    mdl.to(DEVICE)
    mdl.eval()

    return mdl


# ───────────────────────────────────────────────
# Load Model
# ───────────────────────────────────────────────

labels = load_labels()
model = None
model_load_error = None

try:
    model = load_model()
    print(f"✅ Model loaded on {DEVICE}")
except Exception as e:
    model_load_error = str(e)
    print(f"❌ Model load failed: {e}")

# ───────────────────────────────────────────────
# Preprocess
# ───────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────


def read_imagefile(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


def validate_upload_size(data):
    if len(data) / (1024 * 1024) > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail="File too large")


# ───────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def homepage():
    with open(os.path.join(FRONTEND_DIR, "index.html")) as f:
        return f.read()


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model": {
            "loaded": model is not None,
            "error": model_load_error,
            "device": str(DEVICE),
            "classes": labels,
        },
        "server": {
            "total_predictions": total_predictions if 'total_predictions' in globals() else 0,
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global total_predictions

    if model is None:
        raise HTTPException(status_code=503, detail=model_load_error)

    data = await file.read()
    validate_upload_size(data)

    image = read_imagefile(data)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    t0 = time.time()

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    idx = torch.argmax(probs).item()
    confidence = float(probs[idx].item())

    total_predictions += 1

    probs_dict = {labels[i]: float(probs[i].item()) for i in range(len(labels))}

    return {
        "predicted_class": labels[idx],
        "confidence": confidence,
        "probabilities": probs_dict,
        "metadata": {
            "inference_ms": round((time.time() - t0) * 1000, 2),
            "device": str(DEVICE),
            "prediction_id": total_predictions,
        }
    }