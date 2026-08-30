# Food AI Vision Platform

This project is organized into separate frontend, backend, and nginx proxy services, deploying a PyTorch Vision Transformer (ViT) to detect food spoilage and AI-generated food.

## Project Structure

- `backend/` — FastAPI backend service
  - `app.py` — API service and static file server
  - `requirements.txt` — backend dependencies
  - `Dockerfile` — backend container image definition
  - `labels.json` — class label mapping
- `frontend/` — static frontend UI
  - `index.html` — upload page
  - `style.css` — styling for the UI
  - `script.js` — upload and prediction logic
- `nginx/` — Nginx reverse proxy configuration
  - `default.conf` — Nginx server block routing requests
- `docker-compose.yml` — Docker orchestration for all services
- `freshsense/` — a second, independent service around the same model: an
  asymmetric safety policy, a retrieval-grounded explanation layer and a test
  suite. See [freshsense/README.md](freshsense/README.md) and the section below.
- `vit_food_detection_model.pth.zip` — The PyTorch model weights (requires extraction)

## Quick Start (Docker Compose)

The fastest way to get the code running is via Docker Compose. From the root of the project, run:

```bash
# 1. Provide the model to the backend service
unzip vit_food_detection_model.pth.zip -d backend/

# 2. Build and start the containers
docker-compose up --build -d
```

Then, open your web browser and navigate to `http://localhost`.

## Local Development (Without Docker)

1. **Install backend dependencies:**
   ```bash
   cd backend
   python3 -m pip install -r requirements.txt
   ```

2. **Prepare the Model:**
   You must have the `vit_food_detection_model.pth` file placed inside the `backend/` folder.
   ```bash
   # From the project root, extract the model directly into the backend folder:
   unzip vit_food_detection_model.pth.zip -d backend/
   ```

3. **Start the backend server:**
   ```bash
   uvicorn app:app --reload
   ```

4. **Access the Application:**
   Open your browser to `http://127.0.0.1:8000`.

## Docker Compose Deployment

The recommended way to run this application is via Docker Compose, which spins up both the FastAPI application and an Nginx reverse proxy.

1. **Prepare the Model File:**
   You must extract the `vit_food_detection_model.pth` file and place it inside the `backend/` directory prior to building the containers.
   ```bash
   unzip vit_food_detection_model.pth.zip -d backend/
   ```

2. **Start the Services:**
   From the root of the repository, run:
   ```bash
   docker-compose up --build -d
   ```

3. **Access the Application:**
   Open your browser to `http://localhost`. The Nginx proxy listens on port 80 and routes traffic to the FastAPI app.

4. **Stopping Services:**
   To stop the deployed services, run:
   ```bash
   docker-compose down
   ```

## Available API Endpoints

- `GET /` - Serves the frontend UI
- `POST /predict` - Upload an image for ViT classification
- `GET /health` - System health check
- `GET /history` - Recent predictions history
- `GET /labels` - Available detection classes

## Notes

- The backend restricts image uploads to a maximum of 10MB.
- Nginx allows payloads up to 15MB.
- The default detection classes supported by the provided ViT model are `ai_detection` and `spoilage_detection`.

---

## Two services, one model

This repository holds two different things built on the same ViT classifier. They
are deliberately separate.

**`backend/` + `frontend/` + `nginx/`** is the deployed demo: upload an image,
get a class and a confidence back. It is the shortest path from model to
something you can click.

**`freshsense/`** treats the same model as one component inside a system that has
to be trusted:

- **The decision policy is separate from the model.** Probabilities are not
  verdicts. Spoilage is condemned on weak evidence, food is cleared only on
  strong evidence, and the band between them abstains to a human. On the
  held-out test split this takes spoiled items wrongly released from one to
  **zero**, at a 1.6% referral rate.
- **Explanations are grounded and guard-railed.** A retrieval layer over a
  food-safety knowledge base supplies cited passages; anything the language
  model returns is validated before it leaves the process, and an explanation
  citing a source that was not retrieved is discarded rather than shown.
- **It runs with no API key.** A deterministic offline explainer is a supported
  mode, not a stub, which is what keeps the test suite hermetic.
- **The checkpoint is 8 KB, not 343 MB.** Only the trained head is ours; the
  backbone is public ImageNet weights.
- 133 fast tests at 87% line coverage, a Dockerfile, and CI.

Reported numbers and the model's limitations are in
[freshsense/docs/MODEL_CARD.md](freshsense/docs/MODEL_CARD.md); the design
reasoning is in
[freshsense/docs/ARCHITECTURE.md](freshsense/docs/ARCHITECTURE.md).

```bash
cd freshsense
pip install -e ".[dev]"
freshsense assess path/to/food.jpg --hint "strawberries"
```
