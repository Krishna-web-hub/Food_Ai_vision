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
