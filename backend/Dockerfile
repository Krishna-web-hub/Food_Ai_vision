FROM python:3.11-slim

# Create base app directory
WORKDIR /app

# Copy requirements from backend directory
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (including backend/ and frontend/) 
COPY . ./

# Set working directory to backend so uvicorn can find app.py
WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
