#!/bin/bash

# Exit script immediately if a command fails
set -e

echo "========================================="
echo " Starting Food AI Vision Deployment"
echo "========================================="

# 1. Ensure required tools are installed
if ! command -v docker &> /dev/null; then
    echo "⬇️ Docker not found. Installing Docker and dependencies..."
    # Assuming standard Ubuntu/Debian based server
    sudo apt update
    sudo apt install -y docker.io docker-compose unzip
else
    echo "✅ Docker is already installed."
    sudo apt install -y unzip &> /dev/null || true
fi

# 2. Extract model if it doesn't exist
if [ ! -f "backend/vit_food_detection_model.pth" ]; then
    echo "🔍 Model file missing in backend directory. Looking for archive..."
    if [ -f "vit_food_detection_model.pth.zip" ]; then
        echo "📦 Extracting model from zip into backend directory..."
        unzip -o vit_food_detection_model.pth.zip -d backend/
    else
        echo "❌ ERROR: vit_food_detection_model.pth.zip not found! Please upload it to the server root."
        exit 1
    fi
else
    echo "✅ Model file already loaded."
fi

# 3. Stop running containers
echo "🛑 Stopping existing containers (if updating)..."
sudo docker-compose down || true

# 4. Rebuild and launch containers
echo "🚀 Building and bringing up containers in the background..."
sudo docker-compose up --build -d

echo "========================================="
echo "🎉 Deployment successful!"
echo "Your app is now running on port 80."
echo ""
echo "Next Steps for Public Websites (HTTPS):"
echo "1. Point your domain A-Record to this Server's IP."
echo "2. SSH into your server, and install Certbot: sudo apt install certbot python3-certbot-nginx"
echo "3. Temporarily stop Nginx container: sudo docker-compose stop nginx"
echo "4. Obtain Certificate: sudo certbot --nginx -d yourdomain.com"
echo "========================================="
