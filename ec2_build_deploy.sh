#!/bin/bash
# EC2 Build and Deploy Script for Wan2.2 S2V Docker Image
# Run this on your EC2 instance with GPU support

set -e

echo "======================================================================"
echo "Wan2.2 S2V Docker Build & Deploy Script"
echo "======================================================================"
echo ""

# Configuration
DOCKER_USERNAME="your_dockerhub_username"  # CHANGE THIS
IMAGE_NAME="wan2-s2v-runpod"
IMAGE_TAG="latest"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Clone repository if not already present
if [ ! -d "storystudio-wan2-s2v" ]; then
    echo "Cloning repository..."
    git clone https://github.com/romantony/storystudio-wan2-s2v.git
    cd storystudio-wan2-s2v
else
    echo "Repository already exists, pulling latest changes..."
    cd storystudio-wan2-s2v
    git pull origin main
fi

echo ""
echo "======================================================================"
echo "Building Docker Image (this will take 20-30 minutes)"
echo "======================================================================"
echo ""

# Build the Docker image
docker build -t ${FULL_IMAGE} .

echo ""
echo "======================================================================"
echo "Build Complete! Image size:"
echo "======================================================================"
docker images ${FULL_IMAGE}

echo ""
echo "======================================================================"
echo "Testing image locally (optional - press Ctrl+C to skip)"
echo "======================================================================"
echo ""
read -p "Test locally? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker run --rm --gpus all \
        -p 8000:8000 \
        ${FULL_IMAGE}
fi

echo ""
echo "======================================================================"
echo "Pushing to Docker Hub"
echo "======================================================================"
echo ""

# Login to Docker Hub (you'll be prompted for credentials)
docker login

# Push the image
docker push ${FULL_IMAGE}

echo ""
echo "======================================================================"
echo "✓ Deployment Complete!"
echo "======================================================================"
echo ""
echo "Docker Image: ${FULL_IMAGE}"
echo ""
echo "Next Steps:"
echo "1. Go to RunPod Serverless -> Your Endpoint -> Settings"
echo "2. Update Container Image to: ${FULL_IMAGE}"
echo "3. Set Container Start Command to: python3 -u handler.py"
echo "4. Configure environment variables:"
echo "   - MODEL_CACHE_DIR=/runpod-volume/models"
echo "   - R2_ACCOUNT_ID=620baa808df08b1a30d448989365f7dd"
echo "   - R2_ACCESS_KEY_ID=a69ca34cdcdeb60bad5ed1a07a0dd29d"
echo "   - R2_SECRET_ACCESS_KEY=751a95202a9fa1eb9ff7d45e0bba5b57b0c2d1f0d45129f5f67c2486d5d4ae24"
echo "   - R2_BUCKET_NAME=storystudio"
echo "   - R2_PUBLIC_URL=parentearn.com"
echo "5. Save and deploy"
echo ""
echo "Test with:"
echo "curl -X POST https://api.runpod.ai/v2/1tggndzlc063rw/run \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -H \"Authorization: Bearer YOUR_API_KEY\" \\"
echo "  -d '{\"input\": {\"image\": \"https://parentearn.com/VideoGen/test_image.png\", \"audio\": \"https://parentearn.com/VideoGen/test_audio.mp3\", \"resolution\": \"480p\", \"sample_steps\": 30}}'"
echo ""
