#!/bin/bash
# Build script for Wan2.2 S2V Docker image

set -e

# Configuration
IMAGE_NAME="wan2-s2v"
VERSION="1.0.0"
DOCKER_USERNAME="${DOCKER_USERNAME:-your-dockerhub-username}"

echo "================================================"
echo "Building Wan2.2 S2V Docker Image"
echo "================================================"
echo "Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo ""


REPO_URL="https://github.com/romantony/storystudio-wan2-s2v.git"
REPO_DIR="storystudio-wan2-s2v"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# Clone repo if not present
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository from $REPO_URL ..."
    git clone $REPO_URL $REPO_DIR
fi

cd $REPO_DIR

# Build the image
echo "Building Docker image..."
docker build -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} .

# Tag as latest
echo "Tagging as latest..."
docker tag ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

echo ""
echo "✓ Build complete!"
echo ""
echo "Images created:"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo ""
echo "Next steps:"
echo "  1. Test locally: docker run --gpus all -p 8000:8000 ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo "  2. Push to Docker Hub: ./deploy.sh"
