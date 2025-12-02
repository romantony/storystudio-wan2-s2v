#!/bin/bash
# Deploy script for pushing Docker image to Docker Hub

set -e

# Configuration
IMAGE_NAME="wan2-s2v"
VERSION="1.0.0"
DOCKER_USERNAME="${DOCKER_USERNAME:-your-dockerhub-username}"

echo "================================================"
echo "Deploying Wan2.2 S2V to Docker Hub"
echo "================================================"
echo "Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# Check if logged in to Docker Hub
echo "Checking Docker Hub authentication..."
if ! docker info | grep -q "Username"; then
    echo "Not logged in to Docker Hub. Please login:"
    docker login
fi

# Push version tag
echo ""
echo "Pushing ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}

# Push latest tag
echo "Pushing ${DOCKER_USERNAME}/${IMAGE_NAME}:latest..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

echo ""
echo "✓ Deployment complete!"
echo ""
echo "Image available at:"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo ""
echo "Next steps:"
echo "  1. Go to RunPod Console: https://www.runpod.io/console/serverless"
echo "  2. Create a new template using: ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo "  3. Deploy your serverless endpoint"
echo "  4. Test with the client examples in the documentation"
