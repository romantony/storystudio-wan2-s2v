#!/bin/bash
# Build and push Docker image for Wan2.2 S2V
set -e

DOCKER_USERNAME="romantony"
IMAGE_NAME="wan2-s2v"
VERSION="1.0.0"

echo "================================================"
echo "Building and Pushing Wan2.2 S2V Docker Image"
echo "================================================"

# Find workspace root by looking for BUILD.bazel
WORKSPACE_DIR="${BUILD_WORKSPACE_DIRECTORY:-$(pwd)}"
if [ ! -f "$WORKSPACE_DIR/Dockerfile" ]; then
    # If running from bazel, use the actual workspace
    WORKSPACE_DIR="/workspace/storystudio-wan2-s2v"
fi

echo "Building Docker image from: $WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# Build the image
echo "Building ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}..."
docker build -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} .

# Tag as latest
echo "Tagging as latest..."
docker tag ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

echo ""
echo "✓ Build complete!"
echo ""

# Push to Docker Hub
echo "Pushing to Docker Hub..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

echo ""
echo "================================================"
echo "✓ Deployment Complete!"
echo "================================================"
echo "Images available at:"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo ""
echo "Next: Create RunPod serverless endpoint with this image"
