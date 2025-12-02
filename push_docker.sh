#!/bin/bash
# Push Docker image to Docker Hub
set -e

echo "Pushing romantony/wan2-s2v to Docker Hub..."
docker push romantony/wan2-s2v:1.0.0
docker push romantony/wan2-s2v:latest
echo "✓ Push complete!"
