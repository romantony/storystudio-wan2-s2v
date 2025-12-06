#!/bin/bash
# Wan2.2 S2V Warm Model Architecture Startup Script
# This script starts the model server and handler together

set -e

echo "=========================================="
echo "Wan2.2 S2V Warm Model Startup"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create necessary directories
mkdir -p /runpod-volume/models /runpod-volume/huggingface /workspace/outputs

# Check if model is available
MODEL_DIR="/runpod-volume/models/Wan2.2-S2V-14B"
if [ ! -d "$MODEL_DIR" ]; then
    echo "WARNING: Model not found at $MODEL_DIR"
    echo "Model will be downloaded on first request"
fi

# Apply patches
cd /app
echo "Applying FlashAttention patches..."
python3 -c "from patches.apply_patches import apply_flashattention_patches; apply_flashattention_patches()"
echo "✓ Patches applied"

# Start the handler (which will start model server on first request)
echo "Starting handler v2 (Warm Model Architecture)..."
exec python3 -u handler_v2.py
