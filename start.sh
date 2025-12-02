#!/bin/bash
set -e

echo "================================================"
echo "Starting Wan2.2 S2V Server"
echo "================================================"

# Display GPU information
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Apply patches
echo ""
echo "Applying FlashAttention patches..."
python3 patches/apply_patches.py
if [ $? -ne 0 ]; then
    echo "ERROR: Patching failed!"
    exit 1
fi

# Check if model exists
echo ""
echo "Checking model cache..."
if [ -d "/runpod-volume/models/Wan-AI/Wan2.2-S2V-14B" ]; then
    echo "✓ Model found in volume cache"
else
    echo "⚠ Model not in cache - will download on first request (~49GB)"
fi

# Start FastAPI server
echo ""
echo "Starting FastAPI server on port 8000..."
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
