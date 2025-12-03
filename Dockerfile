# Wan2.2 S2V RunPod Serverless Dockerfile
# Using RunPod's official PyTorch image for proper CUDA integration
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/runpod-volume \
    HF_HOME=/runpod-volume/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install additional system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies (PyTorch already included in base image)
RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    diffusers==0.31.0 \
    accelerate==1.1.1 \
    peft==0.17.0 \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    opencv-python-headless==4.10.0.84 \
    imageio==2.36.1 \
    imageio-ffmpeg==0.5.1 \
    fastapi==0.115.5 \
    "uvicorn[standard]==0.32.1" \
    python-multipart==0.0.20 \
    pydantic==2.10.3 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    pillow==11.0.0 \
    einops==0.8.0 \
    omegaconf==2.3.0 \
    safetensors==0.4.5 \
    tokenizers==0.21.0 \
    huggingface-hub==0.30.0 \
    easydict==1.13 \
    ftfy==6.3.1 \
    requests==2.32.3 \
    boto3==1.35.76 \
    runpod==1.7.5 \
    decord \
    dashscope \
    filelock \
    "packaging>=20.0" \
    "pyyaml>=5.1" \
    regex \
    tqdm && \
    rm -rf /root/.cache/pip && \
    pip cache purge

# Clone Wan2.2 repository (code only, no models)
RUN git clone https://github.com/Wan-Video/Wan2.2.git && \
    rm -rf Wan2.2/.git

# Copy application files
COPY patches/ ./patches/
COPY app.py .
COPY handler.py .
COPY start.sh .

# Make scripts executable
RUN chmod +x start.sh

# Create directories for models and outputs
RUN mkdir -p /runpod-volume/models /runpod-volume/huggingface /workspace/outputs

# Expose port for FastAPI
EXPOSE 8000

# Health check - use simple check that doesn't import torch
# (importing torch before handler.py corrupts CUDA context)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "print('healthy')" || exit 1

# Default command
CMD ["python3", "-u", "handler.py"]
