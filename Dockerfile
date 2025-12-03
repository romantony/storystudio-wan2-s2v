# Wan2.2 S2V RunPod Serverless Dockerfile
# Optimized for A100-80GB with FlashAttention compatibility patches
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH" \
    MODEL_CACHE_DIR=/runpod-volume \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential \
    cmake \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    rm -rf /root/.cache/pip

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies in batches to save space
COPY requirements.txt .

# Install PyTorch first (largest packages) and clean up
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121 && \
    rm -rf /root/.cache/pip && \
    pip cache purge

# Install transformers and related packages
RUN pip install --no-cache-dir \
    transformers==4.47.0 \
    diffusers==0.31.0 \
    accelerate==1.1.1 \
    peft==0.17.0 && \
    rm -rf /root/.cache/pip && \
    pip cache purge

# Install remaining dependencies
RUN pip install --no-cache-dir \
    librosa==0.10.2 \
    soundfile==0.12.1 \
    opencv-python-headless==4.10.0.84 \
    imageio==2.36.1 \
    imageio-ffmpeg==0.5.1 \
    fastapi==0.115.5 \
    uvicorn[standard]==0.32.1 \
    python-multipart==0.0.20 \
    pydantic==2.10.3 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    pillow==11.0.0 \
    einops==0.8.0 \
    omegaconf==2.3.0 \
    safetensors==0.4.5 \
    tokenizers==0.21.0 \
    huggingface-hub==0.26.5 \
    easydict==1.13 \
    ftfy==6.3.1 \
    requests==2.32.3 \
    boto3==1.35.76 \
    runpod==1.7.5 && \
    rm -rf /root/.cache/pip && \
    pip cache purge

# Install optional packages separately
RUN pip install --no-cache-dir decord || echo "Warning: decord installation failed, continuing without it" && \
    pip install --no-cache-dir dashscope || echo "Warning: dashscope installation failed, continuing without it" && \
    rm -rf /root/.cache/pip && \
    pip cache purge

# Clone Wan2.2 repository
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
RUN mkdir -p /runpod-volume/models /workspace/outputs

# Expose port for FastAPI (for testing/pod mode)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command - runs RunPod serverless handler
CMD ["python3", "-u", "handler.py"]
