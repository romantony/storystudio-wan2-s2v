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
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip and install build tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install RunPod SDK for serverless compatibility
RUN pip install --no-cache-dir runpod

# Clone Wan2.2 repository
RUN git clone https://github.com/Wan-Video/Wan2.2.git

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
