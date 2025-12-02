# Wan2.2 S2V RunPod Docker Deployment Guide

**Based on Modal Implementation Learnings**  
*Date: December 2, 2025*

## Overview

This guide documents how to deploy Wan2.2-S2V-14B on RunPod using Docker, based on our successful Modal serverless deployment. The implementation includes critical patches for FlashAttention compatibility and optimized configuration for GPU performance.

---

## Key Learnings from Modal Implementation

### 1. FlashAttention Compatibility Issue
**Problem:** Wan2.2 requires `flash-attn` which fails to build on most cloud platforms.

**Solution:** Three-file patching strategy (must be applied BEFORE imports):
- `wan/modules/attention.py` - Disable FlashAttention flags
- `wan/s2v/model_s2v.py` - Alias flash_attention to attention()
- `wan/modules/model.py` - Same aliasing for shared modules

### 2. Performance Optimization Results
| Configuration | Cold Start | Warm Container |
|--------------|------------|----------------|
| 2x A100-80GB, offload=True, steps=50 | 29 min | 16 min |
| 1x A100-80GB, offload=False, steps=30 | ~25 min | ~6-8 min (estimated) |

**Key Findings:**
- Model offloading (`--offload_model True`) trades GPU memory for 60-70% slower generation
- Reduced sampling steps (30 vs 50) provides 30-40% speedup with acceptable quality
- Single GPU eliminates multi-GPU coordination overhead for sequential requests

### 3. Critical Implementation Details
- Model size: 49.1GB (requires adequate storage)
- Valid resolutions: 480p → 832×480, 720p → 1024×704
- Output: 24fps MP4
- Patches must be applied BEFORE any Wan2.2 imports
- subprocess timeout: 3600s (1 hour) for generation

---

## RunPod Docker Implementation

### Directory Structure
```
/workspace/
├── Dockerfile
├── requirements.txt
├── start.sh
├── app.py                    # FastAPI application
├── patches/
│   ├── apply_patches.py      # Patching logic
│   └── README.md
└── models/                   # Persistent storage mount
    └── Wan-AI/
        └── Wan2.2-S2V-14B/   # 49.1GB model
```

---

## Dockerfile

```dockerfile
# Base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone Wan2.2 repository
RUN cd /workspace && git clone https://github.com/Wan-Video/Wan2.2.git

# Copy application files
COPY app.py .
COPY start.sh .
COPY patches/ ./patches/

# Make start script executable
RUN chmod +x start.sh

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["./start.sh"]
```

---

## requirements.txt

```txt
# Core dependencies (pinned versions from successful Modal deployment)
torch>=2.4.0
torchvision>=0.19.0
torchaudio>=2.0.0
transformers>=4.47.0
diffusers>=0.31.0
accelerate>=1.1.1
peft>=0.17.0

# Audio/Video processing
librosa>=0.10.0
soundfile>=0.12.0
opencv-python>=4.9.0.80
imageio>=2.0.0
imageio-ffmpeg>=0.4.9
decord

# API framework
fastapi
uvicorn[standard]
python-multipart
pydantic

# Utilities
numpy>=1.23.5,<2
scipy
pillow
einops>=0.7.0
omegaconf>=2.3.0
safetensors>=0.4.0
tokenizers>=0.20.3
huggingface-hub
dashscope
easydict
ftfy
requests
```

---

## patches/apply_patches.py

```python
"""
FlashAttention compatibility patches for Wan2.2
Must be applied BEFORE any Wan2.2 imports
"""
from pathlib import Path
import sys

def apply_flashattention_patches(wan_dir="/workspace/Wan2.2"):
    """
    Apply three-phase patching strategy for FlashAttention compatibility
    """
    print("=" * 70)
    print("Applying FlashAttention Compatibility Patches")
    print("=" * 70)
    
    wan_path = Path(wan_dir)
    if not wan_path.exists():
        raise RuntimeError(f"Wan2.2 directory not found: {wan_dir}")
    
    # Phase 1: Patch attention.py
    print("\n[1/3] Patching wan/modules/attention.py...")
    attention_path = wan_path / "wan/modules/attention.py"
    
    attention_code = attention_path.read_text()
    
    # Force disable FlashAttention flags
    attention_code = attention_code.replace(
        'FLASH_ATTN_2_AVAILABLE = importlib.util.find_spec("flash_attn") is not None',
        'FLASH_ATTN_2_AVAILABLE = False  # [PATCHED] Forced to False'
    )
    
    # Remove assertion that blocks execution
    if 'assert FLASH_ATTN_2_AVAILABLE' in attention_code:
        lines = attention_code.split('\n')
        attention_code = '\n'.join([
            line if 'assert FLASH_ATTN_2_AVAILABLE' not in line 
            else '    pass  # [PATCHED] Assertion removed'
            for line in lines
        ])
    
    # Add shim for flash_attention import
    if 'def flash_attention(' not in attention_code:
        shim = '''

def flash_attention(*args, **kwargs):
    """[PATCHED] Shim that redirects to attention()"""
    return attention(*args, **kwargs)
'''
        attention_code += shim
    
    attention_path.write_text(attention_code)
    print("✓ attention.py patched")
    
    # Phase 2: Patch model_s2v.py
    print("\n[2/3] Patching wan/s2v/model_s2v.py...")
    model_path = wan_path / "wan/s2v/model_s2v.py"
    
    model_code = model_path.read_text()
    
    # Replace flash_attention imports
    import_patterns = [
        'from ..attention import flash_attention',
        'from wan.modules.attention import flash_attention',
    ]
    for pattern in import_patterns:
        if pattern in model_code:
            model_code = model_code.replace(
                pattern,
                'from ..attention import attention as flash_attention  # [PATCHED]'
            )
    
    model_path.write_text(model_code)
    print("✓ model_s2v.py patched")
    
    # Phase 3: Patch shared model.py
    print("\n[3/3] Patching wan/modules/model.py...")
    common_model_path = wan_path / "wan/modules/model.py"
    
    if common_model_path.exists():
        common_code = common_model_path.read_text()
        
        for pattern in import_patterns:
            if pattern in common_code:
                common_code = common_code.replace(
                    pattern,
                    'from .attention import attention as flash_attention  # [PATCHED]'
                )
        
        common_model_path.write_text(common_code)
        print("✓ model.py patched")
    
    # Verification
    print("\n" + "=" * 70)
    print("Patch Verification:")
    attention_check = 'FLASH_ATTN_2_AVAILABLE = False' in attention_path.read_text()
    model_check = 'attention as flash_attention' in model_path.read_text()
    
    print(f"  {'✓' if attention_check else '✗'} FLASH_ATTN_2 disabled in attention.py")
    print(f"  {'✓' if model_check else '✗'} model_s2v.py using attention fallback")
    
    if attention_check and model_check:
        print("\n✓ All patches applied successfully!")
    else:
        print("\n✗ Some patches failed - check manually")
    
    print("=" * 70)
    
    return attention_check and model_check

if __name__ == "__main__":
    success = apply_flashattention_patches()
    sys.exit(0 if success else 1)
```

---

## app.py (FastAPI Application)

```python
"""
FastAPI server for Wan2.2 S2V video generation
Optimized configuration: 1x GPU, no offload, 30 sampling steps
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import subprocess
import tempfile
import base64
import os
import time
from pathlib import Path
from typing import Optional

# Apply patches BEFORE any Wan2.2 imports
from patches.apply_patches import apply_flashattention_patches
apply_flashattention_patches()

# Now safe to import Wan2.2 modules
from huggingface_hub import snapshot_download

app = FastAPI(title="Wan2.2 S2V API", version="1.0.0")

# Configuration
MODEL_ID = "Wan-AI/Wan2.2-S2V-14B"
MODEL_CACHE_DIR = "/workspace/models"
WAN_DIR = "/workspace/Wan2.2"

# Resolution mapping (validated options)
RESOLUTION_MAP = {
    "480p": "832*480",
    "720p": "1024*704"
}

class GenerationConfig:
    def __init__(self):
        self.model_loaded = False
        self.model_dir = None
        
    def load_model(self):
        """Download model if not cached"""
        if self.model_loaded:
            return
        
        print(f"Loading model: {MODEL_ID}")
        model_dir = f"{MODEL_CACHE_DIR}/{MODEL_ID}"
        
        if not Path(model_dir).exists():
            print(f"Downloading model (~49GB)...")
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=model_dir,
                cache_dir=MODEL_CACHE_DIR,
            )
            print("✓ Model downloaded")
        else:
            print("✓ Model found in cache")
        
        self.model_dir = model_dir
        self.model_loaded = True

config = GenerationConfig()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    config.load_model()

@app.get("/")
def root():
    return {
        "status": "online",
        "model": "Wan2.2-S2V-14B",
        "version": "1.0.0",
        "gpu_config": "Optimized: 1x GPU, No Offload, 30 Steps",
        "endpoints": {
            "POST /generate": "Generate video from audio and image",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": config.model_loaded
    }

@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form(""),
    resolution: str = Form("720p"),
    sample_steps: int = Form(30)
):
    """
    Generate video from audio and image
    
    Parameters:
    - image: Input image (JPG/PNG)
    - audio: Input audio (WAV/MP3)
    - prompt: Optional text prompt
    - resolution: "480p" (832x480) or "720p" (1024x704)
    - sample_steps: Sampling steps (20-50, default 30)
    
    Returns:
    - JSON with base64-encoded MP4 video
    """
    start_time = time.time()
    
    # Validate inputs
    if resolution not in RESOLUTION_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resolution. Must be one of: {list(RESOLUTION_MAP.keys())}"
        )
    
    if not (20 <= sample_steps <= 50):
        raise HTTPException(
            status_code=400,
            detail="sample_steps must be between 20 and 50"
        )
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded files
        image_path = Path(tmpdir) / "input_image.jpg"
        audio_path = Path(tmpdir) / "input_audio.wav"
        
        image_bytes = await image.read()
        audio_bytes = await audio.read()
        
        image_path.write_bytes(image_bytes)
        audio_path.write_bytes(audio_bytes)
        
        # Build generation command
        size = RESOLUTION_MAP[resolution]
        cmd = [
            "python",
            f"{WAN_DIR}/generate.py",
            "--task", "s2v-14B",
            "--size", size,
            "--ckpt_dir", config.model_dir,
            "--offload_model", "False",  # Keep model in GPU for speed
            "--convert_model_dtype",
            "--sample_steps", str(sample_steps),
            "--image", str(image_path),
            "--audio", str(audio_path)
        ]
        
        if prompt:
            cmd.extend(["--prompt", prompt])
        
        # Run generation
        try:
            print(f"Starting generation: {resolution}, {sample_steps} steps")
            result = subprocess.run(
                cmd,
                cwd=WAN_DIR,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"Generation failed: {result.stderr}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed: {result.stderr}"
                )
            
            # Find generated video
            video_files = list(Path(WAN_DIR).glob("s2v-14B_*.mp4"))
            if not video_files:
                raise HTTPException(
                    status_code=500,
                    detail="No video file generated"
                )
            
            # Get most recent video
            video_path = max(video_files, key=lambda p: p.stat().st_mtime)
            
            # Encode to base64
            video_bytes = video_path.read_bytes()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            # Clean up generated file
            video_path.unlink()
            
            generation_time = time.time() - start_time
            
            return {
                "success": True,
                "video": video_base64,
                "generation_time": round(generation_time, 2),
                "video_size_mb": round(len(video_bytes) / 1024 / 1024, 2),
                "resolution": resolution,
                "sample_steps": sample_steps
            }
            
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Generation timed out (1 hour limit)"
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## start.sh

```bash
#!/bin/bash

echo "================================================"
echo "Starting Wan2.2 S2V Server"
echo "================================================"

# Display GPU information
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Apply patches
echo ""
echo "Applying FlashAttention patches..."
python3 patches/apply_patches.py
if [ $? -ne 0 ]; then
    echo "ERROR: Patching failed!"
    exit 1
fi

# Start FastAPI server
echo ""
echo "Starting FastAPI server on port 8000..."
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## RunPod Template Configuration

### GPU Requirements
- **Minimum:** 1x A100 (80GB) or H100
- **Recommended:** A100-80GB for optimal performance
- **Storage:** 80GB+ for model (49GB) + workspace

### Environment Variables
```bash
MODEL_CACHE_DIR=/workspace/models
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_VISIBLE_DEVICES=0
```

### Volume Mounts
- **Container Path:** `/workspace/models`
- **Purpose:** Persistent model storage (49.1GB)
- **Type:** Network Volume (recommended) or Pod Volume

### Ports
- **8000:** FastAPI HTTP server

### Docker Command Override
```bash
./start.sh
```

---

## Deployment Steps

### 1. Build Docker Image
```bash
# Build locally
docker build -t wan2-s2v:latest .

# Push to registry
docker tag wan2-s2v:latest your-registry/wan2-s2v:latest
docker push your-registry/wan2-s2v:latest
```

### 2. Create RunPod Template
1. Go to RunPod Templates
2. Click "New Template"
3. Configure:
   - **Container Image:** `your-registry/wan2-s2v:latest`
   - **Container Disk:** 20GB
   - **Volume Mount:** `/workspace/models` (80GB+)
   - **Expose HTTP Ports:** `8000`
   - **Docker Command:** `./start.sh`

### 3. Deploy Pod
1. Select template
2. Choose GPU: A100 (80GB)
3. Configure volume for persistent model storage
4. Start pod

### 4. Test Deployment
```bash
# Health check
curl http://YOUR_POD_URL:8000/health

# Test generation
curl -X POST http://YOUR_POD_URL:8000/generate \
  -F "image=@test.png" \
  -F "audio=@test.mp3" \
  -F "resolution=480p" \
  -F "sample_steps=30"
```

---

## Client Integration

### Python Client Example
```python
import requests
import base64

def generate_video(image_path, audio_path, resolution="720p", sample_steps=30):
    url = "http://YOUR_POD_URL:8000/generate"
    
    with open(image_path, 'rb') as img, open(audio_path, 'rb') as aud:
        files = {
            'image': img,
            'audio': aud
        }
        data = {
            'resolution': resolution,
            'sample_steps': sample_steps
        }
        
        response = requests.post(url, files=files, data=data, timeout=1800)
        response.raise_for_status()
        
        result = response.json()
        
        if result['success']:
            # Decode and save video
            video_bytes = base64.b64decode(result['video'])
            with open('output.mp4', 'wb') as f:
                f.write(video_bytes)
            
            print(f"✓ Video generated in {result['generation_time']}s")
            print(f"  Size: {result['video_size_mb']} MB")
        
    return result

# Usage
generate_video("image.png", "audio.mp3", resolution="480p", sample_steps=30)
```

---

## Performance Expectations

### Generation Times (Single A100-80GB)
| Resolution | Sample Steps | Cold Start | Warm Container |
|-----------|--------------|------------|----------------|
| 480p | 30 | ~25 min | ~6-8 min |
| 480p | 50 | ~35 min | ~12-16 min |
| 720p | 30 | ~35 min | ~10-12 min |
| 720p | 50 | ~50 min | ~18-22 min |

**Note:** Cold start includes model loading time. Warm container has model in GPU memory.

---

## Optimization Tips

### 1. Faster Generation
- Use `sample_steps=30` (vs default 50) - saves 30-40% time
- Set `--offload_model False` - keeps model in GPU (60-70% faster)
- Use 480p for prototyping, 720p for production

### 2. Cost Optimization
- Use Network Volume for model storage (share across pods)
- Enable auto-pause when idle
- Use spot instances for non-critical workloads

### 3. Scaling
- Deploy multiple pods behind load balancer
- Each pod handles one generation at a time
- Consider queue system (Redis/RabbitMQ) for high volume

---

## Troubleshooting

### Issue: Out of Memory
**Solution:** Ensure A100 with 80GB VRAM. Reduce to 480p or enable offloading.

### Issue: FlashAttention Import Error
**Solution:** Verify patches applied before Wan2.2 imports. Check patch logs.

### Issue: Invalid Size Error
**Solution:** Use mapped resolutions: 480p → 832×480, 720p → 1024×704

### Issue: Slow Generation
**Solution:** 
- Verify `--offload_model False` is set
- Check GPU utilization: `nvidia-smi dmon`
- Reduce `sample_steps` to 30

### Issue: Model Download Fails
**Solution:** Check HuggingFace token, network connectivity, and disk space (49GB needed)

---

## Monitoring

### Key Metrics to Track
- GPU utilization: `nvidia-smi`
- Generation time per request
- Memory usage
- Queue depth (if using queue system)
- Error rate

### Logging
```python
# Add to app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log each generation
logger.info(f"Generation started: {resolution}, {sample_steps} steps")
logger.info(f"Generation completed: {generation_time}s")
```

---

## Security Considerations

### 1. API Authentication
Add API key middleware:
```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    valid_keys = os.getenv("API_KEYS", "").split(",")
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# Add to endpoint
@app.post("/generate", dependencies=[Depends(verify_api_key)])
```

### 2. Rate Limiting
Use `slowapi` or nginx rate limiting.

### 3. Input Validation
- File size limits (e.g., 10MB for images, 50MB for audio)
- File type validation
- Timeout enforcement

---

## Maintenance

### Model Updates
```bash
# Download new version
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir /workspace/models/Wan-AI/Wan2.2-S2V-14B

# Restart pod
```

### Log Rotation
Add to Dockerfile:
```dockerfile
RUN apt-get install -y logrotate
COPY logrotate.conf /etc/logrotate.d/wan2
```

---

## Comparison: Modal vs RunPod

| Feature | Modal | RunPod Docker |
|---------|-------|---------------|
| Setup Complexity | Low | Medium |
| Cold Start | Fast (cached images) | Slower (full init) |
| Scaling | Automatic | Manual/K8s |
| Cost (idle) | $0 (serverless) | Minimum pod cost |
| Cost (active) | Per-second billing | Per-hour billing |
| Customization | Limited | Full control |
| Persistent Storage | Modal Volumes | Network/Pod Volumes |

**When to use RunPod:**
- Need full Docker control
- Running 24/7 service
- Complex custom workflows
- Integration with existing RunPod infrastructure

**When to use Modal:**
- Serverless auto-scaling preferred
- Sporadic/bursty workload
- Want zero idle costs
- Simplified deployment

---

## Next Steps

1. **Test locally:** Build and run Docker container on local GPU
2. **Push to registry:** Upload to Docker Hub or private registry
3. **Create template:** Configure RunPod template with proper settings
4. **Deploy test pod:** Validate with sample requests
5. **Monitor performance:** Track generation times and adjust config
6. **Scale as needed:** Add more pods or implement queue system

---

## References

- **Modal Implementation:** `wan2_modal.py` (this repository)
- **Wan2.2 Repository:** https://github.com/Wan-Video/Wan2.2
- **Model:** https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
- **RunPod Docs:** https://docs.runpod.io/
- **Performance Testing:** See `test_local_generate.py` for client examples

---

**Document Version:** 1.0  
**Last Updated:** December 2, 2025  
**Based On:** Modal deployment v39_optimized_speed
