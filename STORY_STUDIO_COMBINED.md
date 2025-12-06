# Story Studio Combined Container

## Overview

This document outlines the architecture for a combined RunPod serverless container that includes:
1. **Z-Image Model** - Fast image generation via ComfyUI
2. **Wan2.2 S2V 14B** - Image + Audio → Talking Video generation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Story Studio Combined Container                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐      ┌─────────────────────────────┐  │
│  │   Z-Image (ComfyUI) │      │      Wan2.2 S2V 14B         │  │
│  ├─────────────────────┤      ├─────────────────────────────┤  │
│  │ • Fast image gen    │      │ • Image + Audio → Video     │  │
│  │ • ~5-10 sec/image   │      │ • ~2 min/video (warm)       │  │
│  │ • ~8GB VRAM         │      │ • ~67GB VRAM                │  │
│  └─────────────────────┘      └─────────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Smart Model Router                    │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ • Loads models on-demand                                │    │
│  │ • Unloads inactive model to free VRAM                   │    │
│  │ • Batches same-type requests for efficiency             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Workflow: Complete Story Project

### Typical Project: 10 Scenes

```
User Request:
{
  "project_id": "story_123",
  "scenes": [
    {"scene": 1, "prompt": "A hero standing on a mountain", "audio_url": "..."},
    {"scene": 2, "prompt": "The hero meets a dragon", "audio_url": "..."},
    ...
  ]
}

Processing Flow:
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: Image Generation (Z-Image loaded, ~2 min total)    │
├──────────────────────────────────────────────────────────────┤
│ Scene 1 → Image 1 (10 sec)                                   │
│ Scene 2 → Image 2 (10 sec)                                   │
│ ...                                                          │
│ Scene 10 → Image 10 (10 sec)                                 │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 2: Model Swap (~3-5 min one-time)                      │
├──────────────────────────────────────────────────────────────┤
│ Unload Z-Image from VRAM                                     │
│ Load Wan2.2 S2V 14B into VRAM                                │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: Video Generation (Wan2.2 loaded, ~20 min total)     │
├──────────────────────────────────────────────────────────────┤
│ Image 1 + Audio 1 → Video 1 (2 min)                          │
│ Image 2 + Audio 2 → Video 2 (2 min)                          │
│ ...                                                          │
│ Image 10 + Audio 10 → Video 10 (2 min)                       │
└──────────────────────────────────────────────────────────────┘

Total Time: ~25-30 minutes for complete 10-scene project
(vs 150+ minutes with cold starts)
```

## API Endpoints

### 1. Generate Image

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {api_key}" \
  -d '{
    "input": {
      "task": "image",
      "prompt": "A professional businesswoman in a modern office, portrait style",
      "negative_prompt": "blurry, low quality",
      "width": 832,
      "height": 480,
      "steps": 20
    }
  }'
```

**Response:**
```json
{
  "id": "job-123",
  "status": "COMPLETED",
  "output": {
    "image_url": "https://parentearn.com/StoryStudio/img_20251204_123456.png",
    "generation_time": 8.5
  }
}
```

### 2. Generate Video (Image + Audio → Talking Video)

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {api_key}" \
  -d '{
    "input": {
      "task": "video",
      "image": "https://parentearn.com/StoryStudio/img_123.png",
      "audio": "https://parentearn.com/StoryStudio/audio_123.mp3",
      "prompt": "Person speaking naturally with subtle expressions",
      "resolution": "480p",
      "sample_steps": 20
    }
  }'
```

**Response:**
```json
{
  "id": "job-456",
  "status": "COMPLETED",
  "output": {
    "video_url": "https://parentearn.com/VideoGen/20251204_123456_s2v_480p.mp4",
    "generation_time": 120.5,
    "resolution": "480p"
  }
}
```

### 3. Batch Process (Full Project)

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {api_key}" \
  -d '{
    "input": {
      "task": "batch",
      "project_id": "story_123",
      "scenes": [
        {
          "id": "scene_1",
          "prompt": "A hero standing on a mountain at sunrise",
          "audio_url": "https://example.com/audio1.mp3"
        },
        {
          "id": "scene_2", 
          "prompt": "The hero meets a friendly dragon",
          "audio_url": "https://example.com/audio2.mp3"
        }
      ],
      "image_settings": {
        "width": 832,
        "height": 480,
        "steps": 20
      },
      "video_settings": {
        "resolution": "480p",
        "sample_steps": 20
      }
    }
  }'
```

**Response:**
```json
{
  "id": "batch-789",
  "status": "COMPLETED",
  "output": {
    "project_id": "story_123",
    "scenes": [
      {
        "id": "scene_1",
        "image_url": "https://parentearn.com/StoryStudio/scene_1_img.png",
        "video_url": "https://parentearn.com/StoryStudio/scene_1_video.mp4"
      },
      {
        "id": "scene_2",
        "image_url": "https://parentearn.com/StoryStudio/scene_2_img.png",
        "video_url": "https://parentearn.com/StoryStudio/scene_2_video.mp4"
      }
    ],
    "total_time": 1520.5,
    "images_time": 18.2,
    "videos_time": 1498.3
  }
}
```

## Repository Structure

```
storystudio-combined/
├── Dockerfile
├── handler.py              # Main RunPod handler with task routing
├── models/
│   ├── image_generator.py  # Z-Image / ComfyUI wrapper
│   └── video_generator.py  # Wan2.2 S2V wrapper
├── utils/
│   ├── model_manager.py    # Smart model loading/unloading
│   ├── storage.py          # R2/S3 upload utilities
│   └── download.py         # URL download utilities
├── comfyui/
│   └── workflows/
│       └── z_image.json    # ComfyUI workflow for Z-Image
├── patches/
│   └── apply_patches.py    # FlashAttention patches for Wan2.2
├── requirements.txt
└── README.md
```

## Docker Configuration

### Dockerfile

```dockerfile
# Combined Story Studio Container
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/runpod-volume \
    HF_HOME=/runpod-volume/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libglib2.0-0 libgl1-mesa-glx \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    git wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and setup ComfyUI for Z-Image
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && pip install -r requirements.txt

# Clone Wan2.2 for video generation
RUN git clone https://github.com/Wan-Video/Wan2.2.git && \
    rm -rf Wan2.2/.git

# Copy application files
COPY . .
RUN chmod +x *.sh 2>/dev/null || true

# Create directories
RUN mkdir -p /runpod-volume/models /workspace/outputs

EXPOSE 8000

CMD ["python3", "-u", "handler.py"]
```

### requirements.txt

```
# RunPod
runpod==1.7.5

# PyTorch (included in base image)
# torch==2.4.0

# Transformers & Diffusers
transformers==4.51.3
diffusers==0.31.0
accelerate==1.1.1

# Image processing
pillow==11.0.0
opencv-python-headless==4.10.0.84

# Video processing
imageio==2.36.1
imageio-ffmpeg==0.5.1

# Audio processing
librosa==0.10.2
soundfile==0.12.1

# Storage
boto3==1.35.76
requests==2.32.3

# Utilities
huggingface-hub==0.30.0
safetensors==0.4.5
einops==0.8.0
omegaconf==2.3.0
numpy==1.26.4
scipy==1.14.1

# ComfyUI dependencies
aiohttp
tqdm
pyyaml
```

## Model Manager Implementation

```python
# models/model_manager.py
import gc
import torch
from enum import Enum
from typing import Optional

class ActiveModel(Enum):
    NONE = "none"
    Z_IMAGE = "z_image"
    WAN2_S2V = "wan2_s2v"

class ModelManager:
    def __init__(self):
        self.active_model: ActiveModel = ActiveModel.NONE
        self.z_image_pipeline = None
        self.wan2_loaded = False
    
    def ensure_z_image_loaded(self):
        """Load Z-Image model, unload Wan2.2 if needed"""
        if self.active_model == ActiveModel.Z_IMAGE:
            return  # Already loaded
        
        # Unload Wan2.2 if loaded
        if self.active_model == ActiveModel.WAN2_S2V:
            print("Unloading Wan2.2 S2V...")
            self._unload_wan2()
        
        # Load Z-Image
        print("Loading Z-Image model...")
        self._load_z_image()
        self.active_model = ActiveModel.Z_IMAGE
    
    def ensure_wan2_loaded(self):
        """Load Wan2.2, unload Z-Image if needed"""
        if self.active_model == ActiveModel.WAN2_S2V:
            return  # Already loaded
        
        # Unload Z-Image if loaded
        if self.active_model == ActiveModel.Z_IMAGE:
            print("Unloading Z-Image...")
            self._unload_z_image()
        
        # Load Wan2.2
        print("Loading Wan2.2 S2V 14B...")
        self._load_wan2()
        self.active_model = ActiveModel.WAN2_S2V
    
    def _load_z_image(self):
        # Load via ComfyUI or diffusers
        from diffusers import StableDiffusionXLPipeline
        self.z_image_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
    
    def _unload_z_image(self):
        if self.z_image_pipeline:
            del self.z_image_pipeline
            self.z_image_pipeline = None
        self._clear_cuda_cache()
    
    def _load_wan2(self):
        # Wan2.2 is loaded per-request via subprocess
        # Just mark as active
        self.wan2_loaded = True
    
    def _unload_wan2(self):
        # Wan2.2 runs in subprocess, just clear any cached state
        self.wan2_loaded = False
        self._clear_cuda_cache()
    
    def _clear_cuda_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"VRAM cleared. Available: {torch.cuda.memory_allocated()/1e9:.1f}GB used")

# Global instance
model_manager = ModelManager()
```

## Handler Implementation

```python
# handler.py
import runpod
import time
from models.model_manager import model_manager, ActiveModel

def generate_image(job_input: dict) -> dict:
    """Generate image using Z-Image model"""
    model_manager.ensure_z_image_loaded()
    
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "")
    width = job_input.get("width", 832)
    height = job_input.get("height", 480)
    steps = job_input.get("steps", 20)
    
    start_time = time.time()
    
    # Generate image
    image = model_manager.z_image_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps
    ).images[0]
    
    # Save and upload
    from utils.storage import upload_image_to_r2
    image_url = upload_image_to_r2(image)
    
    return {
        "image_url": image_url,
        "generation_time": round(time.time() - start_time, 2)
    }

def generate_video(job_input: dict) -> dict:
    """Generate video using Wan2.2 S2V"""
    model_manager.ensure_wan2_loaded()
    
    # Use existing Wan2.2 S2V logic from storystudio-wan2-s2v
    from models.video_generator import run_wan2_generation
    return run_wan2_generation(job_input)

def process_batch(job_input: dict) -> dict:
    """Process full project: all images, then all videos"""
    scenes = job_input.get("scenes", [])
    image_settings = job_input.get("image_settings", {})
    video_settings = job_input.get("video_settings", {})
    
    results = []
    start_time = time.time()
    
    # Phase 1: Generate all images (model loaded once)
    print(f"Phase 1: Generating {len(scenes)} images...")
    images_start = time.time()
    
    for scene in scenes:
        img_result = generate_image({
            "prompt": scene["prompt"],
            **image_settings
        })
        scene["image_url"] = img_result["image_url"]
    
    images_time = time.time() - images_start
    print(f"Images complete in {images_time:.1f}s")
    
    # Phase 2: Generate all videos (model swapped once)
    print(f"Phase 2: Generating {len(scenes)} videos...")
    videos_start = time.time()
    
    for scene in scenes:
        vid_result = generate_video({
            "image": scene["image_url"],
            "audio": scene["audio_url"],
            "prompt": scene.get("video_prompt", "Person speaking naturally"),
            **video_settings
        })
        scene["video_url"] = vid_result["video_url"]
        results.append({
            "id": scene["id"],
            "image_url": scene["image_url"],
            "video_url": scene["video_url"]
        })
    
    videos_time = time.time() - videos_start
    
    return {
        "project_id": job_input.get("project_id"),
        "scenes": results,
        "total_time": round(time.time() - start_time, 2),
        "images_time": round(images_time, 2),
        "videos_time": round(videos_time, 2)
    }

def handler(job: dict) -> dict:
    """Main RunPod handler - routes to appropriate task"""
    job_input = job.get("input", {})
    task = job_input.get("task", "video")  # Default to video for backward compat
    
    try:
        if task == "image":
            return generate_image(job_input)
        elif task == "video":
            return generate_video(job_input)
        elif task == "batch":
            return process_batch(job_input)
        else:
            return {"error": f"Unknown task: {task}. Use 'image', 'video', or 'batch'"}
    except Exception as e:
        return {"error": str(e)}

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
```

## GPU Requirements

| Configuration | VRAM Needed | Recommended GPU |
|---------------|-------------|-----------------|
| Z-Image only | ~8-12 GB | RTX 4090, A10 |
| Wan2.2 only | ~67 GB | A100 80GB, H100 |
| Combined (swap) | ~67 GB | A100 80GB, H100 |

## Cost Optimization

### Warm Worker Strategy

```python
# In your backend application
async def process_project(project_id: str, scenes: list):
    # 1. Enable warm worker before batch
    await runpod_api.update_endpoint(workers_min=1)
    
    # 2. Wait for worker to be ready
    await wait_for_worker_ready()
    
    # 3. Submit batch job
    result = await runpod_api.run({
        "task": "batch",
        "project_id": project_id,
        "scenes": scenes
    })
    
    # 4. Optionally disable warm worker after completion
    # Or keep warm if more projects expected
    await runpod_api.update_endpoint(workers_min=0)
    
    return result
```

### Estimated Costs (A100 80GB @ $1.89/hr)

| Scenario | Time | Cost |
|----------|------|------|
| Single video (cold) | 15 min | $0.47 |
| Single video (warm) | 2 min | $0.06 |
| 10-scene project (batch) | 30 min | $0.95 |
| 10 videos (cold starts) | 150 min | $4.73 |

**Batch processing saves ~80% on multi-scene projects!**

## Next Steps

1. **Create new repository**: `storystudio-combined`
2. **Port Wan2.2 video generation** from `storystudio-wan2-s2v`
3. **Integrate Z-Image/ComfyUI** for image generation
4. **Implement model manager** for smart VRAM handling
5. **Add batch processing endpoint**
6. **Test full project workflow**

## Related Repositories

- **storystudio-wan2-s2v** - Current video-only implementation
- **ComfyUI** - https://github.com/comfyanonymous/ComfyUI
- **Z-Image Examples** - https://comfyanonymous.github.io/ComfyUI_examples/z_image/
