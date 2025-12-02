# Wan2.2 S2V RunPod Serverless Deployment

Docker deployment for Wan2.2-S2V-14B on RunPod serverless infrastructure. This implementation includes critical FlashAttention compatibility patches and optimized configuration for production use.

## 🚀 Quick Start

### Prerequisites

- Docker Hub account
- Docker installed locally (optional, for local testing)
- RunPod account with serverless access

### 1. Build Docker Image

**Linux/Mac:**
```bash
export DOCKER_USERNAME=your-dockerhub-username
./build.sh
```

**Windows:**
```cmd
set DOCKER_USERNAME=your-dockerhub-username
build.bat
```

### 2. Deploy to Docker Hub

**Linux/Mac:**
```bash
./deploy.sh
```

**Windows:**
```cmd
deploy.bat
```

### 3. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **"New Endpoint"**
3. Configure:
   - **Container Image:** `your-dockerhub-username/wan2-s2v:latest`
   - **GPU Type:** A100 (80GB) or H100
   - **Container Disk:** 20GB
   - **Volume Mount:** `/runpod-volume` (recommended 80GB+ for model cache)
   - **Environment Variables:**
     - `MODEL_CACHE_DIR=/runpod-volume/models`
4. Deploy and wait for initialization

## 📋 Features

- ✅ **Serverless Ready:** RunPod serverless handler with auto-scaling
- ✅ **FlashAttention Patched:** Works without FlashAttention compilation
- ✅ **Model Caching:** Persistent volume support (49GB model)
- ✅ **Optimized Performance:** 1x GPU, no offload, 30 sampling steps
- ✅ **Multiple Modes:** Serverless handler + FastAPI server
- ✅ **Resolution Support:** 480p (832×480) and 720p (1024×704)
- ✅ **Base64 I/O:** Compatible with standard serverless workflows

## 🏗️ Architecture

```
wan2-s2v/
├── Dockerfile              # Multi-stage build with CUDA 12.1
├── requirements.txt        # Pinned dependencies
├── handler.py             # RunPod serverless handler
├── app.py                 # FastAPI server (pod mode)
├── start.sh               # Startup script
├── patches/               # FlashAttention compatibility
│   ├── apply_patches.py
│   └── README.md
├── build.sh / build.bat   # Build scripts
├── deploy.sh / deploy.bat # Deployment scripts
└── docs/                  # Documentation
```

## 📊 Performance Expectations

### Generation Times (Single A100-80GB)

| Resolution | Sample Steps | Cold Start | Warm Container |
|-----------|--------------|------------|----------------|
| 480p | 30 | ~25 min | ~6-8 min |
| 720p | 30 | ~35 min | ~10-12 min |
| 480p | 50 | ~35 min | ~12-16 min |
| 720p | 50 | ~50 min | ~18-22 min |

**Note:** Cold start includes model download time on first run.

## 🔧 Usage

### Serverless API (RunPod)

```python
import runpod
import base64

# Configure RunPod
runpod.api_key = "your-runpod-api-key"

# Read and encode inputs
with open("image.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

with open("audio.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Submit job
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")
run_request = endpoint.run({
    "image": image_base64,
    "audio": audio_base64,
    "resolution": "720p",
    "sample_steps": 30,
    "prompt": "A person talking"
})

# Wait for completion
status = run_request.status()
print(f"Status: {status}")

# Get result
result = run_request.output()
if "video" in result:
    # Decode and save video
    video_bytes = base64.b64decode(result["video"])
    with open("output.mp4", "wb") as f:
        f.write(video_bytes)
    print(f"Generated in {result['generation_time']}s")
```

### Pod Mode (FastAPI)

If running as a standard pod with the FastAPI server:

```python
import requests
import base64

url = "http://your-pod-url:8000/generate"

with open("image.png", "rb") as img, open("audio.mp3", "rb") as aud:
    files = {
        "image": img,
        "audio": aud
    }
    data = {
        "resolution": "720p",
        "sample_steps": 30
    }
    
    response = requests.post(url, files=files, data=data, timeout=1800)
    result = response.json()
    
    if result["success"]:
        video_bytes = base64.b64decode(result["video"])
        with open("output.mp4", "wb") as f:
            f.write(video_bytes)
```

## 🐳 Local Testing

Test the Docker image locally before deploying:

```bash
# Run with GPU support
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/models:/runpod-volume/models \
    your-dockerhub-username/wan2-s2v:latest

# Test health endpoint
curl http://localhost:8000/health

# Test generation
curl -X POST http://localhost:8000/generate \
    -F "image=@test.png" \
    -F "audio=@test.mp3" \
    -F "resolution=480p" \
    -F "sample_steps=30"
```

## ⚙️ Configuration

### Environment Variables

- `MODEL_CACHE_DIR`: Model storage path (default: `/runpod-volume/models`)
- `PYTORCH_CUDA_ALLOC_CONF`: PyTorch CUDA allocator config
- `CUDA_VISIBLE_DEVICES`: GPU selection

### Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| image | base64 string | Yes | - | Input image (JPG/PNG) |
| audio | base64 string | Yes | - | Input audio (WAV/MP3) |
| resolution | string | No | 720p | "480p" or "720p" |
| sample_steps | integer | No | 30 | 20-50, affects quality/speed |
| prompt | string | No | "" | Optional text prompt |

### Output Format

```json
{
    "video": "base64_encoded_mp4_video",
    "generation_time": 420.5,
    "video_size_mb": 12.3,
    "resolution": "720p",
    "sample_steps": 30
}
```

## 🔍 Troubleshooting

### Issue: Out of Memory
- **Solution:** Use A100-80GB or H100, reduce to 480p

### Issue: FlashAttention Import Error
- **Solution:** Patches should auto-apply. Check logs for patch verification

### Issue: Model Download Slow/Fails
- **Solution:** Use persistent volume, ensure adequate storage (49GB)

### Issue: Generation Timeout
- **Solution:** Increase timeout to 3600s, reduce sample_steps to 30

### View Logs

In RunPod console, check endpoint logs for detailed error messages.

## 📈 Optimization Tips

### Speed
- Use `sample_steps=30` instead of 50 (30-40% faster)
- Keep `offload_model=False` (60-70% faster)
- Use 480p for prototyping

### Cost
- Use persistent network volume for model storage
- Configure auto-pause when idle
- Use spot instances for non-critical workloads

## 🔒 Security

### API Authentication (Recommended)

Add authentication to the FastAPI server by modifying `app.py`:

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    valid_keys = os.getenv("API_KEYS", "").split(",")
    if x_api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

@app.post("/generate", dependencies=[Depends(verify_api_key)])
```

## 📚 Documentation

- [RunPod Serverless Docs](https://docs.runpod.io/serverless/overview)
- [Wan2.2 Repository](https://github.com/Wan-Video/Wan2.2)
- [Model on HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
- [Detailed Deployment Guide](DEPLOYMENT_GUIDE.md)

## 📝 License

This deployment configuration is provided as-is. Wan2.2 model has its own license from Wan-AI.

## 🤝 Support

For issues specific to this deployment:
1. Check the troubleshooting section
2. Review RunPod logs
3. Verify Docker image build succeeded

For Wan2.2 model issues:
- See [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)

---

**Version:** 1.0.0  
**Last Updated:** December 2, 2025  
**Based On:** Modal deployment with performance optimizations
