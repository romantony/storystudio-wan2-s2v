# Wan2.2 S2V Deployment Guide

Complete step-by-step guide for deploying Wan2.2-S2V-14B to RunPod serverless.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building the Docker Image](#building-the-docker-image)
3. [Deploying to Docker Hub](#deploying-to-docker-hub)
4. [Setting Up RunPod Serverless](#setting-up-runpod-serverless)
5. [Testing Your Deployment](#testing-your-deployment)
6. [Production Considerations](#production-considerations)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Prerequisites

### Required Accounts

1. **Docker Hub Account**
   - Sign up at [hub.docker.com](https://hub.docker.com)
   - Create an access token:
     - Go to Account Settings > Security
     - Click "New Access Token"
     - Name it "RunPod" with Read/Write permissions
     - Save the token securely

2. **RunPod Account**
   - Sign up at [runpod.io](https://www.runpod.io)
   - Add payment method
   - Navigate to Serverless section

### Required Tools

- **Docker:** [Install Docker](https://docs.docker.com/get-docker/)
- **Git:** [Install Git](https://git-scm.com/downloads)
- **Python 3.8+:** For testing scripts

---

## Building the Docker Image

### Step 1: Clone or Download Repository

```bash
# If this is a git repository
git clone <repository-url>
cd storystudio-wan2-s2v

# Or navigate to the directory
cd c:\runpod\storystudio-wan2-s2v
```

### Step 2: Set Docker Username

**Linux/Mac:**
```bash
export DOCKER_USERNAME=your-dockerhub-username
```

**Windows (Command Prompt):**
```cmd
set DOCKER_USERNAME=your-dockerhub-username
```

**Windows (PowerShell):**
```powershell
$env:DOCKER_USERNAME="your-dockerhub-username"
```

### Step 3: Build the Image

The build process will:
- Install CUDA 12.1 and Python 3.11
- Install all dependencies (~5GB)
- Clone Wan2.2 repository
- Apply FlashAttention patches
- Create optimized container

**Linux/Mac:**
```bash
chmod +x build.sh deploy.sh
./build.sh
```

**Windows:**
```cmd
build.bat
```

**Build time:** ~10-20 minutes depending on internet speed

**Expected output:**
```
Building Docker image...
[+] Building 650.2s (15/15) FINISHED
✓ Build complete!

Images created:
  - your-username/wan2-s2v:1.0.0
  - your-username/wan2-s2v:latest
```

### Step 4: Verify Build

```bash
# Check image exists
docker images | grep wan2-s2v

# Check image size (should be ~12-15GB)
docker images your-username/wan2-s2v:latest
```

---

## Deploying to Docker Hub

### Step 1: Login to Docker Hub

```bash
docker login
# Enter your Docker Hub username and access token (not password!)
```

### Step 2: Push to Docker Hub

**Linux/Mac:**
```bash
./deploy.sh
```

**Windows:**
```cmd
deploy.bat
```

**Push time:** ~20-40 minutes depending on upload speed (image is ~12-15GB)

**Expected output:**
```
Pushing your-username/wan2-s2v:1.0.0...
The push refers to repository [docker.io/your-username/wan2-s2v]
...
✓ Deployment complete!
```

### Step 3: Verify on Docker Hub

1. Go to [hub.docker.com](https://hub.docker.com)
2. Navigate to your repositories
3. Verify `wan2-s2v` repository exists
4. Check that tags `latest` and `1.0.0` are present

---

## Setting Up RunPod Serverless

### Step 1: Create Serverless Endpoint

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **"New Endpoint"**

### Step 2: Configure Template

**Basic Settings:**
- **Name:** `wan2-s2v-serverless`
- **Container Image:** `your-dockerhub-username/wan2-s2v:latest`
- **Container Registry Credentials:** (if using private registry)

**Compute Configuration:**
- **GPU Type:** Select `A100 SXM 80GB` or `H100`
- **Container Disk:** `20 GB` (minimum)
- **Volume Size:** `80 GB` (for model storage)
- **Volume Mount Path:** `/runpod-volume`

**Environment Variables:**
```
MODEL_CACHE_DIR=/runpod-volume/models
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Advanced Settings:**
- **Max Workers:** Start with `1` (can scale later)
- **Idle Timeout:** `5 minutes` (to save costs)
- **Execution Timeout:** `3600 seconds` (1 hour)
- **Max Concurrent Requests:** `1` per worker

### Step 3: Deploy Endpoint

1. Review configuration
2. Click **"Deploy"**
3. Wait for endpoint to initialize (~5-10 minutes)
4. Note your **Endpoint ID** (e.g., `abc123def456`)

### Step 4: Get API Key

1. In RunPod Console, go to **Settings > API Keys**
2. Create new API key if needed
3. Copy the API key securely

---

## Testing Your Deployment

### Method 1: Using RunPod Python SDK

```python
import runpod
import base64
from pathlib import Path

# Configure
runpod.api_key = "your-runpod-api-key"
ENDPOINT_ID = "your-endpoint-id"

# Prepare inputs
image_path = "test_image.png"
audio_path = "test_audio.mp3"

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Submit job
endpoint = runpod.Endpoint(ENDPOINT_ID)
job = endpoint.run({
    "image": image_b64,
    "audio": audio_b64,
    "resolution": "480p",  # Start with 480p for faster testing
    "sample_steps": 30
})

# Monitor status
print(f"Job ID: {job.job_id}")
print("Status:", job.status())

# Wait for completion (this will take 6-25 minutes)
result = job.output(timeout=3600)

# Save result
if "video" in result:
    video_bytes = base64.b64decode(result["video"])
    with open("output.mp4", "wb") as f:
        f.write(video_bytes)
    print(f"✓ Video generated in {result['generation_time']}s")
    print(f"  Size: {result['video_size_mb']} MB")
else:
    print(f"✗ Error: {result.get('error', 'Unknown error')}")
```

### Method 2: Using curl (for testing)

First, encode your files to base64:

```bash
# Linux/Mac
IMAGE_B64=$(base64 -w 0 test_image.png)
AUDIO_B64=$(base64 -w 0 test_audio.mp3)

# Windows (PowerShell)
$IMAGE_B64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("test_image.png"))
$AUDIO_B64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("test_audio.mp3"))
```

Then make API request:

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "'$IMAGE_B64'",
      "audio": "'$AUDIO_B64'",
      "resolution": "480p",
      "sample_steps": 30
    }
  }'
```

### Expected Timeline

1. **Job Submission:** Immediate (< 1 second)
2. **Cold Start:** 5-10 minutes (model download + initialization)
3. **Generation:** 6-25 minutes depending on resolution/steps
4. **Total (first run):** 15-35 minutes
5. **Subsequent runs:** 6-25 minutes (warm container)

---

## Production Considerations

### 1. Cost Optimization

**GPU Selection:**
- A100 80GB: ~$2.50/hour
- H100: ~$4.00/hour (faster but more expensive)

**Strategies:**
- Use aggressive idle timeout (5 minutes)
- Scale workers based on demand
- Consider spot instances for non-critical workloads

**Estimated costs (A100):**
- Single 720p video (30 steps): ~$0.50-$1.00
- Single 480p video (30 steps): ~$0.30-$0.50

### 2. Performance Tuning

**For Speed:**
```json
{
    "resolution": "480p",
    "sample_steps": 30
}
```

**For Quality:**
```json
{
    "resolution": "720p",
    "sample_steps": 50
}
```

**Balanced:**
```json
{
    "resolution": "720p",
    "sample_steps": 30
}
```

### 3. Scaling

**Horizontal Scaling:**
- Increase max workers in endpoint settings
- Each worker handles 1 video at a time
- RunPod auto-scales based on queue depth

**Queue Management:**
- Jobs automatically queue when workers are busy
- Monitor queue depth in RunPod console
- Consider implementing client-side retry logic

### 4. Error Handling

```python
import time

def generate_video_with_retry(endpoint, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            job = endpoint.run(input_data)
            result = job.output(timeout=3600)
            
            if "error" in result:
                print(f"Attempt {attempt + 1} failed: {result['error']}")
                if attempt < max_retries - 1:
                    time.sleep(60)  # Wait before retry
                    continue
            
            return result
            
        except Exception as e:
            print(f"Attempt {attempt + 1} exception: {e}")
            if attempt < max_retries - 1:
                time.sleep(60)
                continue
            raise
    
    return {"error": "Max retries exceeded"}
```

### 5. Monitoring Setup

**Key Metrics:**
- Request count
- Average generation time
- Error rate
- Queue depth
- Worker utilization

**Logging:**
- View logs in RunPod console
- Set up alerts for errors
- Track costs in billing section

---

## Monitoring and Maintenance

### Daily Checks

1. **Verify Endpoint Status**
   - Check RunPod console dashboard
   - Ensure endpoint is active
   - Review error logs

2. **Monitor Costs**
   - Check daily spending
   - Review worker utilization
   - Adjust timeouts if needed

### Weekly Maintenance

1. **Review Performance**
   - Check average generation times
   - Analyze error patterns
   - Optimize based on usage

2. **Update Model Cache**
   - Model is cached in persistent volume
   - No action needed unless model updates

### Model Updates

When Wan2.2 releases new version:

1. Update model ID in `handler.py` and `app.py`
2. Rebuild Docker image: `./build.sh`
3. Push to Docker Hub: `./deploy.sh`
4. Update RunPod endpoint with new image
5. Test with new model version

### Troubleshooting Common Issues

**Issue: Endpoint fails to initialize**
- Check Docker image exists on Docker Hub
- Verify image tag is correct
- Check volume mount path is `/runpod-volume`

**Issue: Generation times out**
- Increase execution timeout to 3600s
- Check GPU type (needs A100 80GB minimum)
- Verify model is cached in volume

**Issue: Out of memory errors**
- Ensure using A100 80GB (not 40GB)
- Try 480p instead of 720p
- Check container disk is adequate (20GB+)

**Issue: High costs**
- Reduce idle timeout (5 minutes recommended)
- Scale down workers during off-peak
- Use 480p for non-critical requests

---

## Advanced Configuration

### Using Private Docker Registry

If using a private Docker registry (e.g., AWS ECR, Google GCR):

1. In RunPod template settings
2. Enable "Container Registry Credentials"
3. Enter credentials:
   - Username
   - Password/Token
   - Registry URL

### Custom Environment Variables

Add to endpoint configuration:

```
HUGGING_FACE_HUB_TOKEN=your_hf_token      # If model requires auth
LOG_LEVEL=INFO                             # Logging verbosity
CUSTOM_MODEL_PATH=/runpod-volume/custom   # Custom model location
```

### Health Check Configuration

The Docker image includes health checks. To test:

```bash
docker run --gpus all -p 8000:8000 wan2-s2v:latest

# In another terminal
curl http://localhost:8000/health
```

---

## Security Best Practices

1. **API Key Management**
   - Never commit API keys to code
   - Use environment variables
   - Rotate keys regularly

2. **Docker Image Security**
   - Keep base images updated
   - Scan for vulnerabilities
   - Use specific version tags (not just `latest`)

3. **Network Security**
   - Use HTTPS for all API calls
   - Implement rate limiting
   - Consider VPC if available

4. **Input Validation**
   - Validate file sizes (image < 10MB, audio < 50MB)
   - Check file types
   - Sanitize prompt text

---

## Cost Calculator

Estimate your costs:

```python
def estimate_cost(num_videos, resolution="720p", steps=30):
    # A100 80GB: ~$2.50/hour
    gpu_cost_per_hour = 2.50
    
    # Time estimates (minutes)
    times = {
        ("480p", 30): 7,
        ("720p", 30): 11,
        ("480p", 50): 15,
        ("720p", 50): 20
    }
    
    time_per_video = times.get((resolution, steps), 11) / 60  # Convert to hours
    cost_per_video = time_per_video * gpu_cost_per_hour
    
    total_cost = num_videos * cost_per_video
    
    print(f"Estimated cost for {num_videos} videos:")
    print(f"  Resolution: {resolution}, Steps: {steps}")
    print(f"  Time per video: {time_per_video * 60:.1f} minutes")
    print(f"  Cost per video: ${cost_per_video:.2f}")
    print(f"  Total cost: ${total_cost:.2f}")
    
    return total_cost

# Example
estimate_cost(100, "720p", 30)
```

---

## Support and Resources

### Documentation
- [RunPod Serverless Docs](https://docs.runpod.io/serverless/overview)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)

### Community
- [RunPod Discord](https://discord.gg/runpod)
- [RunPod Forums](https://community.runpod.io/)

### Issues
- RunPod platform: RunPod support
- Wan2.2 model: GitHub issues
- This deployment: Check logs and troubleshooting section

---

**Version:** 1.0.0  
**Last Updated:** December 2, 2025  
**Deployment Type:** RunPod Serverless with Docker Hub
