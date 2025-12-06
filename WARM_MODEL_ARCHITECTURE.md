# Wan2.2 S2V Warm Model Architecture

## Overview

The Warm Model Architecture keeps the Wan2.2 S2V-14B model loaded in GPU memory between requests, eliminating the cold start delay (~10-15 minutes) for subsequent video generations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RunPod Serverless Container                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    handler_v2.py                         │    │
│  │  • Receives RunPod job requests                         │    │
│  │  • Downloads image/audio from URLs                      │    │
│  │  • Sends generation request to model server             │    │
│  │  • Uploads result to R2                                 │    │
│  └───────────────────────┬─────────────────────────────────┘    │
│                          │                                       │
│                    Unix Socket IPC                               │
│                 /tmp/wan2_s2v_model_server.sock                  │
│                          │                                       │
│  ┌───────────────────────▼─────────────────────────────────┐    │
│  │                   model_server.py                        │    │
│  │  • Loads model once at container startup                │    │
│  │  • Keeps model warm in GPU memory                       │    │
│  │  • Processes generation requests                        │    │
│  │  • Uses WanS2V pipeline from Wan2.2                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    GPU Memory (80-141GB)                 │    │
│  │  • Wan2.2 S2V-14B model (~67GB)                         │    │
│  │  • T5 encoder (~20GB)                                   │    │
│  │  • VAE (~2GB)                                           │    │
│  │  • Wav2Vec2 audio encoder (~1GB)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Comparison

| Metric | Cold Start (v1) | Warm Model (v2) |
|--------|-----------------|-----------------|
| First request | ~15 min | ~15 min (model loading) |
| Subsequent requests | ~15 min | ~3-5 min |
| Model loading | Every request | Once per container |
| GPU memory usage | Transient | Persistent |

## Files

### `model_server.py`
The persistent model server that:
- Loads the Wan2.2 S2V-14B model at startup
- Creates a Unix socket at `/tmp/wan2_s2v_model_server.sock`
- Handles generation requests from the handler
- Keeps the model warm between requests

### `handler_v2.py`
The RunPod serverless handler that:
- Starts the model server on first request
- Communicates with model server via Unix socket
- Handles input validation and file downloads
- Uploads generated videos to R2

### `start_warm.sh`
Startup script for the warm model architecture.

## Socket Protocol

The model server uses JSON messages with newline delimiters.

### Health Check Request
```json
{"action": "health"}
```

### Health Check Response
```json
{
  "success": true,
  "model_loaded": true,
  "status": "ready"
}
```

### Generation Request
```json
{
  "action": "generate",
  "image_path": "/tmp/input_image.jpg",
  "audio_path": "/tmp/input_audio.wav",
  "prompt": "A person speaking naturally",
  "resolution": "480p",
  "sample_steps": 20
}
```

### Generation Response
```json
{
  "success": true,
  "video_path": "/workspace/Wan2.2/s2v-14B_20250615_123456_1.mp4",
  "generation_time": 180.5,
  "total_time": 185.2,
  "video_size_mb": 1.75
}
```

### Status Request
```json
{"action": "status"}
```

### Status Response
```json
{
  "success": true,
  "model_loaded": true,
  "vram_allocated_gb": 67.5,
  "generation_count": 5
}
```

## Configuration

### Resolution Options
| Resolution | Max Area | Shift | Typical Size |
|------------|----------|-------|--------------|
| 480p | 399,360 (832×480) | 3.0 | ~1.5MB |
| 720p | 921,600 (1280×720) | 5.0 | ~3.5MB |

### Sample Steps
- **Minimum**: 15 (fastest, lower quality)
- **Default**: 20 (good balance)
- **Maximum**: 50 (highest quality, slowest)

## Deployment

### Build Docker Image
```bash
docker build -t ghcr.io/yourusername/storystudio-wan2-s2v:v2.0.0 .
docker push ghcr.io/yourusername/storystudio-wan2-s2v:v2.0.0
```

### RunPod Configuration

1. **GPU**: H200 SXM (141GB) recommended, or H100/A100 80GB
2. **Network Volume**: 100GB at `/runpod-volume` with model pre-downloaded
3. **Active Workers**: 1 (to keep model warm)
4. **Max Workers**: 1-3 (depending on budget)

### Environment Variables (Optional)
```
HANDLER_VERSION=v2  # Use warm model architecture (default)
HANDLER_VERSION=v1  # Use original subprocess handler
```

## API Usage

The API is identical to v1. The only change is in response:

```json
{
  "video_url": "https://parentearn.com/VideoGen/...",
  "generation_time": 180.5,
  "total_time": 185.2,
  "video_size_mb": 1.75,
  "resolution": "480p",
  "sample_steps": 20,
  "warm_model": true  // New field indicating warm model was used
}
```

## Troubleshooting

### Model Server Not Starting
Check logs for CUDA initialization errors:
```bash
# View RunPod logs
curl -X GET "https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}" \
  -H "Authorization: Bearer {api_key}"
```

### Socket Connection Failed
The handler will automatically start the model server if it's not running. Check for:
- CUDA out of memory errors
- Model checkpoint not found at `/runpod-volume/models/Wan2.2-S2V-14B`

### Generation Timeout
Default timeout is 30 minutes. For long audio clips, generation may take longer. Consider:
- Using 480p resolution
- Reducing sample_steps to 15-20
- Splitting audio into shorter clips

## Cost Optimization

1. **Keep 1 active worker** running to avoid cold starts
2. **Use 480p** for faster generation (~3 min vs ~5 min for 720p)
3. **Batch requests** when possible to amortize worker startup cost
4. **Monitor worker activity** and disable during off-hours

### Estimated Costs (RunPod)
| GPU | Hourly Rate | Cost per Video (warm) |
|-----|-------------|----------------------|
| H200 SXM | ~$4.49/hr | ~$0.22 (3 min) |
| H100 80GB | ~$3.99/hr | ~$0.33 (5 min) |
| A100 80GB | ~$1.99/hr | ~$0.40 (12 min) |

## Version History

- **v2.0.0** (2025-01-15): Warm model architecture with persistent model server
- **v1.2.5** (2025-01-14): Subprocess wrapper with safetensors patches
- **v1.0.0** (2025-01-10): Initial release
