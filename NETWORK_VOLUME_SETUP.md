# RunPod Network Volume Configuration for Wan2.2 S2V

## Network Volume Setup

**Volume Name:** storystudio  
**Size:** 70 GB  
**Mount Path:** `/runpod-volume`  
**Purpose:** Persistent model cache across workers

## Directory Structure

```
/runpod-volume/
├── models/                          # Model cache directory
│   └── Wan-AI/
│       └── Wan2.2-S2V-14B/         # ~49GB model files
│           ├── config.json
│           ├── model.safetensors
│           └── ...
├── huggingface/                     # HuggingFace cache
│   └── hub/
└── outputs/                         # Optional: temporary outputs
```

## Environment Variables for RunPod Endpoint

Set these in your RunPod serverless endpoint configuration:

```bash
MODEL_CACHE_DIR=/runpod-volume/models
HF_HOME=/runpod-volume/huggingface

# R2 Storage (already configured)
R2_ACCOUNT_ID=620baa808df08b1a30d448989365f7dd
R2_ACCESS_KEY_ID=a69ca34cdcdeb60bad5ed1a07a0dd29d
R2_SECRET_ACCESS_KEY=751a95202a9fa1eb9ff7d45e0bba5b57b0c2d1f0d45129f5f67c2486d5d4ae24
R2_BUCKET_NAME=storystudio
R2_PUBLIC_URL=parentearn.com
```

## RunPod Endpoint Configuration

1. **Container Image:** `romantony/wan2-s2v:1.1.0`
2. **Container Disk:** 20 GB (reduced from 60GB)
3. **Network Volume:** 
   - Select: `storystudio`
   - Mount path: `/runpod-volume`
4. **GPU:** A100 80GB
5. **Max Workers:** 3 (or as needed)
6. **Idle Timeout:** 5 seconds
7. **Execution Timeout:** 600 seconds (10 minutes)

## First Run Behavior

### Initial Model Download (One-time per volume)
- **Duration:** 10-15 minutes
- **Process:** Downloads Wan2.2-S2V-14B from HuggingFace Hub
- **Location:** Saves to `/runpod-volume/models/Wan-AI/Wan2.2-S2V-14B/`
- **Size:** ~49 GB

### Subsequent Runs
- **Duration:** <30 seconds startup
- **Process:** Loads model from cached files
- **Benefit:** Fast cold starts, shared across all workers

## Model Download Details

The model is downloaded automatically on first handler initialization:

```python
# In handler.py (lines 46-58)
def load_model(self):
    if not Path(model_dir).exists():
        print(f"Downloading model (~49GB)...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=model_dir,
            cache_dir=MODEL_CACHE_DIR,
        )
```

## Monitoring Model Download

You can monitor the download progress in RunPod logs:

```
Loading model: Wan-AI/Wan2.2-S2V-14B
Downloading model (~49GB)...
Fetching 15 files: 100%|██████████| 15/15
✓ Model downloaded
Model loaded from /runpod-volume/models/Wan-AI/Wan2.2-S2V-14B
```

## Cost Breakdown

- **Network Volume:** $0.10/GB/month = $7/month for 70GB
- **Storage Used:** ~49GB for model + ~2GB for cache = ~51GB
- **Available Space:** 19GB for future growth

## Benefits

1. ✅ **Reduced Docker Image:** 67GB → 15GB (78% reduction)
2. ✅ **Faster Builds:** GitHub Actions succeeds in ~10 minutes
3. ✅ **Shared Cache:** All workers use same model copy
4. ✅ **No Repeated Downloads:** Model persists across worker lifecycles
5. ✅ **Easy Updates:** Delete model folder to force re-download of new version

## Troubleshooting

### Issue: Model not found after first run
**Solution:** Check that network volume is properly attached at `/runpod-volume`

### Issue: Disk space errors
**Solution:** Network volume has 70GB, model needs ~49GB. Current usage should be ~51GB.

### Issue: Slow first run
**Expected:** First run takes 10-15 minutes to download model. This is normal.

### Issue: Need to update model
**Solution:**
```bash
# SSH into RunPod pod or use serverless job
rm -rf /runpod-volume/models/Wan-AI/Wan2.2-S2V-14B
# Next run will re-download latest version
```

## Verification

After deploying v1.1.0, test with:

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "image": "https://parentearn.com/VideoGen/test_image.png",
      "audio": "https://parentearn.com/VideoGen/test_audio.mp3",
      "prompt": "A professional businesswoman in modern office",
      "resolution": "480p",
      "sample_steps": 30
    }
  }'
```

**First run:** Expect 15-20 minute response (downloading + inference)  
**Subsequent runs:** Expect 2-3 minute response (inference only)
