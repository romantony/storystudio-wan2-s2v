# Docker Image Size Optimization Options

## Current Problem
Docker image exceeds 50GB, causing build failures on GitHub Actions (disk space limit).

## Size Breakdown (Estimated)
- Base CUDA image (devel): ~8GB
- PyTorch + CUDA binaries: ~6GB
- Python dependencies: ~4GB
- Wan2.2 repository code: ~500MB
- **Wan2.2-S2V-14B model: ~49GB** ← Main culprit
- **Total: ~67GB**

---

## Solution Options

### ✅ **OPTION 1: Runtime Model Download (RECOMMENDED)**
**Target Size: 15-18GB** (73% reduction)

**Changes:**
- Remove model from Docker image
- Download model at first container startup
- Store on RunPod Network Volume (persistent)

**Implementation:**
```dockerfile
# Don't pre-download model in Dockerfile
# Model downloads via handler.py at runtime
```

**Handler already supports this!** (line 46-58 in handler.py)

**Pros:**
- Image under 20GB ✅
- Fast builds (5-10 minutes)
- Model cached on network volume, shared across workers
- Only download once per region
- Free model updates (just pull latest)

**Cons:**
- First worker startup: +10-15 min (one-time per volume)
- Requires RunPod Network Volume setup
- Network volume cost: ~$5/month for 50GB

**RunPod Setup:**
1. Create Network Volume (50GB minimum)
2. Attach to endpoint
3. Model downloads on first invocation
4. Subsequent workers use cached model

---

### **OPTION 2: Use Runtime Base Image Instead of Devel**
**Target Size: 55-60GB** (10% reduction)

**Changes:**
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# Instead of: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
```

**Pros:**
- Removes build tools (~3-5GB): gcc, g++, make, cmake, headers
- Cleaner runtime environment

**Cons:**
- May break packages that need compilation (scipy, librosa)
- Still includes 49GB model (doesn't solve main issue)

---

### **OPTION 3: Multi-Stage Build**
**Target Size: 60-63GB** (6% reduction)

**Implementation:**
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder
# Install everything, download model

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime
# Copy only necessary files from builder
```

**Pros:**
- Removes build artifacts
- Slightly cleaner

**Cons:**
- Complex Dockerfile
- Still includes 49GB model
- Minimal size benefit

---

### **OPTION 4: Model Compression**
**Target Size: 40-50GB** (25% reduction)

**Changes:**
- Quantize model to 8-bit or 4-bit (bitsandbytes)
- Use model compression techniques

**Pros:**
- Smaller model size
- Faster inference (maybe)

**Cons:**
- Quality degradation
- Requires model re-training/conversion
- Complex setup
- May not work with Wan2.2 architecture

---

### **OPTION 5: Build on Self-Hosted Runner**
**Target Size: 67GB** (no reduction, but solves build issue)

**Changes:**
- Use self-hosted GitHub Actions runner with 100GB+ disk
- Or use EC2 for manual builds

**Pros:**
- Can build any size image
- No disk space limits

**Cons:**
- Cost: $0.50-$2 per build (EC2 t3.xlarge)
- Manual setup required
- No automation benefits

---

## **RECOMMENDED APPROACH: Option 1 + Option 2**

### Combined Optimization
1. **Use runtime base image** (saves 3-5GB)
2. **Download model at runtime** (saves 49GB)
3. **Result: ~12-15GB image** ✅

### Implementation Steps:

1. **Update Dockerfile** (already created as `Dockerfile.optimized`)
2. **Test locally** (optional)
3. **Replace current Dockerfile**
4. **Update RunPod endpoint**:
   - Create Network Volume (50GB)
   - Attach to endpoint
   - Set environment: `MODEL_CACHE_DIR=/runpod-volume/models`
5. **First run downloads model** (~10-15 min)
6. **Subsequent runs use cached model** (<30 sec startup)

### Cost Comparison:
- **Current approach**: Build fails, can't deploy
- **Option 1 + 2**: $5/month network volume + faster builds
- **Option 5**: $0.50-$2 per build + $5/month volume

### Build Time Comparison:
- **Current**: 50+ minutes (if it worked)
- **Optimized**: 8-12 minutes ✅

---

## Decision Matrix

| Option | Image Size | Build Time | First Startup | Quality | Cost | Complexity |
|--------|-----------|------------|---------------|---------|------|-----------|
| **Current** | 67GB | N/A (fails) | 30s | 100% | $0 | Low |
| **Option 1** | 15GB | 10 min | 15 min | 100% | $5/mo | Low |
| **Option 2** | 60GB | 45 min | 30s | 100% | $0 | Low |
| **Option 1+2** | 12GB | 8 min | 15 min | 100% | $5/mo | Low |
| **Option 3** | 63GB | 50 min | 30s | 100% | $0 | Medium |
| **Option 4** | 45GB | 40 min | 30s | 80-90% | $0 | High |
| **Option 5** | 67GB | 50 min | 30s | 100% | $1-2/build | Medium |

---

## Next Steps

**To implement Option 1+2:**
```bash
# 1. Use optimized Dockerfile
mv Dockerfile Dockerfile.old
mv Dockerfile.optimized Dockerfile

# 2. Commit and push
git add Dockerfile
git commit -m "feat: optimize image size using runtime base and runtime model download"
git push origin main

# 3. Wait for build (~10 minutes)

# 4. Create RunPod Network Volume
# - Name: wan2-model-cache
# - Size: 50GB
# - Region: Same as your endpoint

# 5. Update RunPod endpoint
# - Attach network volume at /runpod-volume
# - Verify env var: MODEL_CACHE_DIR=/runpod-volume/models

# 6. First test run
# - Takes ~15 minutes (downloads model)
# - Model cached on volume
# - Future runs: <30 seconds startup
```
