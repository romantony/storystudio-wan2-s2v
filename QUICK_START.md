# Quick Start Guide - Wan2.2 S2V on RunPod

## 🚀 Deploy in 4 Steps

### Step 1: Set Your Docker Username
```bash
# Linux/Mac
export DOCKER_USERNAME=your-dockerhub-username

# Windows
set DOCKER_USERNAME=your-dockerhub-username
```

### Step 2: Build Docker Image
```bash
# Linux/Mac
./build.sh

# Windows
build.bat
```

### Step 3: Push to Docker Hub
```bash
# Login first
docker login

# Deploy
# Linux/Mac: ./deploy.sh
# Windows: deploy.bat
```

### Step 4: Create RunPod Endpoint

1. Go to https://www.runpod.io/console/serverless
2. Click "New Endpoint"
3. Configure:
   - **Image:** `your-dockerhub-username/wan2-s2v:latest`
   - **GPU:** A100 80GB
   - **Container Disk:** 20GB
   - **Volume:** 80GB at `/runpod-volume`
   - **Env Var:** `MODEL_CACHE_DIR=/runpod-volume/models`
4. Deploy!

---

## 📝 Test Your Endpoint

Update `test_client.py` with your credentials:
```python
RUNPOD_API_KEY = "your-runpod-api-key"
ENDPOINT_ID = "your-endpoint-id"
```

Run:
```bash
python test_client.py
```

---

## 📊 What to Expect

- **First run:** 15-35 minutes (cold start + generation)
- **After warm:** 6-25 minutes (generation only)
- **Cost per video:** ~$0.30-$1.00 (A100)
- **Resolution options:** 480p or 720p
- **Sample steps:** 20-50 (30 recommended)

---

## 📚 Need More Info?

- Full setup: See `DEPLOYMENT_GUIDE.md`
- Usage examples: See `README.md`
- Troubleshooting: Check both guides above

---

## 🏗️ Project Structure

```
wan2-s2v/
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── handler.py             # RunPod serverless handler
├── app.py                 # FastAPI server (alternative)
├── start.sh               # Startup script
├── patches/               # FlashAttention fixes
├── build.sh / build.bat   # Build scripts
├── deploy.sh / deploy.bat # Deploy scripts
├── test_client.py         # Test script
├── README.md              # Full documentation
├── DEPLOYMENT_GUIDE.md    # Detailed setup guide
└── QUICK_START.md         # This file
```

---

## ⚡ Key Features

✅ Serverless auto-scaling  
✅ FlashAttention compatibility patches  
✅ Model caching (49GB persistent)  
✅ Optimized for A100 80GB  
✅ Base64 I/O for easy integration  
✅ Both serverless & pod modes  

---

## 🆘 Common Issues

**Build fails:** Make sure Docker is running  
**Push fails:** Login with `docker login` first  
**Endpoint fails:** Check image name and GPU type  
**Out of memory:** Use A100 80GB, not 40GB  
**Slow generation:** Normal - takes 6-25 minutes  

---

**Ready to deploy?** Start with Step 1 above! 🎬
