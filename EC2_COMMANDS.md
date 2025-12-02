# EC2 Build and Deploy Commands for Wan2.2 S2V

## Prerequisites on EC2

```bash
# 1. SSH to your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 2. Install Docker (if not already installed)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# 3. Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 4. Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 5. Log out and back in for docker group to take effect
exit
ssh -i your-key.pem ubuntu@your-ec2-ip
```

## Build & Deploy Commands

```bash
# Clone the repository
git clone https://github.com/romantony/storystudio-wan2-s2v.git
cd storystudio-wan2-s2v

# IMPORTANT: Set your Docker Hub username
DOCKER_USERNAME="your_dockerhub_username"  # CHANGE THIS!

# Build the Docker image (takes 20-30 minutes)
docker build -t ${DOCKER_USERNAME}/wan2-s2v-runpod:latest .

# Check image size (should be around 20-25GB)
docker images ${DOCKER_USERNAME}/wan2-s2v-runpod:latest

# (Optional) Test locally before pushing
docker run --rm --gpus all \
  -e MODEL_CACHE_DIR=/runpod-volume/models \
  -e R2_ACCOUNT_ID="$R2_ACCOUNT_ID" \
  -e R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
  -e R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
  -e R2_BUCKET_NAME="$R2_BUCKET_NAME" \
  -e R2_PUBLIC_URL="$R2_PUBLIC_URL" \
  -p 8000:8000 \
  ${DOCKER_USERNAME}/wan2-s2v-runpod:latest

# Login to Docker Hub
docker login

# Push to Docker Hub (takes 10-20 minutes depending on upload speed)
docker push ${DOCKER_USERNAME}/wan2-s2v-runpod:latest
```

## Configure RunPod Endpoint

After pushing to Docker Hub:

1. Go to https://www.runpod.io/console/serverless
2. Select your endpoint (ID: 1tggndzlc063rw)
3. Click "Edit Template" or "Settings"
4. Update:
   - **Container Image**: `your_dockerhub_username/wan2-s2v-runpod:latest`
   - **Container Start Command**: `python3 -u handler.py`
   - **Container Disk**: 50 GB (for model storage)
   - **GPU Type**: A100 80GB (or A100 40GB minimum)
   
5. Set Environment Variables:
   ```
  MODEL_CACHE_DIR=/runpod-volume/models
  R2_ACCOUNT_ID=<your_r2_account_id>
  R2_ACCESS_KEY_ID=<your_r2_access_key_id>
  R2_SECRET_ACCESS_KEY=<your_r2_secret_access_key>
  R2_BUCKET_NAME=<your_r2_bucket_name>
  R2_PUBLIC_URL=<your_public_domain>
   ```

6. Click "Save" and wait for deployment

## Test the Endpoint

```bash
# Submit a test job
curl -X POST https://api.runpod.ai/v2/1tggndzlc063rw/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "input": {
      "image": "https://parentearn.com/VideoGen/test_image.png",
      "audio": "https://parentearn.com/VideoGen/test_audio.mp3",
      "resolution": "480p",
      "sample_steps": 30
    }
  }'

# Get the job ID from response, then check status
curl -s "https://api.runpod.ai/v2/1tggndzlc063rw/status/YOUR_JOB_ID" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## Quick One-Liner (after setting DOCKER_USERNAME)

```bash
# Complete build and push in one command
DOCKER_USERNAME="your_dockerhub_username" && \
git clone https://github.com/romantony/storystudio-wan2-s2v.git && \
cd storystudio-wan2-s2v && \
docker build -t ${DOCKER_USERNAME}/wan2-s2v-runpod:latest . && \
docker login && \
docker push ${DOCKER_USERNAME}/wan2-s2v-runpod:latest
```

## Monitoring Build Progress

```bash
# In another terminal, watch docker build progress
docker stats

# Check docker logs if build seems stuck
docker ps -a
docker logs <container_id>
```

## Troubleshooting

**Build fails with "out of disk space":**
```bash
# Clean up old docker images/containers
docker system prune -a
```

**Build is slow:**
- Use an EC2 instance with good network bandwidth (c5.2xlarge or better)
- Consider using spot instances to save cost during build

**GPU not detected:**
```bash
# Verify NVIDIA drivers
nvidia-smi

# Verify Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
