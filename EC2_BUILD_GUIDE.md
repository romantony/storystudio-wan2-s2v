# EC2 Docker Build and Deploy Guide

Simple guide for building and deploying Wan2.2 S2V Docker image on AWS EC2.

## Prerequisites

- AWS EC2 instance (Ubuntu 22.04 recommended)
- Docker Hub account and access token
- Sufficient storage (at least 100GB)

## Setup on EC2

### 1. SSH into your EC2 instance

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 2. Install Docker

```bash
# Update system
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (optional, to avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker installation
docker --version
```

### 3. Clone Repository

```bash
cd ~
git clone https://github.com/romantony/storystudio-wan2-s2v.git
cd storystudio-wan2-s2v
```

### 4. Login to Docker Hub

```bash
docker login -u romantony
# Enter your Docker Hub access token when prompted
```

## Build and Deploy

### Option 1: Using build.sh script

```bash
export DOCKER_USERNAME=romantony
./build.sh
```

### Option 2: Using deploy.sh (builds and pushes)

```bash
export DOCKER_USERNAME=romantony
./build.sh
./deploy.sh
```

### Option 3: Manual commands

```bash
# Build
docker build -t romantony/wan2-s2v:1.0.0 .
docker tag romantony/wan2-s2v:1.0.0 romantony/wan2-s2v:latest

# Push
docker push romantony/wan2-s2v:1.0.0
docker push romantony/wan2-s2v:latest
```

## Build Time Expectations

- Build time: 20-40 minutes (depends on EC2 instance and internet speed)
- Image size: ~12-15GB
- Push time: 10-30 minutes (depends on upload bandwidth)

## EC2 Instance Recommendations

- **Minimum:** t3.large (2 vCPU, 8GB RAM)
- **Recommended:** t3.xlarge (4 vCPU, 16GB RAM) or c5.2xlarge for faster builds
- **Storage:** 100GB EBS volume
- **Region:** Choose closest to your location for faster Docker Hub uploads

## After Deployment

Once pushed to Docker Hub, you can:

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Create endpoint with `romantony/wan2-s2v:latest`
3. Configure:
   - GPU: A100 80GB
   - Container Disk: 20GB
   - Volume: 80GB at `/runpod-volume`
   - Env: `MODEL_CACHE_DIR=/runpod-volume/models`
4. Deploy and test!

## Troubleshooting

### Build fails with "no space left on device"
- Increase EBS volume size or clean up Docker: `docker system prune -a`

### Push fails with "denied: requested access to the resource is denied"
- Verify Docker Hub login: `docker login -u romantony`
- Use access token, not password

### Build is very slow
- Use a larger EC2 instance (c5.2xlarge or similar)
- Check internet connectivity

## Cost Estimates (AWS)

- t3.xlarge spot instance: ~$0.05/hour
- 100GB EBS volume: ~$10/month
- Data transfer out: ~$0.09/GB

**Total for one build:** ~$0.10-$0.30

## Clean Up

After successful push, you can:

```bash
# Remove local images to free space
docker rmi romantony/wan2-s2v:1.0.0 romantony/wan2-s2v:latest

# Or clean everything
docker system prune -a
```

Then stop/terminate your EC2 instance if not needed anymore.
