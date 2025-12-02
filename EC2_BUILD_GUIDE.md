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

## Quick Commands (Run on EC2 via Git Bash)

```bash
# 1. Update repository
git pull origin main

# 2. Login to Docker Hub
docker login -u romantony

# 3. Build Docker image with version (takes 20-30 minutes)
docker build -t romantony/wan2-s2v:1.0.2 .

# 4. Push to Docker Hub (takes 10-20 minutes)
docker push romantony/wan2-s2v:1.0.2
```

## Complete One-Liner

```bash
git pull origin main && docker build -t romantony/wan2-s2v:1.0.2 . && docker push romantony/wan2-s2v:1.0.2
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
2. Create endpoint with `romantony/wan2-s2v:1.0.2`
3. Configure:
   - GPU: A100 80GB (or A100 40GB minimum)
   - Container Disk: 60GB (model included in image)
   - Container Start Command: `python3 -u handler.py`
4. Set Environment Variables:
   - `R2_ACCOUNT_ID=620baa808df08b1a30d448989365f7dd`
   - `R2_ACCESS_KEY_ID=a69ca34cdcdeb60bad5ed1a07a0dd29d`
   - `R2_SECRET_ACCESS_KEY=751a95202a9fa1eb9ff7d45e0bba5b57b0c2d1f0d45129f5f67c2486d5d4ae24`
   - `R2_BUCKET_NAME=storystudio`
   - `R2_PUBLIC_URL=parentearn.com`
5. Deploy and test!

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
docker rmi romantony/wan2-s2v:1.0.2

# Or clean everything
docker system prune -a
```

Then stop/terminate your EC2 instance if not needed anymore.
