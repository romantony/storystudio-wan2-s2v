# Bazel Build and Deploy Guide for RunPod

This guide shows how to build and deploy the Wan2.2 S2V Docker image using Bazel on RunPod.

## Prerequisites

1. RunPod PyTorch pod with sufficient storage
2. Docker Hub account and access token

## Installation Steps

### 1. Install Bazel

```bash
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
cp ./bazelisk-linux-amd64 /usr/local/bin/bazel
```

### 2. Clone Repository

```bash
git clone https://github.com/romantony/storystudio-wan2-s2v.git
cd storystudio-wan2-s2v
```

### 3. Login to Docker Hub

```bash
docker login -u romantony
# Enter your Docker Hub access token when prompted
```

## Build and Deploy

### Option 1: Build and Push (Recommended)

```bash
bazel run //:push_image
```

This will:
- Build the Docker image using Bazel
- Push to `romantony/wan2-s2v:latest` and `romantony/wan2-s2v:1.0.0`

### Option 2: Build Tarball for Local Testing

```bash
bazel build //:wan2_s2v_tarball
docker load < bazel-bin/wan2_s2v_tarball/tarball.tar
docker run --gpus all -p 8000:8000 romantony/wan2-s2v:latest
```

## Bazel Files Structure

```
storystudio-wan2-s2v/
├── MODULE.bazel           # Bazel workspace and dependencies
├── BUILD.bazel            # Main build rules
├── .bazelrc              # Bazel configuration
└── patches/
    └── BUILD.bazel       # Patches build rules
```

## Troubleshooting

### Error: "not invoked from within a workspace"
- Ensure `MODULE.bazel` exists in your project root
- Run `bazel` commands from the repository root directory

### Error: "Cannot connect to Docker daemon"
- Start Docker: `service docker start`
- Verify: `docker info`

### Error: "Unauthorized" during push
- Login again: `docker login -u romantony`
- Use access token, not password

## Benefits of Bazel on RunPod

✅ Works on GPU pods where direct Docker builds are restricted  
✅ Reproducible builds with locked dependencies  
✅ Efficient caching and incremental builds  
✅ Direct push to Docker Hub  

## Next Steps

After pushing your image:
1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Create endpoint with `romantony/wan2-s2v:latest`
3. Configure GPU (A100 80GB), volume, and environment variables
4. Deploy and test!
