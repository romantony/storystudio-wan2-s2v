"""
RunPod Serverless Handler for Wan2.2 S2V
Compatible with RunPod's serverless architecture
Using RunPod's official PyTorch base image for proper CUDA integration
"""
# FIRST: Set CUDA environment variables before ANY imports
# This must happen before any module that might import torch
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

# Now import other modules (but NOT torch - it will be imported lazily)
import runpod
import subprocess
import tempfile
import base64
import time
import boto3
import requests
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from huggingface_hub import snapshot_download

# NOTE: Do NOT import torch at module level!
# It will be imported in load_model() after CUDA env vars are set.

# Configuration
MODEL_ID = "Wan-AI/Wan2.2-S2V-14B"
# Network volume path - RunPod mounts at /runpod-volume by default
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")
WAN_DIR = "/workspace/Wan2.2"

# R2 Configuration
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "620baa808df08b1a30d448989365f7dd")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "a69ca34cdcdeb60bad5ed1a07a0dd29d")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "751a95202a9fa1eb9ff7d45e0bba5b57b0c2d1f0d45129f5f67c2486d5d4ae24")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "storystudio")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "parentearn.com")
R2_FOLDER = "VideoGen"

# Resolution mapping
RESOLUTION_MAP = {
    "480p": "832*480",
    "720p": "1024*704"
}

# Global model config
class ModelConfig:
    def __init__(self):
        self.model_loaded = False
        self.model_dir = None
        
    def load_model(self):
        """Download model if not cached - NO CUDA/torch imports here!"""
        if self.model_loaded:
            return
        
        print(f"Loading model: {MODEL_ID}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
        
        # NOTE: Do NOT import torch here! 
        # The subprocess wrapper will handle CUDA initialization properly.
        # Importing torch in the main process corrupts CUDA context.
        
        model_dir = f"{MODEL_CACHE_DIR}/{MODEL_ID}"
        
        if not Path(model_dir).exists():
            print(f"Downloading model (~49GB)...")
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=model_dir,
                cache_dir=MODEL_CACHE_DIR,
            )
            print("✓ Model downloaded")
        else:
            print("✓ Model found in cache")
        
        self.model_dir = model_dir
        self.model_loaded = True

model_config = ModelConfig()

# Initialize R2 client
def get_r2_client():
    """Initialize and return R2 S3 client"""
    return boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )

def upload_to_r2(file_path: str, filename: str) -> str:
    """Upload file to R2 and return public URL"""
    s3_client = get_r2_client()
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    r2_key = f"{R2_FOLDER}/{timestamp}_{filename}"
    
    # Upload file
    with open(file_path, 'rb') as f:
        s3_client.upload_fileobj(f, R2_BUCKET_NAME, r2_key)
    
    # Return public URL
    public_url = f"https://{R2_PUBLIC_URL}/{r2_key}"
    return public_url

def generate_video(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function
    
    Expected job input:
    {
        "image": "base64_encoded_image OR url",
        "audio": "base64_encoded_audio OR url",
        "prompt": "optional text prompt",
        "resolution": "480p" or "720p",
        "sample_steps": 20-50 (default 30)
    }
    
    Returns:
    {
        "video_url": "https://parentearn.com/VideoGen/...",
        "generation_time": float,
        "video_size_mb": float,
        "resolution": str,
        "sample_steps": int
    }
    """
    job_input = job.get("input", {})
    start_time = time.time()
    
    try:
        # Validate inputs
        if "image" not in job_input or "audio" not in job_input:
            return {"error": "Missing required inputs: image and audio"}
        
        # Get parameters
        image_input = job_input["image"]
        audio_input = job_input["audio"]
        prompt = job_input.get("prompt", "")
        resolution = job_input.get("resolution", "720p")
        sample_steps = job_input.get("sample_steps", 30)
        
        # Validate parameters
        if resolution not in RESOLUTION_MAP:
            return {"error": f"Invalid resolution. Must be one of: {list(RESOLUTION_MAP.keys())}"}
        
        if not (20 <= sample_steps <= 50):
            return {"error": "sample_steps must be between 20 and 50"}
        
        # Ensure model is loaded
        if not model_config.model_loaded:
            model_config.load_model()
        
        # Handle image input (URL or base64)
        if image_input.startswith(('http://', 'https://')):
            print(f"Downloading image from URL: {image_input}")
            response = requests.get(image_input, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
        else:
            image_bytes = base64.b64decode(image_input)
        
        # Handle audio input (URL or base64)
        if audio_input.startswith(('http://', 'https://')):
            print(f"Downloading audio from URL: {audio_input}")
            response = requests.get(audio_input, timeout=30)
            response.raise_for_status()
            audio_bytes = response.content
        else:
            audio_bytes = base64.b64decode(audio_input)
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded files
            image_path = Path(tmpdir) / "input_image.jpg"
            audio_path = Path(tmpdir) / "input_audio.wav"
            
            image_path.write_bytes(image_bytes)
            audio_path.write_bytes(audio_bytes)
            
            # Build generation command
            size = RESOLUTION_MAP[resolution]
            
            # Create wrapper script to handle CUDA initialization for subprocess
            # This is needed because safetensors rust backend has issues with cuda:0
            wrapper_script = Path(tmpdir) / "run_generate.py"
            wrapper_script.write_text(f'''#!/usr/bin/env python
import os
import sys

# Set CUDA environment before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Initialize PyTorch CUDA context first
import torch
torch.cuda.init()
torch.cuda.set_device(0)
_ = torch.zeros(1, device="cuda:0")
print(f"CUDA initialized: {{torch.cuda.get_device_name(0)}}", flush=True)

# Patch safetensors to load to CPU (workaround for rust backend cuda:0 issue)
from safetensors import torch as safetensors_torch
_original_load = safetensors_torch.load_file
def _patched_load(filename, device="cpu"):
    # Always load to CPU first, PyTorch will move to GPU as needed
    return _original_load(filename, device="cpu")
safetensors_torch.load_file = _patched_load
print("Patched safetensors to load via CPU", flush=True)

# Change to Wan2.2 directory and add to path
os.chdir("{WAN_DIR}")
sys.path.insert(0, "{WAN_DIR}")

# Run generate.py
exec(open("generate.py").read())
''')
            
            # Build command using wrapper
            cmd = [
                "python", "-u",
                str(wrapper_script),
                "--task", "s2v-14B",
                "--size", size,
                "--ckpt_dir", model_config.model_dir,
                "--offload_model", "False",  # Keep model in GPU for speed
                "--convert_model_dtype",
                "--sample_steps", str(sample_steps),
                "--image", str(image_path),
                "--audio", str(audio_path)
            ]
            
            if prompt:
                cmd.extend(["--prompt", prompt])
            
            # Run generation
            print(f"Starting generation: {resolution}, {sample_steps} steps")
            print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            
            # NOTE: Do NOT import torch in main process - it corrupts CUDA context
            # The subprocess wrapper handles all CUDA initialization
            print(f"Running generation subprocess...")
            
            # Run subprocess with wrapper script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"Generation failed: {result.stderr}")
                print(f"Generation stdout: {result.stdout}")
                return {"error": f"Generation failed: {result.stderr}"}
            
            # Find generated video
            video_files = list(Path(WAN_DIR).glob("s2v-14B_*.mp4"))
            if not video_files:
                return {"error": "No video file generated"}
            
            # Get most recent video
            video_path = max(video_files, key=lambda p: p.stat().st_mtime)
            
            # Get video size before upload
            video_size_mb = round(video_path.stat().st_size / 1024 / 1024, 2)
            
            # Upload to R2
            print(f"Uploading video to R2...")
            video_filename = f"s2v_{resolution}_{sample_steps}steps.mp4"
            video_url = upload_to_r2(str(video_path), video_filename)
            
            # Clean up generated file
            video_path.unlink()
            
            generation_time = time.time() - start_time
            
            return {
                "video_url": video_url,
                "generation_time": round(generation_time, 2),
                "video_size_mb": video_size_mb,
                "resolution": resolution,
                "sample_steps": sample_steps
            }
            
    except subprocess.TimeoutExpired:
        return {"error": "Generation timed out (1 hour limit)"}
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Initialize on container startup
print("Initializing Wan2.2 S2V handler...")
print("Note: PyTorch/CUDA will be initialized on first job (lazy loading)")

# Apply FlashAttention patches (must be done before model loading)
print("Applying FlashAttention compatibility patches...")
from patches.apply_patches import apply_flashattention_patches
apply_flashattention_patches()
print("✓ Patches applied")

# Model loading happens on first job request to avoid cold start timeout
print("✓ Handler ready (model will be loaded on first request)")

# Start RunPod serverless handler
runpod.serverless.start({"handler": generate_video})