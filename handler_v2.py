#!/usr/bin/env python
"""
Wan2.2 S2V RunPod Handler - Version 2 (Warm Model Architecture)

This handler:
1. Starts the model server as a background process on first request
2. Communicates with the model server via Unix socket
3. Keeps the model warm in GPU memory between requests
4. Reduces generation time from ~15min (cold) to ~5min (warm)

Version: 2.0.0 (Warm Model Architecture)
"""

import json
import os
import socket
import subprocess
import sys
import time
import base64
import tempfile
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import runpod
import boto3
from botocore.client import Config

# Configuration
SOCKET_PATH = "/tmp/wan2_s2v_model_server.sock"
MODEL_SERVER_SCRIPT = "/app/model_server.py"
SERVER_START_TIMEOUT = 600  # 10 minutes for model loading
GENERATION_TIMEOUT = 1800   # 30 minutes for generation

# Resolution options
RESOLUTION_MAP = {
    "480p": "832*480",
    "720p": "1280*720",
}

# R2 Configuration (Cloudflare R2 / S3 compatible)
R2_ENDPOINT_URL = "https://30e9fb89be1cacd4913f30a6652213ae.r2.cloudflarestorage.com"
R2_ACCESS_KEY_ID = "d17ddbf9403c50f7e1a88a59ef19c16b"
R2_SECRET_ACCESS_KEY = "07eed86f6c28df20a9b65f5c02c4cf5e70fd2ef4ab8fd1b48b6549c1faab7dd5"
R2_BUCKET_NAME = "parentearn"
R2_PUBLIC_URL = "parentearn.com"
R2_FOLDER = "VideoGen"

# Track model server process
model_server_process: Optional[subprocess.Popen] = None


def get_s3_client():
    """Create S3/R2 client for uploads"""
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"}
        ),
        region_name="auto"
    )


def upload_to_r2(file_path: str, filename: str) -> str:
    """Upload file to R2 and return public URL"""
    s3_client = get_s3_client()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    r2_key = f"{R2_FOLDER}/{timestamp}_{filename}"
    
    with open(file_path, 'rb') as f:
        s3_client.upload_fileobj(f, R2_BUCKET_NAME, r2_key)
    
    return f"https://{R2_PUBLIC_URL}/{r2_key}"


def is_server_running() -> bool:
    """Check if the model server is running and responsive"""
    if not os.path.exists(SOCKET_PATH):
        return False
    
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(SOCKET_PATH)
        
        # Send health check
        request = json.dumps({"action": "health"}) + "\n"
        sock.sendall(request.encode("utf-8"))
        
        # Read response
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk or b"\n" in response:
                break
            response += chunk
        
        sock.close()
        
        data = json.loads(response.decode("utf-8").strip())
        return data.get("success", False) and data.get("model_loaded", False)
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def start_model_server() -> bool:
    """Start the model server as a background process"""
    global model_server_process
    
    print("=" * 60)
    print("STARTING MODEL SERVER")
    print("=" * 60)
    
    # Clean up any existing socket
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    
    # Start model server as background process
    # Use a log file instead of pipes to avoid blocking
    log_file = open("/tmp/model_server.log", "w")
    model_server_process = subprocess.Popen(
        [sys.executable, "-u", MODEL_SERVER_SCRIPT],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    
    print(f"Model server started (PID: {model_server_process.pid})")
    print("Model server log: /tmp/model_server.log")
    
    # Wait for server to be ready (model loading takes ~10 minutes)
    start_time = time.time()
    last_log_time = time.time()
    last_log_pos = 0
    
    while time.time() - start_time < SERVER_START_TIMEOUT:
        # Print new log output
        try:
            with open("/tmp/model_server.log", "r") as f:
                f.seek(last_log_pos)
                new_output = f.read()
                if new_output:
                    for line in new_output.strip().split("\n"):
                        if line:
                            print(f"[Model Server] {line}")
                last_log_pos = f.tell()
        except:
            pass
        
        # Check if server is ready
        if os.path.exists(SOCKET_PATH):
            if is_server_running():
                elapsed = time.time() - start_time
                print(f"✓ Model server ready after {elapsed:.1f}s")
                return True
        
        # Check if process died
        if model_server_process.poll() is not None:
            print(f"✗ Model server exited with code {model_server_process.returncode}")
            # Print remaining log output
            try:
                with open("/tmp/model_server.log", "r") as f:
                    f.seek(last_log_pos)
                    remaining = f.read()
                    if remaining:
                        print(f"[Model Server] {remaining}")
            except:
                pass
            return False
        
        # Log progress every 30 seconds
        if time.time() - last_log_time > 30:
            elapsed = time.time() - start_time
            print(f"Waiting for model server... ({elapsed:.0f}s)")
            last_log_time = time.time()
        
        time.sleep(1)
    
    print(f"✗ Model server failed to start within {SERVER_START_TIMEOUT}s")
    return False


def ensure_model_server() -> bool:
    """Ensure model server is running, starting it if necessary"""
    if is_server_running():
        print("Model server already running")
        return True
    
    print("Model server not running, starting...")
    return start_model_server()


def send_to_model_server(request: Dict[str, Any], timeout: int = GENERATION_TIMEOUT) -> Dict[str, Any]:
    """Send request to model server via Unix socket"""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(SOCKET_PATH)
        
        # Send request
        request_str = json.dumps(request) + "\n"
        sock.sendall(request_str.encode("utf-8"))
        
        # Read response
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if b"\n" in response:
                break
        
        sock.close()
        
        return json.loads(response.decode("utf-8").strip())
        
    except socket.timeout:
        return {"success": False, "error": "Generation timed out"}
    except Exception as e:
        return {"success": False, "error": f"Socket error: {str(e)}"}


def generate_video(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function
    
    Expected job input:
    {
        "image": "base64_encoded_image OR url",
        "audio": "base64_encoded_audio OR url",
        "prompt": "optional text prompt",
        "resolution": "480p" or "720p",
        "sample_steps": 15-50 (default 20)
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
        resolution = job_input.get("resolution", "480p")  # Default to 480p for speed
        sample_steps = job_input.get("sample_steps", 20)
        
        # Validate parameters
        if resolution not in RESOLUTION_MAP:
            return {"error": f"Invalid resolution. Must be one of: {list(RESOLUTION_MAP.keys())}"}
        
        if not (15 <= sample_steps <= 50):
            return {"error": "sample_steps must be between 15 and 50"}
        
        # Ensure model server is running
        if not ensure_model_server():
            return {"error": "Failed to start model server"}
        
        # Download/decode inputs to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Handle image input
            image_path = Path(tmpdir) / "input_image.jpg"
            if image_input.startswith(('http://', 'https://')):
                print(f"Downloading image from URL: {image_input}")
                response = requests.get(image_input, timeout=30)
                response.raise_for_status()
                image_path.write_bytes(response.content)
            else:
                image_path.write_bytes(base64.b64decode(image_input))
            
            # Handle audio input
            audio_path = Path(tmpdir) / "input_audio.wav"
            if audio_input.startswith(('http://', 'https://')):
                print(f"Downloading audio from URL: {audio_input}")
                response = requests.get(audio_input, timeout=30)
                response.raise_for_status()
                audio_path.write_bytes(response.content)
            else:
                audio_path.write_bytes(base64.b64decode(audio_input))
            
            print(f"Starting generation: {resolution}, {sample_steps} steps")
            
            # Send generation request to model server
            request = {
                "action": "generate",
                "image_path": str(image_path),
                "audio_path": str(audio_path),
                "prompt": prompt,
                "resolution": resolution,
                "sample_steps": sample_steps
            }
            
            response = send_to_model_server(request)
            
            if not response.get("success"):
                return {"error": response.get("error", "Unknown error")}
            
            # Upload video to R2
            video_path = response.get("video_path")
            if not video_path or not os.path.exists(video_path):
                return {"error": "Video file not found"}
            
            print(f"Uploading video to R2...")
            video_filename = f"s2v_{resolution}_{sample_steps}steps.mp4"
            video_url = upload_to_r2(video_path, video_filename)
            
            # Clean up generated file
            os.unlink(video_path)
            
            total_time = time.time() - start_time
            
            return {
                "video_url": video_url,
                "generation_time": response.get("generation_time", 0),
                "total_time": round(total_time, 2),
                "video_size_mb": response.get("video_size_mb", 0),
                "resolution": resolution,
                "sample_steps": sample_steps,
                "warm_model": True  # Indicates warm model was used
            }
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Initialize on container startup
print("=" * 60)
print("Wan2.2 S2V Handler v2.0.0 (Warm Model Architecture)")
print("=" * 60)

# Apply FlashAttention patches (must be done before any model imports)
print("Applying FlashAttention compatibility patches...")
from patches.apply_patches import apply_flashattention_patches
apply_flashattention_patches()
print("✓ Patches applied")

# Pre-start model server to minimize first request latency
print("Pre-starting model server...")
if ensure_model_server():
    print("✓ Model server running - ready for requests!")
else:
    print("⚠ Model server will be started on first request")

print("=" * 60)

# Start RunPod serverless handler
runpod.serverless.start({"handler": generate_video})
