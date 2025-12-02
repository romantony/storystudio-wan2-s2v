"""
RunPod Serverless Handler for Wan2.2 S2V
Compatible with RunPod's serverless architecture
"""
import runpod
import subprocess
import tempfile
import base64
import os
import time
from pathlib import Path
from typing import Dict, Any

# Apply patches BEFORE any Wan2.2 imports
from patches.apply_patches import apply_flashattention_patches
apply_flashattention_patches()

from huggingface_hub import snapshot_download

# Configuration
MODEL_ID = "Wan-AI/Wan2.2-S2V-14B"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")
WAN_DIR = "/workspace/Wan2.2"

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
        """Download model if not cached"""
        if self.model_loaded:
            return
        
        print(f"Loading model: {MODEL_ID}")
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

def generate_video(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function
    
    Expected job input:
    {
        "image": "base64_encoded_image",
        "audio": "base64_encoded_audio",
        "prompt": "optional text prompt",
        "resolution": "480p" or "720p",
        "sample_steps": 20-50 (default 30)
    }
    
    Returns:
    {
        "video": "base64_encoded_video",
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
        image_base64 = job_input["image"]
        audio_base64 = job_input["audio"]
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
        
        # Decode base64 inputs
        image_bytes = base64.b64decode(image_base64)
        audio_bytes = base64.b64decode(audio_base64)
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded files
            image_path = Path(tmpdir) / "input_image.jpg"
            audio_path = Path(tmpdir) / "input_audio.wav"
            
            image_path.write_bytes(image_bytes)
            audio_path.write_bytes(audio_bytes)
            
            # Build generation command
            size = RESOLUTION_MAP[resolution]
            cmd = [
                "python",
                f"{WAN_DIR}/generate.py",
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
            result = subprocess.run(
                cmd,
                cwd=WAN_DIR,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"Generation failed: {result.stderr}")
                return {"error": f"Generation failed: {result.stderr}"}
            
            # Find generated video
            video_files = list(Path(WAN_DIR).glob("s2v-14B_*.mp4"))
            if not video_files:
                return {"error": "No video file generated"}
            
            # Get most recent video
            video_path = max(video_files, key=lambda p: p.stat().st_mtime)
            
            # Encode to base64
            video_bytes = video_path.read_bytes()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            # Clean up generated file
            video_path.unlink()
            
            generation_time = time.time() - start_time
            
            return {
                "video": video_base64,
                "generation_time": round(generation_time, 2),
                "video_size_mb": round(len(video_bytes) / 1024 / 1024, 2),
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

# Initialize model on container startup
print("Initializing Wan2.2 S2V handler...")
model_config.load_model()
print("✓ Handler ready")

# Start RunPod serverless handler
runpod.serverless.start({"handler": generate_video})
