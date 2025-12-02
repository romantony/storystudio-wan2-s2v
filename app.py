"""
FastAPI server for Wan2.2 S2V video generation
Optimized configuration: 1x GPU, no offload, 30 sampling steps
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import subprocess
import tempfile
import base64
import os
import time
from pathlib import Path
from typing import Optional

# Apply patches BEFORE any Wan2.2 imports
from patches.apply_patches import apply_flashattention_patches
apply_flashattention_patches()

# Now safe to import Wan2.2 modules
from huggingface_hub import snapshot_download

app = FastAPI(title="Wan2.2 S2V API", version="1.0.0")

# Configuration
MODEL_ID = "Wan-AI/Wan2.2-S2V-14B"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")
WAN_DIR = "/workspace/Wan2.2"

# Resolution mapping (validated options)
RESOLUTION_MAP = {
    "480p": "832*480",
    "720p": "1024*704"
}

class GenerationConfig:
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

config = GenerationConfig()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        config.load_model()
        print("✓ Server started successfully")
    except Exception as e:
        print(f"✗ Startup error: {e}")
        raise

@app.get("/")
def root():
    return {
        "status": "online",
        "model": "Wan2.2-S2V-14B",
        "version": "1.0.0",
        "gpu_config": "Optimized: 1x GPU, No Offload, 30 Steps",
        "endpoints": {
            "POST /generate": "Generate video from audio and image",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": config.model_loaded,
        "model_path": config.model_dir
    }

@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form(""),
    resolution: str = Form("720p"),
    sample_steps: int = Form(30)
):
    """
    Generate video from audio and image
    
    Parameters:
    - image: Input image (JPG/PNG)
    - audio: Input audio (WAV/MP3)
    - prompt: Optional text prompt
    - resolution: "480p" (832x480) or "720p" (1024x704)
    - sample_steps: Sampling steps (20-50, default 30)
    
    Returns:
    - JSON with base64-encoded MP4 video
    """
    start_time = time.time()
    
    # Validate inputs
    if resolution not in RESOLUTION_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resolution. Must be one of: {list(RESOLUTION_MAP.keys())}"
        )
    
    if not (20 <= sample_steps <= 50):
        raise HTTPException(
            status_code=400,
            detail="sample_steps must be between 20 and 50"
        )
    
    # Ensure model is loaded
    if not config.model_loaded:
        config.load_model()
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded files
        image_path = Path(tmpdir) / "input_image.jpg"
        audio_path = Path(tmpdir) / "input_audio.wav"
        
        image_bytes = await image.read()
        audio_bytes = await audio.read()
        
        image_path.write_bytes(image_bytes)
        audio_path.write_bytes(audio_bytes)
        
        # Build generation command
        size = RESOLUTION_MAP[resolution]
        cmd = [
            "python",
            f"{WAN_DIR}/generate.py",
            "--task", "s2v-14B",
            "--size", size,
            "--ckpt_dir", config.model_dir,
            "--offload_model", "False",  # Keep model in GPU for speed
            "--convert_model_dtype",
            "--sample_steps", str(sample_steps),
            "--image", str(image_path),
            "--audio", str(audio_path)
        ]
        
        if prompt:
            cmd.extend(["--prompt", prompt])
        
        # Run generation
        try:
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
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed: {result.stderr}"
                )
            
            # Find generated video
            video_files = list(Path(WAN_DIR).glob("s2v-14B_*.mp4"))
            if not video_files:
                raise HTTPException(
                    status_code=500,
                    detail="No video file generated"
                )
            
            # Get most recent video
            video_path = max(video_files, key=lambda p: p.stat().st_mtime)
            
            # Encode to base64
            video_bytes = video_path.read_bytes()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            
            # Clean up generated file
            video_path.unlink()
            
            generation_time = time.time() - start_time
            
            return {
                "success": True,
                "video": video_base64,
                "generation_time": round(generation_time, 2),
                "video_size_mb": round(len(video_bytes) / 1024 / 1024, 2),
                "resolution": resolution,
                "sample_steps": sample_steps
            }
            
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Generation timed out (1 hour limit)"
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
