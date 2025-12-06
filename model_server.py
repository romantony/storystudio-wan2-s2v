#!/usr/bin/env python
"""
Wan2.2 S2V Model Server - Keeps model warm in GPU memory

This server:
1. Loads the Wan2.2 S2V-14B model once at startup
2. Keeps it warm in GPU memory
3. Handles generation requests via Unix socket IPC
4. Eliminates cold start delays for subsequent requests

Socket protocol: JSON messages with newline delimiter
Request: {"action": "generate", "image_path": "...", "audio_path": "...", ...}
Response: {"success": true, "video_path": "..."} or {"success": false, "error": "..."}

Version: 2.0.0
"""

import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Socket configuration
SOCKET_PATH = "/tmp/wan2_s2v_model_server.sock"
MODEL_DIR = "/runpod-volume/models/Wan2.2-S2V-14B"
WAN_DIR = "/workspace/Wan2.2"

# Resolution mapping (width * height = max_area)
RESOLUTION_MAP = {
    "480p": {"max_area": 480 * 832, "shift": 3.0},
    "720p": {"max_area": 720 * 1280, "shift": 5.0},
}


class ModelServer:
    """Persistent model server that keeps Wan2.2 S2V model warm in GPU memory"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.model_loaded = False
        self.is_running = False
        self.socket_server = None
        self.lock = threading.Lock()
        
        # Generation counter for unique filenames
        self.generation_count = 0
        
    def load_model(self):
        """Load the Wan2.2 S2V model into GPU memory"""
        if self.model_loaded:
            print("Model already loaded")
            return True
            
        print("=" * 60)
        print("LOADING WAN2.2 S2V-14B MODEL INTO GPU MEMORY")
        print("=" * 60)
        
        try:
            start_time = time.time()
            
            # Set CUDA environment
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            # Initialize PyTorch CUDA context first
            import torch
            torch.cuda.init()
            torch.cuda.set_device(0)
            _ = torch.zeros(1, device="cuda:0")
            print(f"✓ CUDA initialized: {torch.cuda.get_device_name(0)}")
            print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Patch safetensors to load to CPU first (workaround for rust backend cuda:0 issue)
            from safetensors import torch as safetensors_torch
            _original_load = safetensors_torch.load_file
            def _patched_load(filename, device="cpu"):
                return _original_load(filename, device="cpu")
            safetensors_torch.load_file = _patched_load
            print("✓ Patched safetensors to load via CPU")
            
            # Add Wan2.2 to path
            sys.path.insert(0, WAN_DIR)
            os.chdir(WAN_DIR)
            
            # Apply FlashAttention patches
            sys.path.insert(0, "/app")
            from patches.apply_patches import apply_flashattention_patches
            apply_flashattention_patches()
            print("✓ FlashAttention patches applied")
            
            # Import Wan2.2 modules
            print("Importing Wan2.2 modules...")
            from wan.configs import WAN_CONFIGS
            from wan.speech2video import WanS2V
            from wan.utils.utils import save_video, merge_video_audio
            print("✓ Wan2.2 modules imported")
            
            # Store utility functions
            self.save_video = save_video
            self.merge_video_audio = merge_video_audio
            
            # Get S2V config
            self.config = WAN_CONFIGS["s2v-14B"]
            print(f"✓ Config loaded: sample_fps={self.config.sample_fps}")
            
            # Load model - keep in GPU memory
            print(f"Loading model from {MODEL_DIR}...")
            print("  This may take 10-15 minutes on first load...")
            
            self.model = WanS2V(
                config=self.config,
                checkpoint_dir=MODEL_DIR,
                device_id=0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_sp=False,
                t5_cpu=False,  # Keep T5 on GPU for speed
                init_on_cpu=True,  # Init on CPU then move to GPU
                convert_model_dtype=True,  # Use bfloat16 for efficiency
            )
            
            # Log VRAM usage
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"✓ Model loaded: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
            load_time = time.time() - start_time
            print(f"✓ Model loaded in {load_time:.1f} seconds")
            print("=" * 60)
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            traceback.print_exc()
            return False
    
    def generate_video(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video from image + audio using the warm model"""
        with self.lock:  # Ensure only one generation at a time
            try:
                import torch
                
                start_time = time.time()
                
                # Extract parameters
                image_path = request.get("image_path")
                audio_path = request.get("audio_path")
                prompt = request.get("prompt", "")
                resolution = request.get("resolution", "480p")
                sample_steps = request.get("sample_steps", 20)
                
                print(f"\n{'='*60}")
                print(f"GENERATING VIDEO: {resolution}, {sample_steps} steps")
                print(f"Image: {image_path}")
                print(f"Audio: {audio_path}")
                print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
                print(f"{'='*60}")
                
                # Get resolution config
                res_config = RESOLUTION_MAP.get(resolution, RESOLUTION_MAP["480p"])
                max_area = res_config["max_area"]
                shift = res_config["shift"]
                
                # Generate video using WanS2V
                print("Starting generation...")
                gen_start = time.time()
                
                video = self.model.generate(
                    input_prompt=prompt,
                    ref_image_path=image_path,
                    audio_path=audio_path,
                    enable_tts=False,
                    tts_prompt_audio=None,
                    tts_prompt_text=None,
                    tts_text=None,
                    num_repeat=None,  # Auto-detect from audio length
                    pose_video=None,
                    max_area=max_area,
                    shift=shift,
                    sample_solver='unipc',
                    sampling_steps=sample_steps,
                    guide_scale=4.5,
                    n_prompt="",
                    seed=-1,  # Random seed
                    offload_model=False,  # Keep model in VRAM
                    init_first_frame=False,
                )
                
                gen_time = time.time() - gen_start
                print(f"✓ Generation complete in {gen_time:.1f}s")
                
                if video is None:
                    return {
                        "success": False,
                        "error": "Video generation returned None"
                    }
                
                # Save video
                self.generation_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"s2v-14B_{timestamp}_{self.generation_count}.mp4"
                output_path = os.path.join(WAN_DIR, output_filename)
                temp_video_path = output_path.replace(".mp4", "_noaudio.mp4")
                
                # Save video frames (without audio)
                print(f"Saving video to {output_path}...")
                self.save_video(
                    tensor=video[None],  # Add batch dimension
                    save_file=temp_video_path,
                    fps=self.config.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                )
                
                # Merge audio with video
                print("Merging audio...")
                self.merge_video_audio(
                    video_path=temp_video_path,
                    audio_path=audio_path,
                    output_path=output_path
                )
                
                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                
                total_time = time.time() - start_time
                video_size = os.path.getsize(output_path) / (1024 * 1024)
                
                print(f"✓ Video saved: {video_size:.2f}MB")
                print(f"✓ Total generation time: {total_time:.1f}s")
                
                # Report VRAM after generation
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"VRAM after generation: {allocated:.1f}GB")
                
                return {
                    "success": True,
                    "video_path": output_path,
                    "generation_time": round(gen_time, 2),
                    "total_time": round(total_time, 2),
                    "video_size_mb": round(video_size, 2)
                }
                
            except Exception as e:
                print(f"✗ Generation failed: {e}")
                traceback.print_exc()
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def handle_client(self, conn: socket.socket):
        """Handle a single client connection"""
        try:
            # Read request (newline-delimited JSON)
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
            
            if not data:
                return
            
            # Parse request
            request_str = data.decode("utf-8").strip()
            print(f"\nReceived request: {request_str[:200]}...")
            
            request = json.loads(request_str)
            action = request.get("action")
            
            # Handle different actions
            if action == "health":
                response = {
                    "success": True,
                    "model_loaded": self.model_loaded,
                    "status": "ready" if self.model_loaded else "loading"
                }
            elif action == "generate":
                if not self.model_loaded:
                    response = {"success": False, "error": "Model not loaded"}
                else:
                    response = self.generate_video(request)
            elif action == "status":
                import torch
                response = {
                    "success": True,
                    "model_loaded": self.model_loaded,
                    "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if self.model_loaded else 0,
                    "generation_count": self.generation_count
                }
            else:
                response = {"success": False, "error": f"Unknown action: {action}"}
            
            # Send response
            response_str = json.dumps(response) + "\n"
            conn.sendall(response_str.encode("utf-8"))
            
        except Exception as e:
            print(f"Error handling client: {e}")
            traceback.print_exc()
            try:
                error_response = json.dumps({"success": False, "error": str(e)}) + "\n"
                conn.sendall(error_response.encode("utf-8"))
            except:
                pass
        finally:
            conn.close()
    
    def run(self):
        """Start the model server"""
        print("=" * 60)
        print("WAN2.2 S2V MODEL SERVER STARTING")
        print("=" * 60)
        
        # Remove existing socket file
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        
        # Load model first
        if not self.load_model():
            print("Failed to load model, exiting")
            sys.exit(1)
        
        # Create Unix socket server
        self.socket_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket_server.bind(SOCKET_PATH)
        self.socket_server.listen(5)
        
        # Make socket world-readable
        os.chmod(SOCKET_PATH, 0o777)
        
        print(f"✓ Model server listening on {SOCKET_PATH}")
        print("=" * 60)
        
        self.is_running = True
        
        # Handle signals for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down...")
            self.is_running = False
            self.socket_server.close()
            if os.path.exists(SOCKET_PATH):
                os.unlink(SOCKET_PATH)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Accept connections
        while self.is_running:
            try:
                conn, _ = self.socket_server.accept()
                # Handle in separate thread to allow concurrent status checks
                thread = threading.Thread(target=self.handle_client, args=(conn,))
                thread.daemon = True
                thread.start()
            except socket.error as e:
                if self.is_running:
                    print(f"Socket error: {e}")
                break


def main():
    """Main entry point"""
    server = ModelServer()
    server.run()


if __name__ == "__main__":
    main()
