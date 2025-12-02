# Test Client for Wan2.2 S2V

import runpod
import base64
from pathlib import Path
import time

# Configuration - UPDATE THESE VALUES
RUNPOD_API_KEY = "your-runpod-api-key"
ENDPOINT_ID = "your-endpoint-id"

# Test files
IMAGE_PATH = "test_image.png"
AUDIO_PATH = "test_audio.mp3"
OUTPUT_PATH = "output.mp4"

def encode_file(file_path):
    """Encode file to base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def test_wan2_serverless():
    """Test Wan2.2 S2V serverless endpoint"""
    
    print("=" * 60)
    print("Wan2.2 S2V Serverless Test")
    print("=" * 60)
    
    # Configure RunPod
    runpod.api_key = RUNPOD_API_KEY
    
    # Encode inputs
    print("\n1. Encoding inputs...")
    image_b64 = encode_file(IMAGE_PATH)
    audio_b64 = encode_file(AUDIO_PATH)
    print(f"   ✓ Image: {len(image_b64)} bytes (base64)")
    print(f"   ✓ Audio: {len(audio_b64)} bytes (base64)")
    
    # Create endpoint
    print("\n2. Connecting to endpoint...")
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Submit job
    print("\n3. Submitting generation job...")
    start_time = time.time()
    
    job = endpoint.run({
        "image": image_b64,
        "audio": audio_b64,
        "resolution": "480p",  # Start with 480p for faster testing
        "sample_steps": 30,
        "prompt": ""
    })
    
    print(f"   ✓ Job ID: {job.job_id}")
    
    # Monitor status
    print("\n4. Waiting for generation...")
    print("   This may take 6-25 minutes depending on cold/warm start")
    
    last_status = None
    while True:
        status = job.status()
        if status != last_status:
            print(f"   Status: {status}")
            last_status = status
        
        if status in ["COMPLETED", "FAILED"]:
            break
        
        time.sleep(10)  # Check every 10 seconds
    
    # Get result
    print("\n5. Retrieving result...")
    result = job.output()
    
    if "error" in result:
        print(f"   ✗ Error: {result['error']}")
        return False
    
    if "video" not in result:
        print(f"   ✗ No video in result: {result}")
        return False
    
    # Decode and save video
    print("\n6. Saving video...")
    video_bytes = base64.b64decode(result["video"])
    with open(OUTPUT_PATH, "wb") as f:
        f.write(video_bytes)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("✓ TEST SUCCESSFUL!")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Generation time: {result['generation_time']}s")
    print(f"Video size: {result['video_size_mb']} MB")
    print(f"Resolution: {result['resolution']}")
    print(f"Sample steps: {result['sample_steps']}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    # Verify files exist
    if not Path(IMAGE_PATH).exists():
        print(f"ERROR: Image file not found: {IMAGE_PATH}")
        print("Please provide a test image file")
        exit(1)
    
    if not Path(AUDIO_PATH).exists():
        print(f"ERROR: Audio file not found: {AUDIO_PATH}")
        print("Please provide a test audio file")
        exit(1)
    
    # Run test
    success = test_wan2_serverless()
    exit(0 if success else 1)
