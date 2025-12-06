# StoryStudio Video Generation API

Complete API documentation for generating talking videos using Wan2.2 S2V.

## Overview

This API enables:
1. **Dynamic Worker Management** - Enable/disable active workers to optimize costs
2. **Video Generation** - Create talking videos from image + audio
3. **Job Status Tracking** - Monitor generation progress

## Base Configuration

```
ENDPOINT_ID: YOUR_ENDPOINT_ID
BASE_URL: https://api.runpod.ai/v2
API_KEY: YOUR_RUNPOD_API_KEY
```

---

## 1. Worker Management API

### 1.1 Enable Active Worker (Before Project Starts)

Call this before starting video generation to warm up the GPU and load the model.

```bash
curl -X POST "https://api.runpod.ai/graphql" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {API_KEY}" \
  -d '{
    "query": "mutation { updateEndpoint(input: { id: \"{ENDPOINT_ID}\", workersMin: 1, idleTimeout: 300 }) { id workersMin idleTimeout } }"
  }'
```

**JavaScript/TypeScript:**
```javascript
async function enableActiveWorker(endpointId, apiKey) {
  const response = await fetch('https://api.runpod.ai/graphql', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      query: `mutation {
        updateEndpoint(input: {
          id: "${endpointId}",
          workersMin: 1,
          idleTimeout: 300
        }) {
          id
          workersMin
          idleTimeout
        }
      }`
    })
  });
  return response.json();
}
```

**Response:**
```json
{
  "data": {
    "updateEndpoint": {
      "id": "1tggndzlc063rw",
      "workersMin": 1,
      "idleTimeout": 300
    }
  }
}
```

### 1.2 Disable Active Worker (After Project Completes)

Call this after all videos are generated and no jobs are pending.

```bash
curl -X POST "https://api.runpod.ai/graphql" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {API_KEY}" \
  -d '{
    "query": "mutation { updateEndpoint(input: { id: \"{ENDPOINT_ID}\", workersMin: 0 }) { id workersMin } }"
  }'
```

**JavaScript/TypeScript:**
```javascript
async function disableActiveWorker(endpointId, apiKey) {
  const response = await fetch('https://api.runpod.ai/graphql', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      query: `mutation {
        updateEndpoint(input: {
          id: "${endpointId}",
          workersMin: 0
        }) {
          id
          workersMin
        }
      }`
    })
  });
  return response.json();
}
```

### 1.3 Check Endpoint Status

```bash
curl -X POST "https://api.runpod.ai/graphql" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {API_KEY}" \
  -d '{
    "query": "query { endpoint(id: \"{ENDPOINT_ID}\") { id name workersMin workersMax idleTimeout workers { id status } } }"
  }'
```

---

## 2. Video Generation API

### 2.1 Submit Video Generation Job

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {API_KEY}" \
  -d '{
    "input": {
      "image": "https://example.com/character.png",
      "audio": "https://example.com/speech.mp3",
      "prompt": "Person speaking naturally with expressive gestures",
      "resolution": "480p",
      "sample_steps": 20
    }
  }'
```

**JavaScript/TypeScript:**
```javascript
async function generateVideo(endpointId, apiKey, params) {
  const response = await fetch(`https://api.runpod.ai/v2/${endpointId}/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      input: {
        image: params.imageUrl,
        audio: params.audioUrl,
        prompt: params.prompt || "Person speaking naturally",
        resolution: params.resolution || "480p",
        sample_steps: params.sampleSteps || 20
      }
    })
  });
  return response.json();
}
```

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | ✅ | - | Image URL or base64 encoded |
| `audio` | string | ✅ | - | Audio URL or base64 encoded |
| `prompt` | string | ❌ | "" | Description of the scene/action |
| `resolution` | string | ❌ | "480p" | "480p" or "720p" |
| `sample_steps` | int | ❌ | 20 | 15-50 (higher = better quality, slower) |

**Response:**
```json
{
  "id": "97a5521f-983a-432c-8da5-99506599aec9-e1",
  "status": "IN_QUEUE"
}
```

### 2.2 Check Job Status

```bash
curl -X GET "https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}" \
  -H "Authorization: Bearer {API_KEY}"
```

**JavaScript/TypeScript:**
```javascript
async function checkJobStatus(endpointId, apiKey, jobId) {
  const response = await fetch(
    `https://api.runpod.ai/v2/${endpointId}/status/${jobId}`,
    {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    }
  );
  return response.json();
}
```

**Response (In Progress):**
```json
{
  "id": "97a5521f-983a-432c-8da5-99506599aec9-e1",
  "status": "IN_PROGRESS",
  "delayTime": 5060,
  "executionTime": 120000
}
```

**Response (Completed):**
```json
{
  "id": "97a5521f-983a-432c-8da5-99506599aec9-e1",
  "status": "COMPLETED",
  "delayTime": 5060,
  "executionTime": 567517,
  "output": {
    "video_url": "https://parentearn.com/VideoGen/20251205_024925_s2v_480p_20steps.mp4",
    "generation_time": 567.46,
    "video_size_mb": 1.71,
    "resolution": "480p",
    "sample_steps": 20
  }
}
```

**Response (Failed):**
```json
{
  "id": "56a7f791-77b9-43f6-867c-c74af1426d6d-e1",
  "status": "FAILED",
  "error": "job timed out after 1 retries"
}
```

### 2.3 Job Status Values

| Status | Description |
|--------|-------------|
| `IN_QUEUE` | Job is waiting for an available worker |
| `IN_PROGRESS` | Job is currently being processed |
| `COMPLETED` | Job finished successfully |
| `FAILED` | Job failed (check `error` field) |
| `CANCELLED` | Job was cancelled |
| `TIMED_OUT` | Job exceeded timeout limit |

---

## 3. Complete Project Workflow

### Full TypeScript Implementation

```typescript
interface VideoRequest {
  imageUrl: string;
  audioUrl: string;
  prompt?: string;
  resolution?: '480p' | '720p';
  sampleSteps?: number;
}

interface VideoResult {
  jobId: string;
  videoUrl: string;
  generationTime: number;
  videoSizeMb: number;
}

class StoryStudioVideoAPI {
  private endpointId: string;
  private apiKey: string;
  private baseUrl = 'https://api.runpod.ai';

  constructor(endpointId: string, apiKey: string) {
    this.endpointId = endpointId;
    this.apiKey = apiKey;
  }

  // Enable active worker before project starts
  async enableActiveWorker(): Promise<void> {
    console.log('Enabling active worker...');
    const response = await fetch(`${this.baseUrl}/graphql`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        query: `mutation {
          updateEndpoint(input: {
            id: "${this.endpointId}",
            workersMin: 1,
            idleTimeout: 300
          }) { id workersMin }
        }`
      })
    });
    
    if (!response.ok) {
      throw new Error('Failed to enable active worker');
    }
    
    console.log('Active worker enabled. Waiting for GPU warm-up...');
    // Wait for worker to initialize (model loading takes ~10-15 min on cold start)
    await this.waitForWorkerReady();
  }

  // Wait for worker to be ready
  private async waitForWorkerReady(maxWaitMs = 60000): Promise<void> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitMs) {
      const status = await this.getEndpointStatus();
      const activeWorkers = status?.workers?.filter(w => w.status === 'RUNNING');
      
      if (activeWorkers?.length > 0) {
        console.log('Worker is ready!');
        return;
      }
      
      console.log('Waiting for worker to start...');
      await this.sleep(5000);
    }
    
    console.log('Worker may still be initializing, proceeding anyway...');
  }

  // Get endpoint status
  async getEndpointStatus(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/graphql`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        query: `query {
          endpoint(id: "${this.endpointId}") {
            id workersMin workersMax
            workers { id status }
          }
        }`
      })
    });
    
    const data = await response.json();
    return data?.data?.endpoint;
  }

  // Submit video generation job
  async submitJob(request: VideoRequest): Promise<string> {
    const response = await fetch(`${this.baseUrl}/v2/${this.endpointId}/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        input: {
          image: request.imageUrl,
          audio: request.audioUrl,
          prompt: request.prompt || 'Person speaking naturally',
          resolution: request.resolution || '480p',
          sample_steps: request.sampleSteps || 20
        }
      })
    });

    const data = await response.json();
    return data.id;
  }

  // Check job status
  async getJobStatus(jobId: string): Promise<any> {
    const response = await fetch(
      `${this.baseUrl}/v2/${this.endpointId}/status/${jobId}`,
      {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`
        }
      }
    );
    return response.json();
  }

  // Wait for job completion
  async waitForCompletion(jobId: string, pollIntervalMs = 10000): Promise<VideoResult> {
    console.log(`Waiting for job ${jobId}...`);
    
    while (true) {
      const status = await this.getJobStatus(jobId);
      
      if (status.status === 'COMPLETED') {
        return {
          jobId,
          videoUrl: status.output.video_url,
          generationTime: status.output.generation_time,
          videoSizeMb: status.output.video_size_mb
        };
      }
      
      if (status.status === 'FAILED') {
        throw new Error(`Job failed: ${status.error}`);
      }
      
      if (status.status === 'TIMED_OUT' || status.status === 'CANCELLED') {
        throw new Error(`Job ${status.status}`);
      }
      
      console.log(`Status: ${status.status}, waiting...`);
      await this.sleep(pollIntervalMs);
    }
  }

  // Disable active worker after project completes
  async disableActiveWorker(): Promise<void> {
    console.log('Disabling active worker...');
    
    await fetch(`${this.baseUrl}/graphql`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({
        query: `mutation {
          updateEndpoint(input: {
            id: "${this.endpointId}",
            workersMin: 0
          }) { id workersMin }
        }`
      })
    });
    
    console.log('Active worker disabled');
  }

  // Process entire project (multiple scenes)
  async processProject(scenes: VideoRequest[]): Promise<VideoResult[]> {
    const results: VideoResult[] = [];
    
    try {
      // Step 1: Enable active worker
      await this.enableActiveWorker();
      
      // Step 2: Submit all jobs
      console.log(`Submitting ${scenes.length} video generation jobs...`);
      const jobIds: string[] = [];
      
      for (const scene of scenes) {
        const jobId = await this.submitJob(scene);
        jobIds.push(jobId);
        console.log(`Submitted job: ${jobId}`);
      }
      
      // Step 3: Wait for all completions
      console.log('Waiting for all jobs to complete...');
      
      for (const jobId of jobIds) {
        const result = await this.waitForCompletion(jobId);
        results.push(result);
        console.log(`Completed: ${result.videoUrl}`);
      }
      
      return results;
      
    } finally {
      // Step 4: Always disable active worker when done
      await this.disableActiveWorker();
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Usage Example
async function main() {
  const api = new StoryStudioVideoAPI(
    'YOUR_ENDPOINT_ID',
    'YOUR_RUNPOD_API_KEY'
  );

  // Process a project with multiple scenes
  const scenes: VideoRequest[] = [
    {
      imageUrl: 'https://parentearn.com/VideoGen/Logan-3.png',
      audioUrl: 'https://parentearn.com/VideoGen/logan-audio-3.mp3',
      prompt: 'Professional man speaking confidently',
      resolution: '480p',
      sampleSteps: 20
    },
    {
      imageUrl: 'https://parentearn.com/VideoGen/test_image2.png',
      audioUrl: 'https://parentearn.com/VideoGen/test_audio2.mp3',
      prompt: 'Person speaking naturally',
      resolution: '480p',
      sampleSteps: 20
    }
  ];

  try {
    const results = await api.processProject(scenes);
    console.log('All videos generated:', results);
  } catch (error) {
    console.error('Project failed:', error);
  }
}

main();
```

---

## 4. Cost Optimization

### Worker States and Costs

| State | Cost | When to Use |
|-------|------|-------------|
| `workersMin: 0` | $0/hr idle | No active projects |
| `workersMin: 1` | ~$1.89/hr (A100) | During project processing |
| `idleTimeout: 300` | Auto-shutdown after 5 min idle | Between jobs |

### Recommended Flow

```
Project Start
     │
     ▼
┌────────────────────────┐
│ Enable Active Worker   │  ← workersMin: 1
│ (Wait for GPU warm-up) │     ~60 sec for worker start
└────────────────────────┘     ~10-15 min for model load (first job)
     │
     ▼
┌────────────────────────┐
│ Submit All Jobs        │  ← Sequential or parallel
│ (Videos generate fast) │     ~2-3 min each with warm model
└────────────────────────┘
     │
     ▼
┌────────────────────────┐
│ Wait for Completions   │  ← Poll status every 10 sec
└────────────────────────┘
     │
     ▼
┌────────────────────────┐
│ Disable Active Worker  │  ← workersMin: 0
│ (Stop paying for idle) │     Worker shuts down
└────────────────────────┘
     │
     ▼
Project Complete
```

### Expected Times

| Scenario | Time per Video |
|----------|----------------|
| Cold start (first video) | ~10-15 min |
| Warm worker (subsequent) | ~2-3 min |
| With Active Worker | ~2-3 min (all videos) |

### Cost Examples

| Project Size | Without Active Worker | With Active Worker |
|--------------|----------------------|-------------------|
| 1 video | ~$0.50 (15 min) | ~$0.50 (15 min) |
| 5 videos | ~$2.50 (75 min) | ~$0.80 (25 min) |
| 10 videos | ~$5.00 (150 min) | ~$1.20 (40 min) |

---

## 5. Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `job timed out` | Generation exceeded 30 min | Reduce resolution or use shorter audio |
| `CUDA not available` | GPU initialization failed | Retry - usually transient |
| `Missing required inputs` | No image or audio | Check request parameters |
| `Invalid resolution` | Wrong resolution value | Use "480p" or "720p" |

### Retry Logic

```javascript
async function submitWithRetry(api, request, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const jobId = await api.submitJob(request);
      return await api.waitForCompletion(jobId);
    } catch (error) {
      if (attempt === maxRetries) throw error;
      console.log(`Attempt ${attempt} failed, retrying...`);
      await new Promise(r => setTimeout(r, 5000));
    }
  }
}
```

---

## 6. Webhook Support (Optional)

Instead of polling, you can use webhooks:

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {API_KEY}" \
  -d '{
    "input": {
      "image": "https://example.com/image.png",
      "audio": "https://example.com/audio.mp3"
    },
    "webhook": "https://your-server.com/webhook/video-complete"
  }'
```

Your webhook will receive a POST with the job result when complete.

---

## 7. Quick Reference

### Endpoints

| Action | Method | URL |
|--------|--------|-----|
| Enable Worker | POST | `https://api.runpod.ai/graphql` |
| Disable Worker | POST | `https://api.runpod.ai/graphql` |
| Submit Job | POST | `https://api.runpod.ai/v2/{id}/run` |
| Check Status | GET | `https://api.runpod.ai/v2/{id}/status/{jobId}` |
| Cancel Job | POST | `https://api.runpod.ai/v2/{id}/cancel/{jobId}` |

### Headers

```
Content-Type: application/json
Authorization: Bearer {API_KEY}
```
