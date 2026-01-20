# Vision Models API Usage Guide

Complete guide for using the Vision Models REST API.

## Installation

1. Install the vision-model-repo package:
```bash
pip install -e .
```

2. Install API server dependencies:
```bash
pip install -r requirements.api.txt
```

## Configuration

Before starting the server, configure the HuggingFace cache directory:

```bash
# Set cache directory (recommended for production)
export HF_ROOT_FOLDER=/path/to/cache

# Optional: Set HuggingFace token for private models
export HF_TOKEN=hf_your_token_here
```

Or copy and edit the `.env.example` file:
```bash
cp .env.example .env
# Edit .env with your settings
```

See [CONFIG.md](CONFIG.md) for detailed configuration options.

## Starting the Server

### Development Mode

```bash
# Using the quick start script
python run_api_server.py --reload

# Or using uvicorn directly
uvicorn vision_model_repo.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
# Multiple workers for production
uvicorn vision_model_repo.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`.

Interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

## API Endpoints Overview

### Health & Status
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /models/status` - Get status of all models

### Model Management
- `POST /models/load` - Load a model into memory
- `POST /models/unload` - Unload a model from memory
- `POST /models/unload-all` - Unload all models

### Inference (Synchronous)
- `POST /inference/sam3` - Instance segmentation
- `POST /inference/grounding-dino` - Object detection
- `POST /inference/depth-anything` - Depth estimation
- `POST /inference/dinov3` - Feature extraction

### Async Jobs
- `POST /jobs/submit/{model_type}` - Submit async job
- `GET /jobs/{job_id}` - Get job status
- `DELETE /jobs/{job_id}` - Delete job
- `GET /jobs` - List all jobs

## Usage Examples

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "loaded_models": ["sam3"],
  "available_memory_gb": 10.5,
  "gpu_available": true
}
```

### 2. Load a Model

```bash
curl -X POST http://localhost:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sam3",
    "device": "cuda"
  }'
```

### 3. Instance Segmentation (Sam3)

#### Using Base64 Images

```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Make request
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [image_base64],
        "prompts": ["person", "car"],
        "threshold": 0.5,
        "output_format": "dense"
    }
)

result = response.json()
print(f"Processing time: {result['processing_time']}s")
print(f"Results: {len(result['results'][0])} detections")
```

#### Using Image URLs

```python
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [
            {
                "data": "https://example.com/image.jpg",
                "format": "url"
            }
        ],
        "prompts": ["object"],
        "threshold": 0.6,
        "output_format": "polygons"
    }
)
```

### 4. Object Detection (GroundingDino)

```python
import requests
import base64

# Load image
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Detect objects
response = requests.post(
    "http://localhost:8000/inference/grounding-dino",
    json={
        "images": [image_base64],
        "prompts": ["person", "car", "traffic light"],
        "box_threshold": 0.4,
        "text_threshold": 0.3
    }
)

result = response.json()
for detection in result["results"][0]["boxes"]:
    print(f"Box: {detection}, Score: {detection['score']}")
```

### 5. Depth Estimation (DepthAnythingV3)

```python
import requests
import base64
from PIL import Image
import io
import numpy as np

# Load image
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Estimate depth
response = requests.post(
    "http://localhost:8000/inference/depth-anything",
    json={
        "images": [image_base64],
        "export_format": "glb"
    }
)

result = response.json()

# Decode depth map
depth_base64 = result["depth_data"]
depth_bytes = base64.b64decode(depth_base64)
depth_image = Image.open(io.BytesIO(depth_bytes))

print(f"Depth map shape: {result['metadata']['shape']}")
depth_image.save("depth_output.png")
```

### 6. Feature Extraction (DinoV3)

```python
import requests
import base64
import numpy as np

# Load image
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Extract features
response = requests.post(
    "http://localhost:8000/inference/dinov3",
    json={
        "images": [image_base64]
    }
)

result = response.json()

# Features are returned as nested lists
features = np.array(result["features"]["last_hidden_state"])
pooled = np.array(result["features"]["pooler_output"])

print(f"Feature shape: {features.shape}")
print(f"Pooled shape: {pooled.shape}")
```

### 7. Batch Processing

Process multiple images in one request:

```python
import requests
import base64

# Load multiple images
images = []
for img_path in ["img1.jpg", "img2.jpg", "img3.jpg"]:
    with open(img_path, "rb") as f:
        images.append(base64.b64encode(f.read()).decode("utf-8"))

# Batch inference
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": images,
        "prompts": ["object"],  # Shared prompt
        "threshold": 0.5
    }
)

result = response.json()
print(f"Processed {len(result['results'])} images in {result['processing_time']}s")
```

### 8. Async Jobs (For Long-Running Tasks)

Submit a job:

```python
import requests
import time

# Submit async job
response = requests.post(
    "http://localhost:8000/jobs/submit/sam3",
    json={
        "images": [image_base64],
        "prompts": ["person", "car"],
        "threshold": 0.5
    }
)

job = response.json()
job_id = job["job_id"]
print(f"Job submitted: {job_id}")

# Poll for completion
while True:
    status_response = requests.get(f"http://localhost:8000/jobs/{job_id}")
    status = status_response.json()

    print(f"Status: {status['status']}")

    if status["status"] in ["completed", "failed"]:
        break

    time.sleep(1)

if status["status"] == "completed":
    print("Result:", status["result"])
else:
    print("Error:", status["error"])
```

### 9. Model Management

```python
import requests

# Load a specific model variant
requests.post(
    "http://localhost:8000/models/load",
    json={
        "model_type": "grounding_dino",
        "model_id": "IDEA-Research/grounding-dino-tiny",
        "device": "cuda"
    }
)

# Check what models are loaded
response = requests.get("http://localhost:8000/models/status")
print(response.json())

# Unload a model to free memory
requests.post(
    "http://localhost:8000/models/unload",
    json={"model_type": "grounding_dino"}
)
```

## Request/Response Formats

### Image Input Formats

The API accepts images in two formats:

1. **Base64 encoded** (default):
   ```json
   {
     "images": ["base64_string_here"]
   }
   ```

2. **Image URL**:
   ```json
   {
     "images": [
       {
         "data": "https://example.com/image.jpg",
         "format": "url"
       }
     ]
   }
   ```

### Segmentation Output Formats

Sam3 supports three output formats:

- `"dense"` - Returns masks as tensors (serialized to lists)
- `"rle"` - Returns RLE-encoded masks (CVAT/COCO format)
- `"polygons"` - Returns polygon vertices

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Not found (job/model not found)
- `500` - Server error (inference failed)

Example error response:
```json
{
  "detail": "Model sam3 not loaded. Call load_model() first."
}
```

## Performance Tips

1. **Preload models**: Load models at startup to avoid first-request latency
2. **Batch processing**: Process multiple images in one request when possible
3. **Use async jobs**: For long-running tasks, use the async job API
4. **Memory management**: Unload unused models to free GPU memory
5. **Model variants**: Use smaller model variants (e.g., "tiny") for faster inference

## Configuration

### Auto-loading Models

By default, models are auto-loaded on first inference request. To disable this:

Modify [src/vision_model_repo/api/server.py](src/vision_model_repo/api/server.py) and change `auto_load=True` to `auto_load=False` in inference endpoints.

### Concurrent Jobs

Adjust the number of concurrent async jobs:

```python
# In jobs.py
_job_queue = JobQueue(max_concurrent_jobs=4)  # Default is 2
```

### CORS Configuration

CORS is enabled for all origins by default. To restrict:

```python
# In server.py, modify CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    ...
)
```

## API Documentation

Full interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Troubleshooting

### Model Loading Errors

If models fail to load, ensure:
1. HuggingFace cache is configured (see CLAUDE.md)
2. CUDA is available if using GPU
3. Sufficient GPU/CPU memory

### Out of Memory

If you encounter OOM errors:
1. Reduce batch size
2. Use smaller model variants
3. Unload unused models
4. Use CPU inference for some models

### Slow Inference

To improve speed:
1. Use GPU if available
2. Preload models at startup
3. Use smaller model variants
4. Enable batch processing
5. Consider async processing for large workloads
