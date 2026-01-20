# Vision Models API

FastAPI-based REST API for serving foundational vision models.

## Features

- **Multiple Vision Models**: Sam3 (segmentation), GroundingDino (detection), DepthAnythingV3 (depth), DinoV3 (features)
- **Dynamic Model Management**: Load/unload models on-demand to manage memory
- **Batch Processing**: Process multiple images in a single request
- **Async Job Queue**: Queue long-running tasks and poll for results
- **Flexible Input**: Accept base64-encoded images or URLs
- **Multiple Output Formats**: Dense masks, RLE, polygons for segmentation
- **Auto-generated Documentation**: Interactive Swagger UI and ReDoc

## Quick Start

1. Install dependencies:
```bash
pip install -e .
pip install -r requirements.api.txt
```

2. Start the server:
```bash
python run_api_server.py --reload
```

3. Open the interactive docs at `http://localhost:8000/docs`

## Architecture

```
api/
├── server.py      # Main FastAPI app with all endpoints
├── models.py      # ModelManager for loading/unloading models
├── schemas.py     # Pydantic models for request/response validation
├── utils.py       # Image decoding and utility functions
└── jobs.py        # Async job queue implementation
```

## Key Components

### ModelManager (`models.py`)

Handles loading, unloading, and access to vision models:

```python
manager = ModelManager()
manager.load_model(ModelType.SAM3, device="cuda")
model = manager.get_model(ModelType.SAM3)
manager.unload_model(ModelType.SAM3)
```

Features:
- Lazy loading (models loaded on first use)
- Memory management (unload to free GPU/CPU memory)
- Auto-detection of CUDA availability
- Model info and memory usage tracking

### JobQueue (`jobs.py`)

Async job processing for long-running tasks:

```python
queue = JobQueue(max_concurrent_jobs=2)
await queue.start_workers()

job_id = queue.submit_job(my_function, *args, **kwargs)
status = queue.get_job_status(job_id)
```

Features:
- Background worker pool
- Job status tracking (pending, running, completed, failed)
- Result caching
- Automatic cleanup of old jobs

### Image Utilities (`utils.py`)

Helper functions for image processing:

```python
# Decode from base64 or URL
image = decode_image(base64_string)
images = decode_images([url1, base64_string, ImageInput(...)])

# Encode to base64
base64_str = encode_image_to_base64(pil_image)

# Serialize tensors for JSON
tensor_list = serialize_tensor(torch_tensor)
```

## API Endpoints

### Health & Status

- `GET /` - Root endpoint with API info
- `GET /health` - Health check with GPU/memory info
- `GET /models/status` - Status of all models

### Model Management

- `POST /models/load` - Load a model into memory
- `POST /models/unload` - Unload a model from memory
- `POST /models/unload-all` - Unload all models

### Synchronous Inference

- `POST /inference/sam3` - Instance segmentation
- `POST /inference/grounding-dino` - Object detection
- `POST /inference/depth-anything` - Depth estimation
- `POST /inference/dinov3` - Feature extraction

### Async Jobs

- `POST /jobs/submit/{model_type}` - Submit async inference job
- `GET /jobs/{job_id}` - Get job status and result
- `DELETE /jobs/{job_id}` - Delete a job
- `GET /jobs` - List all jobs

## Usage Examples

### Python Client

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Segmentation
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [image_b64],
        "prompts": ["person", "car"],
        "threshold": 0.5,
        "output_format": "dense"
    }
)
result = response.json()
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Load model
curl -X POST http://localhost:8000/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_type": "sam3", "device": "cuda"}'
```

See [API_USAGE.md](../../../API_USAGE.md) for comprehensive examples.

## Configuration

### Environment Variables

- `HF_HOME` - HuggingFace cache directory (set via `hf_hub.py`)
- `CUDA_VISIBLE_DEVICES` - Control GPU visibility

### Server Settings

Modify in `server.py`:

```python
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    ...
)

# Job queue
job_queue = JobQueue(max_concurrent_jobs=2)  # Adjust concurrency
```

## Performance Tips

1. **Preload Models**: Load frequently-used models at startup to avoid first-request latency
2. **Batch Processing**: Send multiple images in one request when possible
3. **Use Async Jobs**: For long-running tasks, use the async API to avoid timeouts
4. **Memory Management**: Unload unused models to free GPU memory
5. **Model Variants**: Use smaller variants (e.g., "tiny") for faster inference

## Error Handling

All endpoints return standard HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Not found (model/job not found)
- `500` - Internal error (inference failed)

Example error response:
```json
{
  "detail": "Model sam3 not loaded. Call load_model() first."
}
```

## Deployment

### Development

```bash
python run_api_server.py --reload
```

### Production

```bash
# Using uvicorn with multiple workers
uvicorn vision_model_repo.api.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4

# Or using gunicorn
gunicorn vision_model_repo.api.server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install -e . && pip install -r requirements.api.txt

EXPOSE 8000

CMD ["uvicorn", "vision_model_repo.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t vision-models-api .
docker run -p 8000:8000 --gpus all vision-models-api
```

## Security Considerations

For production deployment:

1. **Restrict CORS**: Update `allow_origins` to specific domains
2. **Add Authentication**: Implement API key or OAuth2 authentication
3. **Rate Limiting**: Add rate limiting middleware
4. **Input Validation**: Validate image sizes and formats
5. **HTTPS**: Use a reverse proxy (nginx) with SSL/TLS

## Troubleshooting

### Models Not Loading

- Check HuggingFace cache is configured
- Verify CUDA is available: `torch.cuda.is_available()`
- Check GPU memory: Look at `GET /health` response

### Out of Memory

- Reduce batch size
- Use smaller model variants
- Unload unused models
- Use CPU inference for some models

### Slow Performance

- Preload models at startup
- Use GPU if available
- Enable batch processing
- Use async jobs for heavy workloads

## Development

To extend the API:

1. **Add new model**: Update `ModelType` enum and `ModelManager`
2. **Add endpoint**: Create request/response schemas and endpoint handler
3. **Add async support**: Implement task function for job queue
4. **Update docs**: Add examples to API_USAGE.md

## Testing

```bash
# Run tests (when implemented)
pytest tests/api/

# Manual testing
python examples/api_client_example.py
```

## License

See repository LICENSE file.
