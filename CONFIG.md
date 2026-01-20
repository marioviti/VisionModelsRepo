# Vision Models API Server Configuration

## Environment Variables

The API server can be configured using environment variables.

### HuggingFace Configuration

#### `HF_ROOT_FOLDER`
- **Default**: `.` (current directory)
- **Description**: Root folder for HuggingFace cache directories
- **Usage**:
  ```bash
  export HF_ROOT_FOLDER=/path/to/cache
  python run_api_server.py
  ```

This will create the following cache structure:
```
{HF_ROOT_FOLDER}/content/hf_cache/
├── transformers/     # Model weights and configs
├── datasets/         # Dataset cache
└── hub/             # HuggingFace Hub cache
```

#### `HF_TOKEN`
- **Default**: `None`
- **Description**: HuggingFace API token for accessing private models
- **Required**: Only if using private/gated models
- **Get token**: https://huggingface.co/settings/tokens
- **Usage**:
  ```bash
  export HF_TOKEN=hf_your_token_here
  python run_api_server.py
  ```

### Server Configuration

#### `HOST`
- **Default**: `0.0.0.0` (all interfaces)
- **Description**: Network interface to bind to
- **Options**:
  - `0.0.0.0` - Accept connections from any network interface
  - `127.0.0.1` - Only accept local connections
  - Specific IP address

#### `PORT`
- **Default**: `8000`
- **Description**: Port number to listen on
- **Usage**:
  ```bash
  export PORT=9000
  python run_api_server.py
  ```

### CUDA/GPU Configuration

#### `CUDA_VISIBLE_DEVICES`
- **Default**: All available GPUs
- **Description**: Control which GPU(s) to use
- **Options**:
  - `0` - Use first GPU only
  - `0,1` - Use first and second GPU
  - `-1` - Disable GPU, use CPU only
- **Usage**:
  ```bash
  # Use only GPU 0
  export CUDA_VISIBLE_DEVICES=0
  python run_api_server.py

  # Use CPU only
  export CUDA_VISIBLE_DEVICES=-1
  python run_api_server.py
  ```

## Configuration File (.env)

You can use a `.env` file for convenience:

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your values:
   ```bash
   HF_ROOT_FOLDER=/data/hf_cache
   HF_TOKEN=hf_your_token_here
   PORT=8000
   CUDA_VISIBLE_DEVICES=0
   ```

3. Load environment variables (using python-dotenv):
   ```bash
   pip install python-dotenv
   ```

   Then in your startup script:
   ```python
   from dotenv import load_dotenv
   load_dotenv()

   # Now run server
   import uvicorn
   from vision_model_repo.api.server import app
   uvicorn.run(app)
   ```

## Configuration in Code

You can also configure programmatically before starting the server:

```python
import os

# Set HuggingFace config
os.environ["HF_ROOT_FOLDER"] = "/data/cache"
os.environ["HF_TOKEN"] = "hf_your_token"

# Start server
import uvicorn
from vision_model_repo.api.server import app

uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Where Settings Are Applied

### HuggingFace Environment Setup

The `set_hf_content_environment()` function is called during server startup in the `lifespan` event:

**File**: [src/vision_model_repo/api/server.py](src/vision_model_repo/api/server.py)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure HuggingFace environment on startup
    set_hf_content_environment(
        root_folder=HF_ROOT_FOLDER,
        hf_token=HF_TOKEN,
        reload_libs=True
    )
    # ... rest of startup
```

This sets the following environment variables:
- `HF_HOME`
- `TRANSFORMERS_CACHE`
- `HF_DATASETS_CACHE`
- `HF_HUB_CACHE`
- `HF_TOKEN` (if provided)

### Reading Configuration

At the top of [server.py](src/vision_model_repo/api/server.py:33-34):

```python
# Configuration: Can be overridden via environment variables
HF_ROOT_FOLDER = os.getenv("HF_ROOT_FOLDER", ".")
HF_TOKEN = os.getenv("HF_TOKEN", None)
```

## Docker Configuration

When using Docker, pass environment variables:

```bash
docker run -p 8000:8000 \
  -e HF_ROOT_FOLDER=/cache \
  -e HF_TOKEN=hf_your_token \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /host/cache:/cache \
  --gpus all \
  vision-models-api
```

Or use docker-compose.yml:

```yaml
version: '3.8'
services:
  api:
    image: vision-models-api
    ports:
      - "8000:8000"
    environment:
      - HF_ROOT_FOLDER=/cache
      - HF_TOKEN=${HF_TOKEN}
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./cache:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Production Recommendations

For production deployments:

1. **Set explicit cache directory**:
   ```bash
   export HF_ROOT_FOLDER=/data/persistent/cache
   ```
   This ensures models aren't re-downloaded on container restart.

2. **Use read-only token** (if needed):
   - Create a HuggingFace token with read-only access
   - Store securely (e.g., AWS Secrets Manager, Kubernetes secrets)

3. **Pin GPU selection**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # Consistent GPU usage
   ```

4. **Pre-download models**:
   ```python
   # Pre-download script
   from transformers import Sam3Model
   Sam3Model.from_pretrained("facebook/sam3")
   # ... other models
   ```

5. **Monitor cache size**:
   - HuggingFace cache can grow large
   - Implement cleanup/rotation policies

## Troubleshooting

### Models not loading
- Check `HF_HOME` is writable
- Verify `HF_TOKEN` if using private models
- Check disk space in cache directory

### Cache location issues
```bash
# Check where cache is being written
python -c "
import os
from vision_model_repo.vision_models.hf_hub import set_hf_content_environment
set_hf_content_environment(root_folder='/tmp/test')
print(os.environ['HF_HOME'])
"
```

### Permission errors
Ensure the user running the server has write access to `HF_ROOT_FOLDER`.

## Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_ROOT_FOLDER` | `.` | Cache directory root |
| `HF_TOKEN` | `None` | HuggingFace API token |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `CUDA_VISIBLE_DEVICES` | All | GPU selection |

For most cases, only `HF_ROOT_FOLDER` needs to be set to a persistent directory.
