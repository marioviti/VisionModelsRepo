# HuggingFace Configuration in API Server - Summary

## Question: Where is `hf_token` and `root_folder` set?

### Answer:

The HuggingFace configuration (`hf_token` and `root_folder`) is set via **environment variables** and applied during **server startup**.

## Configuration Flow

```
Environment Variables → Server Startup → set_hf_content_environment() → HF Cache Directories Created
```

### 1. Environment Variables (Input)

Set before starting the server:

```bash
# Cache directory
export HF_ROOT_FOLDER=/path/to/cache

# Optional: HuggingFace token for private models
export HF_TOKEN=hf_your_token_here
```

**Defaults:**
- `HF_ROOT_FOLDER`: `.` (current directory)
- `HF_TOKEN`: `None` (no token)

### 2. Server Reads Configuration

**File**: [src/vision_model_repo/api/server.py](src/vision_model_repo/api/server.py:33-34)

```python
# Configuration: Can be overridden via environment variables
HF_ROOT_FOLDER = os.getenv("HF_ROOT_FOLDER", ".")
HF_TOKEN = os.getenv("HF_TOKEN", None)
```

### 3. Applied During Startup

**File**: [src/vision_model_repo/api/server.py](src/vision_model_repo/api/server.py:43-46) (in `lifespan` function)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Vision Models API Server")

    # Configure HuggingFace environment
    logger.info(f"Configuring HuggingFace environment: root={HF_ROOT_FOLDER}, token={'***' if HF_TOKEN else 'None'}")
    set_hf_content_environment(root_folder=HF_ROOT_FOLDER, hf_token=HF_TOKEN, reload_libs=True)

    # ... rest of startup
```

### 4. What `set_hf_content_environment()` Does

**File**: [src/vision_model_repo/vision_models/hf_hub.py](src/vision_model_repo/vision_models/hf_hub.py:17-46)

```python
def set_hf_content_environment(root_folder=".", hf_token=None, reload_libs=True):
    # 1. Set environment variables
    if hf_token is not None:
        os.environ["HF_TOKEN"] = hf_token

    os.environ["HF_HOME"] = f"{root_folder}/content/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = f"{root_folder}/content/hf_cache/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{root_folder}/content/hf_cache/datasets"
    os.environ["HF_HUB_CACHE"] = f"{root_folder}/content/hf_cache/hub"

    # 2. Create directories
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    # 3. Reload HuggingFace libraries (if already imported)
    # ... reload logic
```

**Result**: Creates cache directory structure:
```
{HF_ROOT_FOLDER}/content/hf_cache/
├── transformers/     # Model weights
├── datasets/         # Dataset cache
└── hub/              # HuggingFace Hub cache
```

## Usage Examples

### Example 1: Using Default Cache (Current Directory)

```bash
# No configuration needed
python run_api_server.py
```

Cache will be created at: `./content/hf_cache/`

### Example 2: Custom Cache Directory

```bash
# Set cache directory
export HF_ROOT_FOLDER=/data/models

# Start server
python run_api_server.py
```

Cache will be created at: `/data/models/content/hf_cache/`

### Example 3: Using .env File

1. Copy example:
```bash
cp .env.example .env
```

2. Edit `.env`:
```bash
HF_ROOT_FOLDER=/data/persistent/cache
HF_TOKEN=hf_your_token_here
```

3. Load and start (requires python-dotenv):
```python
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from vision_model_repo.api.server import app
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Example 4: Docker

```bash
docker run -p 8000:8000 \
  -e HF_ROOT_FOLDER=/cache \
  -e HF_TOKEN=hf_your_token \
  -v /host/cache:/cache \
  --gpus all \
  vision-models-api
```

## When Is It Set?

**Timing**: During FastAPI application startup, before any models are loaded.

**Lifecycle**:
1. FastAPI app created
2. `lifespan` context manager runs
3. `set_hf_content_environment()` called ← **HERE**
4. Job queue workers started
5. Server ready to accept requests
6. Models loaded on first inference (or via `/models/load`)

## Verification

Check that configuration is applied:

```bash
# Start server with logging
python run_api_server.py

# Look for this log line:
# INFO: Configuring HuggingFace environment: root=/your/path, token=***

# Check the /health endpoint
curl http://localhost:8000/health
```

Or inspect environment variables after startup:

```python
import os
print(os.environ.get("HF_HOME"))
print(os.environ.get("TRANSFORMERS_CACHE"))
```

## Summary Table

| Configuration | How to Set | Where Read | When Applied |
|---------------|------------|------------|--------------|
| `HF_ROOT_FOLDER` | `export HF_ROOT_FOLDER=/path` | [server.py:33](src/vision_model_repo/api/server.py:33) | Server startup (lifespan) |
| `HF_TOKEN` | `export HF_TOKEN=hf_xxx` | [server.py:34](src/vision_model_repo/api/server.py:34) | Server startup (lifespan) |
| Cache directories | Auto-created | [hf_hub.py:22-25](src/vision_model_repo/vision_models/hf_hub.py:22-25) | During `set_hf_content_environment()` |

## Files to Review

1. **[CONFIG.md](CONFIG.md)** - Complete configuration guide
2. **[.env.example](.env.example)** - Example environment file
3. **[src/vision_model_repo/api/server.py](src/vision_model_repo/api/server.py)** - Where config is read and applied
4. **[src/vision_model_repo/vision_models/hf_hub.py](src/vision_model_repo/vision_models/hf_hub.py)** - HF environment setup logic

## Quick Reference

```bash
# Minimal setup (uses defaults)
python run_api_server.py

# Production setup
export HF_ROOT_FOLDER=/data/persistent/cache
export HF_TOKEN=hf_your_private_token
python run_api_server.py

# Verify cache location
ls -la /data/persistent/cache/content/hf_cache/
```
