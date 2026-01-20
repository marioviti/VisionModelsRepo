# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VisionModelsRepo is a unified wrapper library for foundational computer vision models from Hugging Face and Meta. It provides standardized interfaces for depth estimation, segmentation, object detection, and feature extraction models, enabling modular composition in vision pipelines.

## Installation

Install the package in development mode:

```bash
pip install -e .
```

For the API server, install additional dependencies:

```bash
pip install -r requirements.api.txt
```

## Key Architecture

### Package Structure

```
src/vision_model_repo/
├── vision_models/           # Model wrappers organized by task
│   ├── hf_hub.py           # HuggingFace integration utilities
│   ├── depth_estimation/   # DepthAnythingV3
│   ├── segmentation/       # Sam3 + mask format utilities
│   ├── object_detection/   # GroundingDino + bbox format utilities
│   └── feature_extraction/ # DinoV3
├── core/                   # (Reserved for future core logic)
├── io/                     # (Reserved for I/O patterns)
└── pipeline/               # (Reserved for pipeline orchestration)

sam-3d-objects/             # Git submodule for 3D reconstruction
notebook/                   # Usage examples and inference utilities
tests/                      # Test modules
```

### Model Wrapper Design Pattern

All model wrappers follow a consistent interface:

```python
class ModelWrapper:
    collection = ["model-id-1", "model-id-2", ...]  # Available variants

    def __init__(self, model_id: str = collection[0], device=None):
        # Auto-detect device (CUDA if available, else CPU)
        # Load model and processor from HuggingFace

    @torch.inference_mode()
    def __call__(self, images: list, prompts=None, **options):
        # Batch-aware inference
        # Returns structured outputs
```

**Key principles:**
- **Unified calling convention**: `model(images, prompts, **options)`
- **Batch processing**: All models accept `list` of PIL Images
- **Device awareness**: Automatic GPU/CPU selection with manual override
- **Flexible prompts**: Text prompts can be shared across batch or per-image
- **Structured outputs**: Each model returns task-specific predictions with metadata

### Available Models

#### Segmentation: Sam3
- **Path**: [src/vision_model_repo/vision_models/segmentation/sam3.py](src/vision_model_repo/vision_models/segmentation/sam3.py)
- **Model**: Facebook SAM3 (Segment Anything Model 3)
- **Input**: List of PIL Images + text prompts
- **Output**: Dense masks, IoU scores, bounding boxes
- **Features**: Overlap resolution, multiple output formats (dense, RLE, polygons)

#### Depth Estimation: DepthAnythingV3
- **Path**: [src/vision_model_repo/vision_models/depth_estimation/depth_anything3.py](src/vision_model_repo/vision_models/depth_estimation/depth_anything3.py)
- **Model**: Depth Anything V3 (small, base, large, giant variants)
- **Input**: List of PIL Images
- **Output**: Depth maps, confidence maps, camera intrinsics/extrinsics
- **Export Formats**: GLB, NPZ, PLY, GS_PLY, GS_VIDEO

#### Object Detection: GroundingDinoInstanceDetection
- **Path**: [src/vision_model_repo/vision_models/object_detection/grounding_dino.py](src/vision_model_repo/vision_models/object_detection/grounding_dino.py)
- **Model**: IDEA-Research Grounding DINO
- **Input**: List of PIL Images + text phrases
- **Output**: Bounding boxes, confidence scores, text labels
- **Features**: Zero-shot text-prompted detection, configurable thresholds

#### Feature Extraction: DinoV3
- **Path**: [src/vision_model_repo/vision_models/feature_extraction/dino3.py](src/vision_model_repo/vision_models/feature_extraction/dino3.py)
- **Model**: Facebook DINOv3 (ConvNeXt and ViT variants)
- **Input**: List of PIL Images
- **Output**: Last hidden state and pooled features

### SAM 3D Objects Submodule

**Location**: [sam-3d-objects/](sam-3d-objects/) (git submodule)

**Purpose**: Single-image 3D object reconstruction - converts 2D masked objects to full 3D geometry, texture, and pose.

**Inference wrapper**: [notebook/sam_3d_inference.py](notebook/sam_3d_inference.py) provides:
- `Inference` class: Public API for SAM 3D inference
- `MemoryEfficientInferencePipeline`: Optimized pipeline with CPU offloading between stages
- Multi-stage pipeline: Depth → Sparse structure → Sparse latent → Decoding
- Output formats: Gaussian Splat, Mesh, Layout

**Setup**: Follow [sam-3d-objects/doc/setup.md](sam-3d-objects/doc/setup.md) for SAM 3D specific dependencies.

### Utility Modules

#### Segmentation Utils
**Path**: [src/vision_model_repo/vision_models/segmentation/utils.py](src/vision_model_repo/vision_models/segmentation/utils.py)

Mask format conversions:
- **RLE Encoding**: CVAT/COCO-style compressed RLE (zlib + base64)
- **Polygon Conversion**: Dense mask ↔ polygon vertices (OpenCV-based)
- **Bridge Functions**: RLE ↔ polygon conversions

#### Object Detection Utils
**Path**: [src/vision_model_repo/vision_models/object_detection/utils.py](src/vision_model_repo/vision_models/object_detection/utils.py)

Bounding box format conversions:
- **Formats**: xyxy, xywh, cxcywh, yolo (normalized)
- **Function**: `bbox_convert()` with clipping and validation

### HuggingFace Hub Integration

**Path**: [src/vision_model_repo/vision_models/hf_hub.py](src/vision_model_repo/vision_models/hf_hub.py)

Before using models, configure the HuggingFace cache environment:

```python
from vision_model_repo.vision_models.hf_hub import set_hf_content_environment

set_hf_content_environment(root_folder=".", hf_token=None, reload_libs=True)
```

This sets up cache directories (`HF_HOME`, `TRANSFORMERS_CACHE`, etc.) and optionally reloads HF-related libraries.

## REST API Server

A FastAPI-based REST API is available for serving vision models over HTTP.

### API Structure

```
src/vision_model_repo/api/
├── server.py          # Main FastAPI application
├── models.py          # Model management (load/unload)
├── schemas.py         # Pydantic request/response models
├── utils.py           # Image decoding utilities
└── jobs.py            # Async job processing queue
```

### Starting the API Server

```bash
# Development mode with auto-reload
uvicorn vision_model_repo.api.server:app --host 0.0.0.0 --port 8000 --reload

# Production mode with multiple workers
uvicorn vision_model_repo.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

API documentation is auto-generated at `http://localhost:8000/docs`.

### API Features

- **Model Management**: Dynamically load/unload models to manage memory
- **Batch Processing**: Process multiple images in a single request
- **Async Jobs**: Queue long-running tasks and poll for results
- **Health Endpoints**: Check server status and loaded models
- **Multiple Input Formats**: Accept base64-encoded images or URLs
- **Flexible Output Formats**: Support dense masks, RLE, and polygons for segmentation

### Key Endpoints

- `POST /inference/sam3` - Instance segmentation
- `POST /inference/grounding-dino` - Object detection
- `POST /inference/depth-anything` - Depth estimation
- `POST /inference/dinov3` - Feature extraction
- `POST /models/load` - Load a model
- `POST /models/unload` - Unload a model
- `GET /health` - Health check
- `GET /models/status` - Get model status

See [API_USAGE.md](API_USAGE.md) for detailed usage examples and [examples/api_client_example.py](examples/api_client_example.py) for a Python client.

## Common Development Commands

### Running Tests
```bash
pytest tests/
```

### Using the Models Directly

See [notebook/load_vision_models.ipynb](notebook/load_vision_models.ipynb) and [notebook/load_sam3d.ipynb](notebook/load_sam3d.ipynb) for usage examples.

Basic pattern:
```python
from vision_model_repo.vision_models.hf_hub import set_hf_content_environment
from vision_model_repo.vision_models.segmentation.sam3 import Sam3
from PIL import Image

# Configure HuggingFace environment
set_hf_content_environment()

# Load model
model = Sam3(device="cuda")  # or "cpu"

# Run inference
images = [Image.open("image.jpg")]
prompts = ["object to segment"]
results = model(images, prompts)
```

### Git Submodule Management

The SAM 3D Objects submodule must be initialized:

```bash
git submodule update --init --recursive
```

To update the submodule to latest:
```bash
cd sam-3d-objects
git pull origin main
cd ..
git add sam-3d-objects
git commit -m "[UPDATE] sam-3d-objects submodule"
```

## Code Conventions

### Commit Message Format
This repository uses structured commit messages with the format:
```
[TAG] Brief description ALL HAIL THE OMNESSIAH
```

Common tags: `[ADD]`, `[FIX]`, `[UPDATE]`, `[REMOVE]`

Example: `[ADD] overlap resolve in sam3 ALL HAIL THE OMNESSIAH`

### Model Wrapper Implementation

When adding new model wrappers:

1. **Choose the appropriate task directory** under `src/vision_model_repo/vision_models/`
2. **Define a `collection` class attribute** listing available model variants
3. **Implement `__init__`** with:
   - `model_id` parameter (default to first in collection)
   - `device` parameter with auto-detection fallback
   - Load model and processor from HuggingFace
4. **Implement `__call__`** with:
   - `@torch.inference_mode()` decorator
   - Accept `list` of PIL Images as first argument
   - Accept prompts (if applicable) as second argument
   - Return structured outputs (dicts/lists with metadata)
5. **Handle batch processing**: Support both shared and per-image prompts
6. **Add utility functions** if format conversions are needed (see segmentation/utils.py)

### ModelIO Pattern (Implicit)

While `io/` and `pipeline/` directories are currently empty, the codebase implements an implicit ModelIO pattern through consistent interfaces. This minimizes glue code when chaining models:

```python
# Example multi-model pipeline
features = dino3([image])
detections = grounding_dino([image], ["object"])
masks = sam3([image], ["object"])
depth = depth_anything3([image])
```

Future development may formalize this pattern in the `io/` and `pipeline/` modules.

## Important Notes

- **Device management**: All models auto-detect CUDA availability but accept manual `device` parameter
- **Batch semantics**: Models expect `list` of images even for single image inference
- **Memory efficiency**: SAM 3D implements CPU offloading for memory-constrained environments
- **Format flexibility**: Segmentation outputs support multiple formats (dense, RLE, polygon) via `output_format` parameter
- **Threshold tuning**: Detection and segmentation models expose confidence thresholds as kwargs
