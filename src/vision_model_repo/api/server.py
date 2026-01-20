"""
FastAPI server for vision model inference.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import ModelManager, ModelType
from .schemas import (
    Sam3Request, Sam3Response,
    GroundingDinoRequest, GroundingDinoResponse,
    DepthAnythingRequest, DepthAnythingResponse,
    DinoV3Request, DinoV3Response,
    ModelLoadRequest, ModelUnloadRequest,
    HealthResponse, ModelsStatusResponse, ModelInfo,
    JobResponse
)
from .utils import decode_images, serialize_tensor, get_gpu_memory_info, encode_image_to_base64
from .jobs import get_job_queue
from ..vision_models.hf_hub import set_hf_content_environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration: Can be overridden via environment variables
HF_ROOT_FOLDER = os.getenv("HF_ROOT_FOLDER", ".")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Global model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting Vision Models API Server")

    # Configure HuggingFace environment
    logger.info(f"Configuring HuggingFace environment: root={HF_ROOT_FOLDER}, token={'***' if HF_TOKEN else 'None'}")
    set_hf_content_environment(root_folder=HF_ROOT_FOLDER, hf_token=HF_TOKEN, reload_libs=True)

    # Start job queue workers
    job_queue = get_job_queue()
    await job_queue.start_workers()

    logger.info("Server startup complete")

    yield

    # Shutdown
    logger.info("Shutting down server")

    # Stop job queue workers
    await job_queue.stop_workers()

    # Unload all models
    model_manager.unload_all()

    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Vision Models API",
    description="REST API for foundational vision model inference",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health and Status Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Vision Models API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check server health and status."""
    gpu_info = get_gpu_memory_info()

    return HealthResponse(
        status="healthy",
        loaded_models=[mt for mt in ModelType if model_manager.is_loaded(mt)],
        available_memory_gb=gpu_info.get("free_gb"),
        gpu_available=gpu_info.get("available", False)
    )


@app.get("/models/status", response_model=ModelsStatusResponse, tags=["Model Management"])
async def get_models_status():
    """Get status of all models."""
    models_info = {}
    total_memory = 0.0

    for model_type in ModelType:
        info = model_manager.get_model_info(model_type)
        models_info[model_type.value] = ModelInfo(**info)

        if info.get("memory_mb"):
            total_memory += info["memory_mb"]

    return ModelsStatusResponse(
        models=models_info,
        total_memory_mb=total_memory if total_memory > 0 else None
    )


# Model Management Endpoints

@app.post("/models/load", tags=["Model Management"])
async def load_model(request: ModelLoadRequest):
    """Load a model into memory."""
    try:
        model_manager.load_model(
            model_type=request.model_type,
            model_id=request.model_id,
            device=request.device
        )

        return {
            "message": f"Model {request.model_type} loaded successfully",
            "model_info": model_manager.get_model_info(request.model_type)
        }
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload", tags=["Model Management"])
async def unload_model(request: ModelUnloadRequest):
    """Unload a model from memory."""
    try:
        model_manager.unload_model(request.model_type)

        return {
            "message": f"Model {request.model_type} unloaded successfully"
        }
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload-all", tags=["Model Management"])
async def unload_all_models():
    """Unload all models from memory."""
    try:
        model_manager.unload_all()

        return {
            "message": "All models unloaded successfully"
        }
    except Exception as e:
        logger.error(f"Failed to unload models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Inference Endpoints - Sam3 (Segmentation)

@app.post("/inference/sam3", response_model=Sam3Response, tags=["Inference"])
async def sam3_inference(request: Sam3Request):
    """
    Perform instance segmentation using Sam3.

    Supports multiple output formats:
    - dense: Binary masks as nested lists (not recommended for API - use RLE or polygons instead)
    - rle: CVAT/COCO-style RLE encoded masks (recommended)
    - polygons: Polygon vertices (recommended)
    """
    start_time = time.time()

    try:
        # Get or load model
        model = model_manager.get_model(ModelType.SAM3, auto_load=True)

        # Decode images
        images = decode_images(request.images)

        # Run inference with specified output format
        results = model(
            images=images,
            texts=request.prompts,
            threshold=request.threshold,
            mask_threshold=request.mask_threshold,
            resolve_overlaps=request.resolve_overlaps,
            output_format=request.output_format.value
        )

        # Serialize results - deep copy and convert any remaining tensors
        serialized_results = []
        for image_results in results:
            image_serialized = []
            for entry in image_results:
                # Deep copy to avoid modifying original
                result_copy = {
                    "prompt_index": entry.get("prompt_index"),
                    "prompt": entry.get("prompt"),
                    "result": {}
                }

                if "result" in entry:
                    result_data = entry["result"]

                    # Copy and serialize each field
                    for key, value in result_data.items():
                        if hasattr(value, "tolist"):
                            # Convert tensors to lists
                            result_copy["result"][key] = serialize_tensor(value)
                        elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                            # JSON-serializable types
                            result_copy["result"][key] = value
                        else:
                            # Try to convert unknown types
                            try:
                                result_copy["result"][key] = str(value)
                            except:
                                logger.warning(f"Could not serialize field {key} of type {type(value)}")
                                result_copy["result"][key] = None

                image_serialized.append(result_copy)
            serialized_results.append(image_serialized)

        processing_time = time.time() - start_time

        return Sam3Response(
            results=serialized_results,
            model_id=model_manager.get_model_info(ModelType.SAM3)["model_id"],
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Sam3 inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Inference Endpoints - GroundingDino (Object Detection)

@app.post("/inference/grounding-dino", response_model=GroundingDinoResponse, tags=["Inference"])
async def grounding_dino_inference(request: GroundingDinoRequest):
    """
    Perform zero-shot object detection using GroundingDino.
    """
    start_time = time.time()

    try:
        # Get or load model
        model = model_manager.get_model(ModelType.GROUNDING_DINO, auto_load=True)

        # Decode images
        images = decode_images(request.images)

        # Run inference
        results = model(
            images=images,
            texts=request.prompts,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold
        )

        # Serialize results
        serialized_results = []
        for result in results:
            result_copy = {}
            for key, value in result.items():
                if hasattr(value, "tolist"):
                    result_copy[key] = serialize_tensor(value)
                else:
                    result_copy[key] = value
            serialized_results.append(result_copy)

        processing_time = time.time() - start_time

        return GroundingDinoResponse(
            results=serialized_results,
            model_id=model_manager.get_model_info(ModelType.GROUNDING_DINO)["model_id"],
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"GroundingDino inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Inference Endpoints - DepthAnythingV3 (Depth Estimation)

@app.post("/inference/depth-anything", response_model=DepthAnythingResponse, tags=["Inference"])
async def depth_anything_inference(request: DepthAnythingRequest):
    """
    Perform depth estimation using DepthAnythingV3.
    """
    start_time = time.time()

    try:
        # Get or load model
        model = model_manager.get_model(ModelType.DEPTH_ANYTHING, auto_load=True)

        # Decode images
        images = decode_images(request.images)

        # Run inference
        prediction = model(
            images=images,
            export_format=request.export_format.value
        )

        # Extract depth data
        # Note: The prediction object structure depends on export_format
        # For now, we'll serialize the depth map as base64
        depth_map = prediction.depth  # [N, H, W]

        # Convert first depth map to image and encode
        from PIL import Image
        import numpy as np

        depth_normalized = (depth_map[0].cpu().numpy() * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_normalized)
        depth_base64 = encode_image_to_base64(depth_image)

        processing_time = time.time() - start_time

        metadata = {
            "shape": list(depth_map.shape),
        }

        # Add camera parameters if available
        if hasattr(prediction, "intrinsics"):
            metadata["intrinsics"] = serialize_tensor(prediction.intrinsics)
        if hasattr(prediction, "extrinsics"):
            metadata["extrinsics"] = serialize_tensor(prediction.extrinsics)

        return DepthAnythingResponse(
            depth_data=depth_base64,
            format="png_base64",
            model_id=model_manager.get_model_info(ModelType.DEPTH_ANYTHING)["model_id"],
            processing_time=processing_time,
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"DepthAnything inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Inference Endpoints - DinoV3 (Feature Extraction)

@app.post("/inference/dinov3", response_model=DinoV3Response, tags=["Inference"])
async def dinov3_inference(request: DinoV3Request):
    """
    Extract features using DinoV3.
    """
    start_time = time.time()

    try:
        # Get or load model
        model = model_manager.get_model(ModelType.DINOV3, auto_load=True)

        # Decode images
        images = decode_images(request.images)

        # Run inference
        outputs = model(images=images)

        # Serialize outputs
        features = {}
        if hasattr(outputs, "last_hidden_state"):
            features["last_hidden_state"] = serialize_tensor(outputs.last_hidden_state)
        if hasattr(outputs, "pooler_output"):
            features["pooler_output"] = serialize_tensor(outputs.pooler_output)

        # Add shape information
        if "last_hidden_state" in features:
            features["last_hidden_state_shape"] = list(outputs.last_hidden_state.shape)
        if "pooler_output" in features:
            features["pooler_output_shape"] = list(outputs.pooler_output.shape)

        processing_time = time.time() - start_time

        return DinoV3Response(
            features=features,
            model_id=model_manager.get_model_info(ModelType.DINOV3)["model_id"],
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"DinoV3 inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Async Job Endpoints

@app.post("/jobs/submit/{model_type}", response_model=JobResponse, tags=["Async Jobs"])
async def submit_async_job(model_type: str, request: dict):
    """
    Submit an async inference job.
    Useful for long-running tasks.
    """
    try:
        model_type_enum = ModelType(model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")

    # Map model type to inference function
    inference_map = {
        ModelType.SAM3: _sam3_inference_task,
        ModelType.GROUNDING_DINO: _grounding_dino_inference_task,
        ModelType.DEPTH_ANYTHING: _depth_anything_inference_task,
        ModelType.DINOV3: _dinov3_inference_task,
    }

    if model_type_enum not in inference_map:
        raise HTTPException(status_code=400, detail=f"Model type not supported: {model_type}")

    # Submit job
    job_queue = get_job_queue()
    job_id = job_queue.submit_job(
        inference_map[model_type_enum],
        model_type_enum,
        request
    )

    job_status = job_queue.get_job_status(job_id)

    return JobResponse(**job_status)


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Async Jobs"])
async def get_job_status(job_id: str):
    """Get status of an async job."""
    job_queue = get_job_queue()
    job_status = job_queue.get_job_status(job_id)

    if not job_status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobResponse(**job_status)


@app.delete("/jobs/{job_id}", tags=["Async Jobs"])
async def delete_job(job_id: str):
    """Delete an async job."""
    job_queue = get_job_queue()
    success = job_queue.delete_job(job_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {"message": f"Job {job_id} deleted"}


@app.get("/jobs", tags=["Async Jobs"])
async def list_jobs():
    """List all jobs."""
    job_queue = get_job_queue()
    return job_queue.get_all_jobs()


# Helper functions for async jobs

def _sam3_inference_task(model_type: ModelType, request_data: dict):
    """Helper for async Sam3 inference."""
    model = model_manager.get_model(model_type, auto_load=True)
    images = decode_images(request_data["images"])

    results = model(
        images=images,
        texts=request_data["prompts"],
        threshold=request_data.get("threshold", 0.5),
        mask_threshold=request_data.get("mask_threshold", 0.5),
        resolve_overlaps=request_data.get("resolve_overlaps", False),
        output_format=request_data.get("output_format", "dense")
    )

    return {"results": results}


def _grounding_dino_inference_task(model_type: ModelType, request_data: dict):
    """Helper for async GroundingDino inference."""
    model = model_manager.get_model(model_type, auto_load=True)
    images = decode_images(request_data["images"])

    results = model(
        images=images,
        texts=request_data["prompts"],
        box_threshold=request_data.get("box_threshold", 0.4),
        text_threshold=request_data.get("text_threshold", 0.3)
    )

    return {"results": results}


def _depth_anything_inference_task(model_type: ModelType, request_data: dict):
    """Helper for async DepthAnything inference."""
    model = model_manager.get_model(model_type, auto_load=True)
    images = decode_images(request_data["images"])

    prediction = model(
        images=images,
        export_format=request_data.get("export_format", "glb")
    )

    return {"prediction": prediction}


def _dinov3_inference_task(model_type: ModelType, request_data: dict):
    """Helper for async DinoV3 inference."""
    model = model_manager.get_model(model_type, auto_load=True)
    images = decode_images(request_data["images"])

    outputs = model(images=images)

    return {"outputs": outputs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
