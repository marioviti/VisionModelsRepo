"""
Pydantic schemas for API request/response validation.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class ModelType(str, Enum):
    """Available model types."""
    SAM3 = "sam3"
    GROUNDING_DINO = "grounding_dino"
    DEPTH_ANYTHING = "depth_anything"
    DINOV3 = "dinov3"


class ImageFormat(str, Enum):
    """Image input formats."""
    BASE64 = "base64"
    URL = "url"


class SegmentationOutputFormat(str, Enum):
    """Segmentation output formats."""
    DENSE = "dense"
    RLE = "rle"
    POLYGONS = "polygons"


class DepthExportFormat(str, Enum):
    """Depth estimation export formats."""
    GLB = "glb"
    NPZ = "npz"
    PLY = "ply"
    GS_PLY = "gs_ply"
    GS_VIDEO = "gs_video"


# Base Request/Response Models

class ImageInput(BaseModel):
    """Image input with multiple format support."""
    data: str = Field(..., description="Base64 encoded image or URL")
    format: ImageFormat = Field(ImageFormat.BASE64, description="Image format type")


class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Async job response."""
    job_id: str
    status: JobStatus
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None


# Segmentation (Sam3) Schemas

class Sam3Request(BaseModel):
    """Request for Sam3 segmentation."""
    images: List[Union[ImageInput, str]] = Field(..., description="Images (base64 or ImageInput objects)")
    prompts: Union[List[str], List[List[str]]] = Field(..., description="Text prompts (shared or per-image)")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Detection threshold")
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Mask threshold")
    resolve_overlaps: Union[bool, List[float]] = Field(False, description="Resolve overlapping masks")
    output_format: SegmentationOutputFormat = Field(
        SegmentationOutputFormat.RLE,
        description="Output format: 'rle' (recommended, compact), 'polygons' (recommended, editable), or 'dense' (large binary masks)"
    )
    model_id: Optional[str] = Field(None, description="Specific model variant to use")


class Sam3Response(BaseModel):
    """Response for Sam3 segmentation."""
    results: List[List[Dict[str, Any]]] = Field(..., description="Segmentation results per image")
    model_id: str
    processing_time: float


# Object Detection (GroundingDino) Schemas

class GroundingDinoRequest(BaseModel):
    """Request for GroundingDino object detection."""
    images: List[Union[ImageInput, str]] = Field(..., description="Images (base64 or ImageInput objects)")
    prompts: List[str] = Field(..., description="Text phrases for detection")
    box_threshold: float = Field(0.4, ge=0.0, le=1.0, description="Box confidence threshold")
    text_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Text confidence threshold")
    model_id: Optional[str] = Field(None, description="Specific model variant to use")


class GroundingDinoResponse(BaseModel):
    """Response for GroundingDino detection."""
    results: List[Dict[str, Any]] = Field(..., description="Detection results per image")
    model_id: str
    processing_time: float


# Depth Estimation (DepthAnythingV3) Schemas

class DepthAnythingRequest(BaseModel):
    """Request for DepthAnythingV3 depth estimation."""
    images: List[Union[ImageInput, str]] = Field(..., description="Images (base64 or ImageInput objects)")
    export_format: DepthExportFormat = Field(DepthExportFormat.GLB, description="Export format")
    model_id: Optional[str] = Field(None, description="Specific model variant to use")


class DepthAnythingResponse(BaseModel):
    """Response for depth estimation."""
    depth_data: str = Field(..., description="Base64 encoded depth data")
    format: str
    model_id: str
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None


# Feature Extraction (DinoV3) Schemas

class DinoV3Request(BaseModel):
    """Request for DinoV3 feature extraction."""
    images: List[Union[ImageInput, str]] = Field(..., description="Images (base64 or ImageInput objects)")
    model_id: Optional[str] = Field(None, description="Specific model variant to use")


class DinoV3Response(BaseModel):
    """Response for feature extraction."""
    features: Dict[str, Any] = Field(..., description="Extracted features")
    model_id: str
    processing_time: float


# Model Management Schemas

class ModelLoadRequest(BaseModel):
    """Request to load a model."""
    model_type: ModelType
    model_id: Optional[str] = Field(None, description="Specific model variant (uses default if not provided)")
    device: Optional[str] = Field(None, description="Device to load model on (cuda/cpu, auto-detect if not provided)")


class ModelUnloadRequest(BaseModel):
    """Request to unload a model."""
    model_type: ModelType


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    model_type: ModelType
    model_id: str
    device: str
    loaded: bool
    memory_mb: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    loaded_models: List[ModelType]
    available_memory_gb: Optional[float] = None
    gpu_available: bool


class ModelsStatusResponse(BaseModel):
    """Status of all models."""
    models: Dict[str, ModelInfo]
    total_memory_mb: Optional[float] = None
