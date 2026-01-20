"""
Model management for API server.
"""

import torch
import logging
from typing import Optional, Dict
from enum import Enum

from ..vision_models.segmentation.sam3 import Sam3
from ..vision_models.object_detection.grounding_dino import GroundingDinoInstanceDetection
from ..vision_models.depth_estimation.depth_anything3 import DepthAnythingV3
from ..vision_models.feature_extraction.dino3 import DinoV3

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Available model types."""
    SAM3 = "sam3"
    GROUNDING_DINO = "grounding_dino"
    DEPTH_ANYTHING = "depth_anything"
    DINOV3 = "dinov3"


class ModelManager:
    """
    Manages loading, unloading, and access to vision models.
    Implements lazy loading and memory management.
    """

    def __init__(self):
        self._models: Dict[ModelType, Optional[object]] = {
            ModelType.SAM3: None,
            ModelType.GROUNDING_DINO: None,
            ModelType.DEPTH_ANYTHING: None,
            ModelType.DINOV3: None,
        }
        self._model_configs: Dict[ModelType, Dict] = {
            ModelType.SAM3: {},
            ModelType.GROUNDING_DINO: {},
            ModelType.DEPTH_ANYTHING: {},
            ModelType.DINOV3: {},
        }

    def load_model(
        self,
        model_type: ModelType,
        model_id: Optional[str] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Load a model into memory.

        Args:
            model_type: Type of model to load
            model_id: Specific model variant (uses default if None)
            device: Device to load on (auto-detect if None)
        """
        if self._models[model_type] is not None:
            logger.info(f"Model {model_type} already loaded, skipping")
            return

        logger.info(f"Loading {model_type} model...")

        try:
            if model_type == ModelType.SAM3:
                model = Sam3(model_id=model_id or Sam3.collection[0], device=device)
            elif model_type == ModelType.GROUNDING_DINO:
                model = GroundingDinoInstanceDetection(
                    model_id=model_id or GroundingDinoInstanceDetection.collection[0],
                    device=device
                )
            elif model_type == ModelType.DEPTH_ANYTHING:
                model = DepthAnythingV3(
                    model_id=model_id or DepthAnythingV3.collection[0],
                    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
                )
            elif model_type == ModelType.DINOV3:
                model = DinoV3(
                    model=model_id or DinoV3.collection[0],
                    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self._models[model_type] = model
            self._model_configs[model_type] = {
                "model_id": getattr(model, "model_id", model_id or "default"),
                "device": str(getattr(model, "device", device or "auto"))
            }
            logger.info(f"Successfully loaded {model_type}")

        except Exception as e:
            logger.error(f"Failed to load {model_type}: {e}")
            raise

    def unload_model(self, model_type: ModelType) -> None:
        """
        Unload a model from memory.

        Args:
            model_type: Type of model to unload
        """
        if self._models[model_type] is None:
            logger.warning(f"Model {model_type} not loaded, nothing to unload")
            return

        logger.info(f"Unloading {model_type} model...")

        # Delete model and clear CUDA cache
        self._models[model_type] = None
        self._model_configs[model_type] = {}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Successfully unloaded {model_type}")

    def get_model(self, model_type: ModelType, auto_load: bool = False):
        """
        Get a loaded model.

        Args:
            model_type: Type of model to get
            auto_load: If True, automatically load model if not loaded

        Returns:
            The loaded model instance

        Raises:
            ValueError: If model not loaded and auto_load=False
        """
        model = self._models[model_type]

        if model is None:
            if auto_load:
                logger.info(f"Auto-loading {model_type}")
                self.load_model(model_type)
                model = self._models[model_type]
            else:
                raise ValueError(f"Model {model_type} not loaded. Call load_model() first.")

        return model

    def is_loaded(self, model_type: ModelType) -> bool:
        """Check if a model is loaded."""
        return self._models[model_type] is not None

    def get_loaded_models(self) -> list[ModelType]:
        """Get list of currently loaded models."""
        return [mt for mt, model in self._models.items() if model is not None]

    def get_model_info(self, model_type: ModelType) -> Dict:
        """Get information about a model."""
        is_loaded = self.is_loaded(model_type)
        info = {
            "model_type": model_type.value,
            "loaded": is_loaded,
        }

        if is_loaded:
            config = self._model_configs[model_type]
            info.update(config)

            # Try to estimate memory usage
            model = self._models[model_type]
            try:
                if hasattr(model, "model") and hasattr(model.model, "parameters"):
                    params = sum(p.numel() for p in model.model.parameters())
                    # Rough estimate: 4 bytes per parameter (float32)
                    info["memory_mb"] = (params * 4) / (1024 * 1024)
            except Exception:
                pass

        return info

    def unload_all(self) -> None:
        """Unload all models."""
        for model_type in list(self._models.keys()):
            if self.is_loaded(model_type):
                self.unload_model(model_type)
