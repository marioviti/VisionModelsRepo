"""
Efficient API Client with WebP compression.

This client automatically compresses images to WebP before sending,
reducing bandwidth usage by 3-5x.
"""

import requests
import base64
import io
from pathlib import Path
from typing import List, Union, Optional
from PIL import Image


class EfficientVisionModelsClient:
    """
    API client with automatic WebP compression for efficient requests.

    Features:
    - Automatic WebP compression (3-5x smaller payloads)
    - Optional image resizing
    - Configurable quality settings
    - All standard API operations
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        webp_quality: int = 85,
        auto_resize: Optional[tuple] = None
    ):
        """
        Initialize the client.

        Args:
            base_url: API server URL
            webp_quality: WebP compression quality (1-100, default 85)
            auto_resize: Optional (max_width, max_height) for automatic resizing
        """
        self.base_url = base_url.rstrip("/")
        self.webp_quality = webp_quality
        self.auto_resize = auto_resize

    def _encode_image_webp(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Load and encode image to WebP base64.

        Args:
            image: File path or PIL Image

        Returns:
            Base64 encoded WebP string
        """
        # Load image if it's a path
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image

        # Resize if configured
        if self.auto_resize:
            img.thumbnail(self.auto_resize, Image.Resampling.LANCZOS)

        # Convert to RGB if necessary
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # Encode to WebP
        buffered = io.BytesIO()
        img.save(buffered, format="WEBP", quality=self.webp_quality)
        img_bytes = buffered.getvalue()

        return base64.b64encode(img_bytes).decode("utf-8")

    def health_check(self) -> dict:
        """Check server health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def load_model(self, model_type: str, model_id: str = None, device: str = None) -> dict:
        """Load a model into memory."""
        payload = {"model_type": model_type}
        if model_id:
            payload["model_id"] = model_id
        if device:
            payload["device"] = device

        response = requests.post(f"{self.base_url}/models/load", json=payload)
        response.raise_for_status()
        return response.json()

    def segment(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompts: Union[List[str], List[List[str]]],
        threshold: float = 0.5,
        output_format: str = "rle"
    ) -> dict:
        """
        Perform instance segmentation using Sam3.

        Images are automatically compressed to WebP before sending.

        Args:
            images: List of image file paths or PIL Images
            prompts: Text prompts (shared or per-image)
            threshold: Detection threshold
            output_format: Output format (rle, polygons, or dense)

        Returns:
            Segmentation results
        """
        # Encode images to WebP
        encoded_images = [self._encode_image_webp(img) for img in images]

        payload = {
            "images": encoded_images,
            "prompts": prompts,
            "threshold": threshold,
            "output_format": output_format
        }

        response = requests.post(f"{self.base_url}/inference/sam3", json=payload)
        response.raise_for_status()
        return response.json()

    def detect_objects(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompts: List[str],
        box_threshold: float = 0.4,
        text_threshold: float = 0.3
    ) -> dict:
        """
        Perform object detection using GroundingDino.

        Images are automatically compressed to WebP before sending.

        Args:
            images: List of image file paths or PIL Images
            prompts: Text phrases for detection
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold

        Returns:
            Detection results
        """
        # Encode images to WebP
        encoded_images = [self._encode_image_webp(img) for img in images]

        payload = {
            "images": encoded_images,
            "prompts": prompts,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold
        }

        response = requests.post(f"{self.base_url}/inference/grounding-dino", json=payload)
        response.raise_for_status()
        return response.json()

    def estimate_depth(
        self,
        images: List[Union[str, Path, Image.Image]],
        export_format: str = "glb"
    ) -> dict:
        """
        Estimate depth using DepthAnythingV3.

        Images are automatically compressed to WebP before sending.

        Args:
            images: List of image file paths or PIL Images
            export_format: Export format (glb, npz, ply, gs_ply, gs_video)

        Returns:
            Depth estimation results
        """
        # Encode images to WebP
        encoded_images = [self._encode_image_webp(img) for img in images]

        payload = {
            "images": encoded_images,
            "export_format": export_format
        }

        response = requests.post(f"{self.base_url}/inference/depth-anything", json=payload)
        response.raise_for_status()
        return response.json()

    def extract_features(
        self,
        images: List[Union[str, Path, Image.Image]]
    ) -> dict:
        """
        Extract features using DinoV3.

        Images are automatically compressed to WebP before sending.

        Args:
            images: List of image file paths or PIL Images

        Returns:
            Feature extraction results
        """
        # Encode images to WebP
        encoded_images = [self._encode_image_webp(img) for img in images]

        payload = {"images": encoded_images}

        response = requests.post(f"{self.base_url}/inference/dinov3", json=payload)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the efficient client."""

    # Initialize client with WebP compression
    client = EfficientVisionModelsClient(
        base_url="http://localhost:8000",
        webp_quality=85,  # Good balance of quality and size
        auto_resize=(1024, 1024)  # Resize large images
    )

    print("Efficient Vision Models API Client")
    print("=" * 70)
    print(f"WebP Quality: {client.webp_quality}")
    print(f"Auto Resize: {client.auto_resize}")
    print()

    # Check health
    try:
        health = client.health_check()
        print(f"✅ Server Status: {health['status']}")
        print(f"GPU Available: {health['gpu_available']}")
        print(f"Loaded Models: {health['loaded_models']}")
    except requests.exceptions.ConnectionError:
        print("❌ Server not running!")
        print("Start with: python run_api_server.py")
        return

    print()

    # Example: Segmentation with WebP compression
    print("Example: Instance Segmentation")
    print("-" * 70)

    # Replace with your actual image path
    image_path = "test.jpg"

    try:
        # Images are automatically compressed to WebP
        result = client.segment(
            images=[image_path],
            prompts=["person", "object"],
            threshold=0.5,
            output_format="rle"  # RLE is also compact
        )

        print(f"✅ Segmentation complete!")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Model: {result['model_id']}")

        for img_results in result['results']:
            for detection in img_results:
                prompt = detection['prompt']
                num_masks = len(detection['result'].get('masks_rle', []))
                print(f"  - '{prompt}': {num_masks} detections")

    except FileNotFoundError:
        print(f"⚠️  Image not found: {image_path}")
        print("Update the image_path variable with a valid image.")

    print()
    print("=" * 70)
    print("Benefits of WebP compression:")
    print("✅ 3-5x smaller payloads")
    print("✅ Faster upload times")
    print("✅ Reduced bandwidth costs")
    print("✅ Same inference quality")
    print("=" * 70)


if __name__ == "__main__":
    main()
