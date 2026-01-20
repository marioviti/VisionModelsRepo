"""
Example client for Vision Models API.

This script demonstrates how to use the API for various vision tasks.
"""

import requests
import base64
import json
from pathlib import Path
from typing import List, Union


class VisionModelsClient:
    """Client for interacting with Vision Models API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode an image file to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

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

    def unload_model(self, model_type: str) -> dict:
        """Unload a model from memory."""
        response = requests.post(
            f"{self.base_url}/models/unload",
            json={"model_type": model_type}
        )
        response.raise_for_status()
        return response.json()

    def segment(
        self,
        images: List[Union[str, Path]],
        prompts: Union[List[str], List[List[str]]],
        threshold: float = 0.5,
        output_format: str = "dense"
    ) -> dict:
        """
        Perform instance segmentation using Sam3.

        Args:
            images: List of image file paths
            prompts: Text prompts (shared or per-image)
            threshold: Detection threshold
            output_format: Output format (dense, rle, polygons)

        Returns:
            Segmentation results
        """
        encoded_images = [self._encode_image(img) for img in images]

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
        images: List[Union[str, Path]],
        prompts: List[str],
        box_threshold: float = 0.4,
        text_threshold: float = 0.3
    ) -> dict:
        """
        Perform object detection using GroundingDino.

        Args:
            images: List of image file paths
            prompts: Text phrases for detection
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold

        Returns:
            Detection results
        """
        encoded_images = [self._encode_image(img) for img in images]

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
        images: List[Union[str, Path]],
        export_format: str = "glb"
    ) -> dict:
        """
        Estimate depth using DepthAnythingV3.

        Args:
            images: List of image file paths
            export_format: Export format (glb, npz, ply, gs_ply, gs_video)

        Returns:
            Depth estimation results
        """
        encoded_images = [self._encode_image(img) for img in images]

        payload = {
            "images": encoded_images,
            "export_format": export_format
        }

        response = requests.post(f"{self.base_url}/inference/depth-anything", json=payload)
        response.raise_for_status()
        return response.json()

    def extract_features(
        self,
        images: List[Union[str, Path]]
    ) -> dict:
        """
        Extract features using DinoV3.

        Args:
            images: List of image file paths

        Returns:
            Feature extraction results
        """
        encoded_images = [self._encode_image(img) for img in images]

        payload = {"images": encoded_images}

        response = requests.post(f"{self.base_url}/inference/dinov3", json=payload)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the Vision Models API client."""

    # Initialize client
    client = VisionModelsClient(base_url="http://localhost:8000")

    # Check server health
    print("Checking server health...")
    health = client.health_check()
    print(f"Server status: {health['status']}")
    print(f"GPU available: {health['gpu_available']}")
    print(f"Loaded models: {health['loaded_models']}")
    print()

    # Example 1: Instance Segmentation
    print("Example 1: Instance Segmentation")
    print("-" * 50)

    # Note: Replace with actual image paths
    image_path = "path/to/your/image.jpg"

    try:
        result = client.segment(
            images=[image_path],
            prompts=["person", "car"],
            threshold=0.5,
            output_format="dense"
        )

        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Model: {result['model_id']}")
        print(f"Results: {len(result['results'][0])} detections")
        print()

    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        print("Please update the image_path variable with a valid image file.")
        print()

    # Example 2: Object Detection
    print("Example 2: Object Detection")
    print("-" * 50)

    try:
        result = client.detect_objects(
            images=[image_path],
            prompts=["person", "car", "traffic light"],
            box_threshold=0.4
        )

        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Model: {result['model_id']}")

        if result['results']:
            boxes = result['results'][0].get('boxes', [])
            labels = result['results'][0].get('labels', [])
            scores = result['results'][0].get('scores', [])

            print(f"Detected {len(boxes)} objects:")
            for label, score in zip(labels, scores):
                print(f"  - {label}: {score:.2f}")
        print()

    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        print()

    # Example 3: Model Management
    print("Example 3: Model Management")
    print("-" * 50)

    # Load a specific model
    print("Loading GroundingDino model...")
    load_result = client.load_model("grounding_dino", device="cuda")
    print(f"Result: {load_result['message']}")
    print()

    # Unload model
    print("Unloading GroundingDino model...")
    unload_result = client.unload_model("grounding_dino")
    print(f"Result: {unload_result['message']}")
    print()


if __name__ == "__main__":
    main()
