"""
Utility functions for API server.
"""

import base64
import io
import logging
from typing import List, Union
from PIL import Image
import requests
from io import BytesIO

from .schemas import ImageInput, ImageFormat

logger = logging.getLogger(__name__)


def decode_image(image_data: Union[str, ImageInput]) -> Image.Image:
    """
    Decode image from base64 string or URL.

    Args:
        image_data: Base64 string, URL string, or ImageInput object

    Returns:
        PIL Image

    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        # Handle ImageInput object
        if isinstance(image_data, ImageInput):
            if image_data.format == ImageFormat.BASE64:
                return decode_base64_image(image_data.data)
            elif image_data.format == ImageFormat.URL:
                return load_image_from_url(image_data.data)
            else:
                raise ValueError(f"Unsupported image format: {image_data.format}")

        # Handle plain string (assume base64)
        elif isinstance(image_data, str):
            # Try to detect if it's a URL
            if image_data.startswith(("http://", "https://")):
                return load_image_from_url(image_data)
            else:
                return decode_base64_image(image_data)

        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")

    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ValueError(f"Failed to decode image: {e}")


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode a base64 encoded image string.

    Args:
        base64_string: Base64 encoded image (with or without data URI prefix)

    Returns:
        PIL Image
    """
    # Remove data URI prefix if present
    if "," in base64_string and base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]

    # Decode base64
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    return image


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """
    Load an image from a URL.

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        PIL Image
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))

    # Convert to RGB if necessary
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    return image


def decode_images(images_data: List[Union[str, ImageInput]]) -> List[Image.Image]:
    """
    Decode a list of images.

    Args:
        images_data: List of base64 strings, URLs, or ImageInput objects

    Returns:
        List of PIL Images
    """
    return [decode_image(img) for img in images_data]


def encode_image_to_base64(image: Image.Image, format: str = "PNG", quality: int = 85) -> str:
    """
    Encode a PIL Image to base64 string.

    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, WEBP, etc.)
        quality: Quality for lossy formats (1-100, default 85)

    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()

    # Save with quality parameter for lossy formats
    if format.upper() in ("JPEG", "JPG", "WEBP"):
        image.save(buffered, format=format, quality=quality)
    else:
        image.save(buffered, format=format)

    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def compress_image_to_webp(image: Image.Image, quality: int = 85, max_size: tuple = None) -> Image.Image:
    """
    Compress image to WebP format in-memory.

    Args:
        image: PIL Image
        quality: WebP quality (1-100, default 85)
        max_size: Optional (width, height) to resize before compression

    Returns:
        PIL Image in WebP format (still in memory)
    """
    # Resize if max_size specified
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Convert to RGB if necessary (WebP doesn't support all modes)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    # Save to WebP in memory
    buffered = io.BytesIO()
    image.save(buffered, format="WEBP", quality=quality)
    buffered.seek(0)

    # Load back as PIL Image
    return Image.open(buffered)


def encode_image_to_webp_base64(image: Image.Image, quality: int = 85, max_size: tuple = None) -> str:
    """
    Compress image to WebP and encode to base64.

    This is the recommended way to send images to the API for maximum efficiency.

    Args:
        image: PIL Image
        quality: WebP quality (1-100, default 85, recommended 80-90)
        max_size: Optional (width, height) to resize before compression

    Returns:
        Base64 encoded WebP string

    Example:
        >>> from PIL import Image
        >>> img = Image.open("large_photo.jpg")
        >>> webp_b64 = encode_image_to_webp_base64(img, quality=80, max_size=(1024, 1024))
        >>> # webp_b64 is typically 3-5x smaller than JPEG base64
    """
    # Resize if max_size specified
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Convert to RGB if necessary
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    # Encode to WebP base64
    buffered = io.BytesIO()
    image.save(buffered, format="WEBP", quality=quality)
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def serialize_tensor(tensor) -> List:
    """
    Serialize a tensor to a JSON-compatible list.

    Args:
        tensor: PyTorch tensor or numpy array

    Returns:
        Nested list representation
    """
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return tensor.tolist()


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information.

    Returns:
        Dictionary with GPU memory stats (GB)
    """
    import torch

    if not torch.cuda.is_available():
        return {
            "available": False,
            "total_gb": 0,
            "allocated_gb": 0,
            "free_gb": 0
        }

    return {
        "available": True,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
        "allocated_gb": torch.cuda.memory_allocated(0) / (1024 ** 3),
        "free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024 ** 3)
    }
