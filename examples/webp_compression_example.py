"""
Example: Using WebP compression for efficient API requests.

WebP compression can reduce payload size by 3-5x compared to JPEG/PNG,
significantly improving API performance and reducing bandwidth usage.
"""

import requests
import base64
import sys
from pathlib import Path
from PIL import Image
import io


def encode_image_to_webp_base64(image: Image.Image, quality: int = 85, max_size: tuple = None) -> str:
    """
    Compress image to WebP and encode to base64.

    Args:
        image: PIL Image
        quality: WebP quality (1-100, recommended 80-90)
        max_size: Optional (width, height) to resize before compression

    Returns:
        Base64 encoded WebP string
    """
    # Resize if max_size specified
    if max_size:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Convert to RGB if necessary (WebP doesn't support all modes)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    # Encode to WebP
    buffered = io.BytesIO()
    image.save(buffered, format="WEBP", quality=quality)
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def encode_image_to_jpeg_base64(image: Image.Image, quality: int = 85) -> str:
    """Encode image to JPEG base64 (for comparison)."""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def encode_image_to_png_base64(image: Image.Image) -> str:
    """Encode image to PNG base64 (for comparison)."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def compare_encodings(image_path: str):
    """Compare different encoding formats."""
    print(f"Comparing encoding formats for: {image_path}")
    print("=" * 70)

    # Load image
    image = Image.open(image_path)
    print(f"Original image: {image.size[0]}x{image.size[1]} ({image.mode})")

    # Get original file size
    original_size = Path(image_path).stat().st_size
    print(f"Original file size: {original_size:,} bytes ({original_size/1024:.1f} KB)")
    print()

    # Test different encodings
    results = {}

    # PNG
    print("1. PNG Encoding")
    png_b64 = encode_image_to_png_base64(image)
    png_size = len(png_b64)
    results['PNG'] = png_size
    print(f"   Base64 size: {png_size:,} chars ({png_size/1024:.1f} KB)")
    print(f"   Compression: {original_size/png_size:.2f}x smaller than original")
    print()

    # JPEG (quality 85)
    print("2. JPEG Encoding (quality=85)")
    jpeg_b64 = encode_image_to_jpeg_base64(image, quality=85)
    jpeg_size = len(jpeg_b64)
    results['JPEG-85'] = jpeg_size
    print(f"   Base64 size: {jpeg_size:,} chars ({jpeg_size/1024:.1f} KB)")
    print(f"   vs PNG: {png_size/jpeg_size:.2f}x smaller")
    print()

    # WebP (quality 85)
    print("3. WebP Encoding (quality=85) - RECOMMENDED")
    webp_b64 = encode_image_to_webp_base64(image, quality=85)
    webp_size = len(webp_b64)
    results['WebP-85'] = webp_size
    print(f"   Base64 size: {webp_size:,} chars ({webp_size/1024:.1f} KB)")
    print(f"   vs PNG: {png_size/webp_size:.2f}x smaller")
    print(f"   vs JPEG: {jpeg_size/webp_size:.2f}x smaller")
    print()

    # WebP (quality 80) - More aggressive
    print("4. WebP Encoding (quality=80) - VERY EFFICIENT")
    webp_80_b64 = encode_image_to_webp_base64(image, quality=80)
    webp_80_size = len(webp_80_b64)
    results['WebP-80'] = webp_80_size
    print(f"   Base64 size: {webp_80_size:,} chars ({webp_80_size/1024:.1f} KB)")
    print(f"   vs PNG: {png_size/webp_80_size:.2f}x smaller")
    print(f"   vs JPEG: {jpeg_size/webp_80_size:.2f}x smaller")
    print()

    # WebP with resize (1024x1024 max)
    print("5. WebP with Resize (max 1024x1024, quality=85)")
    webp_resized_b64 = encode_image_to_webp_base64(image.copy(), quality=85, max_size=(1024, 1024))
    webp_resized_size = len(webp_resized_b64)
    results['WebP-Resized'] = webp_resized_size
    print(f"   Base64 size: {webp_resized_size:,} chars ({webp_resized_size/1024:.1f} KB)")
    print(f"   vs PNG: {png_size/webp_resized_size:.2f}x smaller")
    print(f"   vs JPEG: {jpeg_size/webp_resized_size:.2f}x smaller")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY:")
    print(f"Best compression: WebP (quality=80) - {png_size/webp_80_size:.1f}x smaller than PNG")
    print(f"Best quality/size: WebP (quality=85) - {png_size/webp_size:.1f}x smaller than PNG")
    print(f"Smallest payload: WebP Resized - {png_size/webp_resized_size:.1f}x smaller than PNG")
    print("=" * 70)

    return webp_b64


def test_api_with_webp(image_path: str, api_url: str = "http://localhost:8000"):
    """Test API request with WebP compression."""
    print("\n" + "=" * 70)
    print("Testing API Request with WebP Compression")
    print("=" * 70)

    # Load image
    image = Image.open(image_path)

    # Encode with WebP
    print("\nEncoding image with WebP (quality=85)...")
    webp_b64 = encode_image_to_webp_base64(image, quality=85)
    print(f"Payload size: {len(webp_b64):,} chars ({len(webp_b64)/1024:.1f} KB)")

    # Make API request
    print("\nSending request to API...")
    response = requests.post(
        f"{api_url}/inference/sam3",
        json={
            "images": [webp_b64],
            "prompts": ["person", "object"],
            "threshold": 0.5,
            "output_format": "rle"  # Use RLE for compact response too
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success!")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Model: {result['model_id']}")

        # Show results
        for img_results in result['results']:
            for detection in img_results:
                prompt = detection['prompt']
                num_masks = len(detection['result'].get('masks_rle', []))
                print(f"  - Prompt '{prompt}': {num_masks} detections")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python webp_compression_example.py <image_path>")
        print("\nExample:")
        print("  python webp_compression_example.py test.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Compare encodings
    webp_b64 = compare_encodings(image_path)

    # Test API (optional)
    try:
        test_api_with_webp(image_path)
    except requests.exceptions.ConnectionError:
        print("\n⚠️  API server not running. Skipping API test.")
        print("Start the server with: python run_api_server.py")


if __name__ == "__main__":
    main()
