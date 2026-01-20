# Image Compression Guide for API Requests

## Overview

Compressing images before sending them to the API can dramatically reduce bandwidth usage and improve performance. **WebP compression is recommended** for API requests.

## Compression Comparison

For a typical 2048x1536 JPEG photo (~1.5MB):

| Format | Payload Size | Reduction | Upload Time (10 Mbps) |
|--------|-------------|-----------|----------------------|
| PNG | ~4.2 MB | Baseline | ~3.4s |
| JPEG (quality 85) | ~1.8 MB | 2.3x smaller | ~1.4s |
| **WebP (quality 85)** | **~0.8 MB** | **5.3x smaller** | **~0.6s** |
| WebP (quality 80) | ~0.6 MB | 7x smaller | ~0.5s |
| WebP + Resize (1024px) | ~0.3 MB | 14x smaller | ~0.24s |

**Recommendation**: Use **WebP with quality 80-85** for the best balance of quality and size.

## Quick Start

### Option 1: Use the Efficient Client (Recommended)

The `EfficientVisionModelsClient` automatically compresses to WebP:

```python
from efficient_api_client import EfficientVisionModelsClient

# Client with automatic WebP compression
client = EfficientVisionModelsClient(
    base_url="http://localhost:8000",
    webp_quality=85,           # Quality (1-100)
    auto_resize=(1024, 1024)   # Optional: resize large images
)

# Images are automatically compressed to WebP
result = client.segment(
    images=["photo1.jpg", "photo2.png"],  # Any format
    prompts=["person", "car"],
    output_format="rle"
)
```

### Option 2: Manual WebP Encoding

Use the utility functions:

```python
from PIL import Image
import io
import base64

def encode_image_to_webp_base64(image_path, quality=85):
    """Compress image to WebP and encode to base64."""
    img = Image.open(image_path)

    # Convert to RGB if necessary
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # Encode to WebP
    buffered = io.BytesIO()
    img.save(buffered, format="WEBP", quality=quality)
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")

# Use it
webp_b64 = encode_image_to_webp_base64("photo.jpg", quality=85)

# Send to API
import requests
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [webp_b64],
        "prompts": ["person"],
        "output_format": "rle"
    }
)
```

## WebP Quality Settings

| Quality | Size | Visual Quality | Use Case |
|---------|------|----------------|----------|
| 100 | Largest | Perfect | Archival, critical applications |
| 90-95 | Large | Excellent | High-quality requirements |
| **80-85** | **Medium** | **Very Good** | **Recommended for API** |
| 70-75 | Small | Good | Low-bandwidth scenarios |
| 60-65 | Very Small | Acceptable | Mobile, very low bandwidth |
| <60 | Tiny | Poor | Not recommended |

**Recommended**: Quality **80-85** provides excellent visual quality with significant size reduction.

## Advanced: Resize Before Compression

For even better efficiency, resize images before sending:

```python
from PIL import Image
import io
import base64

def encode_resized_webp(image_path, max_size=(1024, 1024), quality=85):
    """Resize and compress to WebP."""
    img = Image.open(image_path)

    # Resize maintaining aspect ratio
    img.thumbnail(max_size, Image.Resampling.LANCZOS)

    # Convert to RGB
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # Encode to WebP
    buffered = io.BytesIO()
    img.save(buffered, format="WEBP", quality=quality)
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")

# Use it
webp_b64 = encode_resized_webp("large_photo.jpg", max_size=(1024, 1024), quality=85)
```

**Benefits**:
- 4K image (3840x2160) → 1024x576: ~15x smaller
- 2K image (2048x1536) → 1024x768: ~4x smaller
- Inference quality typically unaffected for detection/segmentation tasks

## Complete Example

```python
from PIL import Image
import requests
import base64
import io

def compress_and_send(image_path, api_url="http://localhost:8000"):
    """Complete example: compress and send to API."""

    # 1. Load image
    img = Image.open(image_path)
    print(f"Original: {img.size[0]}x{img.size[1]}")

    # 2. Resize if large
    max_size = (1024, 1024)
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        print(f"Resized: {img.size[0]}x{img.size[1]}")

    # 3. Convert to RGB
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # 4. Compress to WebP
    buffered = io.BytesIO()
    img.save(buffered, format="WEBP", quality=85)
    img_bytes = buffered.getvalue()

    print(f"Compressed size: {len(img_bytes):,} bytes ({len(img_bytes)/1024:.1f} KB)")

    # 5. Encode to base64
    webp_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # 6. Send to API
    response = requests.post(
        f"{api_url}/inference/sam3",
        json={
            "images": [webp_b64],
            "prompts": ["person", "object"],
            "threshold": 0.5,
            "output_format": "rle"  # Also use compact format
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Processing time: {result['processing_time']:.2f}s")
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

# Use it
compress_and_send("my_photo.jpg")
```

## Test Script

Compare different compression methods:

```bash
python examples/webp_compression_example.py your_image.jpg
```

This will show:
- Size comparison of PNG, JPEG, WebP
- Different quality levels
- Resized vs. original
- Actual size savings

## Server-Side Support

The API server automatically handles WebP-encoded images:

1. **Decode**: Server automatically detects and decodes WebP
2. **Process**: Models process images normally
3. **Quality**: No loss in inference quality

```python
# Server automatically handles this
decoded_image = decode_base64_image(webp_base64_string)
# → Returns PIL Image, ready for inference
```

## Best Practices

### ✅ DO:

1. **Use WebP with quality 80-85** for API requests
2. **Resize large images** before encoding (1024px is usually sufficient)
3. **Use RLE or Polygon output** for compact responses
4. **Batch multiple images** in one request when possible
5. **Cache encoded images** if sending the same image multiple times

### ❌ DON'T:

1. Send PNG-encoded images (3-5x larger than WebP)
2. Send 4K/8K images without resizing
3. Use quality > 90 for API requests (diminishing returns)
4. Forget to convert to RGB before WebP encoding
5. Use dense output format with large images

## Real-World Example

### Before (PNG encoding):
```python
# Large payload
with open("photo.jpg", "rb") as f:
    png_b64 = base64.b64encode(f.read()).decode()

# Payload: 4.2 MB
# Upload time: 3.4s @ 10 Mbps
```

### After (WebP encoding):
```python
# Optimized payload
webp_b64 = encode_resized_webp("photo.jpg", max_size=(1024, 1024), quality=85)

# Payload: 0.3 MB (14x smaller!)
# Upload time: 0.24s @ 10 Mbps
```

**Savings**:
- 93% reduction in payload size
- 14x faster upload
- Same inference quality
- Lower bandwidth costs

## Bandwidth Cost Comparison

For 1000 API requests with 2MB images:

| Encoding | Total Data | Cost @ $0.10/GB | Time @ 10 Mbps |
|----------|-----------|-----------------|----------------|
| PNG | 4.2 GB | $0.42 | 56 minutes |
| JPEG | 1.8 GB | $0.18 | 24 minutes |
| **WebP** | **0.8 GB** | **$0.08** | **11 minutes** |
| WebP + Resize | 0.3 GB | $0.03 | 4 minutes |

## Mobile/Low Bandwidth Scenarios

For mobile or low-bandwidth scenarios:

```python
# Aggressive compression for mobile
client = EfficientVisionModelsClient(
    webp_quality=70,           # Lower quality
    auto_resize=(800, 800)     # Smaller size
)

# Payloads typically < 200 KB
```

## Troubleshooting

### WebP not supported

**Problem**: PIL doesn't have WebP support.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libwebp-dev

# macOS
brew install webp

# Reinstall Pillow
pip uninstall Pillow
pip install Pillow
```

### Quality too low

**Problem**: Visible compression artifacts.

**Solution**: Increase quality to 85-90:
```python
webp_b64 = encode_image_to_webp_base64(img, quality=90)
```

### Server rejects image

**Problem**: Server can't decode WebP.

**Solution**: Ensure Pillow on server has WebP support (see above).

## Summary

**For Best Performance:**

1. Use `EfficientVisionModelsClient` (automatic WebP)
2. Quality: 80-85
3. Resize to 1024px max for detection/segmentation
4. Use RLE/Polygon output formats
5. Batch requests when possible

**Expected Savings:**
- 3-5x smaller payloads
- 3-5x faster uploads
- 3-5x lower bandwidth costs
- Same inference quality

## See Also

- [examples/efficient_api_client.py](examples/efficient_api_client.py) - Client with auto-compression
- [examples/webp_compression_example.py](examples/webp_compression_example.py) - Compression comparison
- [src/vision_model_repo/api/utils.py](src/vision_model_repo/api/utils.py) - Compression utilities
- [SAM3_OUTPUT_FORMATS.md](SAM3_OUTPUT_FORMATS.md) - Output format guide
