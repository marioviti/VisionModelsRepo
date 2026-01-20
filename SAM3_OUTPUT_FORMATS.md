# Sam3 Output Formats Guide

## Overview

Sam3 segmentation supports three output formats. **Use RLE or Polygons for API responses** to avoid large payloads.

| Format | Size | Use Case | Recommended |
|--------|------|----------|-------------|
| **RLE** | Small (~10KB) | Storage, transmission, API responses | ✅ Yes |
| **Polygons** | Medium (~50KB) | Editing, visualization, vector graphics | ✅ Yes |
| **Dense** | Large (~10MB+) | Direct pixel manipulation, research | ❌ No (for API) |

## Format Comparison

### 1. RLE (Run-Length Encoding) - RECOMMENDED

**Best for**: API responses, storage, CVAT/COCO compatibility

**Structure**:
```json
{
  "masks_rle": [
    {
      "size": [height, width],
      "counts": "compressed_rle_string..."
    }
  ]
}
```

**Advantages**:
- ✅ Very compact (~100x smaller than dense)
- ✅ Fast to serialize/deserialize
- ✅ Compatible with CVAT, COCO format
- ✅ Lossless compression

**Usage**:
```python
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [image_base64],
        "prompts": ["person"],
        "output_format": "rle"  # Default
    }
)

result = response.json()
rle_masks = result['results'][0][0]['result']['masks_rle']
```

### 2. Polygons - RECOMMENDED

**Best for**: Editing, visualization, vector graphics

**Structure**:
```json
{
  "masks_polygon": [
    [[x1, y1], [x2, y2], ...],  // Polygon 1
    [[x1, y1], [x2, y2], ...]   // Polygon 2
  ]
}
```

**Advantages**:
- ✅ Human-readable
- ✅ Editable (can modify vertices)
- ✅ Good for vector graphics
- ✅ Reasonable size (~10-50x smaller than dense)

**Usage**:
```python
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [image_base64],
        "prompts": ["person"],
        "output_format": "polygons"
    }
)

result = response.json()
polygons = result['results'][0][0]['result']['masks_polygon']

# Each polygon is a list of [x, y] coordinates
for polygon in polygons:
    print(f"Polygon with {len(polygon)} vertices")
```

### 3. Dense Masks - NOT RECOMMENDED FOR API

**Best for**: Direct pixel manipulation in research/notebook environments

**Structure**:
```json
{
  "masks": [
    [[0, 1, 1, 0, ...], [0, 1, 1, 0, ...], ...],  // Mask 1 (H x W)
    [[0, 0, 1, 1, ...], [0, 0, 1, 1, ...], ...]   // Mask 2 (H x W)
  ]
}
```

**Disadvantages**:
- ❌ Very large (MBs per image)
- ❌ Slow to serialize/transfer
- ❌ Not suitable for web APIs
- ❌ High bandwidth usage

**Usage** (if you really need it):
```python
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [image_base64],
        "prompts": ["person"],
        "output_format": "dense"
    }
)

result = response.json()
dense_masks = result['results'][0][0]['result']['masks']

# Shape: (num_instances, height, width)
# Values: 0 or 1 (binary mask)
```

## Size Comparison Example

For a typical 640x480 image with 3 detected objects:

| Format | Approx. Size | Compression Ratio |
|--------|--------------|-------------------|
| Dense | 2.7 MB | 1x (baseline) |
| Polygons | 15 KB | 180x smaller |
| RLE | 8 KB | 340x smaller |

## Conversion Between Formats

### RLE to Dense (if needed)

```python
import zlib
import base64
import numpy as np

def rle_to_dense(rle_data):
    """Convert RLE mask to dense binary mask."""
    size = rle_data['size']  # [height, width]
    counts = base64.b64decode(rle_data['counts'])
    counts = zlib.decompress(counts)

    # Decode RLE to binary mask
    # ... (use pycocotools or custom decoder)

    return binary_mask
```

### Polygon to Dense

```python
import cv2
import numpy as np

def polygon_to_dense(polygon, size):
    """Convert polygon to dense binary mask."""
    height, width = size
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert polygon to numpy array
    pts = np.array(polygon, dtype=np.int32)

    # Fill polygon
    cv2.fillPoly(mask, [pts], 1)

    return mask
```

## API Request Examples

### Example 1: RLE Format (Default)

```bash
curl -X POST http://localhost:8000/inference/sam3 \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["base64_image_here"],
    "prompts": ["person", "car"],
    "threshold": 0.5,
    "output_format": "rle"
  }'
```

### Example 2: Polygon Format

```bash
curl -X POST http://localhost:8000/inference/sam3 \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["base64_image_here"],
    "prompts": ["person"],
    "threshold": 0.5,
    "output_format": "polygons"
  }'
```

## Response Structure

All formats return the same base structure:

```json
{
  "results": [
    [
      {
        "prompt_index": 0,
        "prompt": "person",
        "result": {
          // Format-specific mask data here
          "iou_scores": [0.95, 0.87],
          "boxes": [[x1, y1, x2, y2], ...]
        }
      }
    ]
  ],
  "model_id": "facebook/sam3",
  "processing_time": 1.23
}
```

## Best Practices

### ✅ DO:
- Use **RLE** for API responses and storage
- Use **Polygons** for editing and visualization
- Compress HTTP responses with gzip
- Cache results when possible

### ❌ DON'T:
- Use **Dense** format over HTTP (too large)
- Request dense masks for multiple images
- Forget to specify `output_format` (defaults to RLE)

## Testing Different Formats

Use the test script:

```bash
python examples/test_sam3_formats.py your_image.jpg
```

This will demonstrate all three formats and show their relative sizes.

## Troubleshooting

### "Unable to serialize unknown type: torch.Tensor"

**Problem**: Dense masks contain PyTorch tensors that can't be serialized.

**Solution**: The server now automatically converts tensors to lists. If you still see this error, ensure you're using the latest server code.

### Large Response Times

**Problem**: Response takes very long to return.

**Solution**:
- Switch from `dense` to `rle` or `polygons`
- Reduce image size
- Use batch processing for multiple images

### Memory Errors

**Problem**: Server runs out of memory.

**Solution**:
- Don't use `dense` format
- Reduce batch size
- Unload unused models

## Summary

**For API Usage:**
1. **Use RLE** (default) - Most efficient
2. **Use Polygons** if you need editability
3. **Avoid Dense** - Too large for network transfer

**For Direct Usage (notebooks, scripts):**
- Dense format is fine when working locally
- RLE/Polygons still recommended for saving results
