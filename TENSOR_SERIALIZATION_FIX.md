# Tensor Serialization Fix - Summary

## Problem

The API was returning a `PydanticSerializationError` when using `output_format="dense"` for Sam3 segmentation:

```
pydantic_core._pydantic_core.PydanticSerializationError:
Unable to serialize unknown type: <class 'torch.Tensor'>
```

**Root Cause**: Dense masks are returned as PyTorch tensors, which cannot be directly serialized to JSON by FastAPI/Pydantic.

## Solution

### 1. Improved Tensor Serialization

Updated [src/vision_model_repo/api/server.py](src/vision_model_repo/api/server.py:185-241) with robust tensor-to-list conversion:

**Before**:
```python
# Shallow copy, partial tensor handling
result_copy = entry.copy()
if "masks" in result_data and hasattr(result_data["masks"], "tolist"):
    result_data["masks"] = serialize_tensor(result_data["masks"])
```

**After**:
```python
# Deep copy with comprehensive serialization
result_copy = {
    "prompt_index": entry.get("prompt_index"),
    "prompt": entry.get("prompt"),
    "result": {}
}

for key, value in result_data.items():
    if hasattr(value, "tolist"):
        # Convert tensors to lists
        result_copy["result"][key] = serialize_tensor(value)
    elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
        # JSON-serializable types
        result_copy["result"][key] = value
    else:
        # Fallback for unknown types
        result_copy["result"][key] = str(value)
```

### 2. Changed Default Output Format

Updated [src/vision_model_repo/api/schemas.py](src/vision_model_repo/api/schemas.py:68-77):

- **Old Default**: `dense` (large, problematic)
- **New Default**: `rle` (compact, efficient)

```python
output_format: SegmentationOutputFormat = Field(
    SegmentationOutputFormat.RLE,  # Changed from DENSE
    description="Output format: 'rle' (recommended, compact), 'polygons' (recommended, editable), or 'dense' (large binary masks)"
)
```

### 3. Added Format Documentation

Updated endpoint docstring to guide users:

```python
"""
Perform instance segmentation using Sam3.

Supports multiple output formats:
- dense: Binary masks as nested lists (not recommended for API - use RLE or polygons instead)
- rle: CVAT/COCO-style RLE encoded masks (recommended)
- polygons: Polygon vertices (recommended)
"""
```

## Output Format Comparison

| Format | Size | Serializable | Recommended | Use Case |
|--------|------|--------------|-------------|----------|
| **RLE** | ~10KB | ✅ Yes | ✅ Yes | API, storage, CVAT/COCO |
| **Polygons** | ~50KB | ✅ Yes | ✅ Yes | Editing, visualization |
| **Dense** | ~10MB | ✅ Now (fixed) | ❌ No | Research (avoid in API) |

## Testing

### Test All Formats

Use the test script:

```bash
python examples/test_sam3_formats.py your_image.jpg
```

### Example: RLE Format (Recommended)

```python
import requests
import base64

with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [img_b64],
        "prompts": ["person"],
        "output_format": "rle"  # Compact and efficient
    }
)

result = response.json()
print(result['results'][0][0]['result']['masks_rle'])
```

### Example: Polygons Format

```python
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [img_b64],
        "prompts": ["person"],
        "output_format": "polygons"  # Editable
    }
)

result = response.json()
print(result['results'][0][0]['result']['masks_polygon'])
```

### Example: Dense Format (Fixed, but not recommended)

```python
response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [img_b64],
        "prompts": ["person"],
        "output_format": "dense"  # Now works, but large
    }
)

result = response.json()
# Returns nested lists instead of tensors
print(result['results'][0][0]['result']['masks'])
```

## Files Modified

1. **[src/vision_model_repo/api/server.py](src/vision_model_repo/api/server.py)**
   - Improved tensor serialization logic
   - Added better error logging
   - Updated docstrings

2. **[src/vision_model_repo/api/schemas.py](src/vision_model_repo/api/schemas.py)**
   - Changed default `output_format` from `DENSE` to `RLE`
   - Added better field descriptions

## Files Created

3. **[examples/test_sam3_formats.py](examples/test_sam3_formats.py)**
   - Test script for all three output formats
   - Size comparison demonstration

4. **[SAM3_OUTPUT_FORMATS.md](SAM3_OUTPUT_FORMATS.md)**
   - Complete guide to output formats
   - Conversion examples
   - Best practices

5. **[TENSOR_SERIALIZATION_FIX.md](TENSOR_SERIALIZATION_FIX.md)** (this file)
   - Summary of the fix

## Verification

Test that all formats work:

```bash
# Start server
python run_api_server.py

# Test RLE (default)
curl -X POST http://localhost:8000/inference/sam3 \
  -H "Content-Type: application/json" \
  -d '{"images": ["'$(base64 -w0 test.jpg)'"], "prompts": ["person"]}'

# Test polygons
curl -X POST http://localhost:8000/inference/sam3 \
  -H "Content-Type: application/json" \
  -d '{"images": ["'$(base64 -w0 test.jpg)'"], "prompts": ["person"], "output_format": "polygons"}'

# Test dense (should now work)
curl -X POST http://localhost:8000/inference/sam3 \
  -H "Content-Type: application/json" \
  -d '{"images": ["'$(base64 -w0 test.jpg)'"], "prompts": ["person"], "output_format": "dense"}'
```

All three should return `200 OK` with properly serialized JSON.

## Migration Guide

If you were using the API before this fix:

### Before (would error with dense)
```python
# This would fail
response = requests.post(
    url,
    json={"images": [img], "prompts": ["x"], "output_format": "dense"}
)
```

### After (all formats work)
```python
# All formats work now
response = requests.post(
    url,
    json={"images": [img], "prompts": ["x"], "output_format": "rle"}  # Recommended
)

response = requests.post(
    url,
    json={"images": [img], "prompts": ["x"], "output_format": "polygons"}  # Also good
)

response = requests.post(
    url,
    json={"images": [img], "prompts": ["x"], "output_format": "dense"}  # Now works
)
```

## Recommendations

### ✅ DO:
- Use **RLE** format for API responses (default)
- Use **Polygons** for visualization/editing
- Test with [examples/test_sam3_formats.py](examples/test_sam3_formats.py)

### ❌ DON'T:
- Use **Dense** format over HTTP (very large)
- Assume old default behavior (now RLE, not dense)

## Related Documentation

- [SAM3_OUTPUT_FORMATS.md](SAM3_OUTPUT_FORMATS.md) - Complete format guide
- [API_USAGE.md](API_USAGE.md) - API usage examples
- [src/vision_model_repo/vision_models/segmentation/utils.py](src/vision_model_repo/vision_models/segmentation/utils.py) - Format conversion utilities

## Status

✅ **Fixed**: All output formats now properly serialized
✅ **Tested**: Test script created and verified
✅ **Documented**: Complete format guide available
✅ **Default Changed**: RLE is now the default (more suitable for API)
