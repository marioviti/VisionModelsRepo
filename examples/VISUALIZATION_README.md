# Segmentation Visualization Utilities

Comprehensive visualization tools for Sam3 segmentation outputs.

## Quick Start

```python
from visualization_utils import visualize_sam3_results
from PIL import Image

# Load image
image = Image.open("photo.jpg")

# Get API results (assume you have this from API call)
results = api_response['results'][0]  # First image

# Visualize!
fig = visualize_sam3_results(
    image,
    results,
    output_format='rle',  # or 'polygons' or 'dense'
    mode='overlay'        # or 'grid' or 'side_by_side'
)
plt.show()
```

## Features

### ✅ Multiple Visualization Modes

1. **Overlay** - Colored masks overlaid on image (most common)
2. **Grid** - Each mask shown separately in a grid
3. **Side-by-side** - Original and segmented images side by side
4. **Montage** - Comprehensive 3-panel view

### ✅ Format Support

- **RLE** (Run-Length Encoding) - Compact format
- **Polygons** - Vector format
- **Dense** - Binary masks

### ✅ Conversion Functions

- `rle_to_dense()` - Convert RLE to binary mask
- `polygon_to_dense()` - Convert polygon to binary mask

### ✅ Customization

- Custom colors
- Adjustable transparency
- Bounding boxes
- Labels and legends
- Multiple figure sizes

## Available Functions

### High-Level Functions (Recommended)

#### `visualize_sam3_results()`

Main visualization function - handles all formats automatically.

```python
fig = visualize_sam3_results(
    image,              # PIL Image, numpy array, or path
    results,            # API results for one image
    output_format='rle', # 'rle', 'polygons', or 'dense'
    mode='overlay',     # 'overlay', 'grid', or 'side_by_side'
    alpha=0.5,          # Transparency (0-1)
    figsize=(12, 8)     # Figure size
)
```

**Modes:**
- `'overlay'` - Colored masks on top of image (default)
- `'grid'` - Individual masks in grid layout
- `'side_by_side'` - Original vs segmented

#### `create_mask_montage()`

Create 3-panel comprehensive view.

```python
fig = create_mask_montage(
    image,
    results,
    output_format='rle'
)
```

Shows:
1. Original image
2. Overlay with all masks
3. Masks only (colored)

#### `show_mask_statistics()`

Print mask statistics.

```python
show_mask_statistics(results, output_format='rle')
```

Output:
```
Mask Statistics
==================================================
Prompt 'person': 3 masks
  IoU scores: min=0.850, max=0.950, avg=0.900
Prompt 'car': 2 masks
  IoU scores: min=0.820, max=0.890, avg=0.855
==================================================
Total: 5 masks detected
```

### Low-Level Functions (Advanced)

#### `visualize_masks_overlay()`

Overlay masks with custom options.

```python
fig = visualize_masks_overlay(
    image,              # numpy array or PIL Image
    masks,              # List of binary masks (H, W)
    labels=['person', 'car'],  # Optional labels
    alpha=0.5,          # Transparency
    colors=None,        # Optional custom colors
    show_boxes=True,    # Draw bounding boxes
    show_labels=True,   # Show labels
    figsize=(12, 8)
)
```

#### `visualize_masks_grid()`

Show masks in grid layout.

```python
fig = visualize_masks_grid(
    image,
    masks,
    labels=['mask1', 'mask2'],
    cols=3,             # Columns in grid
    figsize=(15, 10)
)
```

#### `visualize_polygons()`

Visualize polygon masks.

```python
fig = visualize_polygons(
    image,
    polygons,           # List of polygons
    labels=None,
    colors=None,
    linewidth=2,
    fill_alpha=0.3,
    figsize=(12, 8)
)
```

### Conversion Functions

#### `rle_to_dense()`

Convert RLE mask to dense binary array.

```python
rle_mask = result['result']['masks_rle'][0]
dense_mask = rle_to_dense(rle_mask)
# Returns: numpy array (H, W) with 0s and 1s
```

#### `polygon_to_dense()`

Convert polygon to dense binary array.

```python
polygon = result['result']['masks_polygon'][0]
size = (height, width)
dense_mask = polygon_to_dense(polygon, size)
# Returns: numpy array (H, W) with 0s and 1s
```

## Complete Example

```python
import requests
import base64
from PIL import Image
import matplotlib.pyplot as plt
from visualization_utils import (
    visualize_sam3_results,
    create_mask_montage,
    show_mask_statistics
)

# 1. Load image
image = Image.open("photo.jpg")

# 2. Get segmentation from API
with open("photo.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/inference/sam3",
    json={
        "images": [img_b64],
        "prompts": ["person", "car", "tree"],
        "output_format": "rle"
    }
)

results = response.json()['results'][0]

# 3. Show statistics
show_mask_statistics(results, output_format='rle')

# 4. Visualize - Overlay
fig = visualize_sam3_results(
    image,
    results,
    output_format='rle',
    mode='overlay',
    alpha=0.5
)
plt.show()

# 5. Visualize - Grid
fig = visualize_sam3_results(
    image,
    results,
    output_format='rle',
    mode='grid'
)
plt.show()

# 6. Visualize - Montage
fig = create_mask_montage(image, results, output_format='rle')
plt.show()

# 7. Save
fig.savefig('segmentation_result.png', dpi=150, bbox_inches='tight')
```

## Examples

### Example 1: Quick Overlay Visualization

```python
from visualization_utils import visualize_sam3_results
from PIL import Image

image = Image.open("photo.jpg")
results = api_response['results'][0]

fig = visualize_sam3_results(image, results, output_format='rle')
plt.show()
```

### Example 2: Grid View for Multiple Masks

```python
fig = visualize_sam3_results(
    image,
    results,
    output_format='rle',
    mode='grid',
    figsize=(18, 12)
)
plt.show()
```

### Example 3: Custom Colors

```python
from visualization_utils import visualize_masks_overlay, rle_to_dense

# Extract masks
masks = [rle_to_dense(m) for m in results[0]['result']['masks_rle']]

# Custom colors
colors = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
]

fig = visualize_masks_overlay(
    image,
    masks,
    labels=['Person', 'Car', 'Tree'],
    colors=colors,
    alpha=0.6
)
plt.show()
```

### Example 4: Analyze Individual Masks

```python
from visualization_utils import rle_to_dense
import numpy as np

# Get first mask
rle_mask = results[0]['result']['masks_rle'][0]
mask = rle_to_dense(rle_mask)

# Analyze
area = mask.sum()
coverage = (area / mask.size) * 100
rows, cols = np.where(mask > 0)
bbox = (cols.min(), rows.min(), cols.max(), rows.max())

print(f"Mask area: {area:,} pixels ({coverage:.2f}%)")
print(f"Bounding box: {bbox}")

# Visualize
plt.figure(figsize=(8, 6))
plt.imshow(mask, cmap='gray')
plt.title(f"Mask Area: {area:,} pixels")
plt.axis('off')
plt.show()
```

### Example 5: Export Masks

```python
from visualization_utils import rle_to_dense
from PIL import Image

# Convert and save each mask
for i, entry in enumerate(results):
    masks_rle = entry['result']['masks_rle']

    for j, rle_mask in enumerate(masks_rle):
        # Convert to dense
        mask = rle_to_dense(rle_mask)

        # Save as PNG
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(f"mask_{i}_{j}.png")
```

## Jupyter Notebook Demo

See [visualization_demo.ipynb](visualization_demo.ipynb) for a complete interactive demo.

## Tips

### Performance

- RLE format is most efficient for API requests
- Convert to dense only when needed for visualization
- Use `mode='overlay'` for quick preview
- Use `mode='grid'` for detailed analysis

### Customization

- Adjust `alpha` (0.3-0.7) for best visibility
- Use `figsize` to control output size
- Custom colors: provide RGB tuples (0-1 range)
- Set `show_boxes=False` to hide bounding boxes

### Common Issues

**Q: Masks not showing?**
- Check `output_format` matches your API response
- Verify results are not empty
- Try increasing `alpha` value

**Q: Colors look wrong?**
- Colors are RGB tuples with values 0-1, not 0-255
- Use `get_distinct_colors(n)` for automatic colors

**Q: Conversion errors?**
- Ensure mask format matches conversion function
- RLE needs 'size' and 'counts' fields
- Polygons need (height, width) size parameter

## Files

| File | Description |
|------|-------------|
| [visualization_utils.py](visualization_utils.py) | Main utility module |
| [visualization_demo.ipynb](visualization_demo.ipynb) | Interactive demo notebook |
| [VISUALIZATION_README.md](VISUALIZATION_README.md) | This file |

## See Also

- [SAM3_OUTPUT_FORMATS.md](../SAM3_OUTPUT_FORMATS.md) - Output format guide
- [IMAGE_COMPRESSION_GUIDE.md](../IMAGE_COMPRESSION_GUIDE.md) - Compression guide
- [API_USAGE.md](../API_USAGE.md) - API usage examples
