"""
Visualization utilities for segmentation outputs.

Functions to visualize masks, create overlays, and display results from
Sam3 segmentation (RLE, polygons, and dense formats).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import zlib
from typing import List, Dict, Any, Optional, Tuple, Union


# ============================================================================
# RLE Conversion Functions
# ============================================================================

def rle_to_dense(rle_data: Dict[str, Any]) -> np.ndarray:
    """
    Convert RLE mask to dense binary mask.

    Args:
        rle_data: Dictionary with 'size' [H, W] and 'counts' (base64 compressed)

    Returns:
        Binary mask as numpy array (H, W)
    """
    size = rle_data['size']  # [height, width]
    counts_compressed = rle_data['counts']

    # Decode base64 and decompress
    counts_bytes = base64.b64decode(counts_compressed)
    counts_decompressed = zlib.decompress(counts_bytes)

    # Parse RLE counts (simple alternating runs of 0s and 1s)
    # Format: starts with 0, alternates between 0 and 1
    counts = np.frombuffer(counts_decompressed, dtype=np.uint32)

    # Decode RLE to flat binary array
    height, width = size
    total_pixels = height * width
    mask_flat = np.zeros(total_pixels, dtype=np.uint8)

    current_idx = 0
    current_val = 0  # Starts with 0

    for count in counts:
        mask_flat[current_idx:current_idx + count] = current_val
        current_idx += count
        current_val = 1 - current_val  # Alternate between 0 and 1

    # Reshape to image dimensions
    mask = mask_flat.reshape((height, width))

    return mask


def polygon_to_dense(polygon: List[List[float]], size: Tuple[int, int]) -> np.ndarray:
    """
    Convert polygon to dense binary mask.

    Args:
        polygon: List of [x, y] coordinates
        size: (height, width) of output mask

    Returns:
        Binary mask as numpy array (H, W)
    """
    height, width = size

    # Create PIL image for drawing
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    # Convert polygon to tuple format
    if len(polygon) > 0:
        polygon_tuples = [tuple(point) for point in polygon]
        draw.polygon(polygon_tuples, outline=1, fill=1)

    # Convert to numpy
    mask = np.array(img, dtype=np.uint8)

    return mask


# ============================================================================
# Visualization Functions
# ============================================================================

def get_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """
    Generate n visually distinct colors.

    Args:
        n: Number of colors to generate

    Returns:
        List of RGB tuples (values 0-1)
    """
    import colorsys

    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    return colors


def visualize_masks_overlay(
    image: Union[np.ndarray, Image.Image],
    masks: List[np.ndarray],
    labels: Optional[List[str]] = None,
    alpha: float = 0.5,
    colors: Optional[List[Tuple[float, float, float]]] = None,
    show_boxes: bool = True,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize masks as colored overlay on image.

    Args:
        image: Input image (numpy array or PIL Image)
        masks: List of binary masks (H, W)
        labels: Optional labels for each mask
        alpha: Transparency of mask overlay (0-1)
        colors: Optional list of RGB colors for masks
        show_boxes: Whether to draw bounding boxes
        show_labels: Whether to show labels
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Generate colors if not provided
    if colors is None:
        colors = get_distinct_colors(len(masks))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    ax.axis('off')

    # Overlay each mask
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]

        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[:, :, :3] = color
        colored_mask[:, :, 3] = mask * alpha

        # Overlay
        ax.imshow(colored_mask)

        # Draw bounding box
        if show_boxes:
            # Find bounding box from mask
            rows, cols = np.where(mask > 0)
            if len(rows) > 0:
                y1, y2 = rows.min(), rows.max()
                x1, x2 = cols.min(), cols.max()

                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)

                # Add label
                if show_labels and labels and i < len(labels):
                    label = labels[i]
                    ax.text(
                        x1, y1 - 5, label,
                        color='white', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7)
                    )

    # Add legend
    if labels and show_labels:
        legend_patches = [
            mpatches.Patch(color=colors[i % len(colors)], label=labels[i])
            for i in range(len(labels))
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def visualize_masks_grid(
    image: Union[np.ndarray, Image.Image],
    masks: List[np.ndarray],
    labels: Optional[List[str]] = None,
    cols: int = 3,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize each mask separately in a grid.

    Args:
        image: Input image
        masks: List of binary masks
        labels: Optional labels for each mask
        cols: Number of columns in grid
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    n_masks = len(masks)
    rows = (n_masks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]

        if i < n_masks:
            # Show image with single mask
            ax.imshow(image)
            ax.imshow(masks[i], alpha=0.5, cmap='jet')

            if labels and i < len(labels):
                ax.set_title(labels[i], fontsize=12, fontweight='bold')

            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    return fig


def visualize_masks_side_by_side(
    image: Union[np.ndarray, Image.Image],
    masks: List[np.ndarray],
    labels: Optional[List[str]] = None,
    alpha: float = 0.5
) -> plt.Figure:
    """
    Show original image and overlay side by side.

    Args:
        image: Input image
        masks: List of binary masks
        labels: Optional labels
        alpha: Overlay transparency

    Returns:
        Matplotlib figure
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    colors = get_distinct_colors(len(masks))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Overlay
    axes[1].imshow(image)
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[:, :, :3] = color
        colored_mask[:, :, 3] = mask * alpha
        axes[1].imshow(colored_mask)

    axes[1].set_title(f'Segmentation ({len(masks)} masks)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Legend
    if labels:
        legend_patches = [
            mpatches.Patch(color=colors[i % len(colors)], label=labels[i])
            for i in range(len(labels))
        ]
        axes[1].legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


def visualize_polygons(
    image: Union[np.ndarray, Image.Image],
    polygons: List[List[List[float]]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[float, float, float]]] = None,
    linewidth: float = 2,
    fill_alpha: float = 0.3,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize polygon masks.

    Args:
        image: Input image
        polygons: List of polygons (each polygon is list of [x, y] points)
        labels: Optional labels
        colors: Optional colors
        linewidth: Polygon outline width
        fill_alpha: Fill transparency
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Generate colors if not provided
    if colors is None:
        colors = get_distinct_colors(len(polygons))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    ax.axis('off')

    # Draw each polygon
    for i, polygon in enumerate(polygons):
        if len(polygon) == 0:
            continue

        color = colors[i % len(colors)]

        # Convert to numpy array
        poly_array = np.array(polygon)

        # Create polygon patch
        poly_patch = mpatches.Polygon(
            poly_array,
            closed=True,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=(*color, fill_alpha)
        )
        ax.add_patch(poly_patch)

    # Legend
    if labels:
        legend_patches = [
            mpatches.Patch(color=colors[i % len(colors)], label=labels[i])
            for i in range(len(labels))
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================================
# High-Level Visualization Functions for API Results
# ============================================================================

def visualize_sam3_results(
    image: Union[np.ndarray, Image.Image, str],
    results: List[Dict[str, Any]],
    output_format: str = "rle",
    mode: str = "overlay",
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize Sam3 API results.

    Args:
        image: Input image (array, PIL Image, or file path)
        results: Results from API (single image results)
        output_format: Format of masks ('rle', 'polygons', or 'dense')
        mode: Visualization mode ('overlay', 'grid', or 'side_by_side')
        alpha: Overlay transparency
        figsize: Figure size

    Returns:
        Matplotlib figure

    Example:
        >>> result = api_response['results'][0]  # First image
        >>> fig = visualize_sam3_results(image, result, output_format='rle')
        >>> plt.show()
    """
    # Load image if path
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert image to numpy for visualization
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # Extract masks and labels from results
    all_masks = []
    all_labels = []

    for entry in results:
        prompt = entry.get('prompt', 'unknown')
        result_data = entry.get('result', {})

        if output_format == 'rle':
            # RLE masks
            masks_rle = result_data.get('masks_rle', [])
            for i, rle_mask in enumerate(masks_rle):
                mask = rle_to_dense(rle_mask)
                all_masks.append(mask)
                all_labels.append(f"{prompt} #{i+1}")

        elif output_format == 'polygons':
            # Polygon masks
            masks_polygon = result_data.get('masks_polygon', [])
            for i, polygon in enumerate(masks_polygon):
                mask = polygon_to_dense(polygon, (height, width))
                all_masks.append(mask)
                all_labels.append(f"{prompt} #{i+1}")

        elif output_format == 'dense':
            # Dense masks (already in correct format)
            masks_dense = result_data.get('masks', [])
            for i, mask in enumerate(masks_dense):
                # Convert to numpy if needed
                if isinstance(mask, list):
                    mask = np.array(mask)
                all_masks.append(mask)
                all_labels.append(f"{prompt} #{i+1}")

    # Visualize based on mode
    if mode == 'overlay':
        return visualize_masks_overlay(
            image_np, all_masks, all_labels, alpha=alpha, figsize=figsize
        )
    elif mode == 'grid':
        return visualize_masks_grid(
            image_np, all_masks, all_labels, figsize=figsize
        )
    elif mode == 'side_by_side':
        return visualize_masks_side_by_side(
            image_np, all_masks, all_labels, alpha=alpha
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'overlay', 'grid', or 'side_by_side'")


def create_mask_montage(
    image: Union[np.ndarray, Image.Image],
    results: List[Dict[str, Any]],
    output_format: str = "rle"
) -> plt.Figure:
    """
    Create a comprehensive montage showing multiple visualization modes.

    Args:
        image: Input image
        results: Sam3 results
        output_format: Mask format

    Returns:
        Matplotlib figure with 3 panels
    """
    # Convert image
    if isinstance(image, Image.Image):
        image = np.array(image)

    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Original image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Overlay
    ax2 = fig.add_subplot(1, 3, 2)
    height, width = image.shape[:2]
    all_masks = []
    all_labels = []

    for entry in results:
        prompt = entry.get('prompt', 'unknown')
        result_data = entry.get('result', {})

        if output_format == 'rle':
            masks_rle = result_data.get('masks_rle', [])
            for i, rle_mask in enumerate(masks_rle):
                mask = rle_to_dense(rle_mask)
                all_masks.append(mask)
                all_labels.append(f"{prompt}")
        elif output_format == 'polygons':
            masks_polygon = result_data.get('masks_polygon', [])
            for i, polygon in enumerate(masks_polygon):
                mask = polygon_to_dense(polygon, (height, width))
                all_masks.append(mask)
                all_labels.append(f"{prompt}")
        elif output_format == 'dense':
            masks_dense = result_data.get('masks', [])
            for mask in masks_dense:
                if isinstance(mask, list):
                    mask = np.array(mask)
                all_masks.append(mask)

    # Draw overlay
    ax2.imshow(image)
    colors = get_distinct_colors(len(all_masks))
    for i, mask in enumerate(all_masks):
        color = colors[i]
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[:, :, :3] = color
        colored_mask[:, :, 3] = mask * 0.5
        ax2.imshow(colored_mask)

    ax2.set_title(f'Overlay ({len(all_masks)} masks)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Panel 3: Masks only
    ax3 = fig.add_subplot(1, 3, 3)
    combined_mask = np.zeros_like(image)
    for i, mask in enumerate(all_masks):
        color = (np.array(colors[i]) * 255).astype(np.uint8)
        combined_mask[mask > 0] = color

    ax3.imshow(combined_mask)
    ax3.set_title('Masks Only', fontsize=14, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()
    return fig


# ============================================================================
# Utility Functions
# ============================================================================

def save_visualization(fig: plt.Figure, output_path: str, dpi: int = 150):
    """
    Save figure to file.

    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: Resolution
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")


def show_mask_statistics(results: List[Dict[str, Any]], output_format: str = "rle"):
    """
    Print statistics about detected masks.

    Args:
        results: Sam3 results
        output_format: Mask format
    """
    print("Mask Statistics")
    print("=" * 50)

    total_masks = 0
    for entry in results:
        prompt = entry.get('prompt', 'unknown')
        result_data = entry.get('result', {})

        if output_format == 'rle':
            n_masks = len(result_data.get('masks_rle', []))
        elif output_format == 'polygons':
            n_masks = len(result_data.get('masks_polygon', []))
        elif output_format == 'dense':
            n_masks = len(result_data.get('masks', []))
        else:
            n_masks = 0

        print(f"Prompt '{prompt}': {n_masks} masks")

        # Show IoU scores if available
        if 'iou_scores' in result_data:
            scores = result_data['iou_scores']
            if isinstance(scores, list) and len(scores) > 0:
                print(f"  IoU scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={np.mean(scores):.3f}")

        total_masks += n_masks

    print("=" * 50)
    print(f"Total: {total_masks} masks detected")
