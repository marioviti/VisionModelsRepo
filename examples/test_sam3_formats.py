"""
Test different Sam3 output formats via the API.

This script demonstrates the three output formats:
1. RLE (recommended) - Compact, efficient for storage
2. Polygons (recommended) - Editable, human-readable
3. Dense (not recommended for API) - Large binary masks
"""

import requests
import base64
import json
from pathlib import Path


def test_sam3_formats(image_path: str, api_url: str = "http://localhost:8000"):
    """Test all three Sam3 output formats."""

    # Load and encode image
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    print(f"Testing Sam3 output formats with image: {image_path}")
    print("=" * 70)

    # Test prompts
    prompts = ["person", "object"]

    # Test 1: RLE Format (Recommended)
    print("\n1. RLE Format (Compact, efficient)")
    print("-" * 70)
    response = requests.post(
        f"{api_url}/inference/sam3",
        json={
            "images": [image_b64],
            "prompts": prompts,
            "threshold": 0.5,
            "output_format": "rle"
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Processing time: {result['processing_time']:.2f}s")
        print(f"Model: {result['model_id']}")

        # Show RLE structure
        if result['results'] and result['results'][0]:
            first_result = result['results'][0][0]
            print(f"Prompt: {first_result['prompt']}")

            if 'result' in first_result and 'masks_rle' in first_result['result']:
                rle_masks = first_result['result']['masks_rle']
                print(f"Number of masks: {len(rle_masks)}")
                if rle_masks:
                    print(f"RLE structure example: {list(rle_masks[0].keys())}")
                    print(f"RLE data (first 100 chars): {str(rle_masks[0])[:100]}...")
            else:
                print("No masks found")

        # Calculate response size
        response_size = len(response.content)
        print(f"Response size: {response_size:,} bytes ({response_size/1024:.1f} KB)")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

    # Test 2: Polygon Format (Recommended)
    print("\n2. Polygon Format (Editable, human-readable)")
    print("-" * 70)
    response = requests.post(
        f"{api_url}/inference/sam3",
        json={
            "images": [image_b64],
            "prompts": prompts,
            "threshold": 0.5,
            "output_format": "polygons"
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Processing time: {result['processing_time']:.2f}s")

        # Show polygon structure
        if result['results'] and result['results'][0]:
            first_result = result['results'][0][0]
            print(f"Prompt: {first_result['prompt']}")

            if 'result' in first_result and 'masks_polygon' in first_result['result']:
                polygon_masks = first_result['result']['masks_polygon']
                print(f"Number of masks: {len(polygon_masks)}")
                if polygon_masks:
                    print(f"Polygon structure: List of polygons")
                    print(f"First polygon vertices (sample): {str(polygon_masks[0])[:150]}...")
            else:
                print("No masks found")

        # Calculate response size
        response_size = len(response.content)
        print(f"Response size: {response_size:,} bytes ({response_size/1024:.1f} KB)")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

    # Test 3: Dense Format (Not recommended for API)
    print("\n3. Dense Format (Large binary masks - NOT RECOMMENDED)")
    print("-" * 70)
    print("⚠️  Warning: Dense masks return large nested arrays")
    response = requests.post(
        f"{api_url}/inference/sam3",
        json={
            "images": [image_b64],
            "prompts": prompts,
            "threshold": 0.5,
            "output_format": "dense"
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Processing time: {result['processing_time']:.2f}s")

        # Show dense structure
        if result['results'] and result['results'][0]:
            first_result = result['results'][0][0]
            print(f"Prompt: {first_result['prompt']}")

            if 'result' in first_result and 'masks' in first_result['result']:
                dense_masks = first_result['result']['masks']
                print(f"Number of masks: {len(dense_masks)}")
                if dense_masks:
                    mask_shape = f"{len(dense_masks)} x {len(dense_masks[0])} x {len(dense_masks[0][0])}"
                    print(f"Dense mask shape: {mask_shape} (instances x height x width)")
                    print(f"Sample values: {dense_masks[0][0][:10]}...")
            else:
                print("No masks found")

        # Calculate response size
        response_size = len(response.content)
        print(f"Response size: {response_size:,} bytes ({response_size/1024:.1f} KB)")
        print(f"⚠️  Dense format is {response_size/1024:.0f}x larger than RLE!")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

    # Summary
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("✅ Use 'rle' for most cases (compact, efficient)")
    print("✅ Use 'polygons' for editing or visualization")
    print("❌ Avoid 'dense' for API responses (very large)")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image
        test_images = [
            "test.jpg",
            "image.jpg",
            "../test.jpg",
            "../../test.jpg"
        ]

        image_path = None
        for img in test_images:
            if Path(img).exists():
                image_path = img
                break

        if not image_path:
            print("Usage: python test_sam3_formats.py <image_path>")
            print("\nNo test image found. Please provide an image path.")
            sys.exit(1)

    test_sam3_formats(image_path)
