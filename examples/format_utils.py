import base64
import zlib
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# ---------------------------
# RLE (COCO/CVAT) <-> dense
# ---------------------------

def dense_to_cvat_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Dense (H,W) binary mask -> CVAT/COCO-style compressed RLE (zlib+base64 of uint32 counts).
    COCO uses Fortran order (column-major), i.e. flatten(mask.T).

    Returns:
      {"encoding":"rle","size":[H,W],"counts":"..."}
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got shape {mask.shape}")
    m = (mask > 0).astype(np.uint8)

    # COCO expects column-major order
    flat = m.T.reshape(-1)

    counts: List[int] = []
    prev = 0
    run = 0
    for v in flat:
        if int(v) == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = int(v)
    counts.append(run)

    counts_bytes = np.asarray(counts, dtype=np.uint32).tobytes()
    compressed = zlib.compress(counts_bytes)
    b64 = base64.b64encode(compressed).decode("ascii")

    return {"encoding": "rle", "size": [int(m.shape[0]), int(m.shape[1])], "counts": b64}


def cvat_rle_to_dense(rle: Dict[str, Any]) -> np.ndarray:
    """
    CVAT/COCO-style compressed RLE -> dense (H,W) uint8 mask with values {0,1}.
    """
    if rle.get("encoding") != "rle":
        raise ValueError("rle['encoding'] must be 'rle'")

    h, w = rle["size"]
    counts_b64 = rle["counts"]

    counts_bytes = zlib.decompress(base64.b64decode(counts_b64))
    counts = np.frombuffer(counts_bytes, dtype=np.uint32)

    # Reconstruct flat array in COCO order (column-major on original),
    # then reshape back to (W,H) and transpose -> (H,W)
    total = int(counts.sum())
    flat = np.empty(total, dtype=np.uint8)

    idx = 0
    val = 0
    for c in counts:
        c = int(c)
        if c:
            flat[idx : idx + c] = val
            idx += c
        val ^= 1

    # flat is for mask.T flattened, so reshape to (W,H) then transpose
    mT = flat.reshape((w, h))
    m = mT.T
    return m.astype(np.uint8)


# ---------------------------
# dense <-> polygons
# ---------------------------

def dense_to_polygons(mask: np.ndarray, epsilon: float = 1.5) -> List[List[List[float]]]:
    """
    Dense binary mask -> list of polygons (external contours).
    Each polygon is [[x,y], [x,y], ...].
    Requires opencv-python.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("dense_to_polygons requires opencv-python (pip install opencv-python)") from e

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got shape {mask.shape}")

    m = ((mask > 0).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[List[List[float]]] = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        poly = approx.reshape(-1, 2).astype(float).tolist()
        if len(poly) >= 3:
            polys.append(poly)
    return polys


def polygons_to_dense(
    polygons: List[List[List[float]]],
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Polygons -> dense binary mask (H,W) uint8 {0,1}.
    Requires opencv-python.

    Args:
      polygons: list of polygons, each poly = [[x,y],...]
      size: (H,W)
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("polygons_to_dense requires opencv-python (pip install opencv-python)") from e

    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)

    if not polygons:
        return mask

    pts_list = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        pts_list.append(pts)

    if pts_list:
        cv2.fillPoly(mask, pts_list, 1)

    return mask


# ---------------------------
# Convenience bridges
# ---------------------------

def rle_to_polygons(rle: Dict[str, Any], epsilon: float = 1.5) -> List[List[List[float]]]:
    """RLE -> dense -> polygons"""
    return dense_to_polygons(cvat_rle_to_dense(rle), epsilon=epsilon)


def polygons_to_rle(polygons: List[List[List[float]]], size: Tuple[int, int]) -> Dict[str, Any]:
    """Polygons -> dense -> RLE"""
    dense = polygons_to_dense(polygons, size=size)
    return dense_to_cvat_rle(dense)
