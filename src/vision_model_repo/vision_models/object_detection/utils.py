from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Literal, List, Optional

Number = float

Fmt = Literal["xyxy", "xywh", "cxcywh", "yolo"]


@dataclass(frozen=True)
class ImageSize:
    w: int
    h: int


def _to_list4(x: Iterable[Number]) -> List[float]:
    x = list(x)
    if len(x) != 4:
        raise ValueError(f"Expected 4 numbers, got {len(x)}")
    return [float(v) for v in x]


def _clip_xyxy(xyxy: List[float], size: ImageSize) -> List[float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(x1, float(size.w)))
    x2 = max(0.0, min(x2, float(size.w)))
    y1 = max(0.0, min(y1, float(size.h)))
    y2 = max(0.0, min(y2, float(size.h)))
    # ensure ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _xywh_to_xyxy(xywh: List[float]) -> List[float]:
    x, y, w, h = xywh
    return [x, y, x + w, y + h]


def _cxcywh_to_xyxy(cxcywh: List[float]) -> List[float]:
    cx, cy, w, h = cxcywh
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _xyxy_to_xywh(xyxy: List[float]) -> List[float]:
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def _xyxy_to_cxcywh(xyxy: List[float]) -> List[float]:
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    return [x1 + w / 2.0, y1 + h / 2.0, w, h]


def _yolo_to_xyxy(yolo: List[float], size: ImageSize) -> List[float]:
    # yolo is normalized cx,cy,w,h
    cx_n, cy_n, w_n, h_n = yolo
    cx = cx_n * size.w
    cy = cy_n * size.h
    w = w_n * size.w
    h = h_n * size.h
    return _cxcywh_to_xyxy([cx, cy, w, h])


def _xyxy_to_yolo(xyxy: List[float], size: ImageSize) -> List[float]:
    cx, cy, w, h = _xyxy_to_cxcywh(xyxy)
    # normalize
    return [cx / size.w, cy / size.h, w / size.w, h / size.h]


def bbox_convert(
    box: Iterable[Number],
    src: Fmt,
    dst: Fmt,
    *,
    size: Optional[ImageSize] = None,
    clip: bool = True,
) -> List[float]:
    """
    Convert bbox between formats.

    Args:
        box: 4 numbers in `src` format.
        src: "xyxy" | "xywh" | "cxcywh" | "yolo"
        dst: "xyxy" | "xywh" | "cxcywh" | "yolo"
        size: required when src or dst is "yolo" OR when clip=True
        clip: if True, clip to image bounds (requires size)

    Returns:
        list[float] box in `dst` format.
    """
    b = _to_list4(box)

    needs_size = (src == "yolo") or (dst == "yolo") or clip
    if needs_size and size is None:
        raise ValueError("`size=ImageSize(w,h)` is required for yolo conversions and/or clipping.")

    # 1) src -> xyxy (absolute pixels)
    if src == "xyxy":
        xyxy = b
    elif src == "xywh":
        xyxy = _xywh_to_xyxy(b)
    elif src == "cxcywh":
        xyxy = _cxcywh_to_xyxy(b)
    elif src == "yolo":
        xyxy = _yolo_to_xyxy(b, size=size)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown src format: {src}")

    # 2) clip in xyxy space
    if clip:
        xyxy = _clip_xyxy(xyxy, size=size)  # type: ignore[arg-type]

    # 3) xyxy -> dst
    if dst == "xyxy":
        return xyxy
    if dst == "xywh":
        return _xyxy_to_xywh(xyxy)
    if dst == "cxcywh":
        return _xyxy_to_cxcywh(xyxy)
    if dst == "yolo":
        return _xyxy_to_yolo(xyxy, size=size)  # type: ignore[arg-type]

    raise ValueError(f"Unknown dst format: {dst}")
