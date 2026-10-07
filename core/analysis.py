from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def _level(value: float) -> str:
    if value >= 75:
        return "High"
    if value >= 45:
        return "Medium"
    return "Low"


def analyze_damage(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Extract lightweight, transparent structural features from image and damage mask."""
    mask_binary = (mask > 0).astype(np.uint8)
    height, width = mask_binary.shape
    masked_pixels = int(mask_binary.sum())
    if masked_pixels == 0:
        raise ValueError("Paint a damage region before analyzing the image.")

    ys, xs = np.where(mask_binary > 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    box_width, box_height = x2 - x1, y2 - y1
    area_pct = masked_pixels / float(width * height) * 100.0

    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    box_perimeter = max(1.0, 2.0 * (box_width + box_height))
    boundary_complexity = float(min(100.0, perimeter / box_perimeter * 100.0))

    kernel = np.ones((25, 25), dtype=np.uint8)
    dilated = cv2.dilate(mask_binary, kernel)
    context_ring = (dilated > 0) & (mask_binary == 0)
    context_pixels = int(context_ring.sum())

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(edges[context_ring].mean() / 255.0 * 100.0) if context_pixels else 0.0

    min_line_length = max(10, min(width, height) // 20)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=30,
        minLineLength=min_line_length,
        maxLineGap=8,
    )
    line_count = int(len(lines)) if lines is not None else 0
    structural_content = float(min(100.0, line_count * 5.0 + edge_density * 2.0))

    # Model recommendation heuristic
    if area_pct >= 25.0:
        recommendation = "MAT"
        reason = "Large damaged area"
    elif structural_content >= 45.0 or edge_density >= 12.0:
        recommendation = "ZITS"
        reason = "Strong nearby edge/line structure"
    else:
        recommendation = "LaMa"
        reason = "Small or general-purpose damage"

    return {
        "mask_area_pct": area_pct,
        "mask_width": box_width,
        "mask_height": box_height,
        "mask_aspect_ratio": box_width / float(max(1, box_height)),
        "boundary_complexity": boundary_complexity,
        "edge_density": edge_density,
        "line_count": line_count,
        "structural_content": structural_content,
        "context_pixels": context_pixels,
        "context_support": _level(min(100.0, context_pixels / max(1, masked_pixels) * 25.0)),
        "recommended_model": recommendation,
        "recommendation_reason": reason,
    }
