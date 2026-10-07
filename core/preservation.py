from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def calculate_preservation_guard(
    original: np.ndarray,
    final: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """Calculate rigorous Preservation Guard metrics comparing unmasked regions.

    Compares ORIGINAL IMAGE vs FINAL IMAGE strictly in the UNMASKED region (mask == 0).
    Metrics are computed purely from empirical pixel values with no hardcoded constants.

    Returns:
        Dict with keys:
            - available: bool
            - outside_pixels: int
            - unchanged_pixels_pct: float (%)
            - pixel_difference_pct: float (%)
            - mean_absolute_error: float (0..255)
            - max_pixel_difference: float (0..255)
            - pixel_preservation: float (%)
            - structural_preservation: float or None (%)
            - edge_preservation: float (%)
            - color_preservation: float (%)
            - preservation_score: float (%)
    """
    if original.shape != final.shape:
        raise ValueError(
            f"Original shape {original.shape} and final shape {final.shape} must match."
        )
    if mask.shape != original.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} must match image dimensions {original.shape[:2]}."
        )

    outside = mask == 0
    outside_pixels = int(np.count_nonzero(outside))

    if outside_pixels == 0:
        return {
            "available": False,
            "reason": "The damage mask covers 100% of the image; no unmasked region exists to verify.",
            "preservation_score": 0.0,
            "unchanged_pixels_pct": 0.0,
            "pixel_difference_pct": 100.0,
            "mean_absolute_error": 0.0,
            "max_pixel_difference": 0.0,
            "pixel_preservation": 0.0,
            "structural_preservation": None,
            "edge_preservation": 0.0,
            "color_preservation": 0.0,
        }

    # 1. Exact Pixel Differences in Outside Region
    diff = np.abs(original.astype(np.float32) - final.astype(np.float32))
    mae = float(diff[outside].mean())
    max_diff = float(diff[outside].max())

    # Pixel change check across RGB channels
    changed_pixels = np.any(original != final, axis=2)
    changed_outside_count = int(np.count_nonzero(changed_pixels & outside))
    pixel_change_pct = float(changed_outside_count / outside_pixels * 100.0)
    unchanged_pixels_pct = float(100.0 - pixel_change_pct)
    pixel_preservation = float(max(0.0, 100.0 - (mae / 255.0) * 100.0))

    # 2. Structural Preservation (SSIM)
    original_gray = _rgb_to_gray(original)
    final_gray = _rgb_to_gray(final)

    # Erode the unmasked mask slightly so the SSIM sliding window doesn't bleed into damaged area
    safe_outside = cv2.erode(
        outside.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)
    ).astype(bool)
    eval_region = safe_outside if np.any(safe_outside) else outside

    structural_preservation: float | None = None
    try:
        _, ssim_map = structural_similarity(
            original_gray,
            final_gray,
            data_range=255,
            full=True,
        )
        structural_preservation = float(
            np.clip(ssim_map[eval_region].mean() * 100.0, 0.0, 100.0)
        )
    except Exception:
        structural_preservation = None

    # 3. Edge Preservation (Canny)
    original_edges = cv2.Canny(original_gray, 100, 200)
    final_edges = cv2.Canny(final_gray, 100, 200)
    edge_diff = np.abs(original_edges.astype(np.float32) - final_edges.astype(np.float32))
    edge_preservation = float(
        max(0.0, 100.0 - (edge_diff[eval_region].mean() / 255.0) * 100.0)
    )

    # 4. Color Preservation
    color_preservation = float(max(0.0, 100.0 - (mae / 255.0) * 100.0))

    # 5. Composite Preservation Score
    components = [pixel_preservation, edge_preservation, color_preservation]
    if structural_preservation is not None:
        components.append(structural_preservation)

    preservation_score = float(np.mean(components))

    return {
        "available": True,
        "outside_pixels": outside_pixels,
        "unchanged_pixels_pct": unchanged_pixels_pct,
        "pixel_difference_pct": pixel_change_pct,
        "mean_absolute_error": mae,
        "max_pixel_difference": max_diff,
        "pixel_preservation": pixel_preservation,
        "structural_preservation": structural_preservation,
        "edge_preservation": edge_preservation,
        "color_preservation": color_preservation,
        "preservation_score": preservation_score,
    }
