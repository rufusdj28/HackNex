from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def _level_rating(value: float) -> str:
    if value >= 75:
        return "High"
    if value >= 50:
        return "Good"
    if value >= 30:
        return "Moderate"
    return "Low"


def evaluate_reconstruction_confidence(
    original_image: np.ndarray,
    reconstructed_image: np.ndarray,
    mask: np.ndarray,
    analysis_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Calculate contextual reconstruction confidence from empirical image properties.

    Evaluates:
    - Damaged area percentage & mask scale
    - Available surrounding context support
    - Boundary complexity
    - Seam boundary continuity (color/gradient transition at the mask perimeter)
    - Structural continuity (edge continuation across seam)
    - Texture consistency (frequency variance comparison between inpaint and surrounding context)

    Returns:
        Dict with keys:
            - score: float (0..100)
            - context_support: str ("High" | "Good" | "Moderate" | "Low")
            - boundary_continuity: str ("High" | "Good" | "Moderate" | "Low")
            - structural_continuity: str ("High" | "Good" | "Moderate" | "Low")
            - boundary_continuity_score: float
            - texture_consistency_score: float
            - heuristic_note: str
    """
    H, W = original_image.shape[:2]
    binary_mask = (mask > 0).astype(np.uint8)
    masked_pixels = int(np.count_nonzero(binary_mask))

    if masked_pixels == 0:
        return {
            "score": 100.0,
            "context_support": "High",
            "boundary_continuity": "High",
            "structural_continuity": "High",
            "boundary_continuity_score": 100.0,
            "texture_consistency_score": 100.0,
            "heuristic_note": "No damaged pixels specified.",
        }

    total_pixels = H * W
    area_pct = (masked_pixels / float(total_pixels)) * 100.0

    # 1. Damage Area Factor (smaller damage = higher confidence)
    area_score = float(max(0.0, 100.0 - min(100.0, area_pct * 1.6)))

    # 2. Context Ring and Surrounding Pixels
    kernel = np.ones((15, 15), dtype=np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    context_ring = (dilated_mask > 0) & (binary_mask == 0)
    context_pixels = int(np.count_nonzero(context_ring))

    context_ratio = context_pixels / float(max(1, masked_pixels))
    context_score = float(min(100.0, context_ratio * 35.0 + 20.0))

    # 3. Boundary Complexity
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    ys, xs = np.where(binary_mask > 0)
    bw = int(xs.max() - xs.min()) + 1
    bh = int(ys.max() - ys.min()) + 1
    box_perimeter = max(1.0, 2.0 * (bw + bh))
    complexity_pct = float(min(100.0, (perimeter / box_perimeter) * 100.0))
    boundary_score = float(max(10.0, 100.0 - min(80.0, complexity_pct * 0.7)))

    # 4. Boundary Seam Continuity (measure transition across mask boundary)
    # Inner border (1-2 px inside mask) vs Outer border (1-2 px outside mask)
    inner_erode = cv2.erode(binary_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    inner_boundary = (binary_mask > 0) & (inner_erode == 0)

    outer_dilate = cv2.dilate(binary_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    outer_boundary = (outer_dilate > 0) & (binary_mask == 0)

    if np.any(inner_boundary) and np.any(outer_boundary):
        inner_mean = reconstructed_image[inner_boundary].mean(axis=0)
        outer_mean = original_image[outer_boundary].mean(axis=0)
        seam_color_diff = float(np.mean(np.abs(inner_mean - outer_mean)))
        boundary_continuity_score = float(max(0.0, 100.0 - (seam_color_diff / 255.0) * 200.0))
    else:
        boundary_continuity_score = 75.0

    # 5. Texture & High-Frequency Consistency
    # Compare Laplacian variance of reconstructed area vs context ring
    recon_gray = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(recon_gray, cv2.CV_64F)

    mask_bool = binary_mask > 0
    if np.count_nonzero(mask_bool) > 16 and np.count_nonzero(context_ring) > 16:
        recon_var = float(np.var(laplacian[mask_bool]))
        ctx_var = float(np.var(laplacian[context_ring]))
        if ctx_var > 1e-4:
            ratio = min(recon_var, ctx_var) / max(recon_var, ctx_var)
            texture_score = float(np.clip(ratio * 100.0, 10.0, 100.0))
        else:
            texture_score = 80.0
    else:
        texture_score = 75.0

    # 6. Structural Edge Continuity
    gray_full = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_full, 100, 200)
    edge_density_context = (
        float(edges[context_ring].mean() / 255.0 * 100.0) if context_pixels else 0.0
    )
    edge_density_recon = (
        float(edges[mask_bool].mean() / 255.0 * 100.0) if masked_pixels else 0.0
    )

    if edge_density_context > 1.0:
        struct_ratio = min(edge_density_recon, edge_density_context) / max(
            edge_density_recon, edge_density_context
        )
        structural_score = float(np.clip(struct_ratio * 100.0, 20.0, 100.0))
    else:
        # Plain/smooth area
        structural_score = 85.0

    # Composite Contextual Confidence Score
    confidence_score = (
        0.25 * area_score
        + 0.20 * context_score
        + 0.20 * boundary_continuity_score
        + 0.15 * structural_score
        + 0.10 * boundary_score
        + 0.10 * texture_score
    )
    confidence_score = float(np.clip(confidence_score, 0.0, 100.0))

    return {
        "score": confidence_score,
        "context_support": _level_rating(context_score),
        "boundary_continuity": _level_rating(boundary_continuity_score),
        "structural_continuity": _level_rating(structural_score),
        "boundary_continuity_score": boundary_continuity_score,
        "texture_consistency_score": texture_score,
        "heuristic_note": "Contextual reconstruction confidence derived from area, context support, seam continuity, and structural consistency.",
    }
