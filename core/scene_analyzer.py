from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.preprocessing import compute_adaptive_crop_coords, extract_mask_bbox


# Transparent profile classification thresholds
THRESHOLD_TEXTURE_LOW = 0.20
THRESHOLD_TEXTURE_HIGH = 0.55

THRESHOLD_STRUCTURE_LOW = 0.25
THRESHOLD_STRUCTURE_HIGH = 0.55

THRESHOLD_DAMAGE_TINY = 0.01
THRESHOLD_DAMAGE_SMALL = 0.08
THRESHOLD_DAMAGE_MEDIUM = 0.25


@dataclass
class SceneAnalysisResult:
    """Structured result returned by the Structure / Scene Analyzer."""
    image: Dict[str, int]
    mask: Dict[str, Any]
    context: Dict[str, Any]
    edges: Dict[str, Any]
    lines: Dict[str, Any]
    texture: Dict[str, Any]
    color: Dict[str, Any]
    structure: Dict[str, Any]
    scene_profile: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_orientation_entropy(angles: List[float], num_bins: int = 18) -> float:
    """Calculate normalized Shannon entropy (0..1) of line angle distribution."""
    if not angles:
        return 0.0
    # Bin angles from 0 to 180 degrees into num_bins
    counts, _ = np.histogram(angles, bins=num_bins, range=(0.0, 180.0))
    probs = counts / float(len(angles))
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(num_bins)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def analyze_scene(
    image: np.ndarray,
    mask: np.ndarray,
    return_debug: bool = False,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """Analyze the visual situation surrounding the damaged region.

    Extracts:
    1. Damage geometry (area, bbox, perimeter, compactness, complexity)
    2. Context region and quality
    3. Edge analysis (Canny pixel count, density, magnitude)
    4. Line / structural analysis (Hough lines, density, orientations, entropy)
    5. Texture complexity (local variance, Laplacian variance)
    6. Color statistics (mean RGB, std RGB, variance)
    7. Structural complexity score
    8. Qualitative scene profile (texture, structure, geometry, damage_scale)

    Args:
        image: Source RGB image (H, W, 3) uint8.
        mask: Damage mask (H, W) uint8 (where >0 is damage).
        return_debug: If True, returns a visualization image showing context, edges, and lines.

    Returns:
        (analysis_dict, debug_image_or_none)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be 3-channel RGB uint8, got shape {image.shape}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} must match image shape {image.shape[:2]}")

    H, W = image.shape[:2]
    binary_mask = (mask > 0).astype(np.uint8) * 255
    masked_pixels = int(np.count_nonzero(binary_mask))
    total_pixels = H * W
    area_ratio = float(masked_pixels / float(max(1, total_pixels)))

    # Handle empty mask gracefully
    if masked_pixels == 0:
        empty_res = SceneAnalysisResult(
            image={"width": W, "height": H},
            mask={
                "area_pixels": 0,
                "area_ratio": 0.0,
                "bbox": [0, 0, 0, 0],
                "width": 0,
                "height": 0,
                "aspect_ratio": 1.0,
                "perimeter": 0.0,
                "compactness": 0.0,
                "boundary_complexity": 0.0,
                "perimeter_to_area_ratio": 0.0,
            },
            context={"quality": 0.0, "context_pixels": 0, "crop_coords": [0, 0, W, H]},
            edges={"density": 0.0, "pixel_count": 0, "mean_magnitude": 0.0},
            lines={
                "count": 0,
                "density": 0.0,
                "dominant_orientation": 0.0,
                "orientation_entropy": 0.0,
                "horizontal_proportion": 0.0,
                "vertical_proportion": 0.0,
                "diagonal_proportion": 0.0,
            },
            texture={"variance": 0.0, "laplacian_variance": 0.0, "complexity": 0.0},
            color={"mean_rgb": [0.0, 0.0, 0.0], "std_rgb": [0.0, 0.0, 0.0], "variance": 0.0},
            structure={"complexity": 0.0},
            scene_profile={
                "texture": "low",
                "structure": "low",
                "geometry": "weak",
                "damage_scale": "tiny",
            },
        )
        debug_img = image.copy() if return_debug else None
        return empty_res.to_dict(), debug_img

    # 1. Damage Geometry
    bbox = extract_mask_bbox(binary_mask)
    bx1, by1, bx2, by2 = bbox
    box_w = bx2 - bx1
    box_h = by2 - by1
    aspect_ratio = float(box_w / float(max(1, box_h)))

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    compactness = float(np.clip(4.0 * np.pi * masked_pixels / max(1.0, perimeter**2), 0.0, 1.0))
    box_perimeter = max(1.0, 2.0 * (box_w + box_h))
    boundary_complexity = float(min(1.0, (perimeter / box_perimeter) / 2.5))
    perimeter_to_area_ratio = float(perimeter / float(max(1, masked_pixels)))

    # 2. Context Region (matches Phase 1 adaptive context crop)
    crop_coords = compute_adaptive_crop_coords((H, W), bbox, area_ratio)
    cx1, cy1, cx2, cy2 = crop_coords
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    crop_img = image[cy1:cy2, cx1:cx2]
    crop_mask = binary_mask[cy1:cy2, cx1:cx2] > 0

    # Build local context ring immediately surrounding the damage
    ring_radius = max(9, min(51, int(max(box_w, box_h) * 0.25) * 2 + 1))
    dilated_crop_mask = cv2.dilate(
        crop_mask.astype(np.uint8), np.ones((ring_radius, ring_radius), np.uint8)
    )
    context_ring = (dilated_crop_mask > 0) & (~crop_mask)

    # Fallback to all unmasked crop pixels if ring is too small
    if np.count_nonzero(context_ring) < 32:
        context_ring = ~crop_mask

    context_pixels = int(np.count_nonzero(context_ring))

    # 3. Edge Analysis
    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    crop_edges = cv2.Canny(crop_gray, 100, 200)

    edge_px_count = int(np.count_nonzero(crop_edges[context_ring]))
    edge_density = float(edge_px_count / float(max(1, context_pixels)))

    sobel_x = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    mean_edge_mag = float(grad_mag[context_ring].mean()) if context_pixels else 0.0
    norm_edge_mag = float(min(1.0, mean_edge_mag / 128.0))

    # 4. Line / Structural Analysis (Hough Line Transform)
    min_line_len = max(12, int(min(crop_w, crop_h) * 0.12))
    max_line_gap = max(4, int(min_line_len * 0.35))
    hough_lines = cv2.HoughLinesP(
        crop_edges,
        1,
        np.pi / 180,
        threshold=32,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    detected_lines: List[Tuple[int, int, int, int, float]] = []
    angles: List[float] = []
    horiz_count = 0
    vert_count = 0
    diag_count = 0

    if hough_lines is not None:
        for seg in hough_lines:
            x_a, y_a, x_b, y_b = seg[0]
            # Check if line has any point in context region (not entirely inside damage)
            dx = x_b - x_a
            dy = y_b - y_a
            length = np.hypot(dx, dy)
            if length < min_line_len:
                continue
            deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
            angles.append(deg)
            detected_lines.append((x_a, y_a, x_b, y_b, deg))

            # Categorize orientations
            if deg <= 20.0 or deg >= 160.0:
                horiz_count += 1
            elif 70.0 <= deg <= 110.0:
                vert_count += 1
            else:
                diag_count += 1

    total_lines = len(detected_lines)
    line_density = float(min(1.0, total_lines / 40.0))

    if total_lines > 0:
        horiz_prop = float(horiz_count / total_lines)
        vert_prop = float(vert_count / total_lines)
        diag_prop = float(diag_count / total_lines)

        # Dominant orientation from histogram peak
        hist, bin_edges = np.histogram(angles, bins=18, range=(0.0, 180.0))
        dominant_bin = int(np.argmax(hist))
        dominant_orientation = float((bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2.0)
        orientation_entropy = _compute_orientation_entropy(angles, num_bins=18)
    else:
        horiz_prop = 0.0
        vert_prop = 0.0
        diag_prop = 0.0
        dominant_orientation = 0.0
        orientation_entropy = 0.0

    # 5. Texture Complexity
    gray_ctx = crop_gray[context_ring]
    tex_var = float(np.var(gray_ctx)) if context_pixels else 0.0
    norm_tex_var = float(min(1.0, tex_var / 3500.0))

    laplacian = cv2.Laplacian(crop_gray, cv2.CV_64F)
    lap_var = float(np.var(laplacian[context_ring])) if context_pixels else 0.0
    norm_lap_var = float(min(1.0, lap_var / 1200.0))

    texture_complexity = float(
        np.clip(0.40 * norm_tex_var + 0.40 * norm_lap_var + 0.20 * norm_edge_mag, 0.0, 1.0)
    )

    # 6. Color Complexity
    if context_pixels:
        mean_rgb = [float(crop_img[context_ring, c].mean()) for c in range(3)]
        std_rgb = [float(crop_img[context_ring, c].std()) for c in range(3)]
        color_var = float(min(1.0, np.mean(std_rgb) / 100.0))
    else:
        mean_rgb = [0.0, 0.0, 0.0]
        std_rgb = [0.0, 0.0, 0.0]
        color_var = 0.0

    # 7. Context Quality (how much useful context surrounds the damage)
    dist_to_border = min(bx1, by1, W - bx2, H - by2)
    border_support = float(min(1.0, dist_to_border / 80.0))
    ratio_context = float(min(1.0, (context_pixels / float(max(1, masked_pixels))) * 0.5))
    visual_richness = float(0.50 * edge_density + 0.50 * texture_complexity)
    context_quality = float(
        np.clip(
            0.40 * ratio_context + 0.30 * border_support + 0.30 * min(1.0, visual_richness * 2.0),
            0.0,
            1.0,
        )
    )

    # 8. Structural Complexity Score (explainable derived formula)
    # Combines edge density, line density, and orientation concentration
    orientation_alignment = (
        float(1.0 - orientation_entropy) if total_lines >= 3 else 0.0
    )
    structural_complexity = float(
        np.clip(
            0.40 * edge_density
            + 0.35 * line_density
            + 0.25 * orientation_alignment,
            0.0,
            1.0,
        )
    )

    # 9. Overall Scene Profile (Transparent heuristic labels)
    # Texture rating
    if texture_complexity < THRESHOLD_TEXTURE_LOW:
        profile_texture = "low"
    elif texture_complexity < THRESHOLD_TEXTURE_HIGH:
        profile_texture = "medium"
    else:
        profile_texture = "high"

    # Structure rating
    if structural_complexity < THRESHOLD_STRUCTURE_LOW:
        profile_structure = "low"
    elif structural_complexity < THRESHOLD_STRUCTURE_HIGH:
        profile_structure = "medium"
    else:
        profile_structure = "high"

    # Geometry rating
    if total_lines < 3 or line_density < 0.10:
        profile_geometry = "weak"
    elif total_lines >= 10 and (horiz_prop >= 0.45 or vert_prop >= 0.45 or line_density >= 0.25):
        profile_geometry = "strong"
    else:
        profile_geometry = "moderate"

    # Damage scale rating
    if area_ratio < THRESHOLD_DAMAGE_TINY:
        profile_damage_scale = "tiny"
    elif area_ratio < THRESHOLD_DAMAGE_SMALL:
        profile_damage_scale = "small"
    elif area_ratio < THRESHOLD_DAMAGE_MEDIUM:
        profile_damage_scale = "medium"
    else:
        profile_damage_scale = "large"

    result = SceneAnalysisResult(
        image={"width": W, "height": H},
        mask={
            "area_pixels": masked_pixels,
            "area_ratio": round(area_ratio, 4),
            "bbox": [bx1, by1, bx2, by2],
            "width": box_w,
            "height": box_h,
            "aspect_ratio": round(aspect_ratio, 3),
            "perimeter": round(perimeter, 1),
            "compactness": round(compactness, 3),
            "boundary_complexity": round(boundary_complexity, 3),
            "perimeter_to_area_ratio": round(perimeter_to_area_ratio, 4),
        },
        context={
            "quality": round(context_quality, 3),
            "context_pixels": context_pixels,
            "crop_coords": [cx1, cy1, cx2, cy2],
        },
        edges={
            "density": round(edge_density, 3),
            "pixel_count": edge_px_count,
            "mean_magnitude": round(mean_edge_mag, 2),
        },
        lines={
            "count": total_lines,
            "density": round(line_density, 3),
            "dominant_orientation": round(dominant_orientation, 1),
            "orientation_entropy": round(orientation_entropy, 3),
            "horizontal_proportion": round(horiz_prop, 3),
            "vertical_proportion": round(vert_prop, 3),
            "diagonal_proportion": round(diag_prop, 3),
        },
        texture={
            "variance": round(norm_tex_var, 3),
            "laplacian_variance": round(norm_lap_var, 3),
            "complexity": round(texture_complexity, 3),
        },
        color={
            "mean_rgb": [round(c, 1) for c in mean_rgb],
            "std_rgb": [round(c, 1) for c in std_rgb],
            "variance": round(color_var, 3),
        },
        structure={
            "complexity": round(structural_complexity, 3),
        },
        scene_profile={
            "texture": profile_texture,
            "structure": profile_structure,
            "geometry": profile_geometry,
            "damage_scale": profile_damage_scale,
        },
    )

    # 10. Optional Visual Debugging
    debug_image = None
    if return_debug:
        debug_image = crop_img.copy()
        # Draw detected edges as faint lime-green overlay
        edge_mask = crop_edges > 0
        debug_image[edge_mask] = (
            0.5 * debug_image[edge_mask] + 0.5 * np.array([40, 220, 60], dtype=np.float32)
        ).astype(np.uint8)

        # Draw detected Hough lines (yellow)
        for lx1, ly1, lx2, ly2, _ in detected_lines:
            cv2.line(debug_image, (lx1, ly1), (lx2, ly2), (255, 220, 30), 2, cv2.LINE_AA)

        # Draw damage mask as semi-transparent red
        mask_overlay = crop_mask
        debug_image[mask_overlay] = (
            0.4 * debug_image[mask_overlay] + 0.6 * np.array([230, 40, 40], dtype=np.float32)
        ).astype(np.uint8)

        # Draw bounding box of the damage (white)
        local_bx1 = bx1 - cx1
        local_by1 = by1 - cy1
        local_bx2 = bx2 - cx1
        local_by2 = by2 - cy1
        cv2.rectangle(
            debug_image,
            (local_bx1, local_by1),
            (local_bx2, local_by2),
            (255, 255, 255),
            1,
        )

    return result.to_dict(), debug_image
