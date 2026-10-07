from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np


# Configurable routing thresholds
THRESHOLD_THIN_PERIMETER_AREA = 0.12
THRESHOLD_THIN_AREA_MAX = 0.08

THRESHOLD_LARGE_AREA = 0.25
THRESHOLD_LARGE_SCORE_MIN = 0.55

THRESHOLD_STRUCTURAL_SCORE_HIGH = 0.35
THRESHOLD_TEXTURE_SCORE_HIGH = 0.45


@dataclass
class DamageClassification:
    """Structured damage characterization based on scene analysis telemetry."""
    primary_type: str
    confidence: float
    secondary_characteristics: List[str]
    characteristics: List[str]
    scores: Dict[str, float]
    summary_description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_routing_scores(scene_analysis: Dict[str, Any]) -> Dict[str, float]:
    """Derive transparent, normalized routing scores (0.0 to 1.0) from Phase 2 features."""
    mask_info = scene_analysis.get("mask", {})
    edges_info = scene_analysis.get("edges", {})
    lines_info = scene_analysis.get("lines", {})
    texture_info = scene_analysis.get("texture", {})
    structure_info = scene_analysis.get("structure", {})

    area_ratio = float(mask_info.get("area_ratio", 0.0))
    p_to_a = float(mask_info.get("perimeter_to_area_ratio", 0.0))
    aspect_ratio = float(mask_info.get("aspect_ratio", 1.0))
    compactness = float(mask_info.get("compactness", 1.0))

    edge_density = float(edges_info.get("density", 0.0))
    line_count = int(lines_info.get("count", 0))
    line_density = float(lines_info.get("density", 0.0))
    orientation_entropy = float(lines_info.get("orientation_entropy", 0.0))

    tex_complexity = float(texture_info.get("complexity", 0.0))
    struct_complexity = float(structure_info.get("complexity", 0.0))

    # 1. Thin Damage Score (scratch / wire / crack)
    # High perimeter-to-area ratio + elongated aspect ratio + non-compact damage
    norm_p_to_a = (
        float(min(1.0, (p_to_a - 0.05) / 0.25)) if p_to_a > 0.05 else 0.0
    )
    aspect_diff = max(aspect_ratio, 1.0 / max(0.01, aspect_ratio))
    elongation = float(min(1.0, (aspect_diff - 1.0) / 3.0)) if aspect_diff > 1.4 else 0.0
    is_non_compact = compactness < 0.45 or aspect_diff > 2.0

    if area_ratio < THRESHOLD_THIN_AREA_MAX and is_non_compact and (norm_p_to_a > 0.2 or p_to_a >= 0.25):
        thin_damage_score = float(np.clip(0.60 * norm_p_to_a + 0.40 * elongation, 0.0, 1.0))
    else:
        thin_damage_score = 0.0

    # 2. Large Damage Score
    # Area ratio scaled from 0.08 (medium) to 0.30+ (large)
    if area_ratio >= 0.08:
        large_damage_score = float(np.clip((area_ratio - 0.08) / (0.32 - 0.08), 0.0, 1.0))
    else:
        large_damage_score = float(max(0.0, area_ratio / 0.16))

    # 3. Structural Score (architectural, windows, fences, straight lines)
    line_sig = float(min(1.0, line_count / 14.0))
    alignment_bonus = float(1.0 - orientation_entropy) if line_count >= 3 else 0.0
    structural_score = float(
        np.clip(
            0.35 * struct_complexity
            + 0.30 * line_sig
            + 0.20 * min(1.0, edge_density * 8.0)
            + 0.15 * alignment_bonus,
            0.0,
            1.0,
        )
    )

    # 4. Texture Score (foliage, grass, cloth, repetitive grain)
    texture_score = float(np.clip(tex_complexity, 0.0, 1.0))

    # 5. Complexity Score (mixed high structure and high texture)
    complexity_score = float(
        np.clip(
            0.45 * structural_score
            + 0.45 * texture_score
            + 0.10 * (1.0 if (structural_score >= 0.35 and texture_score >= 0.45) else 0.0),
            0.0,
            1.0,
        )
    )

    # 6. Object Removal Score (discrete compact damage in manageable context)
    is_compact = compactness >= 0.25 and area_ratio < 0.20
    object_score = float(
        np.clip(
            0.50 * compactness + 0.50 * (1.0 - min(1.0, area_ratio * 4.0)),
            0.0,
            1.0,
        )
    ) if is_compact else 0.0

    return {
        "thin_damage_score": round(thin_damage_score, 3),
        "large_damage_score": round(large_damage_score, 3),
        "structural_score": round(structural_score, 3),
        "texture_score": round(texture_score, 3),
        "complexity_score": round(complexity_score, 3),
        "object_removal_score": round(object_score, 3),
    }


def characterize_damage(scene_analysis: Dict[str, Any]) -> DamageClassification:
    """Classify the damage situation into an explainable reconstruction category.

    Categories:
    - THIN_LINEAR_DAMAGE
    - LARGE_MISSING_REGION
    - COMPLEX_MIXED_SCENE
    - STRUCTURAL_SCENE
    - TEXTURED_BACKGROUND
    - GENERAL_OBJECT_REMOVAL
    - SIMPLE_BACKGROUND
    """
    mask_info = scene_analysis.get("mask", {})
    area_ratio = float(mask_info.get("area_ratio", 0.0))
    p_to_a = float(mask_info.get("perimeter_to_area_ratio", 0.0))
    compactness = float(mask_info.get("compactness", 1.0))
    edge_density = float(scene_analysis.get("edges", {}).get("density", 0.0))
    lines_count = int(scene_analysis.get("lines", {}).get("count", 0))

    scores = calculate_routing_scores(scene_analysis)
    thin_s = scores["thin_damage_score"]
    large_s = scores["large_damage_score"]
    struct_s = scores["structural_score"]
    tex_s = scores["texture_score"]
    comp_s = scores["complexity_score"]
    obj_s = scores["object_removal_score"]

    secondary: List[str] = []
    characteristics: List[str] = []

    # Characteristic tags
    if area_ratio < 0.01:
        characteristics.append("tiny_damage")
    elif area_ratio < 0.08:
        characteristics.append("small_damage")
    elif area_ratio < 0.25:
        characteristics.append("medium_damage")
    else:
        characteristics.append("large_damage")

    if thin_s >= 0.40:
        characteristics.append("high_perimeter_ratio")
    if struct_s >= 0.35:
        characteristics.append("strong_linear_geometry")
    if tex_s >= 0.45:
        characteristics.append("dense_surface_texture")

    # Priority-based hierarchy for Primary Situation
    # 1. THIN_LINEAR_DAMAGE: narrow scratch or wire (requires non-compact damage)
    if thin_s >= 0.45 and area_ratio < THRESHOLD_THIN_AREA_MAX and compactness < 0.50:
        primary_type = "THIN_LINEAR_DAMAGE"
        confidence = float(np.clip(0.65 + thin_s * 0.30, 0.60, 0.95))
        summary = "Elongated scratch or wire damage with high perimeter-to-area ratio."

    # 2. LARGE_MISSING_REGION: expansive damage requiring broad context
    elif area_ratio >= THRESHOLD_LARGE_AREA or large_s >= THRESHOLD_LARGE_SCORE_MIN:
        primary_type = "LARGE_MISSING_REGION"
        confidence = float(np.clip(0.60 + large_s * 0.35, 0.60, 0.95))
        summary = f"Large missing region covering {area_ratio*100:.1f}% of image area."

    # 3. SIMPLE_BACKGROUND: smooth/plain wall, sky, gradient
    elif struct_s < 0.15 and tex_s < 0.15 and edge_density < 0.02 and lines_count == 0:
        primary_type = "SIMPLE_BACKGROUND"
        confidence = float(np.clip(1.0 - max(struct_s, tex_s), 0.70, 0.98))
        summary = "Smooth background with low structural and texture complexity."

    # 4. COMPLEX_MIXED_SCENE: elevated structure and elevated texture simultaneously
    elif struct_s >= 0.35 and tex_s >= 0.55 and comp_s >= 0.55:
        primary_type = "COMPLEX_MIXED_SCENE"
        confidence = float(np.clip(0.60 + comp_s * 0.30, 0.60, 0.92))
        summary = "Complex scene combining high structural geometry and dense texture."

    # 5. STRUCTURAL_SCENE: prominent straight lines, architectural features
    elif struct_s >= THRESHOLD_STRUCTURAL_SCORE_HIGH or lines_count >= 8:
        primary_type = "STRUCTURAL_SCENE"
        confidence = float(np.clip(0.60 + struct_s * 0.35, 0.60, 0.95))
        summary = f"Strong edge and linear geometry detected ({lines_count} lines)."

    # 6. TEXTURED_BACKGROUND: foliage, brick, fabric, regular patterns
    elif tex_s >= THRESHOLD_TEXTURE_SCORE_HIGH:
        primary_type = "TEXTURED_BACKGROUND"
        confidence = float(np.clip(0.60 + tex_s * 0.30, 0.60, 0.92))
        summary = "Rich surface texture with high local and Laplacian variance."

    # 7. GENERAL_OBJECT_REMOVAL: compact object on standard background
    elif obj_s >= 0.30 and area_ratio < 0.20:
        primary_type = "GENERAL_OBJECT_REMOVAL"
        confidence = float(np.clip(0.60 + obj_s * 0.30, 0.60, 0.90))
        summary = "Discrete compact object on standard background."

    # 8. Fallback to Simple Background
    else:
        primary_type = "SIMPLE_BACKGROUND"
        confidence = float(np.clip(1.0 - max(struct_s, tex_s), 0.70, 0.98))
        summary = "Smooth background with low structural and texture complexity."

    # Populate secondary characteristics
    if primary_type != "THIN_LINEAR_DAMAGE" and thin_s >= 0.35:
        secondary.append("THIN_DAMAGE_COMPONENT")
    if primary_type != "LARGE_MISSING_REGION" and large_s >= 0.35:
        secondary.append("SIGNIFICANT_AREA")
    if primary_type != "STRUCTURAL_SCENE" and struct_s >= 0.25:
        secondary.append("STRUCTURAL_CONTEXT")
    if primary_type != "TEXTURED_BACKGROUND" and tex_s >= 0.35:
        secondary.append("TEXTURED_SURROUNDINGS")
    if primary_type != "SIMPLE_BACKGROUND" and struct_s < 0.15 and tex_s < 0.20:
        secondary.append("LOW_COMPLEXITY_CONTEXT")

    return DamageClassification(
        primary_type=primary_type,
        confidence=round(confidence, 3),
        secondary_characteristics=secondary,
        characteristics=characteristics,
        scores=scores,
        summary_description=summary,
    )
