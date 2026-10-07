from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from core.damage_classifier import DamageClassification, characterize_damage


@dataclass
class ReconstructionPlan:
    """Actionable, explainable reconstruction strategy plan."""
    situation: str
    preferred_model: str
    candidate_models: List[str]
    structural_priority: str  # "high" | "medium" | "low"
    texture_priority: str  # "high" | "medium" | "low"
    context_priority: str  # "high" | "medium" | "low"
    reason: str
    reasoning: List[str]
    routing_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_debug_summary(self) -> str:
        s = self.routing_scores
        candidates_str = " + ".join([m.upper() for m in self.candidate_models])
        return (
            f"=== Reconstruction Strategy Plan ===\n"
            f"Situation:         {self.situation}\n"
            f"Preferred Model:   {self.preferred_model.upper()}\n"
            f"Candidates:        {candidates_str}\n"
            f"Priorities:        Structure={self.structural_priority.upper()} | "
            f"Texture={self.texture_priority.upper()} | Context={self.context_priority.upper()}\n"
            f"Scores:            Struct={s.get('structural_score', 0):.2f} | "
            f"Tex={s.get('texture_score', 0):.2f} | "
            f"Large={s.get('large_damage_score', 0):.2f} | "
            f"Thin={s.get('thin_damage_score', 0):.2f} | "
            f"Complex={s.get('complexity_score', 0):.2f}\n"
            f"Primary Reason:    {self.reason}\n"
            f"Detailed Rationale:\n" + "\n".join([f"  - {r}" for r in self.reasoning])
        )


def create_reconstruction_plan(
    scene_analysis: Dict[str, Any],
    classification: Optional[DamageClassification] = None,
) -> ReconstructionPlan:
    """Synthesize scene analysis and damage classification into a reconstruction plan.

    Routes transparently according to situation:
    - SIMPLE_BACKGROUND       -> LaMa
    - GENERAL_OBJECT_REMOVAL  -> LaMa
    - TEXTURED_BACKGROUND     -> LaMa (with MAT candidate)
    - STRUCTURAL_SCENE        -> ZITS + LaMa
    - LARGE_MISSING_REGION    -> MAT + LaMa
    - THIN_LINEAR_DAMAGE      -> LaMa (with thin-mask preprocessing)
    - COMPLEX_MIXED_SCENE     -> ZITS + LaMa + MAT (multi-model candidate pool)
    """
    if classification is None:
        classification = characterize_damage(scene_analysis)

    situation = classification.primary_type
    scores = classification.scores
    mask_info = scene_analysis.get("mask", {})
    lines_info = scene_analysis.get("lines", {})

    area_pct = float(mask_info.get("area_ratio", 0.0)) * 100.0
    line_count = int(lines_info.get("count", 0))
    dominant_deg = float(lines_info.get("dominant_orientation", 0.0))

    reasoning: List[str] = []

    # CASE A: SIMPLE_BACKGROUND
    if situation == "SIMPLE_BACKGROUND":
        preferred = "lama"
        candidates = ["lama"]
        struct_prio = "low"
        tex_prio = "low"
        context_prio = "medium"
        reason = "LaMa preferred because the damage is in a smooth region with low structural and texture complexity."
        reasoning.append(f"Low edge density ({scene_analysis.get('edges', {}).get('density', 0):.3f}) and minimal texture.")
        reasoning.append("Fourier convolutions in LaMa excel at seamless smooth and gradient inpainting.")

    # CASE B: GENERAL_OBJECT_REMOVAL
    elif situation == "GENERAL_OBJECT_REMOVAL":
        preferred = "lama"
        candidates = ["lama"]
        struct_prio = "low"
        tex_prio = "medium"
        context_prio = "high"
        reason = "LaMa preferred for general-purpose compact object removal with surrounding contextual cues."
        reasoning.append(f"Damage is relatively compact (compactness {mask_info.get('compactness', 0):.2f}) occupying {area_pct:.2f}% of the scene.")
        reasoning.append("Surrounding context provides balanced background evidence for object erasure.")

    # CASE C: TEXTURED_BACKGROUND
    elif situation == "TEXTURED_BACKGROUND":
        preferred = "lama"
        candidates = ["lama", "mat"]
        struct_prio = "low"
        tex_prio = "high"
        context_prio = "high"
        reason = "LaMa preferred with MAT candidate considered for detailed texture and surface synthesis."
        reasoning.append(f"Elevated texture complexity ({scores.get('texture_score', 0):.2f}) with local variance across the damage boundary.")
        reasoning.append("MAT's transformer generator serves as secondary candidate for synthesizing stochastic textures.")

    # CASE D: STRUCTURAL_SCENE
    elif situation == "STRUCTURAL_SCENE":
        preferred = "zits"
        candidates = ["zits", "lama"]
        struct_prio = "high"
        tex_prio = "medium"
        context_prio = "high"
        reason = "ZITS preferred because the damaged region is surrounded by strong edge and linear structural geometry."
        reasoning.append(f"Detected {line_count} significant lines with dominant orientation {dominant_deg:.1f}°.")
        reasoning.append(f"High structural score ({scores.get('structural_score', 0):.2f}) indicates geometric continuation requirement.")
        reasoning.append("ZITS integrates dual-stream edge and line transformer priors before inpainting texture.")

    # CASE E: LARGE_MISSING_REGION
    elif situation == "LARGE_MISSING_REGION":
        preferred = "mat"
        candidates = ["mat", "lama"]
        struct_prio = "medium"
        tex_prio = "medium"
        context_prio = "high"
        reason = "MAT preferred because the missing region occupies a large portion of the image requiring broad contextual hallucination."
        reasoning.append(f"Damaged region covers {area_pct:.1f}% of the total image canvas.")
        reasoning.append("Mask-Aware Transformer (MAT) is purpose-built for large-hole inpainting with global receptive field.")

    # CASE F: THIN_LINEAR_DAMAGE
    elif situation == "THIN_LINEAR_DAMAGE":
        preferred = "lama"
        candidates = ["lama"]
        struct_prio = "medium"
        tex_prio = "low"
        context_prio = "medium"
        reason = "LaMa preferred with thin-mask-preserving preprocessing for elongated scratch or wire damage."
        reasoning.append(f"High perimeter-to-area ratio ({mask_info.get('perimeter_to_area_ratio', 0):.4f}) denotes narrow crack/scratch geometry.")
        reasoning.append("Phase 1 area-downsampling prevents interpolation erasure of subpixel damage lines.")

    # CASE G: COMPLEX_MIXED_SCENE
    else:  # COMPLEX_MIXED_SCENE
        # If structural score dominates, prefer ZITS; if large area, prefer MAT; else LaMa
        if scores.get("structural_score", 0) >= 0.45:
            preferred = "zits"
        elif scores.get("large_damage_score", 0) >= 0.50:
            preferred = "mat"
        else:
            preferred = "lama"
        candidates = ["zits", "lama", "mat"]
        struct_prio = "high"
        tex_prio = "high"
        context_prio = "high"
        reason = f"{preferred.upper()} preferred in multi-candidate pool because both structural geometry and texture complexity are simultaneously elevated."
        reasoning.append(f"High structural score ({scores.get('structural_score', 0):.2f}) and texture score ({scores.get('texture_score', 0):.2f}).")
        reasoning.append("Complex scene requires candidate generation and multi-metric contextual boundary evaluation.")

    return ReconstructionPlan(
        situation=situation,
        preferred_model=preferred,
        candidate_models=candidates,
        structural_priority=struct_prio,
        texture_priority=tex_prio,
        context_priority=context_prio,
        reason=reason,
        reasoning=reasoning,
        routing_scores=scores,
    )
