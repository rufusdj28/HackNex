"""Test suite verifying Damage Characterization and Reconstruction Planning across 9 scenarios."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.damage_classifier import characterize_damage
from core.reconstruction_planner import create_reconstruction_plan
from core.scene_analyzer import analyze_scene
from scripts.test_scene_analyzer import (
    gen_anime_illustration,
    gen_building_lines,
    gen_detailed_texture,
    gen_large_damage,
    gen_plain_background,
    gen_repetitive_pattern,
    gen_road_perspective,
    gen_small_object,
    gen_thin_scratch,
)


def run_planner_tests() -> bool:
    scenarios = [
        ("SCENARIO 1", "Plain background", *gen_plain_background()),
        ("SCENARIO 2", "Detailed texture", *gen_detailed_texture()),
        ("SCENARIO 3", "Building / straight lines", *gen_building_lines()),
        ("SCENARIO 4", "Road / perspective", *gen_road_perspective()),
        ("SCENARIO 5", "Repetitive pattern", *gen_repetitive_pattern()),
        ("SCENARIO 6", "Small object (<1%)", *gen_small_object()),
        ("SCENARIO 7", "Large damage (>35%)", *gen_large_damage()),
        ("SCENARIO 8", "Thin scratch", *gen_thin_scratch()),
        ("SCENARIO 9", "Anime illustration", *gen_anime_illustration()),
    ]

    print("\n========================================================")
    print("RUNNING RECONSTRUCTION PLANNER VALIDATION SUITE")
    print("========================================================")

    for sid, name, img, mask in scenarios:
        # 1. Phase 2 Scene Analysis
        scene_analysis, _ = analyze_scene(img, mask)

        # 2. Phase 3 Damage Characterization
        classification = characterize_damage(scene_analysis)

        # 3. Phase 3 Reconstruction Plan
        plan = create_reconstruction_plan(scene_analysis, classification)

        # Schema verification
        assert plan.situation in [
            "SIMPLE_BACKGROUND",
            "TEXTURED_BACKGROUND",
            "STRUCTURAL_SCENE",
            "LARGE_MISSING_REGION",
            "THIN_LINEAR_DAMAGE",
            "COMPLEX_MIXED_SCENE",
            "GENERAL_OBJECT_REMOVAL",
        ], f"Invalid situation: {plan.situation}"
        assert plan.preferred_model in ["lama", "zits", "mat"]
        assert len(plan.candidate_models) >= 1
        assert len(plan.reason) > 0
        assert len(plan.reasoning) >= 1

        print(f"\n{sid}: {name}")
        print(f"  Situation:           {plan.situation}")
        print(f"  Primary Type:        {classification.primary_type} (conf: {classification.confidence:.2f})")
        print(f"  Secondary Traits:    {classification.secondary_characteristics or ['None']}")
        print(f"  Preferred Model:     {plan.preferred_model.upper()}")
        print(f"  Candidate Pool:      {[m.upper() for m in plan.candidate_models]}")
        print(f"  Structural Priority: {plan.structural_priority.upper()}")
        print(f"  Texture Priority:    {plan.texture_priority.upper()}")
        print(f"  Reason:              {plan.reason}")

    print("\n========================================================")
    print("ALL 9 RECONSTRUCTION PLANNER TESTS PASSED PERFECTLY!")
    print("========================================================\n")
    return True


if __name__ == "__main__":
    success = run_planner_tests()
    if not success:
        sys.exit(1)
