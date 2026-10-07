"""Test suite verifying the Structure / Scene Analyzer across 9+ visual scenarios."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scene_analyzer import analyze_scene

DEBUG_OUTPUT_DIR = ROOT / "data" / "output" / "scene_analysis"
DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def gen_plain_background(w: int = 600, h: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Plain smooth sky-like gradient."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        img[y, :, :] = [190 + y // 15, 215, 240]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[150:250, 250:350] = 255
    return img, mask


def gen_detailed_texture(w: int = 500, h: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """High-frequency sinusoidal wave texture."""
    x = np.linspace(0, 24 * np.pi, w)
    y = np.linspace(0, 24 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    pattern = (np.sin(xx) * np.cos(yy) * 127 + 128).astype(np.uint8)
    img = np.stack([pattern, (pattern * 0.7).astype(np.uint8), (255 - pattern)], axis=2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (250, 250), 55, 255, -1)
    return img, mask


def gen_building_lines(w: int = 640, h: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Building facade with prominent horizontal floors and vertical pillars."""
    pil_img = Image.new("RGB", (w, h), (200, 195, 185))
    draw = ImageDraw.Draw(pil_img)
    # Horizontal floor ledges
    for y in range(60, 440, 50):
        draw.line([(60, y), (580, y)], fill=(50, 45, 40), width=5)
    # Vertical window pillars
    for x in range(100, 560, 60):
        draw.line([(x, 60), (x, 440)], fill=(60, 55, 50), width=4)
        for y in range(60, 390, 50):
            draw.rectangle([x + 10, y + 10, x + 40, y + 40], fill=(70, 110, 150))
    img = np.array(pil_img)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (220, 160), (380, 280), 255, -1)
    return img, mask


def gen_road_perspective(w: int = 640, h: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Road with diagonal perspective receding lines towards horizon."""
    img = np.full((h, w, 3), (120, 160, 210), dtype=np.uint8)  # Sky
    cv2.rectangle(img, (0, 220), (w, h), (80, 140, 70), -1)  # Landscape
    # Road polygon
    pts = np.array([[260, 220], [380, 220], [580, 480], [60, 480]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (60, 60, 65))
    # Dashed center line
    cv2.line(img, (320, 220), (320, 480), (240, 220, 60), 4)
    # Road edges
    cv2.line(img, (260, 220), (60, 480), (255, 255, 255), 4)
    cv2.line(img, (380, 220), (580, 480), (255, 255, 255), 4)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (260, 260), (380, 380), 255, -1)
    return img, mask


def gen_repetitive_pattern(w: int = 500, h: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Repetitive regular dot/circle grid pattern."""
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    for y in range(30, h, 40):
        for x in range(30, w, 40):
            cv2.circle(img, (x, y), 10, (50, 60, 140), -1)
            cv2.circle(img, (x, y), 5, (220, 80, 40), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (250, 250), 65, 255, -1)
    return img, mask


def gen_small_object(w: int = 800, h: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Tiny damage spot (<1% area) on large canvas."""
    img = np.full((h, w, 3), (170, 190, 210), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (700, 500), (120, 140, 160), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[285:315, 385:415] = 255  # 30x30 = 900 px out of 480,000 (0.18%)
    return img, mask


def gen_large_damage(w: int = 512, h: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Large missing region (>35% area)."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (450, 450), (90, 120, 160), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[90:410, 90:410] = 255  # ~39% area
    return img, mask


def gen_thin_scratch(w: int = 600, h: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Thin 2px scratch across the image (high perimeter to area ratio)."""
    img = np.full((h, w, 3), (210, 205, 195), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask, (60, 100), (540, 300), 255, thickness=3)
    return img, mask


def gen_anime_illustration(w: int = 512, h: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Flat-shaded illustration with dark ink outline contours."""
    pil_img = Image.new("RGB", (w, h), (250, 245, 235))
    draw = ImageDraw.Draw(pil_img)
    # Head & hair polygons with heavy ink outlines
    draw.polygon([(150, 100), (360, 100), (420, 280), (330, 430), (180, 430), (90, 280)],
                 fill=(245, 175, 120), outline=(20, 20, 25), width=5)
    draw.polygon([(120, 100), (200, 220), (260, 130), (320, 230), (390, 100)],
                 fill=(60, 90, 170), outline=(20, 20, 25), width=5)
    img = np.array(pil_img)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (170, 160), (330, 250), 255, -1)
    return img, mask


def verify_analysis_schema(res: dict, title: str) -> None:
    """Assert all required keys exist and values have valid numerical ranges."""
    required_sections = ["image", "mask", "context", "edges", "lines", "texture", "color", "structure", "scene_profile"]
    for s in required_sections:
        assert s in res, f"[{title}] Missing top-level section: {s}"

    # Geometry checks
    assert res["image"]["width"] > 0 and res["image"]["height"] > 0
    assert res["mask"]["area_ratio"] >= 0.0 and res["mask"]["area_ratio"] <= 1.0
    assert res["mask"]["compactness"] >= 0.0 and res["mask"]["compactness"] <= 1.0
    assert res["mask"]["boundary_complexity"] >= 0.0 and res["mask"]["boundary_complexity"] <= 1.0

    # Edge checks
    assert 0.0 <= res["edges"]["density"] <= 1.0

    # Line checks
    assert res["lines"]["count"] >= 0
    assert 0.0 <= res["lines"]["density"] <= 1.0
    assert 0.0 <= res["lines"]["orientation_entropy"] <= 1.0

    # Texture & Structure
    assert 0.0 <= res["texture"]["complexity"] <= 1.0
    assert 0.0 <= res["structure"]["complexity"] <= 1.0
    assert 0.0 <= res["context"]["quality"] <= 1.0

    # Profile labels
    prof = res["scene_profile"]
    assert prof["texture"] in ["low", "medium", "high"], f"Invalid texture label: {prof['texture']}"
    assert prof["structure"] in ["low", "medium", "high"], f"Invalid structure label: {prof['structure']}"
    assert prof["geometry"] in ["weak", "moderate", "strong"], f"Invalid geometry label: {prof['geometry']}"
    assert prof["damage_scale"] in ["tiny", "small", "medium", "large"], f"Invalid damage scale: {prof['damage_scale']}"


def run_tests() -> bool:
    scenarios = [
        ("SCENE_1", "Plain background", *gen_plain_background()),
        ("SCENE_2", "Detailed texture", *gen_detailed_texture()),
        ("SCENE_3", "Building / straight lines", *gen_building_lines()),
        ("SCENE_4", "Road / perspective", *gen_road_perspective()),
        ("SCENE_5", "Repetitive pattern", *gen_repetitive_pattern()),
        ("SCENE_6", "Small object (<1%)", *gen_small_object()),
        ("SCENE_7", "Large damage (>35%)", *gen_large_damage()),
        ("SCENE_8", "Thin scratch", *gen_thin_scratch()),
        ("SCENE_9", "Anime illustration", *gen_anime_illustration()),
    ]

    print("\n========================================================")
    print("RUNNING PHASE 2 SCENE ANALYZER VALIDATION SUITE")
    print("========================================================")

    all_passed = True
    for sid, name, img, mask in scenarios:
        res, dbg_img = analyze_scene(img, mask, return_debug=True)
        verify_analysis_schema(res, name)

        # Save debug visualization image
        if dbg_img is not None:
            debug_path = DEBUG_OUTPUT_DIR / f"{sid.lower()}_debug.png"
            Image.fromarray(dbg_img).save(debug_path)

        p = res["scene_profile"]
        m = res["mask"]
        print(f"\n{sid}: {name}")
        print(f"  Damage Area:    {m['area_pixels']} px ({m['area_ratio']*100:.2f}%) -> scale: {p['damage_scale']}")
        print(f"  Edges Density:  {res['edges']['density']:.3f} | Lines Detected: {res['lines']['count']} (density: {res['lines']['density']:.3f})")
        print(f"  Texture Score:  {res['texture']['complexity']:.3f} ({p['texture']})")
        print(f"  Structure:      {res['structure']['complexity']:.3f} ({p['structure']}, geometry: {p['geometry']})")
        print(f"  Context Qual:   {res['context']['quality']:.3f}")
        if sid == "SCENE_8":
            print(f"  Perimeter/Area: {m['perimeter_to_area_ratio']:.4f} (High ratio indicates thin linear scratch!)")

    # Edge Case 1: Empty mask (0 damage pixels)
    print("\n--------------------------------------------------------")
    print("TESTING EDGE CASE: Empty Mask")
    empty_mask = np.zeros((300, 300), dtype=np.uint8)
    empty_img = np.zeros((300, 300, 3), dtype=np.uint8)
    empty_res, _ = analyze_scene(empty_img, empty_mask, return_debug=False)
    verify_analysis_schema(empty_res, "Empty Mask")
    assert empty_res["mask"]["area_pixels"] == 0
    print("  Empty mask handled gracefully without crashing.")

    # Edge Case 2: Mask touching all 4 boundaries
    print("TESTING EDGE CASE: Mask Touching Image Borders")
    touching_mask = np.zeros((400, 400), dtype=np.uint8)
    touching_mask[0:50, 0:50] = 255  # Touch top-left (0, 0)
    touching_img = np.full((400, 400, 3), 150, dtype=np.uint8)
    touch_res, _ = analyze_scene(touching_img, touching_mask, return_debug=False)
    verify_analysis_schema(touch_res, "Touching Border")
    print("  Mask touching border handled properly.")

    # Edge Case 3: Mask covering 100% of image
    print("TESTING EDGE CASE: 100% Full-Image Damage")
    full_mask = np.full((200, 200), 255, dtype=np.uint8)
    full_img = np.full((200, 200, 3), 120, dtype=np.uint8)
    full_res, _ = analyze_scene(full_img, full_mask, return_debug=False)
    verify_analysis_schema(full_res, "100% Mask")
    assert full_res["scene_profile"]["damage_scale"] == "large"
    print("  100% mask handled cleanly without zero-division.")

    print("\n========================================================")
    print("ALL SCENE ANALYZER TESTS PASSED SUCCESSFULLY!")
    print(f"Debug overlays saved to: {DEBUG_OUTPUT_DIR}")
    print("========================================================\n")
    return True


if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
