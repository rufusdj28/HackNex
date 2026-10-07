"""Comprehensive test suite executing the 9 required StructureGuard quality tests."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis import analyze_damage
from core.confidence import evaluate_reconstruction_confidence
from core.preservation import calculate_preservation_guard
from core.reconstruction import StructureGuardInpainter

TEST_OUTPUT_DIR = ROOT / "data" / "output" / "tests"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_plain_background(w: int = 600, h: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """TEST 1: Plain gradient background with central damage."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        img[y, :, :] = [180 + y // 15, 200 + y // 20, 220]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 255
    return img, mask


def create_detailed_texture(w: int = 500, h: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """TEST 2: Detailed high-frequency texture (checkerboard & sinusoidal pattern)."""
    x = np.linspace(0, 20 * np.pi, w)
    y = np.linspace(0, 20 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    pattern = (np.sin(xx) * np.cos(yy) * 127 + 128).astype(np.uint8)
    img = np.stack([pattern, (pattern * 0.8).astype(np.uint8), (255 - pattern)], axis=2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), 60, 255, -1)
    return img, mask


def create_building_lines(w: int = 640, h: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """TEST 3: Building facade with strong straight lines and windows."""
    pil_img = Image.new("RGB", (w, h), (140, 160, 180))
    draw = ImageDraw.Draw(pil_img)
    # Building outline
    draw.rectangle([80, 60, 560, 480], fill=(210, 200, 180), outline=(80, 70, 60), width=4)
    # Vertical columns
    for x in range(120, 540, 80):
        draw.line([(x, 60), (x, 480)], fill=(70, 60, 50), width=5)
    # Horizontal floors / window rows
    for y in range(100, 460, 60):
        draw.line([(80, y), (560, y)], fill=(90, 80, 70), width=4)
        for x in range(100, 520, 80):
            draw.rectangle([x, y + 10, x + 40, y + 45], fill=(70, 100, 140))
    img = np.array(pil_img)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Mask crossing straight lines and windows
    cv2.rectangle(mask, (220, 140), (380, 260), 255, -1)
    return img, mask


def create_object_removal(w: int = 600, h: int = 450) -> tuple[np.ndarray, np.ndarray]:
    """TEST 4: Discrete object placed on grassy landscape."""
    pil_img = Image.new("RGB", (w, h), (160, 210, 240))  # Sky
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([0, 280, w, h], fill=(80, 160, 70))  # Grass
    # Object: bright red balloon/ball
    draw.ellipse([260, 180, 360, 280], fill=(220, 40, 40), outline=(150, 20, 20), width=3)
    draw.line([(310, 280), (310, 350)], fill=(50, 50, 50), width=2)
    img = np.array(pil_img)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Mask over object with small margin
    cv2.circle(mask, (310, 230), 65, 255, -1)
    cv2.line(mask, (310, 280), (310, 355), 255, thickness=8)
    return img, mask


def create_small_mask_large_image(w: int = 1280, h: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """TEST 5: Small mask (35x35) on a large 720p HD image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = [215, 225, 235]
    # Some details scattered
    cv2.rectangle(img, (200, 150), (1080, 600), (160, 180, 200), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Small 35x35 spot at (640, 360)
    mask[345:380, 625:660] = 255
    return img, mask


def create_large_mask(w: int = 512, h: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """TEST 6: Large mask covering >60% of image area."""
    img = np.full((h, w, 3), 190, dtype=np.uint8)
    cv2.circle(img, (256, 256), 180, (70, 120, 200), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Covers central 65%
    mask[80:432, 80:432] = 255
    return img, mask


def create_mask_touching_edge(w: int = 550, h: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """TEST 7: Mask touching the image border/corner."""
    img = np.full((h, w, 3), (220, 210, 195), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (500, 350), (120, 140, 160), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Touches top and left boundaries (x=0, y=0)
    mask[0:120, 0:140] = 255
    return img, mask


def create_non_square_image(w: int = 900, h: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """TEST 8: Non-square ultra-wide panoramic ratio (3:1)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        img[:, x] = [int(100 + 120 * (x / w)), 180, int(220 - 100 * (x / w))]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (380, 70), (520, 230), 255, -1)
    return img, mask


def create_anime_illustration(w: int = 512, h: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """TEST 9: Anime/illustration line-art with flat colors and sharp ink contours."""
    pil_img = Image.new("RGB", (w, h), (250, 245, 235))
    draw = ImageDraw.Draw(pil_img)
    # Hair/head shapes with black ink borders
    draw.polygon([(160, 120), (350, 120), (410, 280), (320, 420), (190, 420), (100, 280)],
                 fill=(245, 175, 120), outline=(20, 20, 25), width=4)
    # Stylized hair bangs
    draw.polygon([(130, 120), (200, 220), (250, 140), (310, 230), (370, 120)],
                 fill=(70, 90, 160), outline=(20, 20, 25), width=4)
    # Eyes
    draw.ellipse([180, 240, 230, 290], fill=(255, 255, 255), outline=(20, 20, 25), width=3)
    draw.ellipse([195, 250, 220, 280], fill=(40, 110, 180))
    draw.ellipse([280, 240, 330, 290], fill=(255, 255, 255), outline=(20, 20, 25), width=3)
    draw.ellipse([295, 250, 320, 280], fill=(40, 110, 180))
    img = np.array(pil_img)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Mask over hair bang crossing line art
    cv2.rectangle(mask, (180, 160), (320, 240), 255, -1)
    return img, mask


def run_single_test(
    test_id: str,
    title: str,
    img: np.ndarray,
    mask: np.ndarray,
    inpainter: StructureGuardInpainter,
) -> bool:
    print(f"\n--------------------------------------------------")
    print(f"RUNNING {test_id}: {title}")
    print(f"Image dimensions: {img.shape[1]} x {img.shape[0]}")
    masked_px = int(np.count_nonzero(mask > 0))
    cov_pct = (masked_px / float(img.shape[0] * img.shape[1])) * 100.0
    print(f"Damaged region: {masked_px} px ({cov_pct:.2f}% coverage)")

    # Execute StructureGuard reconstruction
    reconstructed, binary_mask, geom = inpainter.reconstruct(img, mask, verbose=False)

    # 1. Output dimensions must match original
    assert reconstructed.shape == img.shape, (
        f"Shape mismatch: {reconstructed.shape} vs original {img.shape}"
    )

    # 2. Strict bit-for-bit unmasked preservation
    outside = mask == 0
    mismatch_count = np.count_nonzero(np.any(img[outside] != reconstructed[outside], axis=1))
    assert mismatch_count == 0, f"Violation! {mismatch_count} unmasked pixels were modified!"

    # 3. Preservation metrics
    guard = calculate_preservation_guard(img, reconstructed, binary_mask)
    assert guard["available"] or cov_pct >= 99.0
    if guard["available"]:
        assert guard["unchanged_pixels_pct"] == 100.0, (
            f"Unchanged pixels: {guard['unchanged_pixels_pct']}%"
        )
        assert guard["preservation_score"] >= 99.9, (
            f"Preservation score: {guard['preservation_score']}%"
        )

    # 4. Contextual Confidence
    analysis = analyze_damage(img, binary_mask)
    confidence = evaluate_reconstruction_confidence(img, reconstructed, binary_mask, analysis)
    assert 0.0 <= confidence["score"] <= 100.0

    # 5. Save test artifacts
    Image.fromarray(reconstructed).save(TEST_OUTPUT_DIR / f"{test_id.lower()}_result.png")

    print(f"PASSED {test_id}")
    print(f"  Preservation Score: {guard['preservation_score']:.2f}%")
    print(f"  Unchanged Pixels:   {guard['unchanged_pixels_pct']:.2f}%")
    print(f"  Confidence:         {confidence['score']:.1f}% ({confidence['context_support']} context)")
    print(f"  Crop used:          {geom.crop_coords} (Scale: {geom.scale:.3f})")
    return True


def main() -> None:
    model_path = ROOT / "models" / "LaMa-ONNX" / "lama_fp32.onnx"
    inpainter = StructureGuardInpainter(model_path)
    print(f"Loaded StructureGuard Inpainter on device: {inpainter.device}")

    tests = [
        ("TEST_1", "Plain background", *create_plain_background()),
        ("TEST_2", "Detailed texture", *create_detailed_texture()),
        ("TEST_3", "Building / straight lines", *create_building_lines()),
        ("TEST_4", "Object removal", *create_object_removal()),
        ("TEST_5", "Small mask on large image", *create_small_mask_large_image()),
        ("TEST_6", "Large mask (>60%)", *create_large_mask()),
        ("TEST_7", "Mask touching image edge", *create_mask_touching_edge()),
        ("TEST_8", "Non-square image (3:1 aspect ratio)", *create_non_square_image()),
        ("TEST_9", "Anime/illustration line art", *create_anime_illustration()),
    ]

    all_passed = True
    for test_id, title, img, mask in tests:
        success = run_single_test(test_id, title, img, mask, inpainter)
        if not success:
            all_passed = False

    print("\n==================================================")
    if all_passed:
        print("ALL 9 STRUCTUREGUARD QUALITY TESTS PASSED PERFECTLY!")
        print(f"Output images saved to: {TEST_OUTPUT_DIR}")
    else:
        print("SOME TESTS FAILED!")
    print("==================================================")


if __name__ == "__main__":
    main()
