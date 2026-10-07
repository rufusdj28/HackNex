# PS06 StructureGuard — Structure-Preserving AI Scene Reconstruction

Generative computer vision and scene reconstruction for HackNex 2026 (HNX26PSI06).
Reconstructs damaged or missing regions of an image while preserving 100% of the undamaged areas outside the damage mask.

Mask convention: white (`255`) means reconstruct; black (`0`) means preserve untouched.

---

## Key Capabilities

1. **Context-Aware Cropped Inpainting:**
   - Instead of naively shrinking the entire image into 512×512, StructureGuard extracts the damage bounding box and expands it with an adaptive visual context margin.
   - Prevents spatial detail loss and geometric distortion on high-resolution images, fine textures, building facades, and small damaged areas.
   - Symmetrically balances crop aspect ratios using surrounding image content to avoid wasting model resolution on empty padding.
   - Preserves thin masks, scratches, and wires during scaling without subpixel loss.

2. **Full-Resolution Mask-Only Compositing:**
   - Reconstructed crops are unpadded, resized back to original crop coordinates, and composited strictly inside the original damage mask.
   - Every pixel outside the damage mask remains untouched from the source image.

3. **Preservation Guard:**
   - Evaluates empirical pixel differences, structural similarity (SSIM), edge consistency (Canny), and color fidelity exclusively in the unmasked region.
   - Produces a verifiable Preservation Score derived mathematically from actual image data (no hardcoded scores).

4. **Contextual Reconstruction Confidence:**
   - Computes a contextual confidence score derived from damage area ratio, surrounding context availability, boundary complexity, seam continuity, and texture variance consistency.

5. **Debug & Alignment Inspector:**
   - Logs and displays full transformation coordinates: original dimensions, mask bounding box, context crop bounds, scale factors, padding offsets, and model input size.

---

## Setup on Windows

The project uses the isolated `.venv` environment. Activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Dependencies installed in `.venv`:
- PyTorch 2.5.1 + CUDA 12.4
- ONNX Runtime 1.30.0
- OpenCV 4.11.0.86
- Scikit-Image 0.24.0
- Gradio 5.49.1
- Pillow 9.5.0

---

## Quality Verification Tests (9 Test Cases)

Run the comprehensive quality test suite covering all 9 required verification scenarios:

```powershell
python scripts/run_quality_tests.py
```

Scenarios evaluated:
- **TEST 1:** Plain background
- **TEST 2:** Detailed high-frequency texture
- **TEST 3:** Building facade / straight structural lines
- **TEST 4:** Object removal
- **TEST 5:** Small mask on large 720p HD image
- **TEST 6:** Large mask (>60% area) with full-image fallback
- **TEST 7:** Mask touching image borders (x=0, y=0)
- **TEST 8:** Non-square panoramic image (3:1 aspect ratio)
- **TEST 9:** Anime / illustration line-art

All tests verify:
- Output dimensions strictly match original dimensions.
- Bit-for-bit unmasked pixel preservation (`mismatch_count == 0`).
- Preservation score is 100.0%.
- Reconstruction Confidence is calculated and reported.
- Artifacts saved to `data/output/tests/`.

---

## Gradio Web Application

Launch the interactive Gradio UI:

```powershell
python app/main.py
```

1. Upload an image.
2. Paint the damaged region with the white brush.
3. Click **Reconstruct**.
4. View Original, Mask, Reconstructed results, Damage Analysis, Preservation Guard metrics, Contextual Confidence, and expandable Spatial Alignment Details.
5. Download the reconstructed image.

---

## PyTorch Smoke Test

The original PyTorch smoke test remains available and unchanged:

```powershell
python scripts/create_sample.py
python scripts/test_inpainting.py
```
Output is saved to `data/output/sample_reconstructed.png`.
