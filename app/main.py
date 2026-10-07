from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

# Ensure project root is in sys.path for direct python execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis import analyze_damage
from core.confidence import evaluate_reconstruction_confidence
from core.damage_classifier import characterize_damage
from core.preservation import calculate_preservation_guard
from core.reconstruction import StructureGuardInpainter
from core.reconstruction_planner import create_reconstruction_plan
from core.scene_analyzer import analyze_scene
import json

# Backward-compatible alias
LamaONNX = StructureGuardInpainter

MODEL_PATH = ROOT / "models" / "LaMa-ONNX" / "lama_fp32.onnx"
OUTPUT_DIR = ROOT / "data" / "output"
MODEL_SIZE = (512, 512)


def _as_rgb_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        return None
    if array.shape[2] == 4:
        array = array[..., :3]
    if array.shape[2] != 3:
        return None
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _extract_editor_image(editor_value: Any) -> np.ndarray | None:
    """Read the untouched background image from a Gradio ImageEditor value."""
    if isinstance(editor_value, dict):
        return _as_rgb_array(editor_value.get("background"))
    return _as_rgb_array(editor_value)


def _extract_editor_mask(editor_value: Any, shape: tuple[int, int]) -> np.ndarray:
    """Convert ImageEditor painted layers into a binary mask (255=damage, 0=preserve)."""
    mask = np.zeros(shape, dtype=np.uint8)
    if not isinstance(editor_value, dict):
        return mask

    layers = editor_value.get("layers") or []
    for layer in layers:
        array = np.asarray(layer)
        if array.ndim != 3 or array.shape[:2] != shape:
            continue
        if array.shape[2] >= 4:
            mask[array[..., 3] > 0] = 255
        elif array.shape[2] == 3:
            mask[np.max(array, axis=2) > 8] = 255
    return mask


try:
    INPAINTER = StructureGuardInpainter(MODEL_PATH)
    MODEL_STATUS = f"Model loaded • LaMa ONNX • {INPAINTER.device}"
except Exception as exc:
    INPAINTER = None
    MODEL_STATUS = f"Model unavailable: {exc}"


def reconstruct(editor_value: Any):
    if INPAINTER is None:
        yield None, None, None, None, MODEL_STATUS, "", "", ""
        return

    try:
        image = _extract_editor_image(editor_value)
        if image is None:
            raise ValueError("Upload an image before clicking Reconstruct.")
        mask = _extract_editor_mask(editor_value, image.shape[:2])
        if not np.any(mask > 0):
            raise ValueError("Paint the damaged region on the image first.")

        # 1. Situation-Aware Scene Analysis & Reconstruction Planning (Phase 2 & 3)
        scene_analysis, _ = analyze_scene(image, mask, return_debug=False)
        damage_class = characterize_damage(scene_analysis)
        plan = create_reconstruction_plan(scene_analysis, damage_class)
        analysis_report = _format_analysis_and_plan_report(scene_analysis, plan)

        yield (
            None,
            None,
            None,
            None,
            f"Analyzing scene & planning strategy on {INPAINTER.device}...",
            analysis_report,
            "",
            "",
        )

        # 2. Context-Aware Inpainting with Alignment Tracking (Phase 1)
        reconstructed, binary_mask, geometry = INPAINTER.reconstruct(image, mask, verbose=True)

        # 3. Save Artifact
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"reconstruction_{time.strftime('%Y%m%d_%H%M%S')}.png"
        Image.fromarray(reconstructed).save(output_path)

        # 4. Strict Preservation Guard & Confidence Estimation
        metrics = calculate_preservation_guard(image, reconstructed, binary_mask)
        confidence = evaluate_reconstruction_confidence(
            original_image=image,
            reconstructed_image=reconstructed,
            mask=binary_mask,
        )

        guard_report = _format_guard_report(metrics, confidence)
        debug_info = (
            f"=== Spatial Alignment Info ===\n"
            f"{geometry.format_debug_info()}\n\n"
            f"{plan.format_debug_summary()}\n\n"
            f"=== Scene Analysis Raw Telemetry ===\n"
            f"{json.dumps(scene_analysis, indent=2)}"
        )

        yield (
            image,
            binary_mask,
            reconstructed,
            str(output_path),
            f"Reconstruction complete • {INPAINTER.device}",
            analysis_report,
            guard_report,
            debug_info,
        )
    except Exception as exc:
        yield None, None, None, None, f"Reconstruction failed: {exc}", "", "", ""


def clear_results():
    return None, None, None, None, MODEL_STATUS, "", "", ""


def _format_analysis_and_plan_report(scene: dict[str, Any], plan: Any) -> str:
    m = scene["mask"]
    prof = scene["scene_profile"]
    candidates_str = ", ".join([c.upper() for c in plan.candidate_models])
    situation_title = plan.situation.replace("_", " ").title()

    scene_md = f"""### Scene Analysis

| Attribute | Assessment |
|---|---|
| **Damage Area** | {m['area_ratio'] * 100.0:.2f}% ({m['area_pixels']} px) |
| **Structure** | **{prof['structure'].upper()}** |
| **Texture** | **{prof['texture'].upper()}** |
| **Geometry** | **{prof['geometry'].upper()}** |
| **Damage Scale** | **{prof['damage_scale'].upper()}** |

Context Quality: **{scene['context']['quality'] * 100.0:.1f}%** • Lines Detected: **{scene['lines']['count']}**
"""

    plan_md = f"""### Reconstruction Plan

| Strategy | Directive |
|---|---|
| **Situation** | **{situation_title}** |
| **Recommended Model** | **{plan.preferred_model.upper()}** |
| **Candidate Pool** | {candidates_str} |
| **Priority** | Structure: {plan.structural_priority.upper()} • Texture: {plan.texture_priority.upper()} |

_{plan.reason}_

Active engine: **LaMa ONNX** (multi-candidate execution is advisory in Phase 3).
"""
    return f"{scene_md}\n\n{plan_md}"


def _format_guard_report(metrics: dict[str, Any], confidence: dict[str, Any]) -> str:
    if not metrics.get("available"):
        preservation = f"### Preservation Guard\n\n**Status:** {metrics['reason']}"
    else:
        structural_value = (
            "N/A"
            if metrics["structural_preservation"] is None
            else f"{metrics['structural_preservation']:.2f}%"
        )
        preservation = f"""### Preservation Guard

**Preservation Score: {metrics['preservation_score']:.2f}%**

| Metric | Value |
|---|---:|
| Unchanged Pixels | {metrics['unchanged_pixels_pct']:.2f}% |
| Pixel Difference | {metrics['pixel_difference_pct']:.2f}% |
| Structural Preservation (SSIM) | {structural_value} |
| Edge Preservation | {metrics['edge_preservation']:.2f}% |
| Color Preservation | {metrics['color_preservation']:.2f}% |
| Mean Absolute Difference | {metrics['mean_absolute_error']:.4f} |
| Max Pixel Difference | {metrics['max_pixel_difference']:.1f} |
"""

    conf_block = f"""### Reconstruction Confidence

**Contextual Reconstruction Confidence: {confidence['score']:.1f}%**

| Factor | Rating |
|---|---|
| Context Support | **{confidence['context_support']}** |
| Boundary Continuity | **{confidence['boundary_continuity']}** |
| Structural Continuity | **{confidence['structural_continuity']}** |

_{confidence['heuristic_note']}_
"""
    return f"{preservation}\n\n{conf_block}"


with gr.Blocks(title="StructureGuard — Scene Reconstruction") as demo:
    gr.Markdown(
        """
        # StructureGuard: Scene Reconstruction
        Context-aware scene reconstruction with strict spatial preservation.
        **White/painted = reconstruct · Black/unpainted = preserve untouched**
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                label="Image input — paint the region to reconstruct",
                type="numpy",
                image_mode="RGB",
                sources=["upload", "clipboard"],
                brush=gr.Brush(colors=["#FFFFFF"], default_size=32),
                eraser=gr.Eraser(default_size=32),
                height=512,
            )
            gr.Markdown(
                "Use the ImageEditor toolbar to adjust brush size. "
                "Paint the damaged region with the white brush."
            )
            with gr.Row():
                reconstruct_button = gr.Button("Reconstruct", variant="primary")
                clear_button = gr.Button("Clear results")
            status = gr.Markdown(MODEL_STATUS)

        with gr.Column(scale=1):
            gr.Markdown("### Results")
            with gr.Row():
                original_output = gr.Image(label="Original", type="numpy")
                mask_output = gr.Image(label="Mask", type="numpy")
                reconstructed_output = gr.Image(label="Reconstructed", type="numpy")
            download_output = gr.File(label="Download reconstructed image", interactive=False)
            damage_report = gr.Markdown("### Damage Analysis\nPaint a region and click Reconstruct.")
            guard_report = gr.Markdown("### Preservation Guard\nNo reconstruction evaluated yet.")
            with gr.Accordion("Debug & Spatial Alignment Info", open=False):
                debug_output = gr.Code(label="Crop & Transform Pipeline Details", language="markdown")

    reconstruct_button.click(
        reconstruct,
        inputs=editor,
        outputs=[
            original_output,
            mask_output,
            reconstructed_output,
            download_output,
            status,
            damage_report,
            guard_report,
            debug_output,
        ],
    )
    clear_button.click(
        clear_results,
        inputs=None,
        outputs=[
            original_output,
            mask_output,
            reconstructed_output,
            download_output,
            status,
            damage_report,
            guard_report,
            debug_output,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch()
