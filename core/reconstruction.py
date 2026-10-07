from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort

from core.compositing import composite_mask_only
from core.preprocessing import (
    CropGeometry,
    compute_adaptive_crop_coords,
    extract_mask_bbox,
    letterbox_crop,
    unpad_and_restore_crop,
)


class StructureGuardInpainter:
    """StructureGuard Context-Aware Inpainter using LaMa ONNX."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"LaMa ONNX model not found: {model_path}")

        available = ort.get_available_providers()
        requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in requested if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.image_name = self.session.get_inputs()[0].name
        self.mask_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name
        active = self.session.get_providers()
        self.device = "CUDA" if active and active[0] == "CUDAExecutionProvider" else "CPU"

    def reconstruct(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, CropGeometry]:
        """Perform context-aware cropped reconstruction.

        Args:
            image: (H, W, 3) uint8 RGB image.
            mask: (H, W) uint8 mask where >0 indicates damage.
            verbose: If True, logs geometric bounding and alignment info.

        Returns:
            final_image: (H, W, 3) reconstructed image with undamaged areas 100% preserved.
            binary_mask: (H, W) normalized binary mask (0 or 255).
            geometry: CropGeometry dataclass with debug / alignment metadata.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be an RGB image of shape (H, W, 3).")
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image dimensions {image.shape[:2]}."
            )

        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        masked_pixels = int(np.count_nonzero(binary_mask))
        if masked_pixels == 0:
            raise ValueError("No damaged region detected. Paint a region to reconstruct.")

        H, W = image.shape[:2]
        mask_coverage = masked_pixels / float(W * H)

        # 1. Mask Bounding Box
        bbox = extract_mask_bbox(binary_mask)

        # 2. Adaptive Context Crop
        crop_coords = compute_adaptive_crop_coords((H, W), bbox, mask_coverage)
        cx1, cy1, cx2, cy2 = crop_coords
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        # 3. Paired Crop of Image and Mask
        crop_image = image[cy1:cy2, cx1:cx2]
        crop_mask = binary_mask[cy1:cy2, cx1:cx2]

        # 4. Aspect-ratio preserving resize + padding to 512x512
        (
            padded_image,
            padded_mask,
            scale,
            resized_size,
            padding,
        ) = letterbox_crop(crop_image, crop_mask, target_size=(512, 512))

        geometry = CropGeometry(
            original_size=(W, H),
            bbox=bbox,
            crop_coords=crop_coords,
            crop_size=(crop_w, crop_h),
            scale=scale,
            resized_size=resized_size,
            padding=padding,
            model_size=(512, 512),
            masked_pixels=masked_pixels,
            mask_coverage=mask_coverage,
        )

        if verbose:
            print("\n=== StructureGuard Alignment Info ===")
            print(geometry.format_debug_info())
            print("======================================\n")

        # 5. LaMa ONNX Inference
        image_input = np.transpose(padded_image.astype(np.float32) / 255.0, (2, 0, 1))[None]
        mask_input = (padded_mask.astype(np.float32) / 255.0)[None, None]

        ort_outputs = self.session.run(
            [self.output_name],
            {self.image_name: image_input, self.mask_name: mask_input},
        )[0][0]

        generated = np.transpose(ort_outputs, (1, 2, 0))
        generated = np.clip(generated, 0, 255).astype(np.uint8)

        # 6. Reverse Letterbox Padding and Resize back to Crop Dimensions
        reconstructed_crop = unpad_and_restore_crop(
            generated,
            crop_size=(crop_w, crop_h),
            resized_size=resized_size,
            padding=padding,
        )

        # 7. Mask-Only Compositing into Full-Resolution Original
        final_image = composite_mask_only(
            original_image=image,
            reconstructed_crop=reconstructed_crop,
            crop_coords=crop_coords,
            mask=binary_mask,
        )

        return final_image, binary_mask, geometry
