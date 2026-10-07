from __future__ import annotations

from typing import Tuple

import numpy as np


def composite_mask_only(
    original_image: np.ndarray,
    reconstructed_crop: np.ndarray,
    crop_coords: Tuple[int, int, int, int],
    mask: np.ndarray,
) -> np.ndarray:
    """Composite reconstructed crop back into original image strictly inside the damage mask.

    Pixels outside the damage mask are guaranteed to remain 100% untouched
    from the original image.

    Args:
        original_image: Full-resolution source image (H, W, 3) uint8.
        reconstructed_crop: Reconstructed crop restored to crop dimensions (crop_H, crop_W, 3) uint8.
        crop_coords: Bounding box of the crop (cx1, cy1, cx2, cy2).
        mask: Full-resolution binary mask (H, W) uint8 (where >0 indicates damaged region).

    Returns:
        final_image: Result image (H, W, 3) uint8.
    """
    if original_image.ndim != 3 or original_image.shape[2] != 3:
        raise ValueError("Original image must be (H, W, 3) uint8 RGB.")
    if mask.shape != original_image.shape[:2]:
        raise ValueError("Mask dimensions must match original image.")

    cx1, cy1, cx2, cy2 = crop_coords
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    if reconstructed_crop.shape[:2] != (crop_h, crop_w):
        raise ValueError(
            f"Reconstructed crop shape {reconstructed_crop.shape[:2]} does not match "
            f"crop coordinates dimensions {(crop_h, crop_w)}"
        )

    # Start with an exact bit-for-bit copy of the original image
    final_image = original_image.copy()

    # Extract the crop region of the mask
    crop_mask = mask[cy1:cy2, cx1:cx2] > 0

    # Only replace pixels that are explicitly masked inside the crop
    if np.any(crop_mask):
        crop_region = final_image[cy1:cy2, cx1:cx2]
        crop_region[crop_mask] = reconstructed_crop[crop_mask]
        final_image[cy1:cy2, cx1:cx2] = crop_region

    return final_image
