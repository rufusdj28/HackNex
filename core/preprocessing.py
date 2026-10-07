from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class CropGeometry:
    original_size: Tuple[int, int]  # (width, height)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    crop_coords: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    crop_size: Tuple[int, int]  # (crop_width, crop_height)
    scale: float
    resized_size: Tuple[int, int]  # (resized_width, resized_height)
    padding: Tuple[int, int, int, int]  # (pad_left, pad_top, pad_right, pad_bottom)
    model_size: Tuple[int, int] = (512, 512)
    masked_pixels: int = 0
    mask_coverage: float = 0.0

    def format_debug_info(self) -> str:
        pad_l, pad_t, pad_r, pad_b = self.padding
        return (
            f"Original size: {self.original_size[0]} x {self.original_size[1]}\n"
            f"Mask bounding box: {self.bbox[0]}, {self.bbox[1]}, {self.bbox[2]}, {self.bbox[3]}\n"
            f"Context crop: {self.crop_coords[0]}, {self.crop_coords[1]}, {self.crop_coords[2]}, {self.crop_coords[3]}\n"
            f"Crop size: {self.crop_size[0]} x {self.crop_size[1]}\n"
            f"Scale: {self.scale:.4f}\n"
            f"Padding: left={pad_l}, top={pad_t}, right={pad_r}, bottom={pad_b}\n"
            f"Model input: {self.model_size[0]} x {self.model_size[1]}\n"
            f"Masked pixels: {self.masked_pixels} ({self.mask_coverage * 100.0:.2f}%)"
        )


def extract_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Find the smallest bounding box (x1, y1, x2, y2) containing the damage mask."""
    if mask.ndim != 2:
        raise ValueError(f"Mask must be a 2D array, got ndim={mask.ndim}")

    mask_binary = mask > 0
    if not np.any(mask_binary):
        raise ValueError("No damaged region detected in mask. Paint a region to reconstruct.")

    ys, xs = np.where(mask_binary)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    return x1, y1, x2, y2


def compute_adaptive_crop_coords(
    image_shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    mask_coverage: float,
) -> Tuple[int, int, int, int]:
    """Compute adaptive context-aware crop bounding box.

    Adapts context margin relative to damage size and image dimensions:
    - Small damage (<64px): generous context to provide ample surrounding visual cues.
    - Medium damage (64-256px): balanced context (80-160px).
    - Large damage (>256px): bounded context without cutting off near-border details.
    - Very large damage (>=55% coverage or covering >=80% of width/height): processes full image.
    - Balances aspect ratio within image bounds to minimize empty letterbox padding.
    """
    H, W = image_shape
    bx1, by1, bx2, by2 = bbox
    bw, bh = bx2 - bx1, by2 - by1
    max_dim = max(bw, bh)

    # Full-image fallback if mask covers most of the image or approaches edges
    if mask_coverage >= 0.55 or (bw >= 0.80 * W and bh >= 0.80 * H):
        return (0, 0, W, H)

    # Adaptive context margin calculation
    if max_dim < 64:
        # Small mask: generous context
        margin_x = max(96, int(bw * 2.0))
        margin_y = max(96, int(bh * 2.0))
    elif max_dim < 256:
        # Medium mask: moderate context (approx 64-128px)
        margin_x = max(80, int(bw * 0.5))
        margin_y = max(80, int(bh * 0.5))
    else:
        # Large mask: bounded context
        margin_x = max(64, int(bw * 0.25))
        margin_y = max(64, int(bh * 0.25))

    cx1 = max(0, bx1 - margin_x)
    cy1 = max(0, by1 - margin_y)
    cx2 = min(W, bx2 + margin_x)
    cy2 = min(H, by2 + margin_y)

    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    # Aspect-ratio balancing: expand the shorter axis with available surrounding image context.
    # This prevents wasting the 512x512 canvas on synthetic padding when real image context exists.
    target_dim = max(crop_w, crop_h)

    if crop_w < target_dim and crop_w < W:
        needed_w = min(target_dim, W) - crop_w
        expand_left = min(cx1, needed_w // 2)
        expand_right = min(W - cx2, needed_w - expand_left)
        if expand_left + expand_right < needed_w and cx1 > 0:
            expand_left = min(cx1, needed_w - expand_right)
        cx1 -= expand_left
        cx2 += expand_right

    if crop_h < target_dim and crop_h < H:
        needed_h = min(target_dim, H) - crop_h
        expand_top = min(cy1, needed_h // 2)
        expand_bottom = min(H - cy2, needed_h - expand_top)
        if expand_top + expand_bottom < needed_h and cy1 > 0:
            expand_top = min(cy1, needed_h - expand_bottom)
        cy1 -= expand_top
        cy2 += expand_bottom

    # If the crop covers >=85% of both dimensions, process full image cleanly
    if (cx2 - cx1) >= 0.85 * W and (cy2 - cy1) >= 0.85 * H:
        return (0, 0, W, H)

    return (cx1, cy1, cx2, cy2)


def letterbox_crop(
    crop_image: np.ndarray,
    crop_mask: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int], Tuple[int, int, int, int]]:
    """Resize crop preserving aspect ratio and pad to target_size (512, 512).

    Returns:
        padded_image, padded_mask, scale, (resized_w, resized_h), (pad_left, pad_top, pad_right, pad_bottom)
    """
    crop_h, crop_w = crop_image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / float(crop_w), target_h / float(crop_h))
    resized_w = max(1, min(target_w, int(round(crop_w * scale))))
    resized_h = max(1, min(target_h, int(round(crop_h * scale))))
    resized_size = (resized_w, resized_h)

    pad_left = (target_w - resized_w) // 2
    pad_right = target_w - resized_w - pad_left
    pad_top = (target_h - resized_h) // 2
    pad_bottom = target_h - resized_h - pad_top

    # 1. Resize RGB Image with Lanczos
    resized_image = np.asarray(
        Image.fromarray(crop_image, mode="RGB").resize(
            resized_size, Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )

    # 2. Resize Mask while preserving thin lines/scratches
    if scale < 1.0:
        # Downsampling: use area interpolation so thin lines/scratches are not skipped
        downscaled = cv2.resize(crop_mask, resized_size, interpolation=cv2.INTER_AREA)
        # Any cell with non-zero masked area is retained as damage
        resized_mask = np.where(downscaled > 10, 255, 0).astype(np.uint8)
    else:
        # Upsampling or 1:1: Nearest neighbor preserves sharp binary mask boundaries
        resized_mask = np.asarray(
            Image.fromarray(crop_mask, mode="L").resize(
                resized_size, Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        resized_mask = np.where(resized_mask > 127, 255, 0).astype(np.uint8)

    # Safety: ensure mask was not erased during resize if crop had masked pixels
    if np.any(crop_mask > 0) and not np.any(resized_mask > 0):
        # Fallback: mark center or non-zero downscaled pixels
        nonzero_down = cv2.resize(crop_mask, resized_size, interpolation=cv2.INTER_NEAREST)
        if np.any(nonzero_down > 0):
            resized_mask = np.where(nonzero_down > 0, 255, 0).astype(np.uint8)
        else:
            resized_mask[resized_h // 2, resized_w // 2] = 255

    # 3. Pad RGB Image (reflect mode avoids artificial step boundaries)
    pad_mode = "reflect" if min(resized_image.shape[:2]) > 1 else "edge"
    padded_image = np.pad(
        resized_image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode=pad_mode,
    )

    # 4. Pad Mask (constant 0 so padded area is never treated as damage)
    padded_mask = np.pad(
        resized_mask,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    return (
        padded_image,
        padded_mask,
        scale,
        resized_size,
        (pad_left, pad_top, pad_right, pad_bottom),
    )


def unpad_and_restore_crop(
    generated: np.ndarray,
    crop_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    padding: Tuple[int, int, int, int],
) -> np.ndarray:
    """Reverse letterbox padding and resize back to original crop dimensions."""
    pad_left, pad_top, _, _ = padding
    resized_w, resized_h = resized_size
    crop_w, crop_h = crop_size

    # Remove padding
    unpadded = generated[
        pad_top : pad_top + resized_h,
        pad_left : pad_left + resized_w,
    ]

    # Resize back to exact original crop dimensions
    reconstructed_crop = np.asarray(
        Image.fromarray(unpadded, mode="RGB").resize(
            (crop_w, crop_h), Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )
    return reconstructed_crop
