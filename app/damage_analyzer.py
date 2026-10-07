from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.analysis import _level, analyze_damage
from core.confidence import evaluate_reconstruction_confidence


def estimate_confidence(
    analysis: Dict[str, Any],
    original: np.ndarray | None = None,
    reconstructed: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Return contextual confidence heuristic, evaluating empirical images if provided."""
    if original is not None and reconstructed is not None and mask is not None:
        return evaluate_reconstruction_confidence(original, reconstructed, mask, analysis)

    # Heuristic based on damage properties alone
    area_score = max(0.0, 100.0 - min(100.0, analysis["mask_area_pct"] * 2.0))
    context_score = min(
        100.0,
        analysis["context_pixels"] / max(1, analysis["mask_width"] * analysis["mask_height"]) * 100.0,
    )
    boundary_score = max(0.0, 100.0 - analysis["boundary_complexity"])
    structure_score = analysis["structural_content"]
    score = 0.30 * area_score + 0.30 * context_score + 0.20 * boundary_score + 0.20 * structure_score
    return {
        "score": float(np.clip(score, 0.0, 100.0)),
        "context_support": _level(context_score),
        "boundary_continuity": _level(boundary_score),
        "structural_continuity": _level(structure_score),
        "heuristic_note": "Contextual reconstruction confidence derived from area, context support, and structure.",
    }


__all__ = ["analyze_damage", "estimate_confidence", "_level"]
