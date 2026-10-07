from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.preservation import calculate_preservation_guard


def calculate_preservation_metrics(
    original: np.ndarray, final: np.ndarray, mask: np.ndarray
) -> Dict[str, Any]:
    """Measure preservation using only pixels outside the original damage mask."""
    return calculate_preservation_guard(original, final, mask)


__all__ = ["calculate_preservation_metrics"]
