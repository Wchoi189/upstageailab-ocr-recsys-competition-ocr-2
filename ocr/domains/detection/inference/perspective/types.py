from __future__ import annotations

"""Data types for perspective correction."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LineQualityReport:
    """Advanced quality metrics for fitted rectangle."""

    decision: str
    metrics: dict[str, Any]
    passes: dict[str, bool]
    fail_reasons: list[str]


@dataclass
class MaskRectangleResult:
    """Result of fitting a rectangle to a foreground mask."""

    corners: np.ndarray | None
    raw_corners: np.ndarray | None
    contour_area: float
    hull_area: float
    mask_area: float
    contour: np.ndarray | None
    hull: np.ndarray | None
    reason: str | None = None
    line_quality: LineQualityReport | None = None
    used_epsilon: float | None = None


__all__ = ["LineQualityReport", "MaskRectangleResult"]
