from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Method = Literal["canny", "morph", "hough"]


@dataclass(frozen=True)
class BorderRemovalMetrics:
    method: Method
    processing_time_ms: float | None = None
    confidence: float | None = None
    cropped_area_ratio: float | None = None
    notes: str | None = None


@dataclass(frozen=True)
class BorderRemovalResult:
    image: np.ndarray
    metrics: BorderRemovalMetrics
    debug: dict[str, Any]


class BorderRemover:
    def __init__(
        self,
        method: Method = "canny",
        *,
        min_area_ratio: float = 0.5,
        confidence_threshold: float = 0.8,
        fallback_to_original: bool = True,
    ) -> None:
        self.method = method
        self.min_area_ratio = float(min_area_ratio)
        self.confidence_threshold = float(confidence_threshold)
        self.fallback_to_original = bool(fallback_to_original)

    def remove_border(self, image: np.ndarray) -> BorderRemovalResult:
        if self.method == "canny":
            from .border_removal_canny import remove_border_canny

            return remove_border_canny(
                image,
                min_area_ratio=self.min_area_ratio,
                confidence_threshold=self.confidence_threshold,
                fallback_to_original=self.fallback_to_original,
            )
        if self.method == "morph":
            from .border_removal_morph import remove_border_morph

            return remove_border_morph(
                image,
                min_area_ratio=self.min_area_ratio,
                confidence_threshold=self.confidence_threshold,
                fallback_to_original=self.fallback_to_original,
            )
        if self.method == "hough":
            from .border_removal_hough import remove_border_hough

            return remove_border_hough(
                image,
                min_area_ratio=self.min_area_ratio,
                confidence_threshold=self.confidence_threshold,
                fallback_to_original=self.fallback_to_original,
            )
        raise ValueError(f"Unknown method: {self.method}")
