from __future__ import annotations

import time
from typing import Any

import numpy as np

from .border_remover import BorderRemovalMetrics, BorderRemovalResult


def remove_border_hough(
    image: np.ndarray,
    *,
    min_area_ratio: float,
    confidence_threshold: float,
    fallback_to_original: bool,
) -> BorderRemovalResult:
    start = time.perf_counter()

    debug: dict[str, Any] = {
        "min_area_ratio": float(min_area_ratio),
        "confidence_threshold": float(confidence_threshold),
        "fallback_to_original": bool(fallback_to_original),
    }

    # Placeholder implementation.
    # Expected algorithm: canny -> hough lines -> cluster -> intersections -> crop
    cropped = image
    confidence = 0.0

    processing_ms = (time.perf_counter() - start) * 1000.0
    metrics = BorderRemovalMetrics(method="hough", processing_time_ms=processing_ms, confidence=confidence)

    return BorderRemovalResult(image=cropped, metrics=metrics, debug=debug)
