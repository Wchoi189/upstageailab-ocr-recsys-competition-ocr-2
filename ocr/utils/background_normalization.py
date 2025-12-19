"""Gray-world background normalization for OCR preprocessing.

This module provides color normalization to remove tinted backgrounds
from document images, improving OCR accuracy on documents with colored
backgrounds.
"""

from __future__ import annotations

import numpy as np


def normalize_gray_world(img: np.ndarray) -> np.ndarray:
    """Apply gray-world normalization to remove background color tints.

    Scales color channels so their average matches a neutral gray,
    effectively removing color casts from tinted backgrounds.

    Args:
        img: BGR numpy array (OpenCV convention), shape (H, W, 3)

    Returns:
        Normalized BGR numpy array with same shape and dtype uint8

    Example:
        >>> img = cv2.imread("tinted_document.jpg")
        >>> normalized = normalize_gray_world(img)
        >>> cv2.imwrite("normalized.jpg", normalized)
    """
    # Calculate average color for each channel (B, G, R)
    b_avg, g_avg, r_avg = img.mean(axis=(0, 1))

    # Calculate neutral gray average
    gray_avg = (b_avg + g_avg + r_avg) / 3

    # Handle edge case: avoid division by zero
    if b_avg == 0 or g_avg == 0 or r_avg == 0:
        return img.copy()

    # Calculate scaling factors for each channel
    scale_factors = np.array([gray_avg / b_avg, gray_avg / g_avg, gray_avg / r_avg])

    # Apply scaling and clip to valid range
    result = img.astype(np.float32) * scale_factors
    return np.clip(result, 0, 255).astype(np.uint8)
