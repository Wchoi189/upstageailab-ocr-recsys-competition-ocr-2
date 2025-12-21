"""Sepia enhancement with CLAHE for OCR preprocessing.

This module provides sepia tone enhancement combined with adaptive contrast
(CLAHE) for improving OCR accuracy on low-contrast or aged document images.

Validated Performance (experiment 20251220_154834):
    - Edge improvement: +164.0% vs baseline
    - Contrast boost: +8.2
    - Processing time: ~25ms
"""

from __future__ import annotations

import cv2
import numpy as np


def enhance_sepia(img: np.ndarray) -> np.ndarray:
    """Apply classic sepia tone transformation.

    Standard sepia matrix (Albumentations version) without contrast enhancement.

    Args:
        img: BGR numpy array (OpenCV convention), shape (H, W, 3)

    Returns:
        Sepia-toned BGR numpy array with same shape and dtype uint8

    Example:
        >>> import cv2
        >>> img = cv2.imread("document.jpg")
        >>> sepia_img = enhance_sepia(img)
        >>> cv2.imwrite("sepia.jpg", sepia_img)
    """
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Classic sepia matrix (same as Albumentations)
    # Formula: R' = 0.393*R + 0.769*G + 0.189*B
    #          G' = 0.349*R + 0.686*G + 0.168*B
    #          B' = 0.272*R + 0.534*G + 0.131*B
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],  # Red channel
        [0.349, 0.686, 0.168],  # Green channel
        [0.272, 0.534, 0.131],  # Blue channel
    ])

    sepia = cv2.transform(img_rgb, sepia_matrix)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    # Convert back to BGR
    return cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)


def enhance_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Adaptive contrast enhancement that preserves local details without over-amplifying noise.

    Args:
        img: BGR numpy array (OpenCV convention), shape (H, W, 3)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_size: Size of grid for histogram equalization (default: 8x8)

    Returns:
        Contrast-enhanced BGR numpy array with same shape and dtype uint8

    Example:
        >>> import cv2
        >>> img = cv2.imread("low_contrast.jpg")
        >>> enhanced = enhance_clahe(img)
        >>> cv2.imwrite("enhanced.jpg", enhanced)
    """
    # Convert to LAB color space for luminance-only enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L (luminance) channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_channel)

    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def enhance_sepia_clahe(img: np.ndarray) -> np.ndarray:
    """Apply sepia + CLAHE enhancement (combined - legacy function).

    DEPRECATED: Use enhance_sepia() and enhance_clahe() separately for finer control.

    Args:
        img: BGR numpy array (OpenCV convention), shape (H, W, 3)

    Returns:
        Enhanced BGR numpy array with same shape and dtype uint8
    """
    sepia_img = enhance_sepia(img)
    return enhance_clahe(sepia_img)
