from __future__ import annotations

"""Image preprocessing helpers."""

import logging
from collections.abc import Callable
from typing import Any

import cv2

from ocr.utils.perspective_correction import (
    correct_perspective_from_mask,
    remove_background_and_mask,
)

from .config_loader import PreprocessSettings
from .dependencies import torch, transforms

LOGGER = logging.getLogger(__name__)


def build_transform(settings: PreprocessSettings):
    """Create a torchvision transform pipeline from preprocessing settings.

    BUG-001: Updated to use LongestMaxSize + PadIfNeeded logic (matching postprocessing
    assumptions) instead of transforms.Resize. The actual resize/padding is applied
    in preprocess_image() using OpenCV before converting to tensor.
    """
    if transforms is None:
        raise RuntimeError("Torchvision transforms are not available. Install the vision extras.")

    # BUG-001: Remove Resize from transform pipeline - we'll apply LongestMaxSize + PadIfNeeded
    # manually in preprocess_image() to match postprocessing expectations.
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            # Note: Resize removed - handled in preprocess_image() with LongestMaxSize + PadIfNeeded
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.normalization.mean, std=settings.normalization.std),
        ]
    )


def preprocess_image(
    image: Any,
    transform: Callable[[Any], Any],
    target_size: int = 640,
    return_processed_image: bool = False
) -> Any | tuple[Any, Any]:
    """Apply preprocessing transform to an image and return a batched tensor.

    BUG-001: Applies LongestMaxSize + PadIfNeeded (matching postprocessing assumptions)
    before converting to tensor. This ensures coordinate alignment between preprocessing
    and postprocessing.

    Args:
        image: BGR numpy array (OpenCV convention)
        transform: Torchvision transform pipeline (should not include Resize)
        target_size: Target size for LongestMaxSize and PadIfNeeded (default: 640)
        return_processed_image: If True, also return the processed BGR image before tensor conversion

    Returns:
        Batched tensor ready for model inference, or tuple of (tensor, processed_image_bgr) if return_processed_image=True
    """
    # BUG-001: Apply LongestMaxSize + PadIfNeeded to match postprocessing expectations.
    # This preserves aspect ratio and pads to a square (typically 640x640) with top_left
    # padding position, matching what decode_polygons_with_head and fallback_postprocess expect.
    # Work on a copy to avoid modifying the input image in place.
    processed_image = image.copy()
    original_h, original_w = processed_image.shape[:2]

    # LongestMaxSize: scale longest side to target_size, preserving aspect ratio
    max_side = max(original_h, original_w)
    if max_side > 0:
        scale = target_size / max_side
        scaled_h = int(round(original_h * scale))
        scaled_w = int(round(original_w * scale))
    else:
        scaled_h, scaled_w = original_h, original_w

    # Resize preserving aspect ratio
    if scaled_h != original_h or scaled_w != original_w:
        processed_image = cv2.resize(processed_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # PadIfNeeded: pad to target_size x target_size with top_left position (padding at bottom/right)
    # BUG-001: Keep top_left padding for model compatibility (matches training pipeline)
    # The preview image will be centered separately for display purposes
    pad_h = target_size - scaled_h
    pad_w = target_size - scaled_w
    if pad_h > 0 or pad_w > 0:
        processed_image = cv2.copyMakeBorder(
            processed_image,
            0, pad_h, 0, pad_w,  # top, bottom, left, right (top_left padding)
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )

    # BUG-001: Optionally return the processed image before RGB conversion for preview
    processed_image_bgr = processed_image.copy() if return_processed_image else None

    # Convert BGR to RGB for PIL/torchvision transforms
    image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    tensor = transform(image_rgb)
    if torch is None:
        raise RuntimeError("Torch is not available to create inference batches.")

    result = tensor.unsqueeze(0)
    if return_processed_image:
        return result, processed_image_bgr
    return result


def apply_optional_perspective_correction(
    image_bgr: Any,
    enable_perspective_correction: bool,
    return_matrix: bool = False,
) -> Any | tuple[Any, Any]:
    """
    Optionally apply rembg-based perspective correction before standard transforms.

    Args:
        image_bgr: Input image in BGR format.
        enable_perspective_correction: If False, the image is returned unchanged.
        return_matrix: If True, return tuple of (corrected_image, transform_matrix).

    Returns:
        Potentially perspective-corrected BGR image, or tuple (corrected_image, transform_matrix)
        if return_matrix is True.
    """

    if not enable_perspective_correction:
        if return_matrix:
            import numpy as np
            return image_bgr, np.eye(3, dtype=np.float32)
        return image_bgr

    try:
        image_no_bg, mask = remove_background_and_mask(image_bgr)
        if return_matrix:
            corrected, _result, matrix = correct_perspective_from_mask(
                image_no_bg, mask, return_matrix=True
            )
            return corrected, matrix
        else:
            corrected, _result = correct_perspective_from_mask(image_no_bg, mask)
            return corrected
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Perspective correction failed or unavailable: %s", exc)
        if return_matrix:
            import numpy as np
            return image_bgr, np.eye(3, dtype=np.float32)
        return image_bgr
