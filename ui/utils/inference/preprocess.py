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
    """Create a torchvision transform pipeline from preprocessing settings."""
    if transforms is None:
        raise RuntimeError("Torchvision transforms are not available. Install the vision extras.")

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(settings.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.normalization.mean, std=settings.normalization.std),
        ]
    )


def preprocess_image(image: Any, transform: Callable[[Any], Any]) -> Any:
    """Apply preprocessing transform to an image and return a batched tensor.

    The input ``image`` is expected to be a BGR numpy array (OpenCV convention).
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = transform(image_rgb)
    if torch is None:
        raise RuntimeError("Torch is not available to create inference batches.")
    return tensor.unsqueeze(0)


def apply_optional_perspective_correction(
    image_bgr: Any,
    enable_perspective_correction: bool,
) -> Any:
    """
    Optionally apply rembg-based perspective correction before standard transforms.

    Args:
        image_bgr: Input image in BGR format.
        enable_perspective_correction: If False, the image is returned unchanged.

    Returns:
        Potentially perspective-corrected BGR image.
    """

    if not enable_perspective_correction:
        return image_bgr

    try:
        image_no_bg, mask = remove_background_and_mask(image_bgr)
        corrected, _result = correct_perspective_from_mask(image_no_bg, mask)
        return corrected
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Perspective correction failed or unavailable: %s", exc)
        return image_bgr
