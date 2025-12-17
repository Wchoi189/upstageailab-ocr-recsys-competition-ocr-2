"""Image loading utilities for inference pipeline.

This module consolidates image I/O, EXIF normalization, and format conversion
for the OCR inference pipeline. It provides a clean interface for loading images
from various sources (file paths, PIL images, numpy arrays) and converting them
to the BGR format expected by OpenCV-based models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ocr.utils.image_loading import load_image_optimized
from ocr.utils.orientation import get_exif_orientation, normalize_pil_image

LOGGER = logging.getLogger(__name__)


@dataclass
class LoadedImage:
    """Container for loaded image data and metadata.

    Attributes:
        image: BGR numpy array ready for OpenCV processing (H, W, C)
        orientation: EXIF orientation value (1-8)
        raw_width: Original image width before EXIF normalization
        raw_height: Original image height before EXIF normalization
        canonical_width: Image width after EXIF normalization
        canonical_height: Image height after EXIF normalization
    """

    image: np.ndarray
    orientation: int
    raw_width: int
    raw_height: int
    canonical_width: int
    canonical_height: int


class ImageLoader:
    """Handles image loading, EXIF normalization, and format conversion for inference.

    This class provides a unified interface for loading images from various sources
    and converting them to the BGR numpy array format expected by OpenCV-based models.
    It handles:
    - Loading from file paths (with TurboJPEG optimization for JPEG files)
    - EXIF orientation normalization
    - Format conversion (RGB → BGR, various PIL modes → RGB)
    - Metadata extraction (dimensions, orientation)

    The loader ensures proper resource cleanup for PIL images and provides detailed
    error messages for debugging.
    """

    def __init__(
        self,
        use_turbojpeg: bool = True,
        turbojpeg_fallback: bool = True,
    ):
        """Initialize image loader.

        Args:
            use_turbojpeg: Whether to use TurboJPEG for JPEG files (faster)
            turbojpeg_fallback: Whether to fallback to PIL if TurboJPEG fails
        """
        self.use_turbojpeg = use_turbojpeg
        self.turbojpeg_fallback = turbojpeg_fallback

    def load_from_path(self, image_path: str | Path) -> LoadedImage | None:
        """Load image from file path with EXIF normalization.

        Loads an image file, applies EXIF orientation normalization, converts to BGR
        format, and extracts metadata. This is the primary method for loading images
        for inference.

        Args:
            image_path: Path to image file (JPEG, PNG, etc.)

        Returns:
            LoadedImage with BGR array and metadata, or None if loading fails

        Example:
            >>> loader = ImageLoader()
            >>> loaded = loader.load_from_path("image.jpg")
            >>> if loaded:
            ...     print(f"Image shape: {loaded.image.shape}")
            ...     print(f"Orientation: {loaded.orientation}")
        """
        try:
            # Load image using optimized loader (TurboJPEG for JPEG if available)
            pil_image = load_image_optimized(
                image_path,
                use_turbojpeg=self.use_turbojpeg,
                turbojpeg_fallback=self.turbojpeg_fallback,
            )

            # Extract raw dimensions before normalization
            raw_width, raw_height = pil_image.size

            # Apply EXIF normalization (rotate/flip based on EXIF orientation tag)
            normalized_image, orientation = normalize_pil_image(pil_image)

            # Convert to BGR numpy array for OpenCV
            bgr_image = self._pil_to_bgr_array(normalized_image)

            # Extract canonical dimensions (after normalization)
            canonical_height, canonical_width = bgr_image.shape[:2]

            # Clean up PIL images
            self._cleanup_pil_images(pil_image, normalized_image)

            return LoadedImage(
                image=bgr_image,
                orientation=orientation,
                raw_width=raw_width,
                raw_height=raw_height,
                canonical_width=canonical_width,
                canonical_height=canonical_height,
            )

        except FileNotFoundError:
            LOGGER.error(f"Image file not found: {image_path}")
            return None
        except OSError as e:
            LOGGER.error(f"Failed to read image at path: {image_path} - {e}")
            return None
        except Exception as e:
            LOGGER.error(f"Unexpected error loading image {image_path}: {e}")
            return None

    def load_from_pil(self, pil_image: Image.Image) -> LoadedImage:
        """Load from PIL Image with EXIF normalization.

        Args:
            pil_image: PIL Image object

        Returns:
            LoadedImage with BGR array and metadata
        """
        # Extract raw dimensions
        raw_width, raw_height = pil_image.size

        # Get EXIF orientation
        orientation = get_exif_orientation(pil_image)

        # Apply EXIF normalization
        normalized_image, _ = normalize_pil_image(pil_image)

        # Convert to BGR numpy array
        bgr_image = self._pil_to_bgr_array(normalized_image)

        # Extract canonical dimensions
        canonical_height, canonical_width = bgr_image.shape[:2]

        # Clean up if we created a new image
        if normalized_image is not pil_image:
            normalized_image.close()

        return LoadedImage(
            image=bgr_image,
            orientation=orientation,
            raw_width=raw_width,
            raw_height=raw_height,
            canonical_width=canonical_width,
            canonical_height=canonical_height,
        )

    def load_from_array(
        self,
        image_array: np.ndarray,
        color_space: str = "BGR",
    ) -> LoadedImage:
        """Load from numpy array (already loaded image).

        For images that are already in memory as numpy arrays. Assumes no EXIF
        orientation correction is needed (orientation = 1).

        Args:
            image_array: Numpy array in HWC format
            color_space: Color space of input array ("BGR", "RGB", "GRAY")

        Returns:
            LoadedImage with BGR array and metadata

        Raises:
            ValueError: If color_space is invalid or array format is incorrect
        """
        if image_array.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got shape {image_array.shape}")

        # Convert to BGR if needed
        if color_space == "BGR":
            bgr_image = image_array
        elif color_space == "RGB":
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif color_space == "GRAY":
            if image_array.ndim == 2:
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            else:
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unsupported color_space: {color_space}")

        height, width = bgr_image.shape[:2]

        return LoadedImage(
            image=bgr_image,
            orientation=1,  # No EXIF orientation for array inputs
            raw_width=width,
            raw_height=height,
            canonical_width=width,
            canonical_height=height,
        )

    @staticmethod
    def _pil_to_bgr_array(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to BGR numpy array.

        Handles various PIL image modes by converting to RGB first if needed,
        then converting to BGR for OpenCV compatibility.

        Args:
            pil_image: PIL Image in any mode

        Returns:
            BGR numpy array (H, W, 3)
        """
        # Convert to RGB if not already
        rgb_image = pil_image
        if pil_image.mode != "RGB":
            rgb_image = pil_image.convert("RGB")

        # Convert to numpy array (RGB)
        image_array = np.asarray(rgb_image)

        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Clean up temporary RGB image if we created it
        if rgb_image is not pil_image:
            rgb_image.close()

        return bgr_image

    @staticmethod
    def _cleanup_pil_images(
        original: Image.Image,
        normalized: Image.Image,
    ) -> None:
        """Clean up PIL images to avoid resource leaks.

        Only closes images that were created during normalization,
        not the original image passed in.

        Args:
            original: Original PIL image (not closed)
            normalized: Normalized PIL image (closed if different from original)
        """
        if normalized is not original:
            normalized.close()


__all__ = [
    "ImageLoader",
    "LoadedImage",
]
