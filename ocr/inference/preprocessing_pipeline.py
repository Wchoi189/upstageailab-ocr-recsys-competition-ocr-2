"""Preprocessing pipeline for OCR inference.

This module consolidates image preprocessing logic into a focused pipeline that:
1. Applies optional perspective correction
2. Resizes and pads images to target size
3. Normalizes for model input
4. Generates metadata for coordinate transformations

The pipeline is designed to be testable in isolation and produce consistent
results for both inference and preview generation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config_loader import PreprocessSettings
from .preprocess import apply_optional_perspective_correction, build_transform, preprocess_image
from .preprocessing_metadata import create_preprocessing_metadata

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of preprocessing pipeline.

    Attributes:
        batch: Model input tensor (1, C, H, W)
        preview_image: Processed image in BGR format for preview generation (H, W, C)
        original_shape: Shape of input image before preprocessing (H, W, C)
        metadata: Preprocessing metadata for coordinate transformations
        perspective_matrix: Perspective transform matrix (3, 3) if correction was applied
        original_image: Original image before perspective correction (for inverse transform)
    """

    batch: Any  # torch.Tensor
    preview_image: np.ndarray
    original_shape: tuple[int, int, int]
    metadata: dict[str, Any] | None
    perspective_matrix: np.ndarray | None = None
    original_image: np.ndarray | None = None


class PreprocessingPipeline:
    """Preprocessing pipeline for OCR inference.

    This pipeline encapsulates all image preprocessing steps:
    1. Optional perspective correction (rembg-based)
    2. Resize with aspect ratio preservation (LongestMaxSize)
    3. Padding to square (PadIfNeeded with top_left position)
    4. Normalization for model input
    5. Metadata generation for coordinate transformations

    The pipeline maintains consistency between model input and preview generation,
    ensuring coordinate alignment for polygon overlays.
    """

    def __init__(
        self,
        transform: Callable[[Any], Any] | None = None,
        target_size: int = 640,
        enable_background_normalization: bool = False,
        enable_sepia_enhancement: bool = False,
        enable_clahe: bool = False,
    ):
        """Initialize preprocessing pipeline.

        Args:
            transform: Torchvision transform pipeline (ToTensor + Normalize)
            target_size: Target size for resize and padding (default: 640)
            enable_background_normalization: Whether to apply gray-world normalization
            enable_sepia_enhancement: Whether to apply sepia enhancement
            enable_clahe: Whether to apply CLAHE contrast enhancement
        """
        self._transform = transform
        self._target_size = target_size
        self._enable_background_normalization = enable_background_normalization
        self._enable_sepia_enhancement = enable_sepia_enhancement
        self._enable_clahe = enable_clahe

    def process(
        self,
        image: np.ndarray,
        enable_perspective_correction: bool = False,
        enable_background_removal: bool = False,
        perspective_display_mode: str = "corrected",
        enable_grayscale: bool = False,
        enable_background_normalization: bool | None = None,
        enable_sepia_enhancement: bool | None = None,
        enable_clahe: bool | None = None,
        sepia_display_mode: str = "enhanced",
    ) -> PreprocessingResult | None:
        """Run preprocessing pipeline on an image.

        Args:
            image: Input image in BGR format (H, W, C)
            enable_perspective_correction: Whether to apply perspective correction
            perspective_display_mode: Display mode for perspective correction
                - "corrected": Display corrected image with corrected polygons
                - "original": Display original image with inverse-transformed polygons
            enable_grayscale: Whether to apply grayscale preprocessing
            enable_background_normalization: Override instance background normalization setting
            enable_sepia_enhancement: Override instance sepia enhancement setting
            enable_clahe: Override instance CLAHE enhancement setting

        Returns:
            PreprocessingResult with batch tensor, preview image, and metadata,
            or None if preprocessing fails

        Example:
            >>> pipeline = PreprocessingPipeline(transform=my_transform)
            >>> result = pipeline.process(image_bgr, enable_perspective_correction=True)
            >>> if result:
            ...     predictions = model(result.batch)
            ...     preview = result.preview_image
        """
        if self._transform is None:
            LOGGER.error("Transform not set. Call set_transform() first.")
            return None

        # Stage 1: Optional perspective correction
        original_image_for_display = None
        perspective_transform_matrix = None

        if enable_perspective_correction:
            if perspective_display_mode == "original":
                # Store original for inverse transformation later
                LOGGER.debug("Storing original image before perspective correction")
                original_image_for_display = image.copy()
                image, perspective_transform_matrix = apply_optional_perspective_correction(
                    image,
                    enable_perspective_correction=True,
                    return_matrix=True,
                )
                LOGGER.debug("Perspective correction applied (matrix captured)")
                # Correct and display corrected (default behavior)
                image = apply_optional_perspective_correction(
                    image,
                    enable_perspective_correction=True,
                )
                LOGGER.debug("Perspective correction applied (corrected mode)")

        # Stage 1.5: Optional Background Removal (rembg)
        if enable_background_removal:
            try:
                import cv2
                from rembg import remove

                # rembg expects RGB or BGR, returns RGBA
                # It handles conversion internally, but let's be explicit if needed
                # For now assuming input is BGR as per docstring

                # Run rembg
                # remove() returns the image with alpha channel.
                # We need to composite it over white background to keep 3 channels
                output = remove(image)

                # Convert RGBA to BGR (white background)
                if output.shape[2] == 4:
                    alpha = output[:, :, 3] / 255.0
                    foreground = output[:, :, :3]

                    # Create white background
                    np.ones_like(foreground, dtype=np.uint8) * 255

                    # Composite
                    # output is usually RGB from rembg, but let's check.
                    # rembg doc says "return image as a byte array" if input is bytes, or PIL/ndarray
                    # If ndarray input (cv2), it returns ndarray.
                    # Colorspace: rembg uses PIL internally, so it might return RGB.
                    # Let's verify standard behavior or force conversion.
                    # Best safety: Convert to RGB before sending to rembg to be sure

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    output_rgba = remove(image_rgb)

                    # Extract alpha
                    alpha = output_rgba[:, :, 3] / 255.0
                    foreground_rgb = output_rgba[:, :, :3]

                    # Composite over white
                    bg_color = [255, 255, 255]
                    composite = np.zeros_like(foreground_rgb)
                    for c in range(3):
                        composite[:, :, c] = alpha * foreground_rgb[:, :, c] + (1 - alpha) * bg_color[c]

                    # Convert back to BGR
                    image = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
                    LOGGER.info("Background removal applied via rembg")

            except ImportError:
                LOGGER.warning("rembg not installed. Skipping background removal.")
            except Exception as e:
                LOGGER.error("Background removal failed: %s", e)

        # Stage 2: Optional grayscale conversion (after perspective correction)
        if enable_grayscale:
            import cv2

            # Convert BGR → GRAY
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Convert GRAY → BGR (maintain 3-channel input for model)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            LOGGER.debug("Grayscale preprocessing applied")

        # Stage 3: Capture original shape before preprocessing
        original_shape = image.shape
        original_h, original_w = original_shape[:2]

        # Stage 4: Resize, pad, and normalize
        try:
            # Get both model input tensor and processed preview image
            # Use parameter override if provided, else fall back to instance variable
            use_background_norm = (
                enable_background_normalization if enable_background_normalization is not None else self._enable_background_normalization
            )
            use_sepia = enable_sepia_enhancement if enable_sepia_enhancement is not None else self._enable_sepia_enhancement
            use_clahe = enable_clahe if enable_clahe is not None else self._enable_clahe

            # If sepia is enabled but display mode is "original", we need separate passes
            if use_sepia and sepia_display_mode == "original":
                # Pass 1: With sepia for inference input
                batch, _ = preprocess_image(
                    image,
                    self._transform,
                    target_size=self._target_size,
                    return_processed_image=True,
                    enable_background_normalization=use_background_norm,
                    enable_sepia_enhancement=True,
                    enable_clahe=use_clahe,
                )

                # Pass 2: Without sepia for preview output
                _, preview_image_bgr = preprocess_image(
                    image,
                    self._transform,
                    target_size=self._target_size,
                    return_processed_image=True,
                    enable_background_normalization=use_background_norm,
                    enable_sepia_enhancement=False,
                    enable_clahe=use_clahe,
                )
            else:
                # Standard single pass
                batch, preview_image_bgr = preprocess_image(
                    image,
                    self._transform,
                    target_size=self._target_size,
                    return_processed_image=True,
                    enable_background_normalization=use_background_norm,
                    enable_sepia_enhancement=use_sepia,
                    enable_clahe=use_clahe,
                )

            # Verify preview dimensions
            preview_h, preview_w = preview_image_bgr.shape[:2]
            if preview_h != self._target_size or preview_w != self._target_size:
                LOGGER.warning(
                    f"Preview size mismatch: expected {self._target_size}x{self._target_size}, "
                    f"got {preview_w}x{preview_h}. Original: {original_w}x{original_h}"
                )

            # Stage 5: Generate metadata for coordinate transformations
            metadata = create_preprocessing_metadata(
                (original_h, original_w),
                target_size=self._target_size,
            )

            return PreprocessingResult(
                batch=batch,
                preview_image=preview_image_bgr,
                original_shape=original_shape,
                metadata=metadata,
                perspective_matrix=perspective_transform_matrix,
                original_image=original_image_for_display,
            )

        except Exception:
            LOGGER.exception("Preprocessing failed")
            return None

    def process_for_original_display(
        self,
        original_image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Preprocess original image for inverse perspective display mode.

        When perspective correction is enabled with "original" display mode,
        this method creates a preview from the original (uncorrected) image
        with matching dimensions and metadata.

        Args:
            original_image: Original image before perspective correction (BGR)

        Returns:
            Tuple of (preview_image_bgr, metadata) or None on failure
        """
        if self._transform is None:
            LOGGER.error("Transform not set")
            return None

        try:
            # Preprocess original image to get matching preview
            _, preview_image_bgr = preprocess_image(
                original_image,
                self._transform,
                target_size=self._target_size,
                return_processed_image=True,
                enable_background_normalization=self._enable_background_normalization,
            )

            # Calculate metadata for original image dimensions
            original_h, original_w = original_image.shape[:2]
            metadata = create_preprocessing_metadata(
                (original_h, original_w),
                target_size=self._target_size,
            )

            LOGGER.debug(f"Original preview created: {original_w}x{original_h} → {preview_image_bgr.shape[1]}x{preview_image_bgr.shape[0]}")

            return preview_image_bgr, metadata

        except Exception:
            LOGGER.exception("Failed to create original preview")
            return None

    def set_transform(self, transform: Callable[[Any], Any]) -> None:
        """Set the torchvision transform pipeline.

        Args:
            transform: Transform pipeline (should include ToTensor and Normalize)
        """
        self._transform = transform

    def set_target_size(self, target_size: int) -> None:
        """Set the target size for preprocessing.

        Args:
            target_size: Target size for resize and padding
        """
        self._target_size = target_size

    @classmethod
    def from_settings(
        cls,
        settings: PreprocessSettings,
    ) -> PreprocessingPipeline:
        """Create pipeline from preprocessing settings.

        Args:
            settings: Preprocessing settings from model config

        Returns:
            Configured preprocessing pipeline
        """
        # Extract target size from settings
        target_size = 640  # Default
        if settings.image_size:
            if isinstance(settings.image_size, tuple):
                target_size = max(settings.image_size)
            else:
                target_size = settings.image_size

        # Build transform pipeline
        transform = build_transform(settings)

        return cls(
            transform=transform,
            target_size=target_size,
            enable_background_normalization=settings.enable_background_normalization,
            enable_sepia_enhancement=settings.enable_sepia_enhancement,
            enable_clahe=settings.enable_clahe,
        )


__all__ = [
    "PreprocessingPipeline",
    "PreprocessingResult",
]
