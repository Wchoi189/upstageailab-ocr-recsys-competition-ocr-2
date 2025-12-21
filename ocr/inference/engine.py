from __future__ import annotations

"""High-level inference engine orchestration.

This module provides the InferenceEngine class, which is a thin wrapper around
InferenceOrchestrator. It maintains backward compatibility with existing code
while delegating all work to the new modular architecture.
"""

import logging
import warnings
from typing import Any

import numpy as np

# Suppress Pydantic v2 configuration warnings from dependencies
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:", category=UserWarning)
warnings.filterwarnings("ignore", message="'allow_population_by_field_name' has been renamed to 'validate_by_name'", category=UserWarning)

from .dependencies import OCR_MODULES_AVAILABLE
from .image_loader import ImageLoader
from .orchestrator import InferenceOrchestrator
from .utils import generate_mock_predictions
from .utils import get_available_checkpoints as scan_checkpoints

LOGGER = logging.getLogger(__name__)


class InferenceEngine:
    """OCR Inference Engine for real-time predictions.

    This class is a thin wrapper around InferenceOrchestrator that maintains
    backward compatibility with existing code while delegating to the new
    modular architecture.
    """

    def __init__(self) -> None:
        """Initialize inference engine with orchestrator delegation."""
        self._orchestrator = InferenceOrchestrator()
        self._image_loader = ImageLoader()

        # Expose device for backward compatibility
        self.device = self._orchestrator.model_manager.device

        # Legacy attributes for backward compatibility (deprecated, use orchestrator)
        self.model = None  # Populated after load_model() for backward compat
        self.trainer = None
        self.config: Any | None = None

        LOGGER.info("InferenceEngine initialized with orchestrator (device: %s)", self.device)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

    def cleanup(self) -> None:
        """Clean up resources and prevent memory leaks.

        Delegates to orchestrator for cleanup.
        """
        LOGGER.info("Cleaning up InferenceEngine resources...")
        self._orchestrator.cleanup()

        # Clear legacy attributes
        self.model = None
        self.trainer = None
        self.config = None

        LOGGER.info("InferenceEngine cleanup completed")

    # Public API ---------------------------------------------------------
    def load_model(self, checkpoint_path: str, config_path: str | None = None) -> bool:
        """Load model from checkpoint.

        Delegates to orchestrator for model loading.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config file

        Returns:
            True if model loaded successfully
        """
        if not OCR_MODULES_AVAILABLE:
            LOGGER.error("OCR modules are not installed. Cannot load a real model.")
            return False

        # Delegate to orchestrator
        success = self._orchestrator.load_model(checkpoint_path, config_path)

        if success:
            # Update legacy attributes for backward compatibility
            self.model = self._orchestrator.model_manager.model
            bundle = self._orchestrator.model_manager.get_config_bundle()
            if bundle is not None:
                self.config = bundle.raw_config

        return success

    def update_postprocessor_params(
        self,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
    ) -> None:
        """Update postprocessing parameters.

        Delegates to orchestrator for parameter updates.

        Args:
            binarization_thresh: Binarization threshold
            box_thresh: Box confidence threshold
            max_candidates: Maximum number of candidates
            min_detection_size: Minimum detection size in pixels
        """
        self._orchestrator.update_postprocessor_params(
            binarization_thresh=binarization_thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            min_detection_size=min_detection_size,
        )

    def predict_array(
        self,
        image_array: np.ndarray,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
        return_preview: bool = True,
        enable_perspective_correction: bool | None = None,
        perspective_display_mode: str = "corrected",
        enable_grayscale: bool = False,
        enable_background_normalization: bool = False,
        enable_sepia_enhancement: bool = False,
        enable_clahe: bool = False,
    ) -> dict[str, Any] | None:
        """Predict on an image array (numpy) directly.

        Delegates to orchestrator for prediction.

        Args:
            image_array: Image as numpy array (BGR format)
            binarization_thresh: Binarization threshold override
            box_thresh: Box threshold override
            max_candidates: Max candidates override
            min_detection_size: Minimum detection size override
            return_preview: If True, attaches preview image
            enable_perspective_correction: Enable perspective correction before inference
            perspective_display_mode: "corrected" or "original" display mode
            enable_grayscale: Enable grayscale preprocessing
            enable_sepia_enhancement: Enable sepia tone transformation
            enable_clahe: Enable CLAHE contrast enhancement

        Returns:
            Predictions dict with 'polygons', 'texts', 'confidences' keys
        """
        LOGGER.info("Starting prediction for numpy array with shape: %s", image_array.shape)

        if self.model is None:
            LOGGER.warning("Model not loaded. Returning mock predictions.")
            return generate_mock_predictions()

        # Update postprocessor parameters if provided
        if any(value is not None for value in (binarization_thresh, box_thresh, max_candidates, min_detection_size)):
            self.update_postprocessor_params(
                binarization_thresh=binarization_thresh,
                box_thresh=box_thresh,
                max_candidates=max_candidates,
                min_detection_size=min_detection_size,
            )

        # Delegate to orchestrator
        return self._orchestrator.predict(
            image=image_array,
            return_preview=return_preview,
            enable_perspective_correction=enable_perspective_correction or False,
            perspective_display_mode=perspective_display_mode,
            enable_grayscale=enable_grayscale,
            enable_background_normalization=enable_background_normalization,
            enable_sepia_enhancement=enable_sepia_enhancement,
            enable_clahe=enable_clahe,
        )

    def predict_image(
        self,
        image_path: str,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
        return_preview: bool = True,
        enable_perspective_correction: bool | None = None,
        perspective_display_mode: str = "corrected",
        enable_grayscale: bool = False,
        enable_background_normalization: bool = False,
        enable_sepia_enhancement: bool = False,
        enable_clahe: bool = False,
    ) -> dict[str, Any] | None:
        """Predict on an image file (legacy file path-based API).

        Loads image from file and delegates to orchestrator.

        Args:
            image_path: Path to image file
            binarization_thresh: Binarization threshold override
            box_thresh: Box threshold override
            max_candidates: Max candidates override
            min_detection_size: Minimum detection size override
            return_preview: If True, attaches preview image
            enable_perspective_correction: Enable perspective correction before inference
            perspective_display_mode: "corrected" or "original" display mode
            enable_grayscale: Enable grayscale preprocessing
            enable_sepia_enhancement: Enable sepia tone transformation
            enable_clahe: Enable CLAHE contrast enhancement

        Returns:
            Predictions dict with 'polygons', 'texts', 'confidences' keys
        """
        LOGGER.info(f"Starting prediction for image: {image_path}")

        if self.model is None:
            LOGGER.warning("Model not loaded. Returning mock predictions.")
            return generate_mock_predictions()

        # Update postprocessor parameters if provided
        if any(value is not None for value in (binarization_thresh, box_thresh, max_candidates, min_detection_size)):
            self.update_postprocessor_params(
                binarization_thresh=binarization_thresh,
                box_thresh=box_thresh,
                max_candidates=max_candidates,
                min_detection_size=min_detection_size,
            )

        # Load image with EXIF normalization
        loaded = self._image_loader.load_from_path(image_path)
        if loaded is None:
            LOGGER.error(f"Failed to load image: {image_path}")
            return None

        LOGGER.info(f"Image loaded successfully. Shape: {loaded.image.shape}")

        # Delegate to orchestrator via predict_array
        return self._orchestrator.predict(
            image=loaded.image,
            return_preview=return_preview,
            enable_perspective_correction=enable_perspective_correction or False,
            perspective_display_mode=perspective_display_mode,
            enable_grayscale=enable_grayscale,
            enable_background_normalization=enable_background_normalization,
            enable_sepia_enhancement=enable_sepia_enhancement,
            enable_clahe=enable_clahe,
        )


def run_inference_on_image(
    image_path: str | np.ndarray,
    checkpoint_path: str,
    config_path: str | None = None,
    binarization_thresh: float | None = None,
    box_thresh: float | None = None,
    max_candidates: int | None = None,
    min_detection_size: int | None = None,
    return_preview: bool = True,
) -> dict[str, Any] | None:
    """Run inference on an image (file path or numpy array).

    Supports both legacy file path API and new numpy array API for eliminating
    tempfile overhead.

    Args:
        image_path: Either file path (str) or numpy array (optimized path)
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file
        binarization_thresh: Binarization threshold override
        box_thresh: Box threshold override
        max_candidates: Max candidates override
        min_detection_size: Minimum detection size override
        return_preview: If True, maps polygons to preview space and attaches preview image.

    Returns:
        Predictions dict with 'polygons', 'texts', 'confidences' keys

    See: artifacts/implementation_plans/2025-11-12_plan-004-revised-inference-consolidation.md
    """
    engine = InferenceEngine()
    if not engine.load_model(checkpoint_path, config_path):
        LOGGER.error("Failed to load model in convenience function.")
        return None

    # NEW: Support numpy array input (eliminates tempfile overhead)
    if isinstance(image_path, np.ndarray):
        return engine.predict_array(
            image_array=image_path,
            binarization_thresh=binarization_thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            min_detection_size=min_detection_size,
            return_preview=return_preview,
        )

    # LEGACY: Support file path input (backward compatible)
    return engine.predict_image(
        image_path=image_path,
        binarization_thresh=binarization_thresh,
        box_thresh=box_thresh,
        max_candidates=max_candidates,
        min_detection_size=min_detection_size,
        return_preview=return_preview,
    )


def get_available_checkpoints() -> list[str]:
    return scan_checkpoints()
