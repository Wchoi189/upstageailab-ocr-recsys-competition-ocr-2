"""Inference engine lifecycle management service."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from apps.shared.backend_shared.inference import InferenceEngine

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for managing InferenceEngine lifecycle and predictions.

    Handles engine initialization, checkpoint loading, and inference execution.
    Reuses the same checkpoint if unchanged to avoid redundant model loading.
    """

    def __init__(self):
        """Initialize inference service."""
        self._engine: InferenceEngine | None = None
        self._current_checkpoint: str | None = None

    async def predict(
        self,
        image: np.ndarray,
        checkpoint_path: str,
        confidence_threshold: float,
        nms_threshold: float,
        enable_perspective_correction: bool = False,
        perspective_display_mode: str = "overlay",
        enable_grayscale: bool = False,
        enable_background_normalization: bool = False,
        enable_sepia_enhancement: bool = False,
        enable_clahe: bool = False,
        sepia_display_mode: str = "enhanced",
    ) -> dict:
        """Run inference on an image with specified parameters.

        Args:
            image: Input image as numpy array (BGR format)
            checkpoint_path: Path to model checkpoint
            confidence_threshold: Binarization threshold for text detection
            nms_threshold: Box threshold for non-maximum suppression
            enable_perspective_correction: Whether to apply perspective correction
            perspective_display_mode: Display mode for perspective correction
            enable_grayscale: Whether to convert to grayscale
            enable_background_normalization: Whether to normalize background
            enable_sepia_enhancement: Whether to apply sepia tone transformation
            enable_clahe: Whether to apply CLAHE contrast enhancement
            sepia_display_mode: Display mode for sepia ("enhanced" or "original")

        Returns:
            Inference result dictionary with polygons, texts, confidences, and preview image

        Raises:
            RuntimeError: If model loading or inference fails
        """
        # Lazy import heavy dependencies
        from apps.shared.backend_shared.inference import InferenceEngine

        # Initialize engine on first use
        if self._engine is None:
            logger.info("🔄 First inference request: Initializing InferenceEngine...")
            start_init = time.perf_counter()
            self._engine = InferenceEngine()
            logger.info("✅ InferenceEngine initialized in %.2fs", time.perf_counter() - start_init)

        # Load model if checkpoint changed
        if self._current_checkpoint != checkpoint_path:
            logger.info("🔄 Loading checkpoint: %s", checkpoint_path)
            load_start = time.perf_counter()

            if not self._engine.load_model(checkpoint_path):
                raise RuntimeError(f"Failed to load model checkpoint: {checkpoint_path}")

            load_elapsed = time.perf_counter() - load_start
            logger.info("✅ Model load complete | elapsed=%.2fs", load_elapsed)
            self._current_checkpoint = checkpoint_path
        else:
            logger.debug("♻️ Reusing loaded checkpoint: %s", checkpoint_path)

        # Run inference
        try:
            # Debug: Log preprocessing parameters
            logger.info(
                "🔍 Preprocessing options: perspective=%s, grayscale=%s, bg_norm=%s, sepia=%s, clahe=%s",
                enable_perspective_correction,
                enable_grayscale,
                enable_background_normalization,
                enable_sepia_enhancement,
                enable_clahe,
            )

            # Offload CPU-bound inference to threadpool to avoid blocking event loop
            import asyncio

            loop = asyncio.get_running_loop()

            result = await loop.run_in_executor(
                None,  # Use default executor
                lambda: self._engine.predict_array(
                    image_array=image,
                    binarization_thresh=confidence_threshold,
                    box_thresh=nms_threshold,
                    return_preview=True,
                    enable_perspective_correction=enable_perspective_correction,
                    perspective_display_mode=perspective_display_mode,
                    enable_grayscale=enable_grayscale,
                    enable_background_normalization=enable_background_normalization,
                    enable_sepia_enhancement=enable_sepia_enhancement,
                    enable_clahe=enable_clahe,
                    sepia_display_mode=sepia_display_mode,
                ),
            )

            if result is None:
                raise RuntimeError("Inference returned None")

            return result

        except Exception as e:
            logger.exception("Inference failed: %s", e)
            raise RuntimeError(f"Inference failed: {str(e)}")

    async def warm_up(self, checkpoint_path: str) -> None:
        """Warm up the engine by loading the model in the background.

        This method is designed to be called during startup to pre-load the model
        so that the first inference request is fast ("eliminated cold start").

        Args:
            checkpoint_path: Path to model checkpoint to pre-load
        """
        if self._engine is not None and self._current_checkpoint == checkpoint_path:
            logger.info("🔥 Engine already warmed up with checkpoint: %s", checkpoint_path)
            return

        logger.info("🔥 Starting background model warm-up: %s", checkpoint_path)

        # Lazy import heavy dependencies
        import asyncio

        from apps.shared.backend_shared.inference import InferenceEngine

        # Initialize engine if needed
        if self._engine is None:
            logger.info("🔄 Initializing InferenceEngine for warm-up...")
            start_init = time.perf_counter()
            self._engine = InferenceEngine()
            logger.info("✅ InferenceEngine initialized in %.2fs", time.perf_counter() - start_init)

        # Offload blocking model load to threadpool
        loop = asyncio.get_running_loop()

        try:
            success = await loop.run_in_executor(None, lambda: self._engine.load_model(checkpoint_path))

            if success:
                self._current_checkpoint = checkpoint_path
                logger.info("✅ Background model warm-up complete!")
            else:
                logger.error("❌ Background model warm-up failed during load_model")

        except Exception as e:
            logger.exception("❌ Background model warm-up failed with error: %s", e)

    def cleanup(self) -> None:
        """Clean up engine resources."""
        if self._engine is not None:
            try:
                self._engine.cleanup()
            except Exception as e:
                logger.error("Error during engine cleanup: %s", e)
            finally:
                self._engine = None
                self._current_checkpoint = None
