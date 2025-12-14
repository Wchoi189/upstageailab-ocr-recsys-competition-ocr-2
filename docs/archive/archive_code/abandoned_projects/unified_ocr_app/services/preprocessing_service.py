"""Preprocessing service for unified OCR app.

Orchestrates preprocessing pipeline with rembg integration and caching.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)

# Lazy import rembg to avoid startup overhead
_rembg_session = None
_rembg_available = None


def is_rembg_available() -> bool:
    """Check if rembg is available.

    Returns:
        True if rembg can be imported
    """
    global _rembg_available

    if _rembg_available is None:
        try:
            import importlib.util

            _rembg_available = importlib.util.find_spec("rembg") is not None
            if not _rembg_available:
                logger.warning("rembg not available - background removal disabled")
        except (ImportError, ValueError):
            _rembg_available = False
            logger.warning("rembg not available - background removal disabled")

    return _rembg_available


def get_rembg_session(model: str = "u2net") -> Any:
    """Get or create rembg session (lazy loading).

    Args:
        model: Model name to use

    Returns:
        rembg session or None if unavailable
    """
    global _rembg_session

    if not is_rembg_available():
        return None

    # Create session if needed
    if _rembg_session is None:
        try:
            from rembg import new_session

            logger.info(f"Creating rembg session with model: {model}")
            _rembg_session = new_session(model)
            logger.info("rembg session created successfully")
        except Exception as e:
            logger.error(f"Failed to create rembg session: {e}")
            return None

    return _rembg_session


class PreprocessingService:
    """Service for executing preprocessing pipeline with visualization support."""

    def __init__(self, config: dict[str, Any]):
        """Initialize preprocessing service.

        Args:
            config: Mode configuration from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Import existing pipeline (lazy)
        self._pipeline = None

    def _get_pipeline(self):
        """Get or create pipeline instance (lazy loading)."""
        if self._pipeline is None:
            from ui.preprocessing_viewer.pipeline import PreprocessingViewerPipeline

            self._pipeline = PreprocessingViewerPipeline(logger=self.logger)

        return self._pipeline

    @st.cache_data(show_spinner=False, ttl=3600)
    def process_image(
        _self,
        image: np.ndarray,
        parameters: dict[str, Any],
        _image_hash: str,  # For cache busting
    ) -> dict[str, Any]:
        """Process image through preprocessing pipeline.

        Args:
            image: Input image (BGR format)
            parameters: Preprocessing parameters from config
            _image_hash: Hash for cache busting (not used in computation)

        Returns:
            Dict with 'stages' (stage name -> image) and 'metadata'
        """
        _self.logger.info("Starting preprocessing pipeline")
        start_time = time.time()

        results = {}
        metadata = {}

        try:
            # Store original
            current_image = image.copy()
            results["original"] = current_image

            # Execute each enabled stage
            pipeline_stages = _self.config.get("pipeline", {}).get("stages", [])

            for stage in pipeline_stages:
                stage_id = stage["id"]
                stage_config_key = stage["config_key"]

                # Get stage parameters
                stage_params = parameters.get(stage_config_key, {})

                # Check if stage is enabled
                if not stage_params.get("enable", False):
                    _self.logger.info(f"Skipping disabled stage: {stage_id}")
                    continue

                _self.logger.info(f"Executing stage: {stage_id}")
                stage_start = time.time()

                try:
                    # Process based on stage type
                    if stage_id == "background_removal":
                        current_image = _self._apply_background_removal(current_image, stage_params)
                    elif stage_id == "document_detection":
                        current_image = _self._apply_document_detection(current_image, stage_params)
                    elif stage_id == "perspective_correction":
                        current_image = _self._apply_perspective_correction(current_image, stage_params)
                    elif stage_id == "orientation_correction":
                        current_image = _self._apply_orientation_correction(current_image, stage_params)
                    elif stage_id == "noise_elimination":
                        current_image = _self._apply_noise_elimination(current_image, stage_params)
                    elif stage_id == "brightness_adjustment":
                        current_image = _self._apply_brightness_adjustment(current_image, stage_params)
                    elif stage_id == "enhancement":
                        current_image = _self._apply_enhancement(current_image, stage_params)
                    else:
                        _self.logger.warning(f"Unknown stage: {stage_id}")
                        continue

                    # Store result
                    results[stage_id] = current_image.copy()

                    # Store metadata
                    stage_time = time.time() - stage_start
                    metadata[stage_id] = {
                        "execution_time": stage_time,
                        "output_shape": current_image.shape,
                    }

                    _self.logger.info(f"Stage {stage_id} completed in {stage_time:.2f}s")

                except Exception as e:
                    _self.logger.error(f"Error in stage {stage_id}: {e}", exc_info=True)
                    metadata[stage_id] = {
                        "error": str(e),
                        "execution_time": time.time() - stage_start,
                    }
                    # Continue with last successful image
                    continue

            # Overall timing
            total_time = time.time() - start_time
            _self.logger.info(f"Pipeline completed in {total_time:.2f}s")

            return {
                "stages": results,
                "metadata": {
                    "total_time": total_time,
                    "stages": metadata,
                    "num_stages": len(results) - 1,  # Exclude original
                },
            }

        except Exception as e:
            _self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {
                "stages": {"original": image},
                "metadata": {"error": str(e)},
            }

    def _apply_background_removal(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply background removal using rembg.

        Args:
            image: Input image (BGR format)
            params: Background removal parameters

        Returns:
            Processed image
        """
        if not is_rembg_available():
            self.logger.warning("rembg not available, skipping background removal")
            return image

        try:
            from ocr.datasets.preprocessing.background_removal import BackgroundRemoval

            # Get parameters
            model = params.get("model", "u2net")
            alpha_matting = params.get("alpha_matting", True)
            fg_threshold = params.get("foreground_threshold", 240)
            bg_threshold = params.get("background_threshold", 10)
            erode_size = params.get("erode_size", 10)

            # Create transform
            transform = BackgroundRemoval(
                model=model,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=fg_threshold,
                alpha_matting_background_threshold=bg_threshold,
                alpha_matting_erode_size=erode_size,
                p=1.0,
            )

            # Convert BGR to RGB for processing
            import cv2

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply transform
            result_rgb = transform.apply(image_rgb)

            # Convert back to BGR
            result = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

            return result

        except Exception as e:
            self.logger.error(f"Background removal failed: {e}", exc_info=True)
            return image

    def _apply_document_detection(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply document detection.

        Args:
            image: Input image
            params: Document detection parameters

        Returns:
            Processed image
        """
        try:
            from ocr.datasets.preprocessing.detector import DocumentDetector

            # Get parameters
            min_area_ratio = params.get("min_area_ratio", 0.18)
            use_adaptive = params.get("use_adaptive", True)
            use_fallback_box = params.get("use_fallback_box", True)

            # Create detector
            detector = DocumentDetector(
                logger=self.logger,
                min_area_ratio=min_area_ratio,
                use_adaptive=use_adaptive,
                use_fallback=use_fallback_box,
            )

            # Detect document
            corners, method = detector.detect(image)

            if corners is not None:
                # For visualization, draw detected corners
                import cv2

                result = image.copy()
                pts = corners.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(result, [pts], True, (0, 255, 0), 3)
                return result
            else:
                return image

        except Exception as e:
            self.logger.error(f"Document detection failed: {e}", exc_info=True)
            return image

    def _apply_perspective_correction(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply perspective correction.

        Args:
            image: Input image
            params: Perspective correction parameters

        Returns:
            Processed image
        """
        try:
            from ocr.datasets.preprocessing.detector import DocumentDetector
            from ocr.datasets.preprocessing.perspective import PerspectiveCorrector

            # Get parameters
            use_doctr = params.get("use_doctr_geometry", False)

            # First detect document
            min_area_ratio = 0.18
            detector = DocumentDetector(
                logger=self.logger,
                min_area_ratio=min_area_ratio,
                use_adaptive=True,
                use_fallback=True,
            )

            corners, method = detector.detect(image)

            if corners is None:
                self.logger.warning("No document detected for perspective correction")
                return image

            # Apply perspective correction
            def ensure_doctr(feature: str) -> bool:
                return True

            corrector = PerspectiveCorrector(
                logger=self.logger,
                ensure_doctr=ensure_doctr,
                use_doctr_geometry=use_doctr,
                doctr_assume_horizontal=False,
            )

            # correct() returns tuple[image, matrix, method_name]
            corrected_image, _matrix, _method_name = corrector.correct(image, corners)
            return corrected_image

        except Exception as e:
            self.logger.error(f"Perspective correction failed: {e}", exc_info=True)
            return image

    def _apply_orientation_correction(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply orientation correction.

        Args:
            image: Input image
            params: Orientation correction parameters

        Returns:
            Processed image
        """
        # Placeholder - would use existing orientation corrector
        self.logger.info("Orientation correction not yet implemented in service")
        return image

    def _apply_noise_elimination(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply noise elimination.

        Args:
            image: Input image
            params: Noise elimination parameters

        Returns:
            Processed image
        """
        try:
            from ocr.datasets.preprocessing.advanced_noise_elimination import (
                AdvancedNoiseEliminator,
            )

            eliminator = AdvancedNoiseEliminator()
            # eliminate_noise returns NoiseEliminationResult which contains cleaned_image
            result = eliminator.eliminate_noise(image)
            return result.cleaned_image

        except Exception as e:
            self.logger.error(f"Noise elimination failed: {e}", exc_info=True)
            return image

    def _apply_brightness_adjustment(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply brightness adjustment.

        Args:
            image: Input image
            params: Brightness adjustment parameters

        Returns:
            Processed image
        """
        try:
            from ocr.datasets.preprocessing.intelligent_brightness import (
                IntelligentBrightnessAdjuster,
            )

            adjuster = IntelligentBrightnessAdjuster()
            # adjust_brightness returns BrightnessResult which contains adjusted_image
            result = adjuster.adjust_brightness(image)
            return result.adjusted_image

        except Exception as e:
            self.logger.error(f"Brightness adjustment failed: {e}", exc_info=True)
            return image

    def _apply_enhancement(self, image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Apply image enhancement.

        Args:
            image: Input image
            params: Enhancement parameters

        Returns:
            Processed image
        """
        try:
            from ocr.datasets.preprocessing.enhancement import ImageEnhancer

            method_param = params.get("method", "conservative")
            enhancer = ImageEnhancer()

            # Map method parameter to actual enhancement method
            # enhance() returns tuple[image, list[enhancements]]
            if method_param == "aggressive":
                enhanced_image, _enhancements = enhancer.enhance(image, method="office_lens")
            else:
                # conservative or moderate use default method (anything other than "office_lens")
                enhanced_image, _enhancements = enhancer.enhance(image, method="mild")

            return enhanced_image

        except Exception as e:
            self.logger.error(f"Enhancement failed: {e}", exc_info=True)
            return image
