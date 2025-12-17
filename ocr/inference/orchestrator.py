"""Inference orchestrator for OCR pipeline.

This module provides a thin coordination layer that integrates all inference
components:
- ModelManager: Model lifecycle management
- PreprocessingPipeline: Image preprocessing
- PostprocessingPipeline: Prediction postprocessing
- PreviewGenerator: Preview image generation

The orchestrator follows the single responsibility principle: it only coordinates
the flow between components, delegating all actual work to specialized classes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config_loader import ModelConfigBundle, PostprocessSettings
from .dependencies import OCR_MODULES_AVAILABLE
from .model_manager import ModelManager
from .postprocessing_pipeline import PostprocessingPipeline
from .preprocessing_pipeline import PreprocessingPipeline
from .preview_generator import PreviewGenerator

LOGGER = logging.getLogger(__name__)


class InferenceOrchestrator:
    """Orchestrates OCR inference pipeline.

    This class coordinates the inference workflow by delegating to specialized
    components:
    1. ModelManager handles model loading and lifecycle
    2. PreprocessingPipeline handles image preprocessing
    3. Model performs inference (coordinated by ModelManager)
    4. PostprocessingPipeline handles prediction postprocessing
    5. PreviewGenerator creates preview images with overlays

    The orchestrator maintains minimal state and focuses solely on coordination.
    """

    def __init__(self, device: str | None = None):
        """Initialize inference orchestrator.

        Args:
            device: Device for inference ("cuda" or "cpu", auto-detected if None)
        """
        self.model_manager = ModelManager(device=device)
        self.preprocessing_pipeline: PreprocessingPipeline | None = None
        self.postprocessing_pipeline: PostprocessingPipeline | None = None
        self.preview_generator = PreviewGenerator(jpeg_quality=85)

        LOGGER.info(f"InferenceOrchestrator initialized (device: {self.model_manager.device})")

    def load_model(self, checkpoint_path: str, config_path: str | None = None) -> bool:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional config path (auto-detected if not provided)

        Returns:
            True if model loaded successfully
        """
        success = self.model_manager.load_model(checkpoint_path, config_path)

        if success:
            # Initialize pipelines from config
            bundle = self.model_manager.get_config_bundle()
            if bundle is not None:
                self.preprocessing_pipeline = PreprocessingPipeline.from_settings(bundle.preprocess)
                self.postprocessing_pipeline = PostprocessingPipeline(settings=bundle.postprocess)

        return success

    def predict(
        self,
        image: np.ndarray,
        return_preview: bool = True,
        enable_perspective_correction: bool = False,
        perspective_display_mode: str = "corrected",
        enable_grayscale: bool = False,
    ) -> dict[str, Any] | None:
        """Run inference on image array.

        This is the main orchestration method. It coordinates the full pipeline:
        1. Preprocessing (resize, normalize, metadata)
        2. Model inference
        3. Postprocessing (decode, format)
        4. Preview generation (if requested)

        Args:
            image: Input image as BGR numpy array (H, W, C)
            return_preview: Whether to generate and attach preview image
            enable_perspective_correction: Whether to apply perspective correction
            perspective_display_mode: "corrected" or "original" display mode
            enable_grayscale: Whether to apply grayscale preprocessing

        Returns:
            Predictions dict with polygons, texts, confidences, and optional preview

        Example:
            >>> orchestrator = InferenceOrchestrator()
            >>> orchestrator.load_model("checkpoint.pth")
            >>> result = orchestrator.predict(image_bgr)
            >>> if result:
            ...     print(f"Found {len(result['texts'])} detections")
        """
        if not self.model_manager.is_loaded():
            LOGGER.error("Model not loaded. Call load_model() first.")
            return None

        if self.preprocessing_pipeline is None or self.postprocessing_pipeline is None:
            LOGGER.error("Pipelines not initialized")
            return None

        # Stage 1: Preprocessing
        preprocess_result = self.preprocessing_pipeline.process(
            image,
            enable_perspective_correction=enable_perspective_correction,
            perspective_display_mode=perspective_display_mode,
            enable_grayscale=enable_grayscale,
        )

        if preprocess_result is None:
            LOGGER.error("Preprocessing failed")
            return None

        # Stage 2: Model inference
        if not OCR_MODULES_AVAILABLE:
            LOGGER.error("OCR modules not available for inference")
            return None

        try:
            import torch
            with torch.no_grad():
                predictions = self.model_manager.model(
                    return_loss=False,
                    images=preprocess_result.batch.to(self.model_manager.device)
                )
        except Exception:
            LOGGER.exception("Model inference failed")
            return None

        # Stage 3: Postprocessing
        postprocess_result = self.postprocessing_pipeline.process(
            self.model_manager.model,
            preprocess_result.batch,
            predictions,
            preprocess_result.original_shape,
        )

        if postprocess_result is None:
            LOGGER.error("Postprocessing failed")
            return None

        # Convert to dict format
        result = {
            "polygons": postprocess_result.polygons,
            "texts": postprocess_result.texts,
            "confidences": postprocess_result.confidences,
        }

        # Stage 4: Handle inverse perspective transformation if needed
        if (preprocess_result.perspective_matrix is not None and
            perspective_display_mode == "original" and
            preprocess_result.original_image is not None):

            # Transform polygons back to original space
            from ocr.utils.perspective_correction import transform_polygons_inverse

            if result["polygons"]:
                result["polygons"] = transform_polygons_inverse(
                    result["polygons"],
                    preprocess_result.perspective_matrix,
                )

            # Create preview from original image
            original_preview = self.preprocessing_pipeline.process_for_original_display(
                preprocess_result.original_image
            )
            if original_preview is not None:
                preview_image, metadata = original_preview
                preprocess_result = preprocess_result.__class__(
                    batch=preprocess_result.batch,
                    preview_image=preview_image,
                    original_shape=preprocess_result.original_image.shape,
                    metadata=metadata,
                    perspective_matrix=preprocess_result.perspective_matrix,
                    original_image=preprocess_result.original_image,
                )

        # Stage 5: Preview generation
        if return_preview:
            result = self.preview_generator.attach_preview_to_payload(
                payload=result,
                preview_image=preprocess_result.preview_image,
                metadata=preprocess_result.metadata,
                transform_polygons=True,
                original_shape=(
                    preprocess_result.original_shape[0],
                    preprocess_result.original_shape[1],
                ),
                target_size=self.preprocessing_pipeline._target_size,
            )

        return result

    def update_postprocessor_params(
        self,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
    ) -> None:
        """Update postprocessing parameters.

        Args:
            binarization_thresh: Binarization threshold
            box_thresh: Box confidence threshold
            max_candidates: Maximum number of candidates
            min_detection_size: Minimum detection size in pixels
        """
        if self.postprocessing_pipeline is None:
            LOGGER.warning("Postprocessing pipeline not initialized")
            return

        # Get current settings
        current = self.postprocessing_pipeline._settings
        if current is None:
            LOGGER.warning("No current settings to update")
            return

        # Create updated settings
        updated = PostprocessSettings(
            binarization_thresh=binarization_thresh if binarization_thresh is not None else current.binarization_thresh,
            box_thresh=box_thresh if box_thresh is not None else current.box_thresh,
            max_candidates=int(max_candidates) if max_candidates is not None else current.max_candidates,
            min_detection_size=int(min_detection_size) if min_detection_size is not None else current.min_detection_size,
        )

        self.postprocessing_pipeline.set_settings(updated)

        # Also update model head if available
        if self.model_manager.model is not None:
            head = getattr(self.model_manager.model, "head", None)
            postprocess = getattr(head, "postprocess", None)
            if postprocess is not None:
                if hasattr(postprocess, "thresh") and binarization_thresh is not None:
                    postprocess.thresh = binarization_thresh
                if hasattr(postprocess, "box_thresh") and box_thresh is not None:
                    postprocess.box_thresh = box_thresh
                if hasattr(postprocess, "max_candidates") and max_candidates is not None:
                    postprocess.max_candidates = int(max_candidates)
                if hasattr(postprocess, "min_size") and min_detection_size is not None:
                    postprocess.min_size = int(min_detection_size)

    def cleanup(self) -> None:
        """Clean up resources."""
        self.model_manager.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False


__all__ = [
    "InferenceOrchestrator",
]
