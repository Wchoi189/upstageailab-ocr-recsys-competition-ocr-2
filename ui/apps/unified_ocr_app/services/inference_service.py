"""Inference service for unified OCR app.

Wraps the existing inference functionality for use in unified app.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)


class InferenceResult:
    """Container for inference results."""

    def __init__(
        self,
        image: np.ndarray,
        polygons: list[np.ndarray],
        scores: list[float],
        processing_time: float,
        image_shape: tuple[int, ...],
    ):
        self.image = image
        self.polygons = polygons
        self.scores = scores
        self.processing_time = processing_time
        self.image_shape = image_shape


class InferenceService:
    """Service for running OCR inference with model checkpoints."""

    def __init__(self, config: dict[str, Any]):
        """Initialize inference service.

        Args:
            config: Mode configuration from YAML
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Import existing inference components (lazy)
        self._inference_engine = None

    def _get_inference_engine(self):
        """Get or create inference engine instance (lazy loading)."""
        if self._inference_engine is None:
            try:
                # Import existing inference runner from the inference app
                from ui.apps.inference.services.inference_runner import InferenceService as ExistingInferenceService

                self._inference_engine = ExistingInferenceService()
                self.logger.info("Inference engine initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize inference engine: {e}")
                raise

        return self._inference_engine

    @st.cache_data(show_spinner=False)
    def run_inference(
        _self,
        image: np.ndarray,
        checkpoint: Any,
        hyperparameters: dict[str, float],
        _image_hash: str,  # For cache busting
    ) -> InferenceResult:
        """Run inference on an image.

        Args:
            image: Input image (BGR format)
            checkpoint: Checkpoint object
            hyperparameters: Inference hyperparameters
            _image_hash: Hash for cache busting (not used in computation)

        Returns:
            InferenceResult object
        """
        _self.logger.info("Starting inference")
        start_time = time.time()

        try:
            # Get inference engine
            engine = _self._get_inference_engine()

            # Prepare inference request
            # This will depend on the existing inference service API
            # For now, we'll create a placeholder structure

            result = _self._run_inference_internal(image, checkpoint, hyperparameters, engine)

            processing_time = time.time() - start_time
            _self.logger.info(f"Inference completed in {processing_time:.2f}s")

            return InferenceResult(
                image=image,
                polygons=result.get("polygons", []),
                scores=result.get("scores", []),
                processing_time=processing_time,
                image_shape=image.shape,
            )

        except Exception as e:
            _self.logger.error(f"Inference failed: {e}", exc_info=True)
            raise

    def _run_inference_internal(
        self,
        image: np.ndarray,
        checkpoint: Any,
        hyperparameters: dict[str, float],
        engine: Any,
    ) -> dict[str, Any]:
        """Internal inference execution.

        Args:
            image: Input image (BGR numpy array)
            checkpoint: Checkpoint object
            hyperparameters: Inference hyperparameters
            engine: Inference engine instance

        Returns:
            Dict with 'polygons' and 'scores'

        Note:
            Uses optimized numpy array path (no tempfile overhead).
            See: artifacts/implementation_plans/2025-11-12_plan-004-revised-inference-consolidation.md
        """
        try:
            from ui.utils.inference import run_inference_on_image

            # Get checkpoint path
            checkpoint_path = checkpoint.checkpoint_path if hasattr(checkpoint, "checkpoint_path") else checkpoint

            # NEW: Pass numpy array directly (eliminates tempfile overhead)
            result = run_inference_on_image(
                image_path=image,  # Pass numpy array directly
                checkpoint_path=str(checkpoint_path),
                **hyperparameters,
            )

            # Extract polygons and scores
            polygons = []
            scores = []

            if result and "polygons" in result:
                polygons = result["polygons"]
                scores = result.get("scores", [1.0] * len(polygons))

            return {
                "polygons": polygons,
                "scores": scores,
            }

        except Exception as e:
            self.logger.error(f"Internal inference execution failed: {e}", exc_info=True)
            # Return empty result on failure
            return {"polygons": [], "scores": []}

    def run_batch_inference(
        self,
        image_paths: list[str],
        checkpoint: Any,
        hyperparameters: dict[str, float],
        output_dir: str,
    ) -> dict[str, Any]:
        """Run batch inference on multiple images.

        Args:
            image_paths: List of image file paths
            checkpoint: Checkpoint object
            hyperparameters: Inference hyperparameters
            output_dir: Output directory for results

        Returns:
            Dict with batch processing results
        """
        self.logger.info(f"Starting batch inference on {len(image_paths)} images")

        try:
            # Import existing batch processing functionality
            from ui.apps.inference.models.batch_request import BatchPredictionRequest

            # Create batch request
            request = BatchPredictionRequest(
                input_directory=str(image_paths[0]) if image_paths else "",  # Placeholder
                output_directory=output_dir,
                checkpoint_path=str(checkpoint.checkpoint_path) if hasattr(checkpoint, "checkpoint_path") else None,
            )

            # Use existing inference engine for batch processing
            engine = self._get_inference_engine()

            # Create minimal state
            class MinimalState:
                def __init__(self):
                    self.results = []
                    self.hyperparams = {}

            state = MinimalState()
            state.hyperparams = hyperparameters

            # Run batch prediction
            engine.run_batch_prediction(state, request)

            return {
                "status": "completed",
                "num_images": len(image_paths),
                "output_dir": output_dir,
            }

        except Exception as e:
            self.logger.error(f"Batch inference failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }


@st.cache_data(show_spinner=False, ttl=300)
def load_checkpoints(config: dict[str, Any]) -> list[Any]:
    """Load available model checkpoints with caching.

    Args:
        config: App configuration with paths

    Returns:
        List of CheckpointInfo objects

    Note:
        Cached for 5 minutes to balance freshness and performance.
        Clear cache if checkpoints are added/removed.
    """
    try:
        from pathlib import Path

        from ui.apps.inference.models.config import PathConfig
        from ui.apps.inference.services.checkpoint import CatalogOptions, build_lightweight_catalog

        # Get paths from config
        paths_config = config.get("paths", {})
        outputs_dir = paths_config.get("outputs_dir", "outputs")

        # Create proper PathConfig instance
        path_config = PathConfig(
            outputs_dir=Path(outputs_dir),
            hydra_config_filenames=[
                "config.yaml",
                "hparams.yaml",
                "train.yaml",
                "predict.yaml",
            ],
        )

        # Create catalog options
        options = CatalogOptions.from_paths(path_config)

        # Build catalog
        catalog = build_lightweight_catalog(options)

        logger.info(f"Loaded {len(catalog)} checkpoints")
        return catalog

    except Exception as e:
        logger.error(f"Failed to load checkpoints: {e}", exc_info=True)
        return []
