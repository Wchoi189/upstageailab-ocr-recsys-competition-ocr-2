"""Comparison service for A/B testing and parameter sweeps.

This service orchestrates the execution of multiple configurations for comparison:
- Preprocessing parameter sweeps
- Inference hyperparameter tuning
- End-to-end pipeline comparison
"""

import hashlib
import json
import time
from typing import Any

import cv2
import numpy as np
import streamlit as st

from .config_loader import load_mode_config
from .inference_service import InferenceService
from .preprocessing_service import PreprocessingService


class ComparisonService:
    """Service for running and comparing multiple configurations."""

    def __init__(self) -> None:
        """Initialize comparison service.

        Note: Services are created on-demand with proper configs.
        """
        self._preprocessing_service: PreprocessingService | None = None
        self._inference_service: InferenceService | None = None

    def _get_preprocessing_service(self) -> PreprocessingService:
        """Get or create preprocessing service instance."""
        if self._preprocessing_service is None:
            # Load config without validation for comparison mode
            # This allows us to use custom parameter sets
            from pathlib import Path

            import yaml

            config_path = Path("configs/ui/modes/preprocessing.yaml")
            with open(config_path) as f:
                config = yaml.safe_load(f)

            self._preprocessing_service = PreprocessingService(config)
        return self._preprocessing_service

    def _get_inference_service(self) -> InferenceService:
        """Get or create inference service instance."""
        if self._inference_service is None:
            config = load_mode_config("inference")
            self._inference_service = InferenceService(config)
        return self._inference_service

    def run_preprocessing_comparison(
        self,
        image: np.ndarray,
        configurations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run preprocessing comparison across multiple configurations.

        Args:
            image: Input image as numpy array
            configurations: List of configuration dictionaries with structure:
                {
                    "label": str,
                    "params": dict,  # Preprocessing parameters
                }

        Returns:
            List of results with structure:
            {
                "config_label": str,
                "config_params": dict,
                "image": np.ndarray,  # Processed image
                "metrics": dict,
                "processing_time": float,
            }
        """
        results = []

        for config in configurations:
            config_label = config.get("label", "Unnamed")
            params = config.get("params", {})

            # Run preprocessing with timing
            start_time = time.time()
            try:
                processed_image = self._run_preprocessing_pipeline(image, params)
                processing_time = time.time() - start_time

                # Calculate metrics
                metrics = self._calculate_preprocessing_metrics(
                    image,
                    processed_image,
                    params,
                )

                results.append(
                    {
                        "config_label": config_label,
                        "config_params": params,
                        "image": processed_image,
                        "metrics": metrics,
                        "processing_time": processing_time,
                    }
                )

            except Exception as e:
                st.error(f"Error in configuration '{config_label}': {str(e)}")
                # Add failed result
                results.append(
                    {
                        "config_label": f"{config_label} (Failed)",
                        "config_params": params,
                        "image": image,  # Return original image
                        "metrics": {"error": str(e)},
                        "processing_time": time.time() - start_time,
                    }
                )

        return results

    def run_inference_comparison(
        self,
        image: np.ndarray,
        configurations: list[dict[str, Any]],
        checkpoint_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference comparison across multiple configurations.

        Args:
            image: Input image as numpy array
            configurations: List of configuration dictionaries
            checkpoint_path: Optional checkpoint path (if not in configs)

        Returns:
            List of results with inference outputs and metrics
        """
        results = []

        for config in configurations:
            config_label = config.get("label", "Unnamed")
            params = config.get("params", {})

            # Extract hyperparameters
            text_threshold = params.get("text_threshold", 0.7)
            link_threshold = params.get("link_threshold", 0.4)
            low_text = params.get("low_text", 0.4)

            # Use checkpoint from params or provided
            ckpt = params.get("checkpoint", checkpoint_path)

            if not ckpt:
                st.warning(f"No checkpoint specified for '{config_label}'")
                continue

            # Run inference with timing
            start_time = time.time()
            try:
                # Get inference service
                service = self._get_inference_service()

                # Generate cache key
                cache_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]

                # Prepare hyperparameters
                hyperparameters = {
                    "text_threshold": text_threshold,
                    "link_threshold": link_threshold,
                    "low_text": low_text,
                }

                # Run inference
                # Note: checkpoint is expected to be a CheckpointInfo object
                # If it's a string path, we need to load it
                if isinstance(ckpt, str):
                    # Create a minimal checkpoint object
                    class MinimalCheckpoint:
                        def __init__(self, path):
                            self.checkpoint_path = path

                    checkpoint_obj = MinimalCheckpoint(ckpt)
                else:
                    checkpoint_obj = ckpt

                inference_result = service.run_inference(
                    image=image,
                    checkpoint=checkpoint_obj,
                    hyperparameters=hyperparameters,
                    _image_hash=cache_key,
                )

                processing_time = time.time() - start_time

                # Extract metrics from inference result
                num_detections = len(inference_result.polygons)
                avg_confidence = self._calculate_avg_confidence_from_result(inference_result)

                metrics = {
                    "num_detections": num_detections,
                    "avg_confidence": avg_confidence,
                    "inference_time": processing_time,
                }

                # Create visualization with polygons overlaid
                vis_image = self._create_inference_visualization(
                    image=image,
                    inference_result=inference_result,
                )

                results.append(
                    {
                        "config_label": config_label,
                        "config_params": params,
                        "image": vis_image,
                        "metrics": metrics,
                        "processing_time": processing_time,
                        "inference_result": inference_result,  # Include for downstream use
                    }
                )

            except Exception as e:
                st.error(f"Error in configuration '{config_label}': {str(e)}")
                results.append(
                    {
                        "config_label": f"{config_label} (Failed)",
                        "config_params": params,
                        "image": image,
                        "metrics": {"error": str(e)},
                        "processing_time": time.time() - start_time,
                    }
                )

        return results

    def run_end_to_end_comparison(
        self,
        image: np.ndarray,
        configurations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Run end-to-end comparison with preprocessing and inference.

        Args:
            image: Input image as numpy array
            configurations: List of configuration dictionaries with both
                preprocessing and inference params

        Returns:
            List of results with full pipeline outputs
        """
        results = []

        for config in configurations:
            config_label = config.get("label", "Unnamed")
            params = config.get("params", {})

            # Split preprocessing and inference params
            preprocessing_params = params.get("preprocessing", {})
            inference_params = params.get("inference", {})

            # Run full pipeline with timing
            total_start = time.time()

            try:
                # Step 1: Preprocessing
                preproc_start = time.time()
                processed_image = self._run_preprocessing_pipeline(
                    image,
                    preprocessing_params,
                )
                preproc_time = time.time() - preproc_start

                # Step 2: Inference
                inf_start = time.time()
                checkpoint_path = inference_params.get("checkpoint")

                if checkpoint_path:
                    # Get inference service
                    service = self._get_inference_service()

                    # Generate cache key
                    cache_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]

                    # Prepare hyperparameters
                    hyperparameters = {
                        "text_threshold": inference_params.get("text_threshold", 0.7),
                        "link_threshold": inference_params.get("link_threshold", 0.4),
                        "low_text": inference_params.get("low_text", 0.4),
                    }

                    # Create checkpoint object if needed
                    if isinstance(checkpoint_path, str):

                        class MinimalCheckpoint:
                            def __init__(self, path):
                                self.checkpoint_path = path

                        checkpoint_obj = MinimalCheckpoint(checkpoint_path)
                    else:
                        checkpoint_obj = checkpoint_path

                    # Run inference
                    inference_result = service.run_inference(
                        image=processed_image,
                        checkpoint=checkpoint_obj,
                        hyperparameters=hyperparameters,
                        _image_hash=cache_key,
                    )

                    inf_time = time.time() - inf_start

                    # Combine metrics
                    num_detections = len(inference_result.polygons)
                    avg_confidence = self._calculate_avg_confidence_from_result(inference_result)

                    metrics = {
                        "num_detections": num_detections,
                        "avg_confidence": avg_confidence,
                        "preprocessing_time": preproc_time,
                        "inference_time": inf_time,
                    }

                    vis_image = processed_image

                else:
                    # No inference, just preprocessing
                    metrics = {
                        "preprocessing_time": preproc_time,
                    }
                    vis_image = processed_image

                total_time = time.time() - total_start

                results.append(
                    {
                        "config_label": config_label,
                        "config_params": params,
                        "image": vis_image,
                        "metrics": metrics,
                        "processing_time": total_time,
                    }
                )

            except Exception as e:
                st.error(f"Error in configuration '{config_label}': {str(e)}")
                results.append(
                    {
                        "config_label": f"{config_label} (Failed)",
                        "config_params": params,
                        "image": image,
                        "metrics": {"error": str(e)},
                        "processing_time": time.time() - total_start,
                    }
                )

        return results

    def _run_preprocessing_pipeline(
        self,
        image: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        """Run preprocessing pipeline with given parameters.

        Args:
            image: Input image
            params: Preprocessing parameters

        Returns:
            Processed image
        """
        # Get preprocessing service
        service = self._get_preprocessing_service()

        # Generate cache key from params
        cache_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]

        # Process image
        try:
            result = service.process_image(image, params, cache_key)

            # Extract final processed image
            stages = result.get("stages", {})
            if "final" in stages:
                return stages["final"]
            elif stages:
                # Return the last stage if 'final' not present
                return list(stages.values())[-1]
            else:
                # No processing happened, return original
                return image.copy()

        except Exception as e:
            # Log error and return original image
            st.error(f"Preprocessing pipeline error: {str(e)}")
            return image.copy()

    def _calculate_preprocessing_metrics(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate metrics for preprocessing results."""
        metrics: dict[str, Any] = {}

        # Image size changes
        metrics["image_size"] = f"{processed.shape[1]}x{processed.shape[0]}"

        # Count enabled stages
        enabled_stages = sum(1 for stage_params in params.values() if isinstance(stage_params, dict) and stage_params.get("enable", False))
        metrics["preprocessing_stages"] = enabled_stages

        # Calculate image statistics
        if processed.size > 0:
            metrics["mean_intensity"] = float(np.mean(processed))
            metrics["std_intensity"] = float(np.std(processed))

        return metrics

    def _calculate_avg_confidence(
        self,
        inference_result: dict[str, Any],
    ) -> float:
        """Calculate average confidence from inference results (dict format)."""
        scores = inference_result.get("scores", [])

        if not scores:
            return 0.0

        return float(np.mean(scores))

    def _calculate_avg_confidence_from_result(
        self,
        inference_result: Any,
    ) -> float:
        """Calculate average confidence from InferenceResult object.

        Args:
            inference_result: InferenceResult object with scores attribute

        Returns:
            Average confidence score
        """
        if not hasattr(inference_result, "scores") or not inference_result.scores:
            return 0.0

        return float(np.mean(inference_result.scores))

    def _create_inference_visualization(
        self,
        image: np.ndarray,
        inference_result: Any,
        polygon_color: tuple[int, int, int] = (0, 255, 0),
        polygon_thickness: int = 2,
        show_scores: bool = True,
    ) -> np.ndarray:
        """Create visualization with inference results overlaid.

        Args:
            image: Input image
            inference_result: InferenceResult object with polygons and scores
            polygon_color: Color for polygon lines (BGR format)
            polygon_thickness: Line thickness for polygons
            show_scores: Whether to show confidence scores

        Returns:
            Image with visualization overlay
        """
        viz_image = image.copy()

        # Draw polygons if available
        if hasattr(inference_result, "polygons") and inference_result.polygons:
            for idx, polygon in enumerate(inference_result.polygons):
                if isinstance(polygon, list | np.ndarray):
                    poly_array = np.array(polygon, dtype=np.int32)

                    # Reshape if needed
                    if poly_array.ndim == 1 and poly_array.size >= 8:
                        # Flat array of coordinates [x1, y1, x2, y2, ...]
                        poly_array = poly_array.reshape(-1, 2)
                    elif poly_array.ndim == 2 and poly_array.shape[1] == 2:
                        # Already in correct shape
                        pass
                    else:
                        continue

                    # Draw polygon
                    cv2.polylines(viz_image, [poly_array], True, polygon_color, polygon_thickness)

                    # Add confidence score if available
                    if show_scores and hasattr(inference_result, "scores") and inference_result.scores:
                        if idx < len(inference_result.scores):
                            # Get top-left corner for text placement
                            x, y = poly_array[0]
                            score_text = f"{inference_result.scores[idx]:.2f}"
                            cv2.putText(
                                viz_image,
                                score_text,
                                (int(x), int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                polygon_color,
                                1,
                                cv2.LINE_AA,
                            )

        return viz_image


@st.cache_resource
def get_comparison_service() -> ComparisonService:
    """Get singleton comparison service instance.

    Returns:
        ComparisonService instance
    """
    return ComparisonService()
