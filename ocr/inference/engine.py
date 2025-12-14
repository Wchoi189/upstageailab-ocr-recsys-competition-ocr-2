from __future__ import annotations

"""High-level inference engine orchestration."""

import base64
import logging
import re
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Suppress Pydantic v2 configuration warnings from dependencies
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:", category=UserWarning)
warnings.filterwarnings("ignore", message="'allow_population_by_field_name' has been renamed to 'validate_by_name'", category=UserWarning)

from ocr.utils.orientation import normalize_pil_image, orientation_requires_rotation, remap_polygons
from ocr.utils.orientation_constants import ORIENTATION_INVERSE_INT
from ocr.utils.path_utils import get_path_resolver

from .config_loader import ModelConfigBundle, PostprocessSettings, load_model_config, resolve_config_path
from .dependencies import OCR_MODULES_AVAILABLE, torch
from .model_loader import instantiate_model, load_checkpoint, load_state_dict
from .postprocess import decode_polygons_with_head, fallback_postprocess
from .preprocess import apply_optional_perspective_correction, build_transform, preprocess_image
from .utils import generate_mock_predictions
from .utils import get_available_checkpoints as scan_checkpoints

LOGGER = logging.getLogger(__name__)

_POLYGON_TOKEN_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _run_with_timeout(func: Callable, timeout_seconds: int = 30) -> Any:
    """Run a function with a timeout using threading (thread-safe for Streamlit).

    This implementation uses threading instead of signal.signal() to be compatible
    with Streamlit's threading model. signal.signal() only works in the main thread,
    but Streamlit runs in a separate thread.

    Args:
        func: Function to execute
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function

    Raises:
        TimeoutError: If function execution exceeds timeout
        Exception: Any exception raised by the function
    """
    result = [None]
    exception = [None]

    def _wrapper():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=_wrapper)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread is still running - timeout occurred
        LOGGER.error(f"Function timed out after {timeout_seconds} seconds")
        raise TimeoutError(f"Inference operation timed out after {timeout_seconds} seconds")

    if exception[0] is not None:
        raise exception[0]

    return result[0]


class InferenceEngine:
    """OCR Inference Engine for real-time predictions."""

    def __init__(self) -> None:
        self.model = None
        self.trainer = None
        self.config: Any | None = None
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

        self._current_checkpoint_path: str | None = None

        self._config_bundle: ModelConfigBundle | None = None
        self._transform = None
        self._postprocess_settings: PostprocessSettings | None = None
        LOGGER.info("Using device: %s", self.device)

    # Public API ---------------------------------------------------------
    def load_model(self, checkpoint_path: str, config_path: str | None = None) -> bool:
        if not OCR_MODULES_AVAILABLE:
            LOGGER.error("OCR modules are not installed. Cannot load a real model.")
            return False

        total_start = time.perf_counter()
        normalized_path = str(Path(checkpoint_path).resolve())
        if self.model is not None and self._current_checkpoint_path == normalized_path:
            LOGGER.info("Checkpoint already loaded; reusing cached model: %s", checkpoint_path)
            return True

        # Include configs directory in search paths for config files
        # Use path resolver for config directory (supports environment variable override)
        resolver = get_path_resolver()
        search_dirs = (resolver.config.config_dir,)
        resolved_config = resolve_config_path(checkpoint_path, config_path, search_dirs)
        if resolved_config is None:
            LOGGER.error("Could not find a valid config file for checkpoint: %s", checkpoint_path)
            return False

        LOGGER.info("Loading model from checkpoint: %s", checkpoint_path)
        bundle = load_model_config(resolved_config)
        self._apply_config_bundle(bundle)

        model_config = getattr(bundle.raw_config, "model", None)
        if model_config is None:
            # Fallback: Try to extract model config from root level (for Hydra configs with defaults)
            LOGGER.warning("Configuration missing direct 'model' section, trying root level extraction: %s", resolved_config)
            model_config = bundle.raw_config
            # Check if this config has model-like attributes
            model_attrs = ["architecture", "encoder", "decoder", "head"]
            if not any(hasattr(model_config, attr) for attr in model_attrs):
                LOGGER.error("Configuration has neither 'model' section nor model attributes at root level: %s", resolved_config)
                return False

        try:
            model = instantiate_model(model_config)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to instantiate model from config %s", resolved_config)
            return False

        load_start = time.perf_counter()
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        load_duration = time.perf_counter() - load_start
        LOGGER.info("Checkpoint weights loaded in %.2fs", load_duration)
        if checkpoint is None:
            LOGGER.error("Failed to load checkpoint %s", checkpoint_path)
            return False

        if not load_state_dict(model, checkpoint):
            LOGGER.error("Failed to load state dictionary for checkpoint %s", checkpoint_path)
            return False

        self.model = model.to(self.device)
        assert self.model is not None  # Model loading succeeded
        self.model.eval()
        self.config = bundle.raw_config
        self._current_checkpoint_path = normalized_path
        LOGGER.info(
            "Model ready | checkpoint=%s | device=%s | total_load=%.2fs",
            checkpoint_path,
            self.device,
            time.perf_counter() - total_start,
        )
        return True

    def update_postprocessor_params(
        self,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
    ) -> None:
        if self.model is None:
            LOGGER.warning("Model not loaded, cannot update postprocessor parameters.")
            return

        settings = self._postprocess_settings
        if settings is None:
            return

        updated_settings = PostprocessSettings(
            binarization_thresh=binarization_thresh if binarization_thresh is not None else settings.binarization_thresh,
            box_thresh=box_thresh if box_thresh is not None else settings.box_thresh,
            max_candidates=int(max_candidates) if max_candidates is not None else settings.max_candidates,
            min_detection_size=int(min_detection_size) if min_detection_size is not None else settings.min_detection_size,
        )
        self._postprocess_settings = updated_settings
        if self._config_bundle is not None:
            self._config_bundle = ModelConfigBundle(
                raw_config=self._config_bundle.raw_config,
                preprocess=self._config_bundle.preprocess,
                postprocess=updated_settings,
            )

        head = getattr(self.model, "head", None)
        postprocess = getattr(head, "postprocess", None)
        if postprocess is None:
            LOGGER.info("Inference model lacks a postprocess module; using engine fallbacks.")
            return

        if hasattr(postprocess, "thresh") and binarization_thresh is not None:
            postprocess.thresh = binarization_thresh
        if hasattr(postprocess, "box_thresh") and box_thresh is not None:
            postprocess.box_thresh = box_thresh
        if hasattr(postprocess, "max_candidates") and max_candidates is not None:
            postprocess.max_candidates = int(max_candidates)
        if hasattr(postprocess, "min_size") and min_detection_size is not None:
            postprocess.min_size = int(min_detection_size)

    def predict_array(
        self,
        image_array: np.ndarray,
        binarization_thresh: float | None = None,
        box_thresh: float | None = None,
        max_candidates: int | None = None,
        min_detection_size: int | None = None,
        return_preview: bool = True,
        enable_perspective_correction: bool | None = None,
    ) -> dict[str, Any] | None:
        """Predict on an image array (numpy) directly.

        New optimized path that accepts numpy arrays directly without file I/O.
        Eliminates tempfile overhead from the inference pipeline.

        Args:
            image_array: Image as numpy array (RGB or BGR format)
            binarization_thresh: Binarization threshold override
            box_thresh: Box threshold override
            max_candidates: Max candidates override
            min_detection_size: Minimum detection size override
            return_preview: If True, maps polygons to preview space and attaches preview image.
                          If False, returns polygons in ORIGINAL image space.
            enable_perspective_correction: Enable rembg-based perspective correction before inference

        Returns:
            Predictions dict with 'polygons', 'texts', 'confidences' keys

        See: artifacts/implementation_plans/2025-11-12_plan-004-revised-inference-consolidation.md
        """
        LOGGER.info("Starting prediction for numpy array with shape: %s", image_array.shape)

        if self.model is None:
            LOGGER.warning("Model not loaded. Returning mock predictions.")
            return generate_mock_predictions()

        if any(value is not None for value in (binarization_thresh, box_thresh, max_candidates, min_detection_size)):
            self.update_postprocessor_params(
                binarization_thresh=binarization_thresh,
                box_thresh=box_thresh,
                max_candidates=max_candidates,
                min_detection_size=min_detection_size,
            )

        # Assume image is already in BGR format (OpenCV standard)
        # If RGB, caller should convert before calling this method
        image = image_array

        LOGGER.info(f"Image array received successfully. Shape: {image.shape}")

        return self._predict_from_array(
            image,
            return_preview=return_preview,
            enable_perspective_correction=enable_perspective_correction,
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
    ) -> dict[str, Any] | None:
        """Predict on an image file (legacy file path-based API).

        Kept for backward compatibility. New code should use predict_array() directly.

        Args:
            image_path: Path to image file
            binarization_thresh: Binarization threshold override
            box_thresh: Box threshold override
            max_candidates: Max candidates override
            min_detection_size: Minimum detection size override
            return_preview: If True, maps polygons to preview space and attaches preview image.
                          If False, returns polygons in ORIGINAL image space.
            enable_perspective_correction: Enable rembg-based perspective correction before inference

        Returns:
            Predictions dict with 'polygons', 'texts', 'confidences' keys
        """
        LOGGER.info(f"Starting prediction for image: {image_path}")

        if self.model is None:
            LOGGER.warning("Model not loaded. Returning mock predictions.")
            return generate_mock_predictions()

        if any(value is not None for value in (binarization_thresh, box_thresh, max_candidates, min_detection_size)):
            self.update_postprocessor_params(
                binarization_thresh=binarization_thresh,
                box_thresh=box_thresh,
                max_candidates=max_candidates,
                min_detection_size=min_detection_size,
            )

        orientation = 1
        canonical_width = 0
        canonical_height = 0
        raw_width = 0
        raw_height = 0

        try:
            with Image.open(image_path) as pil_image:
                raw_width, raw_height = pil_image.size
                normalized_image, orientation = normalize_pil_image(pil_image)

                rgb_image = normalized_image
                if normalized_image.mode != "RGB":
                    rgb_image = normalized_image.convert("RGB")

                image_array = np.asarray(rgb_image)
                canonical_height, canonical_width = image_array.shape[:2]
                image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                if rgb_image is not normalized_image:
                    rgb_image.close()
                if normalized_image is not pil_image:
                    normalized_image.close()
        except OSError:
            LOGGER.error("Failed to read image at path: %s", image_path)
            return None

        LOGGER.info(f"Image loaded successfully. Shape: {image.shape}, Orientation: {orientation}")

        return self._predict_from_array(
            image,
            return_preview=return_preview,
            enable_perspective_correction=enable_perspective_correction,
        )

    def _predict_from_array(
        self,
        image: np.ndarray,
        return_preview: bool = True,
        enable_perspective_correction: bool | None = None,
    ) -> dict[str, Any] | None:
        """Internal method: shared prediction logic for both file and array paths.

        Args:
            image: Image as BGR numpy array
            return_preview: Whether to attach preview image and map coordinates to it
            enable_perspective_correction: Enable rembg-based perspective correction before inference

        Returns:
            Predictions dict or None on failure
        """
        bundle = self._config_bundle
        if bundle is None:
            LOGGER.error("Model configuration bundle missing; load_model must be called first.")
            return None

        if self._transform is None:
            self._transform = build_transform(bundle.preprocess)

        # Optional perspective correction stage (guarded by runtime parameter or config flag)
        # Runtime parameter takes precedence over config
        if enable_perspective_correction is not None:
            enable_persp = bool(enable_perspective_correction)
        else:
            # Fall back to config flag
            raw_config = bundle.raw_config
            enable_persp = False
            try:
                # Prefer an explicit flag if available on the config
                enable_persp = bool(getattr(raw_config, "enable_perspective_correction", False))
            except Exception:
                enable_persp = False

        if enable_persp:
            # BUG-001: perspective-corrected image becomes the coordinate space
            # for decoded polygons; keep this version for downstream preview.
            image = apply_optional_perspective_correction(image, enable_perspective_correction=True)

        # BUG-001: Capture original shape BEFORE preprocessing for coordinate mapping.
        # We'll capture the resized/padded preview AFTER preprocessing for consistent output resolution.
        original_shape = image.shape  # Capture shape before preprocessing
        original_h, original_w = original_shape[:2]

        # Initialize metadata (will be populated after preprocessing)
        meta: dict[str, Any] | None = None

        try:
            # BUG-001: Extract target_size from preprocess settings for LongestMaxSize + PadIfNeeded
            target_size = 640  # Default matching postprocessing assumptions
            if bundle.preprocess.image_size:
                if isinstance(bundle.preprocess.image_size, tuple):
                    target_size = max(bundle.preprocess.image_size)
                else:
                    target_size = bundle.preprocess.image_size

            # BUG-001: Get both the batch tensor and the exact processed image used for inference
            # This ensures the preview image matches exactly what the model sees, eliminating
            # any coordinate misalignment from reconstruction differences.
            batch, preview_image_bgr = preprocess_image(
                image,
                self._transform,
                target_size=target_size,
                return_processed_image=True
            )

            # BUG-001: Verify preview image dimensions match expected target_size
            preview_h, preview_w = preview_image_bgr.shape[:2]
            if preview_h != target_size or preview_w != target_size:
                LOGGER.warning(
                    "BUG-001: Preview image size mismatch: expected %dx%d, got %dx%d. Original: %dx%d",
                    target_size, target_size, preview_w, preview_h, original_w, original_h
                )
            else:
                # Calculate expected content area for debugging
                max_side = max(original_h, original_w)
                scale = target_size / max_side if max_side > 0 else 1.0
                resized_h = int(round(original_h * scale))
                resized_w = int(round(original_w * scale))
                LOGGER.debug(
                    "BUG-001: Preview image verified: %dx%d (original: %dx%d, content area: %dx%d, scale: %.4f)",
                    preview_w, preview_h, original_w, original_h, resized_w, resized_h, scale
                )

            # Calculate metadata for data contract (always calculate, even if preview size mismatch)
            max_side = max(original_h, original_w)
            scale = target_size / max_side if max_side > 0 else 1.0
            resized_h = int(round(original_h * scale))
            resized_w = int(round(original_w * scale))

            # Calculate padding (top-left padding: padding at bottom/right)
            pad_h = target_size - resized_h
            pad_w = target_size - resized_w

            # BUG-001: Metadata contract - polygons are mapped to full processed_size frame (640x640).
            # With top_left padding, content is at [0, resized_w] x [0, resized_h] within [0, target_size] x [0, target_size].
            # coordinate_system="pixel" means absolute pixel coordinates relative to processed_size frame.
            # Viewer should only apply display centering (dx/dy), no padding offsets needed.
            meta = {
                "original_size": (original_w, original_h),
                "processed_size": (target_size, target_size),
                "padding": {
                    "top": 0,
                    "bottom": pad_h,
                    "left": 0,
                    "right": pad_w,
                },
                "padding_position": "top_left",  # Content starts at (0,0), padding at bottom/right
                "content_area": (resized_w, resized_h),  # Content dimensions within processed_size frame
                "scale": float(scale),
                "coordinate_system": "pixel",  # BUG-001: Absolute pixels in processed_size frame [0-target_size, 0-target_size]
            }
            LOGGER.debug(
                "BUG-001: Metadata created: original_size=%s, processed_size=%s, padding=%s, content_area=%s, scale=%.4f",
                meta["original_size"], meta["processed_size"], meta["padding"], meta["content_area"], meta["scale"]
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to preprocess image")
            meta = None  # Ensure meta is None on error
            return None

        LOGGER.info(
            "Image preprocessing completed. Batch shape: %s",
            getattr(batch, "shape", "unknown"),
        )

        if torch is None:
            LOGGER.error("Torch is not available to run inference.")
            return None

        LOGGER.info("Starting model inference...")
        start_time = time.time()
        try:
            if self.model is None:
                raise RuntimeError("Model is not loaded")

            # Direct inference without timeout wrapper to avoid threading issues in Streamlit
            # Streamlit has its own timeout mechanism, and nested threading can cause crashes
            with torch.no_grad():
                predictions = self.model(return_loss=False, images=batch.to(self.device))

            inference_time = time.time() - start_time
            LOGGER.info(f"Model inference completed in {inference_time:.2f} seconds")
        except Exception as e:
            LOGGER.exception(f"Model inference failed: {e}")
            return None

        # BUG-001: Postprocessing maps polygons from resized/padded space back to original_shape.
        # We then map them forward to the resized/padded preview space for consistent output resolution.
        decoded = decode_polygons_with_head(self.model, batch, predictions, original_shape)

        # BUG-001: Helper to map polygons from original_shape to resized/padded preview space.
        # Note: x_offset is captured from outer scope (calculated when centering preview image)
        def _map_polygons_to_preview_space(payload: dict[str, Any]) -> dict[str, Any]:
            """Map polygon coordinates from original image space to resized/padded preview space."""
            if not isinstance(payload, dict) or not payload.get("polygons"):
                return payload

            polygons_str = payload["polygons"]
            if not polygons_str:
                return payload

            # BUG-001: Compute forward transform matching postprocessing logic exactly.
            # Postprocessing uses: scale = 640.0 / max(original_h, original_w)
            # Then maps from 640x640 padded space to original by: x_orig = x_padded * (original_w / resized_w)
            # So we need to map from original to padded by: x_padded = x_orig * (resized_w / original_w)
            # For LongestMaxSize, resized_w = round(original_w * scale), so the forward scale is:
            # forward_scale = resized_w / original_w = round(original_w * scale) / original_w
            # This matches the inverse of what postprocessing does.
            max_side = float(max(original_h, original_w))
            if max_side <= 0:
                return payload

            scale = target_size / max_side
            resized_h = int(round(original_h * scale))
            resized_w = int(round(original_w * scale))

            # BUG-001: Use the same calculation method as postprocessing for consistency
            # Calculate forward scales matching the inverse of postprocessing scales
            # This ensures perfect coordinate alignment: forward_scale is the inverse of postprocessing scale
            forward_scale_x = resized_w / float(original_w) if original_w > 0 else scale
            forward_scale_y = resized_h / float(original_h) if original_h > 0 else scale

            # BUG-001: Log coordinate mapping details for debugging misalignment issues
            LOGGER.debug(
                "BUG-001: Coordinate mapping - original: %dx%d, resized content: %dx%d, "
                "forward_scales: x=%.6f, y=%.6f, processed_size=%dx%d, padding: top=%d bottom=%d left=%d right=%d",
                original_w, original_h, resized_w, resized_h, forward_scale_x, forward_scale_y,
                target_size, target_size, 0, pad_h, 0, pad_w
            )

            # Parse and transform polygons
            polygon_groups = polygons_str.split("|")
            transformed_polygons = []

            for polygon_str in polygon_groups:
                coords = polygon_str.strip().split()
                if len(coords) < 6:
                    continue
                try:
                    coord_floats = [float(c) for c in coords]
                    polygon = np.array([[coord_floats[i], coord_floats[i + 1]] for i in range(0, len(coord_floats), 2)], dtype=np.float32)
                    # BUG-001: Apply forward transform to map coordinates from original space to the full
                    # processed_size frame (640x640). With top_left padding, content starts at (0,0), so
                    # scaling to content-space [0-resized_w, 0-resized_h] is equivalent to full-frame coordinates.
                    # This ensures polygons are in absolute pixel coordinates relative to the processed_size frame,
                    # allowing the viewer to only apply display centering (dx/dy) without any padding offsets.
                    coords_2d = polygon.reshape(-1, 2)
                    transformed_coords = coords_2d * np.array([forward_scale_x, forward_scale_y])

                    # BUG-001: Verify coordinates are within the full processed_size frame [0-target_size, 0-target_size]
                    # With top_left padding, content is at [0-resized_w, 0-resized_h] within [0-target_size, 0-target_size],
                    # so coordinates should be within [0-target_size] range. Allow slight overflow due to rounding.
                    if len(transformed_coords) > 0:
                        min_x = min(c[0] for c in transformed_coords)
                        min_y = min(c[1] for c in transformed_coords)
                        max_x = max(c[0] for c in transformed_coords)
                        max_y = max(c[1] for c in transformed_coords)
                        # Coordinates are in full processed_size frame [0-target_size, 0-target_size]
                        # Allow small overflow (1-2 pixels) due to rounding errors
                        tolerance = 2.0
                        if max_x > target_size + tolerance or max_y > target_size + tolerance or min_x < -tolerance or min_y < -tolerance:
                            LOGGER.warning(
                                "BUG-001: Transformed polygon coordinates out of processed_size bounds: "
                                "min=(%.1f, %.1f), max=(%.1f, %.1f), processed_size=%dx%d, "
                                "content_area=[0-%d, 0-%d] (original: %dx%d)",
                                min_x, min_y, max_x, max_y, target_size, target_size, resized_w, resized_h,
                                original_w, original_h
                            )

                    # Convert back to space-separated string (round to nearest integer)
                    transformed_polygons.append(" ".join(str(int(round(c))) for row in transformed_coords for c in row))
                except (ValueError, IndexError):
                    continue

            payload = dict(payload)
            payload["polygons"] = "|".join(transformed_polygons) if transformed_polygons else ""
            return payload

        # BUG-001: attach preview image (base64 JPEG) for coordinate-aligned overlays.
        # Using JPEG instead of PNG reduces file size by ~10x while maintaining acceptable quality for visualization.
        def _attach_preview(payload: dict[str, Any]) -> dict[str, Any]:
            if not isinstance(payload, dict):
                return payload
            try:
                # BUG-001: Map polygons to preview space (resized/padded) for consistent output resolution
                payload = _map_polygons_to_preview_space(payload)

                # BUG-001: Use JPEG encoding with quality=85 to reduce file size (~10x smaller than PNG)
                # while maintaining acceptable visual quality for overlay alignment verification.
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                success, buffer = cv2.imencode(".jpg", preview_image_bgr, encode_params)
                if not success:
                    LOGGER.warning("BUG-001: cv2.imencode failed for preview image")
                    return payload
                payload = dict(payload)
                payload["preview_image_base64"] = base64.b64encode(buffer).decode("ascii")

                # BUG-001: Attach metadata for data contract - always attach if available
                # This is critical for frontend coordinate handling
                if meta is not None:
                    payload["meta"] = meta
                    LOGGER.debug(
                        "BUG-001: Attached meta to preview response: original_size=%s, processed_size=%s, "
                        "coordinate_system=%s",
                        meta.get("original_size"), meta.get("processed_size"), meta.get("coordinate_system")
                    )
                else:
                    LOGGER.warning(
                        "BUG-001: meta is None when attaching preview - coordinate system contract may be incomplete. "
                        "This may cause frontend to fall back to heuristic normalization."
                    )
            except Exception:  # noqa: BLE001
                LOGGER.exception("BUG-001: Failed to encode preview image for overlay alignment")
                # BUG-001: Even if encoding fails, try to attach meta if available
                if meta is not None:
                    payload = dict(payload)
                    payload["meta"] = meta
            return payload

        if decoded is not None:
            LOGGER.info("Primary decoding successful")
            if return_preview:
                return _attach_preview(decoded)
            return decoded

        LOGGER.info("Primary decoding failed, trying fallback postprocessing...")
        try:
            # BUG-001: Use original_shape (captured before preprocessing) for fallback postprocessing
            result = fallback_postprocess(predictions, original_shape, bundle.postprocess)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Error in post-processing for image with shape %s", original_shape)
            return generate_mock_predictions()

        LOGGER.info("Fallback postprocessing completed")
        if return_preview:
            return _attach_preview(result)
        return result

    # Internal helpers ---------------------------------------------------
    def _apply_config_bundle(self, bundle: ModelConfigBundle) -> None:
        self._config_bundle = bundle
        self._postprocess_settings = bundle.postprocess
        self._transform = None

    @staticmethod
    def _remap_predictions_if_needed(
        result: dict[str, Any],
        orientation: int,
        canonical_width: int,
        canonical_height: int,
        raw_width: int,
        raw_height: int,
    ) -> dict[str, Any]:
        if not result:
            return result

        if not orientation_requires_rotation(orientation):
            return result

        polygons_str = result.get("polygons")
        if not polygons_str:
            return result

        inverse_orientation = ORIENTATION_INVERSE_INT.get(orientation, 1)
        if inverse_orientation == 1:
            return result

        remapped_polygons = []
        for polygon_entry in polygons_str.split("|"):
            tokens = _POLYGON_TOKEN_PATTERN.findall(polygon_entry)
            if len(tokens) < 8 or len(tokens) % 2 != 0:
                continue
            try:
                coords = np.array([float(value) for value in tokens], dtype=np.float32).reshape(-1, 2)
            except ValueError:
                continue
            remapped_polygons.append(coords)

        if not remapped_polygons:
            return result

        transformed = remap_polygons(remapped_polygons, raw_width, raw_height, inverse_orientation)
        serialised: list[str] = []
        for polygon in transformed:
            flat = polygon.reshape(-1)
            # Competition format uses space-separated coordinates, not commas
            serialised.append(" ".join(str(int(round(value))) for value in flat))

        updated = dict(result)
        updated["polygons"] = "|".join(serialised)
        return updated


def run_inference_on_image(
    image_path: str | np.ndarray,
    checkpoint_path: str,
    config_path: str | None = None,
    binarization_thresh: float | None = None,
    box_thresh: float | None = None,
    max_candidates: int | None = None,
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
