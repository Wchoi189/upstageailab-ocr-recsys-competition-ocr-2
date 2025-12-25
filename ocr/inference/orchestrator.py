"""Inference orchestrator for OCR pipeline.

This module provides a thin coordination layer that integrates all inference
components:
- ModelManager: Model lifecycle management
- PreprocessingPipeline: Image preprocessing
- PostprocessingPipeline: Prediction postprocessing
- PreviewGenerator: Preview image generation
- TextRecognizer: Text recognition (optional, disabled by default)
- CropExtractor: Crop extraction for recognition

The orchestrator follows the single responsibility principle: it only coordinates
the flow between components, delegating all actual work to specialized classes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config_loader import PostprocessSettings
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

    def __init__(self, device: str | None = None, enable_recognition: bool = False):
        """Initialize inference orchestrator.

        Args:
            device: Device for inference ("cuda" or "cpu", auto-detected if None)
            enable_recognition: Whether to enable text recognition (default: False)
        """
        self.model_manager = ModelManager(device=device)
        self.preprocessing_pipeline: PreprocessingPipeline | None = None
        self.postprocessing_pipeline: PostprocessingPipeline | None = None
        self.preview_generator = PreviewGenerator(jpeg_quality=85)

        # Recognition components (disabled by default)
        self._enable_recognition = enable_recognition
        self._recognizer = None
        self._crop_extractor = None

        # Layout and extraction components (disabled by default)
        self._enable_layout = False
        self._enable_extraction = False
        self._layout_grouper = None
        self._field_extractor = None

        if enable_recognition:
            self._init_recognition_components()

        LOGGER.info(
            "InferenceOrchestrator initialized (device: %s, recognition: %s)",
            self.model_manager.device,
            enable_recognition,
        )

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

    def _init_recognition_components(self) -> None:
        """Initialize text recognition components.

        Lazy initialization of recognition module to avoid loading
        when recognition is disabled.
        """
        try:
            from .crop_extractor import CropConfig, CropExtractor
            from .recognizer import RecognizerConfig, TextRecognizer

            self._crop_extractor = CropExtractor(config=CropConfig())
            self._recognizer = TextRecognizer(config=RecognizerConfig())
            LOGGER.info("Recognition components initialized")
        except Exception as e:
            LOGGER.warning("Failed to initialize recognition: %s", e)
            self._enable_recognition = False
            self._recognizer = None
            self._crop_extractor = None

    def enable_extraction_pipeline(self) -> None:
        """Enable layout + extraction modules.

        This initializes the LineGrouper and ReceiptFieldExtractor components
        to enable full receipt extraction functionality.
        """
        try:
            from .extraction.field_extractor import ExtractorConfig, ReceiptFieldExtractor
            from .layout.grouper import LineGrouper, LineGrouperConfig

            self._enable_layout = True
            self._enable_extraction = True
            self._layout_grouper = LineGrouper(config=LineGrouperConfig())
            self._field_extractor = ReceiptFieldExtractor(config=ExtractorConfig())
            LOGGER.info("Extraction pipeline components initialized")
        except Exception as e:
            LOGGER.warning("Failed to initialize extraction pipeline: %s", e)
            self._enable_layout = False
            self._enable_extraction = False
            self._layout_grouper = None
            self._field_extractor = None

    def predict(
        self,
        image: np.ndarray,
        return_preview: bool = True,
        enable_perspective_correction: bool = False,
        perspective_display_mode: str = "corrected",
        enable_grayscale: bool = False,
        enable_background_normalization: bool = False,
        enable_sepia_enhancement: bool = False,
        enable_clahe: bool = False,
        sepia_display_mode: str = "enhanced",
        enable_extraction: bool = False,
    ) -> dict[str, Any] | None:
        """Run inference on image array.

        This is the main orchestration method. It coordinates the full pipeline:
        1. Preprocessing (resize, normalize, metadata)
        2. Model inference
        3. Postprocessing (decode, format)
        4. Text recognition (if enabled)
        5. Layout grouping (if extraction enabled)
        6. Field extraction (if enabled)
        7. Preview generation (if requested)

        Args:
            image: Input image as BGR numpy array (H, W, C)
            return_preview: Whether to generate and attach preview image
            enable_perspective_correction: Whether to apply perspective correction
            perspective_display_mode: "corrected" or "original" display mode
            enable_grayscale: Whether to apply grayscale preprocessing
            enable_background_normalization: Whether to apply gray-world background normalization
            enable_sepia_enhancement: Whether to apply sepia tone transformation
            enable_clahe: Whether to apply CLAHE contrast enhancement
            sepia_display_mode: Display mode for sepia ("enhanced" or "original")
            enable_extraction: Whether to enable layout + field extraction pipeline

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
            enable_background_normalization=enable_background_normalization,
            enable_sepia_enhancement=enable_sepia_enhancement,
            enable_clahe=enable_clahe,
            sepia_display_mode=sepia_display_mode,
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
                predictions = self.model_manager.model(return_loss=False, images=preprocess_result.batch.to(self.model_manager.device))
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

        # Optional Stage: Text Recognition
        if self._enable_recognition and self._recognizer and self._crop_extractor:
            result = self._run_text_recognition(
                image=image,
                result=result,
            )

        # Stage 5: Layout grouping (if extraction enabled)
        layout_result = None
        if self._enable_layout and self._enable_recognition:
            layout_result = self._run_layout_grouping(result)
            result["layout"] = layout_result.model_dump()

        # Stage 6: Field extraction with hybrid gating (if requested)
        if enable_extraction and self._enable_extraction and layout_result is not None:
            receipt_data = self._run_extraction_with_gating(
                layout_result, image
            )
            result["receipt_data"] = receipt_data.model_dump()

        # Stage 7: Handle inverse perspective transformation if needed
        if (
            preprocess_result.perspective_matrix is not None
            and perspective_display_mode == "original"
            and preprocess_result.original_image is not None
        ):
            # Transform polygons back to original space
            from ocr.utils.perspective_correction import transform_polygons_inverse

            if result["polygons"]:
                result["polygons"] = transform_polygons_inverse(
                    result["polygons"],
                    preprocess_result.perspective_matrix,
                )

            # Create preview from original image
            original_preview = self.preprocessing_pipeline.process_for_original_display(preprocess_result.original_image)
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

        # Stage 8: Preview generation
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

    def _run_text_recognition(
        self,
        image: np.ndarray,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run text recognition on detected regions.

        Args:
            image: Original input image (BGR numpy array)
            result: Detection result with polygons

        Returns:
            Updated result with recognized text
        """
        if not result.get("polygons"):
            return result

        try:
            from .recognizer import RecognitionInput

            # Parse polygons from string format
            polygon_strs = result["polygons"].split("|") if isinstance(result["polygons"], str) else result["polygons"]
            polygons = []
            for poly_str in polygon_strs:
                if isinstance(poly_str, str):
                    coords = list(map(float, poly_str.split()))
                    polygons.append(np.array(coords).reshape(-1, 2))
                else:
                    polygons.append(np.array(poly_str))

            # Extract crops
            crop_results = self._crop_extractor.extract_crops(image, polygons)

            # Prepare recognition inputs (only successful crops)
            recognition_inputs = []
            crop_indices = []  # Track which detections have valid crops
            for i, crop_result in enumerate(crop_results):
                if crop_result.success and crop_result.crop is not None:
                    detection_conf = result["confidences"][i] if i < len(result["confidences"]) else 0.5
                    recognition_inputs.append(
                        RecognitionInput(
                            crop=crop_result.crop,
                            polygon=crop_result.original_polygon,
                            detection_confidence=detection_conf,
                        )
                    )
                    crop_indices.append(i)

            if not recognition_inputs:
                LOGGER.debug("No valid crops for recognition")
                return result

            # Run recognition
            recognition_outputs = self._recognizer.recognize_batch(recognition_inputs)

            # Update result with recognized text
            recognized_texts = list(result["texts"])  # Copy
            recognition_confidences = [0.0] * len(result["texts"])

            for idx, output in zip(crop_indices, recognition_outputs, strict=False):
                if idx < len(recognized_texts):
                    recognized_texts[idx] = output.text
                    recognition_confidences[idx] = output.confidence

            result["recognized_texts"] = recognized_texts
            result["recognition_confidences"] = recognition_confidences

            LOGGER.debug(
                "Recognition complete: %d/%d texts recognized",
                len(recognition_outputs),
                len(polygon_strs),
            )

        except Exception as e:
            LOGGER.warning("Text recognition failed: %s", e)
            # Don't fail the whole pipeline - just skip recognition

        return result

    def _run_layout_grouping(self, result: dict) -> Any:
        """Group recognized text into lines and blocks.

        Args:
            result: Detection/recognition result dict with polygons and texts

        Returns:
            LayoutResult with hierarchical text structure
        """
        from .layout.contracts import BoundingBox, TextElement

        elements = []
        for i, (poly, text, conf) in enumerate(zip(
            result.get("polygons", []),
            result.get("recognized_texts", []),
            result.get("recognition_confidences", []),
            strict=False,
        )):
            # Convert polygon string to coordinates
            coords = self._parse_polygon(poly)
            if not coords:
                continue

            bbox = BoundingBox(
                x_min=min(c[0] for c in coords),
                y_min=min(c[1] for c in coords),
                x_max=max(c[0] for c in coords),
                y_max=max(c[1] for c in coords),
            )
            elements.append(TextElement(
                polygon=coords,
                bbox=bbox,
                text=text,
                confidence=conf,
            ))

        return self._layout_grouper.group_elements(elements)

    def _parse_polygon(self, poly: str | list) -> list[list[float]]:
        """Parse polygon from string or list format.

        Args:
            poly: Polygon as string "x1 y1 x2 y2 ..." or list of coordinates

        Returns:
            List of [x, y] coordinate pairs
        """
        if isinstance(poly, str):
            try:
                coords = list(map(float, poly.split()))
                # Convert flat list to pairs [[x1,y1], [x2,y2], ...]
                return [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
            except (ValueError, IndexError):
                LOGGER.warning("Failed to parse polygon string: %s", poly)
                return []
        elif isinstance(poly, list):
            # Already in list format
            return poly
        else:
            return []

    def _run_extraction_with_gating(
        self,
        layout_result: Any,
        original_image: np.ndarray,
    ) -> Any:
        """Extract receipt data with hybrid rule/VLM gating.

        Args:
            layout_result: LayoutResult from grouping stage
            original_image: Original BGR image

        Returns:
            ReceiptData with extracted fields
        """
        # Try rule-based first (fast path: 80% of receipts)
        receipt = self._field_extractor.extract(layout=layout_result)

        # Gate to VLM if confidence too low or complex layout
        if self._should_use_vlm(receipt, layout_result):
            LOGGER.debug("Gating to VLM extraction (confidence=%.2f)", receipt.extraction_confidence)
            receipt = self._run_vlm_extraction(original_image, layout_result)

        return receipt

    def _should_use_vlm(self, receipt: Any, layout: Any) -> bool:
        """Determine if VLM extraction should be used.

        Args:
            receipt: ReceiptData from rule-based extraction
            layout: LayoutResult

        Returns:
            True if VLM should be used
        """
        # Low confidence from rule-based extraction
        if receipt.extraction_confidence < 0.7:
            return True

        # Complex layout indicators
        if len(layout.blocks) > 5:  # Many separate text blocks
            return True
        if layout.tables:  # Table structures detected
            return True

        return False

    def _run_vlm_extraction(
        self,
        image: np.ndarray,
        layout: Any,
    ) -> Any:
        """Run VLM extraction with fallback to rule-based.

        Args:
            image: Original BGR image
            layout: LayoutResult for context

        Returns:
            ReceiptData from VLM or rule-based fallback
        """
        try:
            import cv2
            from PIL import Image

            from .extraction.vlm_extractor import VLMExtractor

            vlm = VLMExtractor()
            if not vlm.is_server_healthy():
                LOGGER.warning("VLM server unavailable, using rule-based")
                return self._field_extractor.extract(layout=layout)

            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return vlm.extract(pil_image, ocr_context=layout.text)
        except Exception as e:
            LOGGER.warning("VLM extraction failed: %s", e)
            return self._field_extractor.extract(layout=layout)

    def cleanup(self) -> None:
        """Clean up resources."""
        self.model_manager.cleanup()

        # Clean up recognition components
        if self._recognizer is not None:
            self._recognizer.cleanup()
            self._recognizer = None
        self._crop_extractor = None

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
