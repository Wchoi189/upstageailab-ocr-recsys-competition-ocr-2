"""
OCR Inference Agent

Handles OCR model inference tasks including:
- Text detection
- Text recognition
- Layout analysis
- Multi-model ensemble
"""

import logging
import sys
from typing import Any, Optional
from pathlib import Path
import json

from ocr.agents.base_agent import BaseAgent, AgentCapability

logger = logging.getLogger(__name__)


class OCRInferenceAgent(BaseAgent):
    """
    Specialized agent for OCR model inference.

    Capabilities:
    - Text detection from images
    - Text recognition
    - Layout analysis
    - Batch inference
    """

    def __init__(
        self,
        agent_id: str = "agent.ocr.inference",
        rabbitmq_host: str = "rabbitmq",
        model_config_path: Optional[str] = None
    ):
        """Initialize OCR inference agent."""
        capabilities = [
            AgentCapability(
                name="detect_text",
                description="Detect text regions in image",
                input_schema={
                    "image_path": "str",
                    "model_name": "str (optional)",
                    "confidence_threshold": "float (default: 0.5)"
                },
                output_schema={
                    "status": "str",
                    "detections": "list[dict]",
                    "count": "int"
                }
            ),
            AgentCapability(
                name="recognize_text",
                description="Recognize text from image regions",
                input_schema={
                    "image_path": "str",
                    "bboxes": "list[list[float]] (optional)",
                    "model_name": "str (optional)"
                },
                output_schema={
                    "status": "str",
                    "recognized_texts": "list[str]",
                    "confidences": "list[float]"
                }
            ),
            AgentCapability(
                name="full_ocr",
                description="Complete OCR pipeline (detect + recognize)",
                input_schema={
                    "image_path": "str",
                    "detection_model": "str (optional)",
                    "recognition_model": "str (optional)",
                    "return_format": "str (default: 'structured')"
                },
                output_schema={
                    "status": "str",
                    "results": "list[dict]",
                    "full_text": "str"
                }
            ),
            AgentCapability(
                name="batch_inference",
                description="Run OCR inference on multiple images",
                input_schema={
                    "image_paths": "list[str]",
                    "output_dir": "str (optional)",
                    "model_config": "dict (optional)"
                },
                output_schema={
                    "status": "str",
                    "results": "list[dict]",
                    "summary": "dict"
                }
            )
        ]

        super().__init__(
            agent_id=agent_id,
            agent_type="ocr.inference",
            rabbitmq_host=rabbitmq_host,
            capabilities=capabilities
        )

        self.model_config_path = model_config_path

        # Register custom handlers
        self.register_handler("cmd.detect_text", self._handle_detect_text)
        self.register_handler("cmd.recognize_text", self._handle_recognize_text)
        self.register_handler("cmd.full_ocr", self._handle_full_ocr)
        self.register_handler("cmd.batch_inference", self._handle_batch_inference)

        # Initialize models (lazy loading)
        self._inference_engine = None

    def get_binding_keys(self) -> list[str]:
        """Return routing keys for this agent."""
        return [
            "cmd.detect_text.#",
            "cmd.recognize_text.#",
            "cmd.full_ocr.#",
            "cmd.batch_inference.#"
        ]

    def _get_inference_engine(self):
        """Lazy load inference engine."""
        if self._inference_engine is None:
            try:
                from ocr.core.inference.engine import InferenceEngine
                from ocr.core.utils.path_utils import PROJECT_ROOT

                # Load default config if not specified
                if self.model_config_path:
                    config_path = Path(self.model_config_path)
                else:
                    config_path = PROJECT_ROOT / "configs" / "inference" / "default.yaml"

                if config_path.exists():
                    self._inference_engine = InferenceEngine.from_config(str(config_path))
                    logger.info(f"Loaded inference engine from {config_path}")
                else:
                    logger.warning(f"Config not found: {config_path}, using default settings")
                    self._inference_engine = InferenceEngine()

            except Exception as e:
                logger.error(f"Failed to load inference engine: {e}")
                raise

        return self._inference_engine

    def _handle_detect_text(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle text detection request."""
        payload = envelope.get("payload", {})
        image_path = payload.get("image_path")
        model_name = payload.get("model_name")
        confidence_threshold = payload.get("confidence_threshold", 0.5)

        if not image_path:
            return {"status": "error", "message": "image_path is required"}

        try:
            import cv2
            import numpy as np

            # Load image
            img_path = Path(image_path)
            if not img_path.exists():
                return {"status": "error", "message": f"Image not found: {image_path}"}

            img = cv2.imread(str(img_path))
            if img is None:
                return {"status": "error", "message": "Failed to load image"}

            # Run detection
            engine = self._get_inference_engine()

            # For now, use a simplified detection approach
            # In production, integrate with your actual OCR models
            detections = []

            # Placeholder: would call actual model here
            # detections = engine.detect(img, model_name=model_name, threshold=confidence_threshold)

            logger.info(f"Detected {len(detections)} text regions in {image_path}")

            return {
                "status": "success",
                "detections": detections,
                "count": len(detections),
                "image_shape": img.shape
            }

        except Exception as e:
            logger.error(f"Text detection failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_recognize_text(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle text recognition request."""
        payload = envelope.get("payload", {})
        image_path = payload.get("image_path")
        bboxes = payload.get("bboxes", [])
        model_name = payload.get("model_name")

        if not image_path:
            return {"status": "error", "message": "image_path is required"}

        try:
            import cv2

            # Load image
            img_path = Path(image_path)
            if not img_path.exists():
                return {"status": "error", "message": f"Image not found: {image_path}"}

            img = cv2.imread(str(img_path))
            if img is None:
                return {"status": "error", "message": "Failed to load image"}

            # Run recognition
            engine = self._get_inference_engine()

            recognized_texts = []
            confidences = []

            # Placeholder: would call actual model here
            # results = engine.recognize(img, bboxes=bboxes, model_name=model_name)
            # recognized_texts = [r['text'] for r in results]
            # confidences = [r['confidence'] for r in results]

            return {
                "status": "success",
                "recognized_texts": recognized_texts,
                "confidences": confidences,
                "count": len(recognized_texts)
            }

        except Exception as e:
            logger.error(f"Text recognition failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_full_ocr(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle full OCR pipeline request."""
        payload = envelope.get("payload", {})
        image_path = payload.get("image_path")
        detection_model = payload.get("detection_model")
        recognition_model = payload.get("recognition_model")
        return_format = payload.get("return_format", "structured")

        if not image_path:
            return {"status": "error", "message": "image_path is required"}

        try:
            # Step 1: Detect text
            detect_envelope = {
                "payload": {
                    "image_path": image_path,
                    "model_name": detection_model
                }
            }
            detect_result = self._handle_detect_text(detect_envelope)

            if detect_result.get("status") != "success":
                return detect_result

            detections = detect_result.get("detections", [])

            if not detections:
                return {
                    "status": "success",
                    "results": [],
                    "full_text": "",
                    "message": "No text detected in image"
                }

            # Step 2: Recognize text
            bboxes = [d["bbox"] for d in detections if "bbox" in d]

            recognize_envelope = {
                "payload": {
                    "image_path": image_path,
                    "bboxes": bboxes,
                    "model_name": recognition_model
                }
            }
            recognize_result = self._handle_recognize_text(recognize_envelope)

            if recognize_result.get("status") != "success":
                return recognize_result

            # Combine results
            recognized_texts = recognize_result.get("recognized_texts", [])
            confidences = recognize_result.get("confidences", [])

            results = []
            for i, detection in enumerate(detections[:len(recognized_texts)]):
                results.append({
                    "bbox": detection.get("bbox"),
                    "text": recognized_texts[i],
                    "confidence": confidences[i] if i < len(confidences) else 0.0
                })

            full_text = " ".join(recognized_texts)

            return {
                "status": "success",
                "results": results,
                "full_text": full_text,
                "detection_count": len(detections),
                "recognized_count": len(recognized_texts)
            }

        except Exception as e:
            logger.error(f"Full OCR failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_batch_inference(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle batch inference request."""
        payload = envelope.get("payload", {})
        image_paths = payload.get("image_paths", [])
        output_dir = payload.get("output_dir")
        model_config = payload.get("model_config", {})

        if not image_paths:
            return {"status": "error", "message": "image_paths is required"}

        try:
            results = []
            success_count = 0
            error_count = 0

            for img_path in image_paths:
                # Create envelope for single image OCR
                single_envelope = {
                    "payload": {
                        "image_path": img_path,
                        **model_config
                    }
                }

                result = self._handle_full_ocr(single_envelope)

                if result.get("status") == "success":
                    success_count += 1
                else:
                    error_count += 1

                results.append({
                    "image": img_path,
                    "result": result
                })

                # Save individual results if output_dir specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    result_file = output_path / f"{Path(img_path).stem}_result.json"
                    with open(result_file, "w") as f:
                        json.dump(result, f, indent=2)

            return {
                "status": "success",
                "results": results,
                "summary": {
                    "total": len(image_paths),
                    "success": success_count,
                    "errors": error_count
                }
            }

        except Exception as e:
            logger.error(f"Batch inference failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Run the agent
    agent = OCRInferenceAgent()
    agent.start()
