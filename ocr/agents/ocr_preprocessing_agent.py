"""
OCR Preprocessing Agent

Handles image preprocessing tasks for OCR pipeline including:
- Image normalization
- Binarization
- Noise removal
- Orientation correction
- Background removal
"""

import logging
import os
from typing import Any, Optional
from pathlib import Path
import json

from ocr.agents.base_agent import BaseAgent, AgentCapability

logger = logging.getLogger(__name__)


class OCRPreprocessingAgent(BaseAgent):
    """
    Specialized agent for OCR preprocessing tasks.

    Capabilities:
    - Normalize images for OCR
    - Remove background
    - Correct orientation
    - Apply filters and enhancements
    """

    def __init__(
        self,
        agent_id: str = "agent.ocr.preprocessor",
        rabbitmq_host: str = "rabbitmq"
    ):
        """Initialize OCR preprocessing agent."""
        capabilities = [
            AgentCapability(
                name="normalize_image",
                description="Normalize image for OCR processing",
                input_schema={
                    "image_path": "str",
                    "output_path": "str (optional)",
                    "options": {
                        "resize": "bool (default: False)",
                        "target_size": "tuple[int, int] (optional)",
                        "enhance_contrast": "bool (default: True)",
                        "denoise": "bool (default: False)"
                    }
                },
                output_schema={
                    "status": "str",
                    "output_path": "str",
                    "metadata": "dict"
                }
            ),
            AgentCapability(
                name="remove_background",
                description="Remove background from document images",
                input_schema={
                    "image_path": "str",
                    "output_path": "str (optional)",
                    "model": "str (default: 'u2net')"
                },
                output_schema={
                    "status": "str",
                    "output_path": "str",
                    "processing_time": "float"
                }
            ),
            AgentCapability(
                name="batch_preprocess",
                description="Preprocess multiple images in batch",
                input_schema={
                    "image_paths": "list[str]",
                    "output_dir": "str",
                    "options": "dict"
                },
                output_schema={
                    "status": "str",
                    "processed_count": "int",
                    "results": "list[dict]"
                }
            )
        ]

        super().__init__(
            agent_id=agent_id,
            agent_type="ocr.preprocessor",
            rabbitmq_host=rabbitmq_host,
            capabilities=capabilities
        )

        # Register custom handlers
        self.register_handler("cmd.normalize_image", self._handle_normalize_image)
        self.register_handler("cmd.remove_background", self._handle_remove_background)
        self.register_handler("cmd.batch_preprocess", self._handle_batch_preprocess)

    def get_binding_keys(self) -> list[str]:
        """Return routing keys for this agent."""
        return [
            "cmd.normalize_image.#",
            "cmd.remove_background.#",
            "cmd.batch_preprocess.#"
        ]

    def _handle_normalize_image(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle image normalization request."""
        payload = envelope.get("payload", {})
        image_path = payload.get("image_path")
        output_path = payload.get("output_path")
        options = payload.get("options", {})

        if not image_path:
            return {"status": "error", "message": "image_path is required"}

        try:
            import cv2
            import numpy as np
            from PIL import Image

            # Load image
            img_path = Path(image_path)
            if not img_path.exists():
                return {"status": "error", "message": f"Image not found: {image_path}"}

            img = cv2.imread(str(img_path))
            if img is None:
                return {"status": "error", "message": "Failed to load image"}

            original_shape = img.shape

            # Apply preprocessing options
            if options.get("resize", False):
                target_size = options.get("target_size")
                if target_size:
                    img = cv2.resize(img, tuple(target_size))

            if options.get("enhance_contrast", True):
                # Convert to LAB color space
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)

                # Merge channels
                lab = cv2.merge([l, a, b])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            if options.get("denoise", False):
                img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

            # Save output
            if not output_path:
                output_path = str(img_path.parent / f"{img_path.stem}_preprocessed{img_path.suffix}")

            cv2.imwrite(output_path, img)

            return {
                "status": "success",
                "output_path": output_path,
                "metadata": {
                    "original_shape": original_shape,
                    "processed_shape": img.shape,
                    "options_applied": options
                }
            }

        except Exception as e:
            logger.error(f"Image normalization failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_remove_background(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle background removal request."""
        payload = envelope.get("payload", {})
        image_path = payload.get("image_path")
        output_path = payload.get("output_path")
        model = payload.get("model", "u2net")

        if not image_path:
            return {"status": "error", "message": "image_path is required"}

        try:
            import time
            from rembg import remove
            from PIL import Image

            start_time = time.time()

            # Load image
            img_path = Path(image_path)
            if not img_path.exists():
                return {"status": "error", "message": f"Image not found: {image_path}"}

            input_img = Image.open(img_path)

            # Remove background
            output_img = remove(input_img, model_name=model)

            # Save output
            if not output_path:
                output_path = str(img_path.parent / f"{img_path.stem}_nobg.png")

            output_img.save(output_path)

            processing_time = time.time() - start_time

            return {
                "status": "success",
                "output_path": output_path,
                "processing_time": processing_time,
                "model_used": model
            }

        except Exception as e:
            logger.error(f"Background removal failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_batch_preprocess(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """Handle batch preprocessing request."""
        payload = envelope.get("payload", {})
        image_paths = payload.get("image_paths", [])
        output_dir = payload.get("output_dir")
        options = payload.get("options", {})

        if not image_paths:
            return {"status": "error", "message": "image_paths is required"}

        if not output_dir:
            return {"status": "error", "message": "output_dir is required"}

        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            results = []
            processed_count = 0

            for img_path in image_paths:
                img_name = Path(img_path).name
                out_path = str(output_path / img_name)

                # Create envelope for single image processing
                single_envelope = {
                    "payload": {
                        "image_path": img_path,
                        "output_path": out_path,
                        "options": options
                    }
                }

                result = self._handle_normalize_image(single_envelope)

                if result.get("status") == "success":
                    processed_count += 1

                results.append({
                    "image": img_path,
                    "result": result
                })

            return {
                "status": "success",
                "processed_count": processed_count,
                "total_count": len(image_paths),
                "results": results
            }

        except Exception as e:
            logger.error(f"Batch preprocessing failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Run the agent
    agent = OCRPreprocessingAgent()
    agent.start()
