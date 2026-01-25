import cv2
import logging
import numpy as np
from typing import Any

from ocr.core.infrastructure.agents.base_agent import BaseAgent
from ocr.core.infrastructure.communication.iacp_schemas import IACPEnvelope
# NOTE: OCRAgent is legacy/unused - orchestrator API changed
# from ocr.pipelines.orchestrator import OCRProjectOrchestrator

logger = logging.getLogger("OCRAgent")

class OCRAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent.ocr",
            agent_type="ocr.inference",
            capabilities=[], # Add capabilities definition formally later
        )

        # Register custom handlers
        self.register_handler("cmd.ocr.process", self._handle_process_payload)
        self.register_handler("cmd.ocr.inference", self._handle_process_payload)

        # Initialize orchestrator
        try:
            self.logger = logger # Fix self.logger reference
            self.logger.info("Initializing InferenceOrchestrator...")
            self.orchestrator = InferenceOrchestrator(enable_recognition=True)
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    def get_binding_keys(self) -> list[str]:
        return ["cmd.ocr.process", "cmd.ocr.inference"]

    def _handle_process_payload(self, envelope: IACPEnvelope) -> dict[str, Any]:
        """
        Process OCR request.
        """
        payload = envelope.payload
        files = payload.get('files', [])
        options = payload.get('options', {})

        # Determine checkpoint - this is critical.
        # If orchestrator not loaded, we must load it.
        checkpoint_path = options.get('checkpoint_path')

        if not self.orchestrator.model_manager.is_loaded():
            if not checkpoint_path:
                 # Try to resolve a default model path or fail
                 model_dir = self.resolve_path("models/ocr/default.pth") 
                 if model_dir.exists():
                     checkpoint_path = str(model_dir)
                 else:
                     return {"status": "error", "message": "Model not loaded and no checkpoint provided"}

            self.logger.info(f"Loading model from {checkpoint_path}")
            if not self.orchestrator.load_model(checkpoint_path):
                return {"status": "error", "message": f"Failed to load model from {checkpoint_path}"}

        results = []

        for f in files:
            image_path = self.resolve_path(f)
            if not image_path.exists():
                results.append({"file": f, "error": "File not found"})
                continue

            try:
                # Read image using cv2
                image = cv2.imread(str(image_path))
                if image is None:
                    results.append({"file": f, "error": "Failed to read image"})
                    continue

                # Run prediction
                pred = self.orchestrator.predict(
                    image,
                    return_preview=False, # Don't return binary preview in JSON response for now
                    enable_extraction=options.get('enable_extraction', False)
                )

                # Serialize numpy/arrays to lists
                if pred:
                    # Convert numpy confidence arrays to list
                    if "confidences" in pred and isinstance(pred["confidences"], (np.ndarray, list)):
                         pred["confidences"] = [float(c) for c in pred["confidences"]]

                    # Handle polygons if they are lists of numpy arrays
                    # Orchestrator returns them as parsed or string?
                    # The predict method returns dict.

                    results.append({"file": f, "result": pred})
                else:
                    results.append({"file": f, "error": "Inference failed (returned None)"})

            except Exception as e:
                self.logger.exception(f"Error processing {f}")
                results.append({"file": f, "error": str(e)})

        return {
            "status": "success",
            "results": results,
            "count": len(results)
        }

    def shutdown(self):
        super().shutdown()
        if hasattr(self, 'orchestrator'):
            self.orchestrator.cleanup()

if __name__ == "__main__":
    agent = OCRAgent()
    agent.run()
