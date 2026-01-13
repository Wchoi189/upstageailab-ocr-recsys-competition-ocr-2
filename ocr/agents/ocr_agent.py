import cv2
import logging
import numpy as np
from typing import Any

from ocr.agents.base_agent import BaseAgent
from ocr.core.inference.orchestrator import InferenceOrchestrator

logger = logging.getLogger("OCRAgent")

class OCRAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent.ocr",
            binding_keys=["cmd.ocr.process", "cmd.ocr.inference"]
        )

        # Initialize orchestrator lazily or on start?
        # On starup is better to fail fast if model missing.
        try:
            self.logger.info("Initializing InferenceOrchestrator...")
            self.orchestrator = InferenceOrchestrator(enable_recognition=True)
            # Find default model if not configured
            # For now we assume models are loaded dynamically or a default exists
            # In a real scenario we might load a default config here.

            # Using orchestrator as context manager in process_payload is safer for cleanup?
            # But loading models is heavy. We keep it persistent.

            # Loading placeholder model or waiting for explicit load config?
            # For now, we instantiate without model loaded, load on demand or if default exists.

        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    def process_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Process OCR request.
        Expected payload:
        {
            "files": ["path/to/image.jpg"],
            "options": {
                "checkpoint_path": "optional/path/to/model.pth",
                "enable_extraction": false
            }
        }
        """
        files = payload.get('files', [])
        options = payload.get('options', {})

        # Determine checkpoint - this is critical.
        # If orchestrator not loaded, we must load it.
        checkpoint_path = options.get('checkpoint_path')

        if not self.orchestrator.model_manager.is_loaded():
            if not checkpoint_path:
                 # Try to resolve a default model path or fail
                 # For MVP, let's assume there's a known location or fail
                 model_dir = self.resolve_path("models/ocr/default.pth") # Hypothetical default
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
