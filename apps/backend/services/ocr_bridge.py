import os
import io
from pathlib import Path
# Lazy import: torch, numpy, PIL only loaded when needed
from typing import List, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

# Lightweight imports only
from ocr.utils.path_utils import setup_project_paths

# Ensure project paths are set
setup_project_paths()

router = APIRouter(prefix="/ocr", tags=["OCR"])

class OCRBridge:
    """OCR inference bridge using proven InferenceEngine.

    This class wraps ocr.inference.InferenceEngine to provide a consistent
    inference interface for the OCR Inference Console backend. The InferenceEngine
    is a battle-tested implementation now moved to the core ocr package.

    Model loads only on first inference request (lazy loading) for fast server startup.
    """
    def __init__(self, checkpoint_path: str, device: str = None):
        self.checkpoint_path = checkpoint_path
        self.device = device  # InferenceEngine auto-detects device, this is for compatibility
        self.engine = None  # Lazy loading: don't load until first use
        print(f"OCR Bridge initialized with checkpoint: {checkpoint_path}")
        print(f"InferenceEngine will load on first request (lazy loading enabled)")

    def _ensure_engine_loaded(self) -> None:
        """Lazy load InferenceEngine on first use."""
        if self.engine is None:
            print(f"Loading InferenceEngine for checkpoint: {self.checkpoint_path}")
            # Lazy import - only load when model is actually needed
            from ocr.inference import InferenceEngine

            self.engine = InferenceEngine()

            # Load model using InferenceEngine's robust loading logic
            if not self.engine.load_model(self.checkpoint_path, config_path=None):
                self.engine = None
                raise RuntimeError(
                    f"InferenceEngine failed to load model from checkpoint: {self.checkpoint_path}\n"
                    f"Check that the checkpoint exists and has an associated config.yaml file."
                )

            print(f"InferenceEngine loaded successfully")

    def predict(self, image) -> tuple:
        """Run OCR inference on image using InferenceEngine.

        Args:
            image: PIL Image object

        Returns:
            tuple: (boxes, scores) where boxes is list of np.ndarray polygons,
                   scores is list of float confidence values

        Raises:
            RuntimeError: If InferenceEngine fails to load or inference fails
        """
        # Lazy imports
        import numpy as np
        from PIL import Image as PILImage

        self._ensure_engine_loaded()  # Load engine if not already loaded

        if self.engine is None:
            raise RuntimeError("InferenceEngine is not loaded")

        # Convert PIL Image to numpy array (BGR format for OpenCV/InferenceEngine)
        if isinstance(image, PILImage.Image):
            image_array = np.array(image)
            # Convert RGB to BGR (OpenCV standard expected by InferenceEngine)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = image_array[:, :, ::-1]  # RGB -> BGR
        else:
            image_array = image

        # Use InferenceEngine's proven predict_array method
        print(f"Running inference on image with shape: {image_array.shape}")
        # BUG FIX: return_preview=False ensures we get ORIGINAL coordinates, not resized 640x640 preview coords
        result = self.engine.predict_array(image_array, return_preview=False)

        if result is None:
            print("InferenceEngine returned None - using empty predictions")
            return [], []

        # InferenceEngine returns string format:
        # {
        #   "polygons": "x1 y1 x2 y2 ... | x1 y1 ...",
        #   "confidences": [0.95, 0.87, ...],
        #   "texts": [...],
        #   "meta": {...}
        # }

        # Convert string format to array format for backward compatibility
        boxes, scores = self._parse_inference_result(result)

        print(f"Inference complete: {len(boxes)} polygons detected")
        return boxes, scores

    @staticmethod
    def _parse_inference_result(result: dict) -> tuple:
        """Parse InferenceEngine result into box arrays and scores.

        Converts from string format ("x1 y1 x2 y2|...") to array format
        ([[x,y], [x,y], ...]) for backward compatibility with existing endpoint.

        Args:
            result: Dictionary from InferenceEngine with 'polygons' and 'confidences' keys

        Returns:
            tuple: (boxes, scores) where boxes is list of np.ndarray, scores is list of float
        """
        import numpy as np

        polygons_str = result.get("polygons", "")
        confidences = result.get("confidences", [])

        boxes = []
        scores = []

        if not polygons_str:
            return boxes, scores

        # Parse polygons: space-separated coordinates, regions separated by "|"
        polygon_groups = polygons_str.split("|")

        for idx, polygon_str in enumerate(polygon_groups):
            coords = polygon_str.strip().split()
            if len(coords) < 6:  # Need at least 3 points (x1 y1 x2 y2 x3 y3)
                continue

            try:
                # Convert to list of [x, y] pairs
                coord_floats = [float(c) for c in coords]
                polygon = np.array(
                    [[coord_floats[i], coord_floats[i + 1]] for i in range(0, len(coord_floats), 2)],
                    dtype=np.float32
                )

                # Get corresponding confidence
                confidence = confidences[idx] if idx < len(confidences) else 0.0

                boxes.append(polygon)
                scores.append(confidence)

            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse polygon coordinates: {polygon_str} - {e}")
                continue

        return boxes, scores

# Singleton instance
bridge: OCRBridge | None = None

def get_default_checkpoint() -> str:
    """Auto-detect latest checkpoint from outputs/experiments/train/ocr.

    Returns:
        Path to the most recently modified checkpoint file.

    Raises:
        ValueError: If no checkpoints found in the directory.
    """
    ckpt_dir = Path("outputs/experiments/train/ocr")
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

    checkpoints = list(ckpt_dir.rglob("*.ckpt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {ckpt_dir}")

    # Sort by modification time, return latest
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Auto-detected latest checkpoint: {latest}")
    return str(latest)

def get_checkpoint_path() -> str:
    """Get checkpoint path from environment variable or auto-detect.

    Priority:
    1. OCR_CHECKPOINT_PATH environment variable (if set)
    2. Auto-detect latest checkpoint from outputs/experiments/train/ocr

    Returns:
        Absolute path to checkpoint file.
    """
    ckpt_path = os.getenv("OCR_CHECKPOINT_PATH")
    if ckpt_path:
        print(f"Using checkpoint from OCR_CHECKPOINT_PATH: {ckpt_path}")
        # Resolve to absolute path
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(ckpt_path)
        return ckpt_path
    else:
        return get_default_checkpoint()

def get_bridge():
    global bridge
    if bridge is None:
        ckpt_path = get_checkpoint_path()
        bridge = OCRBridge(ckpt_path)
    return bridge

@router.on_event("startup")
async def startup_event():
    """Initialize OCR Bridge on startup (lazy loading - model loads on first request)."""
    try:
        get_bridge()  # Initialize bridge but don't load model yet
        print("OCR Bridge initialized successfully (model will load on first inference request)")
    except Exception as e:
        print(f"Warning: OCR Bridge initialization failed: {e}")
        print("OCR endpoints will be unavailable until checkpoint is configured.")

@router.get("/checkpoints")
def list_checkpoints():
    """List available checkpoints from outputs/experiments/train/ocr."""
    from pathlib import Path

    checkpoints_dir = Path("outputs/experiments/train/ocr")
    if not checkpoints_dir.exists():
        return {"checkpoints": []}

    checkpoints = []
    for ckpt_path in sorted(checkpoints_dir.rglob("*.ckpt")):
        try:
            stat = ckpt_path.stat()
            checkpoints.append({
                "path": str(ckpt_path),
                "name": ckpt_path.stem,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime
            })
        except Exception:
            continue

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x["modified"], reverse=True)
    return {"checkpoints": checkpoints}

@router.get("/health")
def health_check():
    """Health check endpoint - doesn't force model loading."""
    b = get_bridge()
    return {
        "status": "ok",
        "model_loaded": b.engine is not None,  # Check if loaded, don't force load
        "checkpoint_path": b.checkpoint_path,
        "device": b.device if b else "N/A"
    }

@router.post("/predict")
async def predict(file: UploadFile = File(...), checkpoint_path: str = None):
    global bridge

    # If checkpoint_path is provided and different from current, reload the model
    if checkpoint_path and bridge and bridge.checkpoint_path != checkpoint_path:
        print(f"Reloading model with checkpoint: {checkpoint_path}")
        bridge = OCRBridge(checkpoint_path)

    b = get_bridge()
    # Don't check b.model here - lazy loading means it's None until first use
    # The b.predict() method will handle lazy loading and error reporting

    try:
        contents = await file.read()
        from PIL import Image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        boxes, scores = b.predict(image)

        # Serialize response
        predictions = []

        # We want to format the output as per Data Contracts
        # But for now, let's match the simple list of points

        # Convert to string format if required by contract "x1,y1 x2,y2 ..." or keep JSON array?
        # The contract in apps/ocr-inference-console/docs/data_contracts.md says:
        # "polygons": "x1,y1 x2,y2 ... | x1,y1 ..." (String representation)
        # Wait, the contract says "polygons": str (Space-separated coordinates, regions separated by "|")
        # BUT existing implementation in api.py returned JSON array "points": [[x,y]...].
        # I should output BOTH or stick to the contract.
        # The contract I wrote says "polygons": str.
        # Let's check the contract AGAIN.
        # "polygons": str, # Space-separated...

        # Return structured JSON data (array format) for frontend compatibility
        # Frontend TypeScript interface expects:
        # { "filename": str, "predictions": [{ "points": [[x,y]...], "confidence": float }] }

        res_predictions = []
        for box, score in zip(boxes, scores):
            res_predictions.append({
                "points": box.tolist(),
                "confidence": float(score)
            })

        return {
            "filename": file.filename,
            "predictions": res_predictions
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
