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
    """OCR inference bridge with lazy loading.

    Model loads only on first inference request, not at initialization.
    This allows the server to start instantly without blocking on model loading.
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None  # Lazy loading: don't load until first use
        print(f"OCR Bridge initialized with checkpoint: {checkpoint_path} (model will load on first request)")

    def _ensure_model_loaded(self) -> None:
        """Lazy load model on first use."""
        if self.model is None:
            self.model = self._load_model()
            if self.model:
                self.model.eval()
                self.model.to(self.device)

    def _load_model(self):
        """Lazy load model - imports heavy dependencies only when called."""
        # Lazy imports - only load when model is actually needed
        import torch
        from ocr.lightning_modules.ocr_pl import OCRPLModule

        if not os.path.exists(self.checkpoint_path):
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}")
            return None

        print(f"Loading checkpoint from {self.checkpoint_path}...")
        try:
            # 1. Resolve Config Path
            # Structure: outputs/.../checkpoints/epoch-14.ckpt -> outputs/.../config.yaml
            ckpt_path = Path(self.checkpoint_path)
            # Go up two levels: checkpoints/epoch... -> checkpoints/ -> experiment_root/
            exp_root = ckpt_path.parent.parent
            config_path = exp_root / "config.yaml"

            if not config_path.exists():
                print(f"Config not found at {config_path}, trying .hydra subdirectory", flush=True)
                # Fallback: try .hydra subdirectory (legacy structure)
                config_path = exp_root / ".hydra" / "config.yaml"
                if not config_path.exists():
                    print(f"Config also not found at {config_path}", flush=True)
                    return None

            print(f"Using config: {config_path}", flush=True)

            from omegaconf import OmegaConf
            import hydra
            from ocr.models.architecture import OCRModel

            cfg = OmegaConf.load(config_path)

            # Remove optimizer/scheduler config to prevent recursive instantiation error
            if hasattr(cfg.model, "optimizer"):
                del cfg.model.optimizer
            if hasattr(cfg.model, "scheduler"):
                del cfg.model.scheduler

            # 2. Instantiate Model Architecture
            # Note: OCRModel.__init__ takes (cfg) as a single argument.
            # hydra.utils.instantiate tries to unpack cfg.model as kwargs, which fails.
            print("Instantiating model architecture directly...", flush=True)
            model = OCRModel(cfg.model)

            # 3. Create Dummy Dataset (required by OCRPLModule __init__)
            # It checks 'val' and 'test' keys
            dummy_dataset = {}

            # 4. Load Checkpoint with injected dependencies
            print("Loading state dict...", flush=True)
            # We pass required init args as kwargs
            module = OCRPLModule.load_from_checkpoint(
                self.checkpoint_path,
                map_location="cpu",
                # Args for __init__
                model=model,
                dataset=dummy_dataset,
                config=cfg,
                metric_cfg=None # Optional
            )

            return module

        except Exception as e:
            print(f"Error loading model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            try:
                with open("load_error.txt", "w") as f:
                    f.write(str(e) + "\n")
                    f.write(traceback.format_exc())
            except:
                pass
            return None

    def predict(self, image: Image.Image) -> tuple[List[np.ndarray], List[float]]:
        """Run OCR inference on image.

        Model loads automatically on first call (lazy loading).
        """
        # Lazy imports
        import torch
        import numpy as np
        from PIL import Image
        import torchvision.transforms.functional as F
        from ocr.utils.geometry_utils import calculate_inverse_transform

        self._ensure_model_loaded()  # Load model if not already loaded

        if not self.model:
            # Check if there's a load error file with details
            error_details = "Unknown error"
            try:
                if os.path.exists("load_error.txt"):
                    with open("load_error.txt", "r") as f:
                        error_details = f.read()
            except:
                pass
            raise RuntimeError(
                f"Model failed to load from checkpoint: {self.checkpoint_path}\n"
                f"Error details: {error_details}"
            )

        # Store original size for inverse_matrix calculation
        original_w, original_h = image.size

        # Preprocessing: LongestMaxSize + PadIfNeeded (matching training pipeline)
        target_size = 640

        # Step 1: LongestMaxSize - scale longest side to target_size, preserving aspect ratio
        max_side = max(original_w, original_h)
        if max_side > 0:
            scale = target_size / max_side
            scaled_w = int(round(original_w * scale))
            scaled_h = int(round(original_h * scale))
        else:
            scaled_w, scaled_h = original_w, original_h

        # Resize preserving aspect ratio
        if scaled_w != original_w or scaled_h != original_h:
            image = image.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)

        # Step 2: PadIfNeeded - pad to target_size x target_size with top_left position
        pad_w = target_size - scaled_w
        pad_h = target_size - scaled_h

        if pad_w > 0 or pad_h > 0:
            # Create a new image with black padding (top_left position means padding at bottom/right)
            padded_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            padded_image.paste(image, (0, 0))  # Paste at top-left corner
            image = padded_image

        # Step 3: Convert to tensor and normalize
        img_tensor = F.to_tensor(image)
        img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Calculate inverse_matrix for coordinate transformation
        # This maps coordinates from the 640x640 padded space back to original image space
        inverse_matrix = calculate_inverse_transform(
            original_size=(original_w, original_h),
            transformed_size=(target_size, target_size),
            padding_position="top_left"
        )

        batch = {
            "images": img_tensor,
            "image_filename": ["uploaded_image"],
            "inverse_matrix": [inverse_matrix],  # Must be a list for batch processing
        }

        with torch.no_grad():
            preds = self.model(batch)
            print(f"[DEBUG] Model predictions type: {type(preds)}", flush=True)
            print(f"[DEBUG] Model predictions keys: {preds.keys() if isinstance(preds, dict) else 'not a dict'}", flush=True)

            # Access the inner model to get polygons
            # Note: OCRPLModule has model, which wraps the actual architecture
            boxes_batch, scores_batch = self.model.model.get_polygons_from_maps(batch, preds)

            print(f"[DEBUG] boxes_batch length: {len(boxes_batch)}", flush=True)
            print(f"[DEBUG] boxes_batch[0] length: {len(boxes_batch[0]) if len(boxes_batch) > 0 else 0}", flush=True)
            print(f"[DEBUG] scores_batch length: {len(scores_batch)}", flush=True)

            return boxes_batch[0], scores_batch[0]

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
        "model_loaded": b.model is not None,  # Check if loaded, don't force load
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

        # However, for a JSON API, structured data is usually better.
        # But if the InferenceEngine contract (which I copied from docs/pipeline/data_contracts.md) specifies a string,
        # maybe I should follow it.
        # Let's check apps/backend/services/ocr_bridge.py intent.

        # Actually, let's return JSON structured data, it's more flexible.
        # And maybe update the contract documentation to reflect JSON structured data if "polygons" string format was legacy or specific to CSV submission.
        # For now, I'll return structured data (list of lists) which is easier for frontend.
        # I will return:
        # "predictions": [ { "points": [[x,y]...], "score": float } ]

        # Convert to string format matching Inference Studio
        # Format: "x1 y1 x2 y2 x3 y3|x1 y1 x2 y2" (space-separated coords, | separates polygons)
        polygon_strings = []
        confidences = []

        for box, score in zip(boxes, scores):
            # box is a list of [x, y] pairs
            # Flatten to "x1 y1 x2 y2 x3 y3 ..."
            coords = []
            for point in box:
                coords.append(str(int(point[0])))
                coords.append(str(int(point[1])))
            polygon_strings.append(" ".join(coords))
            confidences.append(float(score))

        # Join polygons with | separator
        polygons_str = "|".join(polygon_strings)

        return {
            "polygons": polygons_str,
            "texts": [],  # OCR doesn't return text yet
            "confidences": confidences
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
