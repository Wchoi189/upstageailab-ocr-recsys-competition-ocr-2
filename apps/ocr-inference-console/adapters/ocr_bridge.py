import torch
import numpy as np
import warnings
from PIL import Image
import torchvision.transforms.functional as F
from ocr.lightning_modules.ocr_pl import OCRPLModule
from ocr.utils.path_utils import setup_project_paths

# Setup paths to ensure ocr package is resolvable
setup_project_paths()

class OCRBridge:
    def __init__(self, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

    def _load_model(self) -> OCRPLModule:
        print(f"Loading checkpoint from {self.checkpoint_path}...")

        # Manual loading logic similar to runners/predict.py for compatibility
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        except (RuntimeError, TypeError, ValueError):
             # Fallback
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # We need to instantiate the model first.
        # Since OCRPLModule takes (model, dataset, config), and we don't have the hydra config handy easily without hydra,
        # we might rely on load_from_checkpoint if the checkpoint saves_hyperparameters.
        # But runners/predict.py does it manually.

        # simplified approach: define a minimal config/mock if needed or try standard PL load first
        # If standard PL load fails, we might need a more complex setup.
        # For now, let's try the standard PL load_from_checkpoint which is cleaner
        # and fallback to manual if we encounter the specific issues mentioned in predict.py code comments.

        # However, checking runners/predict.py: it uses get_pl_modules_by_cfg(config).
        # This implies we need the config. The config is usually in the checkpoint or we need to construct it.

        # Let's try loading from checkpoint directly, assuming strict=False to avoid minor mismatches
        try:
            model = OCRPLModule.load_from_checkpoint(self.checkpoint_path, map_location="cpu")
            return model
        except Exception as e:
            print(f"Standard load failed: {e}. Trying fallback... (Note: This is a stub implementation, real implementation needs the Config object)")
            raise e

    def predict(self, image: Image.Image):
        # Preprocess
        # Resize to multiple of 32 (DBNet requirement usually) or use dataset transform
        # For simplicity, we'll basic transform here or assume model handles it?
        # OCRPLModule expects a batch dict.

        w, h = image.size
        # Resize logic (simple for now, match val_dataset typically)
        # In a real app we should use the same transform as validation
        new_w = ((w // 32) * 32)
        new_h = ((h // 32) * 32)
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h))

        img_tensor = F.to_tensor(image) # C, H, W, 0-1
        img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.unsqueeze(0).to(self.device) # B, C, H, W

        batch = {
            "images": img_tensor,
            "image_filename": ["uploaded_image"],
            # polygon/maps are not needed for inference mode usually,
            # but OCRPLModule.predict_step checks for them to decide mode
        }

        with torch.no_grad():
            # OCRPLModule.predict_step signature: batch -> pred
            # But we can also call model(batch) directly?
            # predict_step calls model(return_loss=False) then get_polygons_from_maps

            # call predict_step logic manually or use the method
            # We need to ensure batch has what predict_step needs

            # The predict_step in OCRPLModule:
            # pred = self.model(return_loss=False, **batch)
            # boxes_batch, scores_batch = self.model.get_polygons_from_maps(batch, pred)

            # So let's use the module's method if possible, but we need to bind it to the instance

            preds = self.model(return_loss=False, **batch)
            boxes_batch, scores_batch = self.model.model.get_polygons_from_maps(batch, preds)

            return boxes_batch[0], scores_batch[0]

