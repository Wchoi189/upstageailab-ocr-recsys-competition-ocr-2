# OCR Pipeline Reference

This document maps the core components of the OCR pipeline located in `ocr/` and explains the key entry points for inference.

## Key Modules

### Lightning Module (`ocr/lightning_modules/ocr_pl.py`)
The `OCRPLModule` is the central coordinator for training, validation, and inference.
- **Role**: Wraps the PyTorch model, handles data flow, metrics, and logging.
- **Key Methods**:
  - `predict_step(batch)`: Used for inference. Returns normalized polygon coordinates.
  - `load_from_checkpoint(path)`: Static method (inherited) to load model weights.

### Architecture (`ocr/models/architecture.py`)
- **Role**: Defines the DBNet (Differentiable Binarization) architecture.
- **Components**:
  - Backbone (ResNet variants)
  - Neck (FPN)
  - Head (DB Head)

### Runners (`runners/`)
CLI scripts that serve as the primary interface for the pipeline.
- `predict.py`: Runs inference on a dataset using a checkpoint.
- `train.py`: specific training runner.

## Entry Points

### CLI Prediction
To run predictions from the command line:

```bash
uv run python runners/predict.py \
    preset=example \
    checkpoint_path="path/to/checkpoint.ckpt"
```

### Python API (for Integration)
The inference console connects to the pipeline via the `OCRPLModule`.

```python
from ocr.lightning_modules.ocr_pl import OCRPLModule
import torch

# 1. Load Model
model = OCRPLModule.load_from_checkpoint("path/to/best.ckpt")
model.eval()
model.freeze()

# 2. Prepare Input (Tensor: [1, C, H, W])
# ... preprocessing steps ...

# 3. Inference
with torch.no_grad():
    predictions = model(input_tensor)
```

## Data Flow
1. **Input**: Images (JPG/PNG)
2. **Preprocessing**: Resize, Normalize (defined in `ocr/datasets/transforms`)
3. **Model**: Forward pass -> Probability Map + Threshold Map
4. **Postprocessing**: Polygon extraction (DB Postprocessor)
5. **Output**: List of polygons (x, y coordinates)
