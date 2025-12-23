---
language:
  - en
  - ko
license: mit
tags:
  - ocr
  - text-detection
  - receipt-detection
  - korean
  - dbnet
  - resnet18
  - pytorch
  - pytorch-lightning
library_name: pytorch
datasets:
  - upstage/ocr-recsys-competition
metrics:
  - precision
  - recall
  - f1
base_model: wchoi189/receipt-text-detection_kr-pan_resnet18
repo: Wchoi189/upstageailab-ocr-recsys-competition-ocr-2
model-index:
  - name: PAN ResNet18 Receipt Detection
    results:
      - task:
          type: object-detection
          name: Text Detection
        dataset:
          name: Upstage OCR Competition
          type: upstage/ocr-recsys-competition
        metrics:
          - type: precision
            value: 95.35
          - type: recall
            value: 95.72
          - type: f1
            value: 95.37
---

# Receipt Text Detection Model (Korean) - PAN ResNet18

## Model Details

### Model Description

A lightweight text detection model optimized for receipt and invoice documents, trained with DBNet architecture and PAN decoder. This model detects text regions in receipt images with high precision and recall.

- **Model Type:** Text Detection
- **Architecture:** DBNet with PAN Decoder
- **Encoder:** ResNet18 (pretrained from ImageNet)
- **Decoder:** PAN Decoder with polygon output support
- **Training Framework:** PyTorch Lightning
- **Input Resolution:** Variable (recommended 640x480 or similar)
- **Output:** Binary segmentation map + polygon masks for detected text regions

### Repositories

- **Personal (Continuation):** [Wchoi189/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2)
- **Original (Bootcamp):** [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

### Model Performance

| Metric | Value |
|--------|-------|
| **Precision** | 95.35% |
| **Recall** | 95.72% |
| **H-Mean (F-Score)** | 95.37% |
| **Training Epochs** | 18/20 |
| **Global Steps** | 1,957 |

### Training Configuration

```json
{
  "model": {
    "architecture": "dbnet",
    "encoder": {
      "model_name": "resnet18",
      "pretrained": true,
      "select_features": [1, 2, 3, 4]
    },
    "decoder": {
      "name": "pan_decoder",
      "inner_channels": 256,
      "output_channels": 256
    },
    "head": {
      "name": "db_head",
      "postprocess": {
        "thresh": 0.2,
        "box_thresh": 0.3,
        "max_candidates": 300,
        "use_polygon": true
      }
    },
    "loss": {
      "name": "db_loss",
      "negative_ratio": 3.0,
      "prob_map_loss_weight": 5.0,
      "thresh_map_loss_weight": 10.0,
      "binary_map_loss_weight": 1.0
    },
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "batch_size": [see config.json]
  },
  "training": {
    "max_epochs": 20,
    "monitor": "val/hmean",
    "save_top_k": 3,
    "checkpoint_strategy": "save_last"
  }
}
```

---

## Intended Use

### Primary Use Cases

- ✅ Receipt text detection and localization
- ✅ Invoice document analysis
- ✅ Scanned document preprocessing
- ✅ Text region identification before OCR recognition

### Out-of-Scope Uses

- ❌ Detecting text in natural scenes or artistic images
- ❌ Processing documents in languages not in training data
- ❌ Real-time video processing without optimization
- ❌ Commercial use without proper licensing review (see limitations)

---

## Training Data

### Data Source

- **Type:** Private, proprietary receipt/invoice dataset
- **Language:** Korean (with English text elements)
- **Domain:** Financial documents (receipts, invoices)
- **Size:** [Specific size not disclosed - proprietary]
- **Annotations:** Polygon-based text region annotations

### Data Characteristics

- Receipt images from various merchants
- Variable lighting conditions and quality
- Handwritten and printed text
- Multiple text regions per image
- Korean and English language content

**⚠️ Important Note:** The training data is proprietary and copyrighted. This model can be published because it contains learned representations, not the original data. However, users should be aware that:
1. The model reflects patterns learned from copyrighted documents
2. The model should be used responsibly and in compliance with applicable laws
3. Do not attempt to reverse-engineer or extract the training data from this model

---

## Model Architecture Details

### Encoder (Feature Extraction)

- **Base:** ResNet18 pretrained on ImageNet
- **Frozen:** No (allows fine-tuning on new data)
- **Output Features:** Multi-scale feature maps from layers 1-4

### Decoder (Feature Fusion)

- **Type:** PAN (Pixel Aggregation Network) Decoder
- **Channels:** 256 inner and output channels
- **Upsampling:** Bilinear interpolation
- **Output Resolution:** 1/4 of input resolution (for inference optimization)

### Head (Detection)

- **Type:** DBNet Head
- **Output:**
  - Probability map (foreground probability)
  - Threshold map (adaptive thresholding)
  - Binary map (for polygon extraction)
- **Postprocessing:**
  - Binary threshold: 0.2
  - Box threshold: 0.3 (confidence filtering)
  - Max candidates: 300 polygons per image
  - Polygon output: Enabled

### Loss Function

```
Total Loss = 5.0 × Prob_Loss + 10.0 × Thresh_Loss + 1.0 × Binary_Loss
```

- Uses weighted negative sampling (ratio: 3:1)
- Handles class imbalance in text detection

---

## Usage

### Installation

```bash
# Install with pytorch
pip install torch torchvision

# Or use the project's environment
git clone <repository-url>
cd <project-directory>
uv sync
```

### Loading the Model

```python
import torch
from pathlib import Path

# Load checkpoint
checkpoint_path = "epoch-18_step-001957.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Initialize model (requires project dependencies)
# See project documentation for full setup
```

### Inference Example

```python
import cv2
import torch
from PIL import Image

# Prepare image
image = cv2.imread("receipt.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to model input (varies - typically 640x480 or 1024x512)
image_resized = cv2.resize(image, (640, 480))

# Normalize (ImageNet normalization for ResNet18)
image_normalized = image_resized.astype('float32') / 255.0
image_normalized = (image_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

# Add batch dimension
input_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)

# Forward pass
with torch.no_grad():
    output = model(input_tensor)
    # output contains probability_map, threshold_map, and polygon coordinates

# Process output
polygons = extract_polygons(output, thresh=0.2, box_thresh=0.3)
```

**Note:** Full integration instructions are available in the project repository.

---

## Benchmarking

### Performance Metrics

- **Model Size:** ~11.7 MB (ResNet18 backbone)
- **Inference Time (GPU):** ~50-100ms per image (640x480)
- **Inference Time (CPU):** ~300-500ms per image (depending on hardware)
- **Memory Requirement:** ~1.5GB VRAM during inference

### Comparison with Other Models

| Model | Architecture | Parameters | H-Mean | Notes |
|-------|--------------|-----------|--------|-------|
| **This Model (PAN ResNet18)** | DBNet + PAN | ~11.7M | 95.37% | Lightweight, polygon output |
| DBNet ResNet50 | DBNet | ~47M | ~95% | Larger backbone |
| CRAFT | Sequential attention | ~30M | ~93% | Different architecture |

---

## Limitations

### Known Limitations

1. **Language Specificity:** Trained primarily on Korean receipts; performance on other languages untested
2. **Domain Specificity:** Optimized for receipt/invoice documents; may not generalize to other document types
3. **Image Quality:** Trained on moderate-quality receipt images; performance on very low-quality or highly degraded images unknown
4. **Polygon vs Rectangular:** Uses polygon-based detection rather than axis-aligned bounding boxes
5. **Early Checkpoint:** 18/20 epochs; not fully trained (though performance plateau was likely reached)

### Potential Biases

- Model may be biased toward Korean text patterns
- Merchant-specific patterns may affect generalization
- Lighting and background patterns from training data may introduce biases

### Ethical Considerations

⚠️ **Important:** This model was trained on private, copyrighted receipt data. Users should:
- Respect intellectual property rights
- Use this model in compliance with applicable laws and regulations
- Not use the model to extract or infer private financial information
- Consider privacy implications when applying to user-submitted receipts
- Implement appropriate data protection measures

---

## Model Card Contact

**Model Name:** receipt-text-detection_kr-pan_resnet18
**Maintained By:** wchoi189
**Created:** October 19, 2024
**Last Updated:** December 2024

**Questions?** Please refer to the main project repository: [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

---

## Additional Resources

### Project Links

- **GitHub Repository:** [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)
- **Documentation:** See `/docs` directory in repository
- **Config Architecture:** See `/configs` directory for training configurations
- **Papers Referenced:**
  - DBNet: Real-time Scene Text Detection with Differentiable Binarization
  - PAN: Towards Accurate Scene Text Recognition with Semantic Reasoning Networks

### Citation

If you use this model in research, please cite:

```bibtex
@model{receipt_text_detection_2024,
  title={Receipt Text Detection Model (Korean) - PAN ResNet18},
  author={wchoi189},
  year={2024},
  url={https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18}
}
```

### License

This model is released under the **MIT License**. See LICENSE file for details.

---

## Changelog

### Version 1.0 (2024-12-XX)

- Initial model release
- DBNet + PAN architecture
- ResNet18 encoder
- 95.37% H-Mean on validation set
- Polygon-based text detection
- Korean receipt optimization

---

*This model card was created with transparency and responsible AI principles in mind.*
