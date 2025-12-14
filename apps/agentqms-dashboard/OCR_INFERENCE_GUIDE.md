# OCR Inference Console - Google AI Studio Integration Guide

## Quick Overview

**OCR Sample Dataset Ready for Testing**

- ✅ 3 high-quality synthetic images (receipt + documents)
- ✅ COCO format annotations with text bounding boxes
- ✅ Configuration files for inference setup
- ✅ **Total size: 44 KB** (lightweight, portable)

**Location**: `apps/agentqms-dashboard/ocr_sample_data/`

---

## Dataset Structure

```
ocr_sample_data/
├── images/                          # Sample images for inference
│   ├── sample_001.jpg (15 KB)       # Synthetic receipt
│   ├── sample_002.jpg (12 KB)       # Document 1
│   └── sample_003.jpg (12 KB)       # Document 2
├── annotations.json (2 KB)          # COCO format ground truth
├── config.yaml                      # Inference parameters
├── requirements.txt                 # Python dependencies
└── README.md                        # Detailed usage guide
```

---

## What's Included

### 1. Sample Images

#### Image 1: Synthetic Receipt (15 KB)
- **Size**: 400x600 pixels
- **Content**: Multi-item receipt with prices and total
- **Text Regions**: 2 (header + total)
- **Use Case**: Testing complex layouts, price extraction

#### Image 2-3: Documents (12 KB each)
- **Size**: 400x300 pixels
- **Content**: Simple document with title and body text
- **Text Regions**: 2 each (title + content)
- **Use Case**: Testing simple layouts, document classification

### 2. Annotations (COCO Format)

5 total text regions with:
- Bounding boxes `[x, y, width, height]`
- Ground truth text content
- Language metadata
- Area calculations

**Sample annotation:**
```json
{
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "bbox": [80, 20, 240, 30],
    "area": 7200,
    "text": "SAMPLE RECEIPT",
    "language": "en"
}
```

### 3. Configuration

**config.yaml** includes:
- Model recommendations (CRAFT, TPS-ResNet)
- Input size and preprocessing parameters
- Confidence thresholds
- Post-processing settings

---

## Using in Google AI Studio (Colab)

### Step 1: Download Dataset

```python
# Option A: Clone from GitHub
!git clone https://github.com/your-username/agentqms-dashboard.git
%cd agentqms-dashboard/apps/agentqms-dashboard/ocr_sample_data

# Option B: Download directly
!wget -q https://raw.githubusercontent.com/your-repo/ocr_sample_data.zip
!unzip -q ocr_sample_data.zip

# Verify
!ls -lah
```

### Step 2: Load and Inspect Data

```python
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Load annotations
with open("annotations.json") as f:
    coco = json.load(f)

print(f"✅ Loaded {len(coco['images'])} images")
print(f"✅ Loaded {len(coco['annotations'])} text regions")

# Visualize first image
img_info = coco['images'][0]
img = Image.open(f"images/{img_info['file_name']}")

fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(img)

# Draw annotation boxes
for ann in coco['annotations']:
    if ann['image_id'] == img_info['id']:
        x, y, w, h = ann['bbox']
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y-5, ann['text'], color='red', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))

plt.title(f"Image: {img_info['file_name']}")
plt.axis('off')
plt.tight_layout()
plt.show()
```

### Step 3: Run Inference with Your Model

#### Option A: Using CRAFT + TPS-ResNet

```python
import torch
from torchvision import transforms
import cv2
import numpy as np

# Load model (example with CRAFT)
def load_craft_model():
    # Replace with your actual model loading code
    from craft_pytorch.craft import CRAFT
    model = CRAFT()
    # Load weights
    model.eval()
    return model

model = load_craft_model()

# Prepare image
def preprocess_image(image_path, input_size=320):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std

    # To tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, img

# Run inference
img_tensor, img_original = preprocess_image("images/sample_001.jpg")

with torch.no_grad():
    output = model(img_tensor)

# Post-process results
predictions = []
for region in output:
    if region['confidence'] > 0.5:  # Confidence threshold
        predictions.append(region)

print(f"Detected {len(predictions)} text regions")
```

#### Option B: Using Gemini Vision API

```python
import anthropic
import base64

client = anthropic.Anthropic()

def ocr_with_gemini(image_path):
    """Extract text using Gemini Vision."""

    # Read image
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Call Gemini
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": """Extract all visible text from this image. For each text region, provide:
1. The text content
2. Approximate location (x, y coordinates as percentage of image dimensions)
3. Confidence score (0-1)

Format as JSON array with fields: text, x, y, confidence"""
                    }
                ]
            }
        ]
    )

    return response.content[0].text

# Process all samples
results = []
for i in range(1, 4):
    img_file = f"images/sample_{i:03d}.jpg"
    result = ocr_with_gemini(img_file)
    results.append({
        "image": img_file,
        "ocr_result": result
    })
    print(f"✅ Processed {img_file}")

# Display results
import json
for result in results:
    print(f"\n{result['image']}:")
    print(result['ocr_result'])
```

### Step 4: Evaluate Against Ground Truth

```python
def compute_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Compute precision, recall, F1 against ground truth."""

    tp = 0  # true positives
    fp = 0  # false positives
    fn = 0  # false negatives

    # Track matched annotations
    matched_gt = set()

    # For each prediction, find best matching GT
    for pred in predictions:
        best_iou = 0
        best_idx = -1

        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue

            # Compute IoU
            iou = compute_iou(pred['bbox'], gt['bbox'])

            # Check text match
            text_match = (
                pred.get('text', '').lower() ==
                gt.get('text', '').lower()
            )

            if iou > best_iou and iou >= iou_threshold and text_match:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    # Remaining GT items are false negatives
    fn = len(ground_truth) - len(matched_gt)

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union for bounding boxes."""
    x1_min, y1_min, w1, h1 = bbox1
    x2_min, y2_min, w2, h2 = bbox2

    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max < xi_min or yi_max < yi_min:
        return 0.0

    intersection = (xi_max - xi_min) * (yi_max - yi_min)

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# Evaluate on all images
metrics_all = []
for image_id in range(1, 4):
    # Get ground truth for this image
    gt_anns = [
        ann for ann in coco['annotations']
        if ann['image_id'] == image_id
    ]

    # Get predictions for this image (from your model)
    # predictions = model.predict(f"images/sample_{image_id:03d}.jpg")
    predictions = []  # Replace with actual predictions

    # Compute metrics
    metrics = compute_metrics(predictions, gt_anns)
    metrics['image_id'] = image_id
    metrics_all.append(metrics)

    print(f"\nImage {image_id}:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")

# Overall metrics
overall_tp = sum(m['tp'] for m in metrics_all)
overall_fp = sum(m['fp'] for m in metrics_all)
overall_fn = sum(m['fn'] for m in metrics_all)

overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

print(f"\n{'='*50}")
print(f"Overall Metrics:")
print(f"  Precision: {overall_precision:.3f}")
print(f"  Recall: {overall_recall:.3f}")
print(f"  F1: {overall_f1:.3f}")
```

---

## Exporting Results

### Option 1: Save as JSON

```python
# Save predictions
predictions_output = {
    "dataset": "ocr_sample_data",
    "model": "your_model_name",
    "metrics": {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1
    },
    "predictions": metrics_all
}

with open("inference_results.json", "w") as f:
    json.dump(predictions_output, f, indent=2)

print("✅ Results saved to inference_results.json")
```

### Option 2: Visualize Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, img_id in enumerate(range(1, 4)):
    img_info = coco['images'][idx]
    img = Image.open(f"images/{img_info['file_name']}")

    axes[idx].imshow(img)

    # Draw ground truth boxes
    for ann in coco['annotations']:
        if ann['image_id'] == img_id:
            x, y, w, h = ann['bbox']
            rect = plt.Rectangle((x, y), w, h, fill=False,
                                edgecolor='green', linewidth=2,
                                label='GT')
            axes[idx].add_patch(rect)

    # Draw prediction boxes (if available)
    # for pred in predictions[img_id]:
    #     x, y, w, h = pred['bbox']
    #     rect = plt.Rectangle((x, y), w, h, fill=False,
    #                         edgecolor='red', linewidth=2,
    #                         linestyle='--', label='Pred')
    #     axes[idx].add_patch(rect)

    axes[idx].set_title(img_info['file_name'])
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("ocr_results_visualization.png", dpi=150, bbox_inches='tight')
print("✅ Visualization saved to ocr_results_visualization.png")
plt.show()
```

---

## Integration with Dashboard Console

### Option A: Direct Integration

The `ocr_sample_data/` folder can be loaded directly in the OCR Inference Console:

1. **In Dashboard**: Settings → Data Path → Select `ocr_sample_data/`
2. **Load Images**: Console auto-discovers images in `images/` folder
3. **Load Annotations**: Console reads `annotations.json` automatically
4. **Run Inference**: Execute inference pipeline on loaded data
5. **View Results**: Compare predictions vs ground truth

### Option B: API Integration

```python
import requests

# Upload dataset to dashboard
with open("ocr_sample_data.zip", "rb") as f:
    files = {"dataset": f}
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/upload-dataset",
        files=files
    )

print(f"Upload status: {response.status_code}")
print(f"Dataset ID: {response.json()['dataset_id']}")
```

---

## Key Features

✅ **Lightweight**: 44 KB total size
✅ **Portable**: Works offline, no cloud dependencies
✅ **COCO Format**: Standard annotation format
✅ **Comprehensive**: Receipt + document images
✅ **Evaluation-Ready**: Ground truth annotations included
✅ **Model-Agnostic**: Works with any OCR model
✅ **Scalable**: Easy to extend with more samples

---

## Performance Expectations

On typical hardware:

| Component | Inference Time |
|-----------|-----------------|
| CRAFT (detection) | 15-50ms |
| TPS-ResNet (recognition) | 30-100ms |
| End-to-end OCR | 50-150ms |
| Gemini Vision API | 2-5 seconds |

---

## Troubleshooting

### Images not loading?
```python
from PIL import Image
img = Image.open("images/sample_001.jpg")
print(f"Image size: {img.size}, Mode: {img.mode}")
```

### Annotations format issue?
```python
import json
with open("annotations.json") as f:
    data = json.load(f)
print(f"Images: {len(data['images'])}")
print(f"Annotations: {len(data['annotations'])}")
print(f"Categories: {len(data['categories'])}")
```

### Missing dependencies?
```bash
pip install -r requirements.txt
```

---

## Next Steps

1. **Clone/Download** the dataset
2. **Load** images and annotations
3. **Run inference** with your model
4. **Evaluate** against ground truth
5. **Export** results
6. **Integrate** with Dashboard console (optional)

---

**Dataset Created**: 2025-12-11
**Total Size**: 44 KB
**Use Case**: OCR model testing and evaluation
**Compatibility**: Python 3.9+, Any OCR framework
