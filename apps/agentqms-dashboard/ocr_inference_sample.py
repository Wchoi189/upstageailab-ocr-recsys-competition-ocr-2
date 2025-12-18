#!/usr/bin/env python3
"""
Create lightweight OCR inference sample dataset for Google AI Studio demo.

This script generates:
1. Sample images (synthetic receipts/documents)
2. Annotation data (COCO format with text regions)
3. Configuration file
4. README with inference instructions

Usage:
    python ocr_inference_sample.py [--output sample_data] [--count 3]

Output Structure:
    sample_data/
    ├── images/
    │   ├── sample_001.jpg
    │   ├── sample_002.jpg
    │   └── sample_003.jpg
    ├── annotations.json (COCO format)
    ├── config.yaml (dataset configuration)
    ├── README.md (inference guide)
    └── requirements.txt
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


def create_synthetic_receipt(width: int = 400, height: int = 600) -> Image.Image:
    """Create a synthetic receipt image for OCR testing."""
    if Image is None:
        raise ImportError("Pillow is required. Install with: pip install Pillow")

    # Create white image
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a nicer font, fallback to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw receipt content
    y_pos = 20

    # Header
    draw.text((width // 2 - 40, y_pos), "SAMPLE RECEIPT", fill="black", font=title_font)
    y_pos += 40

    # Store info
    draw.line([(20, y_pos), (width - 20, y_pos)], fill="black", width=1)
    y_pos += 10
    draw.text((30, y_pos), "Store: ABC Mart", fill="black", font=text_font)
    y_pos += 25
    draw.text((30, y_pos), "Address: 123 Main St", fill="black", font=small_font)
    y_pos += 20
    draw.text((30, y_pos), "Phone: (555) 123-4567", fill="black", font=small_font)
    y_pos += 30

    # Items
    draw.line([(20, y_pos), (width - 20, y_pos)], fill="black", width=1)
    y_pos += 10

    items = [
        ("Item 1 - Widget", "$10.99"),
        ("Item 2 - Gadget", "$25.50"),
        ("Item 3 - Component", "$5.00"),
        ("Item 4 - Part", "$12.75"),
    ]

    for item_name, item_price in items:
        # Draw item name and price
        draw.text((30, y_pos), item_name, fill="black", font=text_font)
        draw.text((width - 100, y_pos), item_price, fill="black", font=text_font)
        y_pos += 25

    # Totals
    y_pos += 10
    draw.line([(20, y_pos), (width - 20, y_pos)], fill="black", width=2)
    y_pos += 15
    draw.text((30, y_pos), "Subtotal:", fill="black", font=text_font)
    draw.text((width - 100, y_pos), "$54.24", fill="black", font=text_font)
    y_pos += 25

    draw.text((30, y_pos), "Tax (8%):", fill="black", font=text_font)
    draw.text((width - 100, y_pos), "$4.34", fill="black", font=text_font)
    y_pos += 25

    draw.text((30, y_pos), "TOTAL:", fill="black", font=text_font)
    draw.text((width - 100, y_pos), "$58.58", fill="black", font=text_font)
    y_pos += 30

    # Footer
    draw.line([(20, y_pos), (width - 20, y_pos)], fill="black", width=1)
    y_pos += 10
    draw.text((50, y_pos), "Thank you for your purchase!", fill="black", font=small_font)
    y_pos += 20
    draw.text((70, y_pos), "Date: 2025-12-11 14:30", fill="black", font=small_font)

    return img


def create_sample_image_with_text(width: int = 400, height: int = 300, text: str = "Sample Document") -> Image.Image:
    """Create a sample document image with text."""
    if Image is None:
        raise ImportError("Pillow is required. Install with: pip install Pillow")

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    # Draw title
    draw.text((50, 30), text, fill="black", font=title_font)

    # Draw body text
    body_lines = [
        "This is a sample document for OCR inference testing.",
        "It contains multiple lines of text.",
        "The system should detect and recognize all text regions.",
        "Date: 2025-12-11",
    ]

    y = 100
    for line in body_lines:
        draw.text((50, y), line, fill="black", font=body_font)
        y += 40

    # Add border
    draw.rectangle([(10, 10), (width - 10, height - 10)], outline="black", width=2)

    return img


def create_annotations(image_count: int = 3) -> dict[str, Any]:
    """Create COCO format annotations for sample images."""
    annotations = {
        "info": {
            "description": "OCR Inference Sample Dataset",
            "version": "1.0",
            "year": 2025,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [{"id": 1, "name": "MIT", "url": "https://opensource.org/licenses/MIT"}],
        "images": [],
        "annotations": [],
    }

    annotation_id = 1

    # Generate annotations for each image
    for img_idx in range(image_count):
        image_id = img_idx + 1
        filename = f"sample_{image_id:03d}.jpg"

        # Image metadata
        annotations["images"].append(
            {
                "id": image_id,
                "file_name": filename,
                "height": 600 if img_idx == 0 else 300,  # First is receipt (600px), others are documents
                "width": 400,
                "date_captured": datetime.now().isoformat(),
            }
        )

        # Text region annotations
        if img_idx == 0:  # Receipt image
            regions = [
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [80, 20, 240, 35],  # [x, y, width, height]
                    "area": 240 * 35,
                    "iscrowd": 0,
                    "text": "SAMPLE RECEIPT",
                    "language": "en",
                },
                {
                    "id": annotation_id + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [30, 60, 150, 20],
                    "area": 150 * 20,
                    "iscrowd": 0,
                    "text": "Store: ABC Mart",
                    "language": "en",
                },
                {
                    "id": annotation_id + 2,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [30, 240, 200, 25],
                    "area": 200 * 25,
                    "iscrowd": 0,
                    "text": "TOTAL: $58.58",
                    "language": "en",
                },
            ]
            annotation_id += 3
        else:  # Document images
            regions = [
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [50, 30, 300, 35],
                    "area": 300 * 35,
                    "iscrowd": 0,
                    "text": "Sample Document",
                    "language": "en",
                },
                {
                    "id": annotation_id + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [50, 100, 350, 120],
                    "area": 350 * 120,
                    "iscrowd": 0,
                    "text": "Sample document for OCR inference testing",
                    "language": "en",
                },
            ]
            annotation_id += 2

        annotations["annotations"].extend(regions)

    # Add categories
    annotations["categories"] = [
        {
            "id": 1,
            "name": "text_region",
            "supercategory": "text",
            "description": "Text region for OCR",
        },
    ]

    return annotations


def create_config() -> str:
    """Create dataset configuration YAML."""
    config = """# OCR Inference Sample Dataset Configuration

dataset:
  name: "OCR Inference Sample"
  version: "1.0"
  description: "Lightweight sample dataset for OCR inference testing in Google AI Studio"
  date_created: 2025-12-11

images:
  format: "JPEG"
  width: 400
  height_min: 300
  height_max: 600
  color_space: "RGB"
  samples: 3

annotations:
  format: "COCO"
  version: "1.0"
  license: "MIT"
  languages: ["en"]
  total_regions: 5

inference:
  # Model requirements for inference
  model_type: "CRAFT + TPS-ResNet"
  input_size: 320
  batch_size: 4

  # Preprocessing
  preprocessing:
    normalize: true
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    resize_method: "letterbox"

  # Post-processing
  postprocessing:
    confidence_threshold: 0.5
    text_score_threshold: 0.4
    nms_threshold: 0.4

evaluation:
  metrics: ["F1", "Recall", "Precision"]
  iou_threshold: 0.5
  text_match_threshold: 0.9

usage:
  description: |
    1. Load images from images/ directory
    2. Run inference with your OCR model
    3. Evaluate predictions against annotations.json
    4. Report metrics (F1, precision, recall)
"""
    return config


def create_requirements() -> str:
    """Create requirements.txt for inference."""
    requirements = """# OCR Inference Sample Requirements

# Minimal dependencies for inference
numpy>=1.24.0
Pillow>=9.0.0

# Optional: For advanced workflows
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0

# Optional: For Google AI Studio integration
google-generativeai>=0.3.0
google-auth>=2.25.0

# Optional: For visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Optional: For validation
pydantic>=2.0.0
"""
    return requirements


def create_readme() -> str:
    """Create README with inference instructions."""
    readme = """# OCR Inference Sample Dataset

Lightweight dataset for testing OCR inference in Google AI Studio and other environments.

## Dataset Contents

### Images (3 samples)
- `sample_001.jpg`: Synthetic receipt (600x400px) - complex layout with multiple text regions
- `sample_002.jpg`: Document (300x400px) - simple layout with title and body text
- `sample_003.jpg`: Document (300x400px) - simple layout with mixed content

### Annotations
- `annotations.json`: COCO format with text bounding boxes and ground truth text
- 5 annotated text regions across all images

### Configuration
- `config.yaml`: Dataset configuration and inference parameters
- `requirements.txt`: Python dependencies

## Quick Start

### 1. Load Sample Data

```python
import json
from pathlib import Path
from PIL import Image

# Load annotations
with open("annotations.json") as f:
    coco = json.load(f)

# Load image
img = Image.open("images/sample_001.jpg")
print(f"Image size: {img.size}")

# Get annotations for first image
image_id = 1
region_annotations = [
    ann for ann in coco["annotations"]
    if ann["image_id"] == image_id
]
print(f"Text regions: {len(region_annotations)}")
for ann in region_annotations:
    print(f"  - {ann['text']} (confidence: {ann.get('confidence', 'N/A')})")
```

### 2. Run Inference with Your Model

```python
import torch
from torchvision import transforms

# Load model
model = load_your_ocr_model()  # Replace with your model loading
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("images/sample_001.jpg")
img_tensor = transform(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    predictions = model(img_tensor)

# Process predictions
# predictions should include:
# - bounding boxes: (x, y, w, h)
# - text content: recognized text
# - confidence scores
```

### 3. Evaluate Results

```python
def compute_metrics(predictions, ground_truth, iou_threshold=0.5):
    # Compute F1, Precision, Recall.

    tp = 0  # true positives
    fp = 0  # false positives
    fn = 0  # false negatives

    for gt in ground_truth:
        gt_box = gt["bbox"]

        # Find matching prediction
        matched = False
        for pred in predictions:
            iou = compute_iou(pred["bbox"], gt_box)

            # Check text match (optional)
            text_match = (
                pred.get("text", "").lower() ==
                gt.get("text", "").lower()
            )

            if iou >= iou_threshold:
                if text_match:
                    tp += 1
                    matched = True
                    break

        if not matched:
            fn += 1

    fp = len(predictions) - tp

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"F1": f1, "Precision": precision, "Recall": recall}
```

## Using in Google AI Studio

### 1. Upload Dataset
- Upload `sample_data/` folder to Google Cloud Storage
- Grant permissions to your AI Studio project

### 2. Create Inference Notebook
```python
# In Google Colab/AI Studio
import os
from pathlib import Path

# Download sample dataset
!git clone <your-repo-url>
%cd sample_data

# Load and test
import json
with open("annotations.json") as f:
    coco = json.load(f)

print(f"Loaded {len(coco['images'])} images")
print(f"Loaded {len(coco['annotations'])} annotations")
```

### 3. Run Inference with Gemini Vision
```python
import anthropic
from pathlib import Path
from PIL import Image
import base64

client = anthropic.Anthropic()

# Load image
img_path = "images/sample_001.jpg"
img = Image.open(img_path)

# Encode to base64
with open(img_path, "rb") as f:
    img_data = base64.standard_b64encode(f.read()).decode("utf-8")

# Use Gemini for OCR
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",  # Or your chosen model
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
                        "data": img_data
                    }
                },
                {
                    "type": "text",
                    "text": "Extract all visible text from this image. For each text region, provide: 1) The text content, 2) Approximate location (top-left, top-right, bottom-left, bottom-right, center), 3) Confidence score (0-1)"
                }
            ]
        }
    ]
)

print(response.content[0].text)
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | 3 |
| Total Text Regions | 5 |
| Average Region Size | ~8,400 px² |
| Languages | English |
| Dataset Size | ~150 KB |
| Image Formats | JPEG |
| Annotation Format | COCO |

## Annotation Format

```json
{
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "bbox": [80, 20, 240, 35],
    "area": 8400,
    "iscrowd": 0,
    "text": "SAMPLE RECEIPT",
    "language": "en"
}
```

Fields:
- `bbox`: [x, y, width, height] in pixels
- `area`: bounding box area in pixels²
- `text`: ground truth OCR text
- `language`: text language code

## Extending the Dataset

To add more samples:

```python
from ocr_inference_sample import create_synthetic_receipt, create_annotations

# Create more images
for i in range(4, 10):
    img = create_synthetic_receipt()
    img.save(f"images/sample_{i:03d}.jpg")

# Regenerate annotations
annotations = create_annotations(image_count=9)
with open("annotations.json", "w") as f:
    json.dump(annotations, f, indent=2)
```

## Performance Benchmarks

Expected inference times on different hardware:

| Model | GPU | Time/Image |
|-------|-----|-----------|
| CRAFT (detection) | A100 | 15-20ms |
| TPS-ResNet (recognition) | A100 | 30-40ms |
| End-to-end | A100 | 50-60ms |
| CPU (i7) | N/A | 500-1000ms |

## Known Limitations

- Synthetic images may differ from real-world documents
- Limited language support (English only)
- Small dataset size (3 images) - suitable for testing only
- No complex layouts (tables, multi-column text)

## Related Resources

- [CRAFT Text Detection](https://github.com/clovaai/CRAFT-pytorch)
- [TPS-ResNet Text Recognition](https://github.com/clovaai/deep-text-recognition-benchmark)
- [COCO Dataset Format](https://cocodataset.org/)
- [Google AI Studio](https://aistudio.google.com/)

## License

MIT License - See LICENSE file for details

---

**Created**: 2025-12-11
**Size**: ~150 KB
**Use Case**: OCR inference testing and model evaluation
"""
    return readme


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create lightweight OCR inference sample dataset for Google AI Studio")
    parser.add_argument(
        "--output",
        type=str,
        default="ocr_sample_data",
        help="Output directory for sample data (default: ocr_sample_data)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of sample images to create (default: 3)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image generation (annotation data only)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Creating OCR inference sample dataset...")
    print(f"   Output: {output_dir}")
    print(f"   Images: {args.count}")
    print()

    # Create images
    if not args.no_images and Image is not None:
        print("🖼️  Generating sample images...")

        # Receipt image
        img = create_synthetic_receipt(width=400, height=600)
        img.save(images_dir / "sample_001.jpg")
        print("   ✅ Created: sample_001.jpg (receipt)")

        # Document images
        for i in range(2, args.count + 1):
            img = create_sample_image_with_text(
                width=400,
                height=300,
                text=f"Sample Document #{i}",
            )
            img.save(images_dir / f"sample_{i:03d}.jpg")
            print(f"   ✅ Created: sample_{i:03d}.jpg (document)")
        print()

    # Create annotations
    print("📝 Creating COCO annotations...")
    annotations = create_annotations(image_count=args.count)
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"   ✅ Created: annotations.json ({len(annotations['annotations'])} regions)")
    print()

    # Create config
    print("⚙️  Creating configuration files...")
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(create_config())
    print("   ✅ Created: config.yaml")

    # Create requirements
    requirements_path = output_dir / "requirements.txt"
    with open(requirements_path, "w") as f:
        f.write(create_requirements())
    print("   ✅ Created: requirements.txt")

    # Create README
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(create_readme())
    print("   ✅ Created: README.md")
    print()

    # Summary
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    total_size_kb = total_size / 1024

    print("=" * 60)
    print("✅ OCR INFERENCE SAMPLE DATASET CREATED")
    print("=" * 60)
    print()
    print("📊 Dataset Summary:")
    print(f"   Location: {output_dir.absolute()}")
    print(f"   Total Size: {total_size_kb:.1f} KB")
    print(f"   Images: {args.count}")
    print(f"   Annotations: {len(annotations['annotations'])} text regions")
    print()
    print("📁 Directory Structure:")
    print(f"   {output_dir}/")
    print(f"   ├── images/ ({args.count} JPEG files)")
    print("   ├── annotations.json (COCO format)")
    print("   ├── config.yaml")
    print("   ├── requirements.txt")
    print("   └── README.md")
    print()
    print("🚀 Next Steps:")
    print("   1. Review README.md for usage instructions")
    print(f"   2. Load data: python -c \"import json; print(json.load(open('{output_dir}/annotations.json')))\"")
    print("   3. Run inference with your OCR model")
    print("   4. Export to Google AI Studio or other platforms")
    print()


if __name__ == "__main__":
    main()
