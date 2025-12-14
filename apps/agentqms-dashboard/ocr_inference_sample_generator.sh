#!/bin/bash
# Create lightweight OCR inference sample dataset

OUTPUT_DIR="${1:-ocr_sample_data}"
COUNT="${2:-3}"

mkdir -p "$OUTPUT_DIR/images"

echo "📦 Creating OCR inference sample dataset..."
echo "   Output: $OUTPUT_DIR"
echo "   Samples: $COUNT"
echo ""

# Create sample images using Python PIL
python3 << 'EOFPYTHON'
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
from datetime import datetime

OUTPUT_DIR = "ocr_sample_data"
COUNT = 3

# Helper function to create receipt image
def create_receipt():
    img = Image.new("RGB", (400, 600), color="white")
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_normal = ImageFont.load_default()
    
    # Draw receipt
    draw.text((80, 20), "SAMPLE RECEIPT", fill="black", font=font_large)
    draw.line([(20, 50), (380, 50)], fill="black", width=1)
    
    y = 70
    items = [
        ("Item 1: Widget", "$10.99"),
        ("Item 2: Gadget", "$25.50"),
        ("Item 3: Component", "$5.00"),
        ("Item 4: Part", "$12.75"),
    ]
    
    for name, price in items:
        draw.text((30, y), name, fill="black", font=font_normal)
        draw.text((300, y), price, fill="black", font=font_normal)
        y += 30
    
    draw.line([(20, y), (380, y)], fill="black", width=2)
    y += 20
    draw.text((30, y), "TOTAL: $58.58", fill="black", font=font_large)
    
    return img

# Helper function to create document
def create_document(num):
    img = Image.new("RGB", (400, 300), color="white")
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()
    
    draw.rectangle([(10, 10), (390, 290)], outline="black", width=2)
    draw.text((50, 30), f"Sample Document #{num}", fill="black", font=font_title)
    draw.text((50, 100), "This is a sample document for OCR testing.", fill="black", font=font_body)
    draw.text((50, 140), "Date: 2025-12-11", fill="black", font=font_body)
    
    return img

# Create images
print("🖼️  Generating sample images...")
img = create_receipt()
img.save(f"{OUTPUT_DIR}/images/sample_001.jpg")
print("   ✅ Created: sample_001.jpg (receipt)")

for i in range(2, COUNT + 1):
    img = create_document(i)
    img.save(f"{OUTPUT_DIR}/images/sample_{i:03d}.jpg")
    print(f"   ✅ Created: sample_{i:03d}.jpg (document)")

# Create annotations
print("\n📝 Creating COCO annotations...")
annotations = {
    "info": {
        "description": "OCR Inference Sample",
        "version": "1.0",
        "year": 2025,
        "date_created": datetime.now().isoformat(),
    },
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "text_region", "supercategory": "text"}],
}

# Image 1 (receipt) annotations
annotations["images"].append({
    "id": 1,
    "file_name": "sample_001.jpg",
    "height": 600,
    "width": 400,
})

annotations["annotations"].extend([
    {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [80, 20, 240, 30],
        "area": 7200,
        "text": "SAMPLE RECEIPT",
        "language": "en",
    },
    {
        "id": 2,
        "image_id": 1,
        "category_id": 1,
        "bbox": [30, 240, 200, 30],
        "area": 6000,
        "text": "TOTAL: $58.58",
        "language": "en",
    },
])

# Images 2+ annotations
for i in range(2, COUNT + 1):
    annotations["images"].append({
        "id": i,
        "file_name": f"sample_{i:03d}.jpg",
        "height": 300,
        "width": 400,
    })
    
    annotations["annotations"].extend([
        {
            "id": len(annotations["annotations"]) + 1,
            "image_id": i,
            "category_id": 1,
            "bbox": [50, 30, 300, 30],
            "area": 9000,
            "text": f"Sample Document #{i}",
            "language": "en",
        },
        {
            "id": len(annotations["annotations"]) + 2,
            "image_id": i,
            "category_id": 1,
            "bbox": [50, 100, 350, 25],
            "area": 8750,
            "text": "This is a sample document for OCR testing.",
            "language": "en",
        },
    ])

with open(f"{OUTPUT_DIR}/annotations.json", "w") as f:
    json.dump(annotations, f, indent=2)

print(f"   ✅ Created: annotations.json ({len(annotations['annotations'])} regions)")

# Create config
print("\n⚙️  Creating configuration files...")
with open(f"{OUTPUT_DIR}/config.yaml", "w") as f:
    f.write("""dataset:
  name: "OCR Inference Sample"
  version: "1.0"
  samples: 3
  format: "COCO"

images:
  format: "JPEG"
  color_space: "RGB"
  width: 400
  height: "300-600"

annotations:
  format: "COCO v1"
  total_regions: 5

inference:
  model_types: ["CRAFT", "TPS-ResNet"]
  input_size: 320
  confidence_threshold: 0.5
""")
print("   ✅ Created: config.yaml")

with open(f"{OUTPUT_DIR}/requirements.txt", "w") as f:
    f.write("""Pillow>=9.0.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
google-generativeai>=0.3.0
""")
print("   ✅ Created: requirements.txt")

print("\n✅ Dataset creation complete!")
print(f"\nDataset Summary:")
print(f"  Location: {OUTPUT_DIR}/")
print(f"  Images: {COUNT}")
print(f"  Annotations: {len(annotations['annotations'])} regions")

EOFPYTHON

echo ""
echo "✅ OCR INFERENCE SAMPLE CREATED"
echo ""
ls -lh "$OUTPUT_DIR"
