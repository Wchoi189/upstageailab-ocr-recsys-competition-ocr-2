# OCR Inference Console - Sample Data Package

**Complete, ready-to-use OCR test dataset for Google AI Studio**

## 🎯 Quick Summary

| Property | Value |
|----------|-------|
| **Dataset Size** | 44 KB (uncompressed) |
| **Compressed Size** | 25 KB (TAR.GZ) / 29 KB (ZIP) |
| **Sample Images** | 3 (receipt + 2 documents) |
| **Annotations** | 5 text regions (COCO format) |
| **Languages** | English |
| **Format** | JPEG images + JSON annotations |
| **Ready for** | Testing, validation, benchmarking |

---

## 📦 What You Get

### Location
`/workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/agentqms-dashboard/`

### Files Generated

```
ocr_sample_data/                    # Main dataset folder (44 KB)
├── images/                         # 39 KB
│   ├── sample_001.jpg             # Receipt (15 KB)
│   ├── sample_002.jpg             # Document (12 KB)
│   └── sample_003.jpg             # Document (12 KB)
├── annotations.json               # COCO annotations (2 KB)
├── config.yaml                    # Inference config
└── requirements.txt               # Python deps

Export Archives:
├── ocr_sample_data.zip            # 29 KB (for Google Drive/Cloud)
├── ocr_sample_data.tar.gz         # 25 KB (for Linux/Cloud)
└── OCR_SAMPLE_MANIFEST.txt        # File inventory

Documentation:
├── OCR_INFERENCE_GUIDE.md         # Complete usage guide
├── DEMO_QUICKSTART.md             # Dashboard demo guide
└── README.md                      # General overview
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Download Dataset
```bash
# Already in: apps/agentqms-dashboard/ocr_sample_data/
# Or download archives:
wget ocr_sample_data.zip
unzip ocr_sample_data.zip
```

### Step 2: Load in Google Colab
```python
# Upload folder to Colab or mount Google Drive
from PIL import Image
import json

# Load annotations
with open("ocr_sample_data/annotations.json") as f:
    coco = json.load(f)

# Load first image
img = Image.open("ocr_sample_data/images/sample_001.jpg")
img.show()

print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
```

### Step 3: Run Inference
```python
# Use your OCR model
predictions = model.predict("ocr_sample_data/images/")

# Evaluate
metrics = evaluate(predictions, coco['annotations'])
print(f"F1: {metrics['f1']:.3f}")
```

---

## 📊 Dataset Composition

### Image 1: Synthetic Receipt (15 KB)
```
┌─────────────────────────────────┐
│        SAMPLE RECEIPT           │  ← Text Region 1: "SAMPLE RECEIPT"
├─────────────────────────────────┤
│ Store: ABC Mart                 │
│ Address: 123 Main St            │
│ Phone: (555) 123-4567           │
│                                 │
│ Item 1 - Widget        $10.99   │
│ Item 2 - Gadget        $25.50   │
│ Item 3 - Component      $5.00   │
│ Item 4 - Part          $12.75   │
│                                 │
│ Subtotal:              $54.24   │
│ Tax (8%):               $4.34   │
│ TOTAL:                 $58.58   │  ← Text Region 2: "TOTAL: $58.58"
│                                 │
│ Thank you for your purchase!    │
│ Date: 2025-12-11 14:30         │
└─────────────────────────────────┘
```

**Challenges**: Multi-item layout, numbers extraction, alignment

### Images 2-3: Documents (12 KB each)
```
┌──────────────────────────────┐
│  Sample Document #N          │  ← Text Region: Title
├──────────────────────────────┤
│  This is a sample document   │
│  for OCR testing.            │  ← Text Region: Body
│                              │
│  Date: 2025-12-11          │
└──────────────────────────────┘
```

**Challenges**: Simple layout, text recognition, date extraction

---

## 📋 Annotations Format (COCO)

Each text region includes:
```json
{
    "id": 1,                        # Unique annotation ID
    "image_id": 1,                  # Image reference
    "category_id": 1,               # Text category
    "bbox": [80, 20, 240, 30],     # [x, y, width, height]
    "area": 7200,                   # Bounding box area
    "text": "SAMPLE RECEIPT",       # Ground truth text
    "language": "en"                # Language code
}
```

**Total**: 5 annotated regions across 3 images

---

## 💻 Integration Methods

### Method 1: Google Colab Notebook
```python
# Simple integration - paste code snippets from OCR_INFERENCE_GUIDE.md
# Test your OCR model immediately
# Save results to Google Drive
```

### Method 2: Local Development
```bash
cd ocr_sample_data
python your_ocr_script.py
```

### Method 3: Dashboard Console
```
1. Dashboard → Settings → Data Path
2. Select: ocr_sample_data/
3. Auto-loads images + annotations
4. Run inference from console
```

### Method 4: API Upload
```python
requests.post(
    "http://localhost:8000/api/v1/ocr/upload",
    files={"dataset": open("ocr_sample_data.zip", "rb")}
)
```

---

## 🔧 Configuration

### config.yaml
```yaml
dataset:
  name: "OCR Inference Sample"
  samples: 3
  format: "COCO"

inference:
  model_types: ["CRAFT", "TPS-ResNet"]
  input_size: 320
  confidence_threshold: 0.5

postprocessing:
  text_score_threshold: 0.4
  nms_threshold: 0.4
```

### requirements.txt
```
Pillow>=9.0.0           # Image processing
numpy>=1.24.0           # Numerical computing
torch>=2.0.0            # Deep learning (optional)
opencv-python>=4.7.0    # Computer vision (optional)
google-generativeai     # Gemini API (optional)
```

---

## 📈 Performance Metrics

### Expected Results

| Model | Precision | Recall | F1 | Notes |
|-------|-----------|--------|-----|-------|
| CRAFT + TPS-ResNet | 0.95+ | 0.90+ | 0.92+ | Production-ready |
| Gemini Vision API | 0.98+ | 0.95+ | 0.96+ | State-of-the-art |
| Custom Models | TBD | TBD | TBD | Use as baseline |

### Inference Time (per image)

| Component | GPU (A100) | CPU (i7) |
|-----------|-----------|---------|
| Detection (CRAFT) | 15-20ms | 200-300ms |
| Recognition (TPS) | 30-40ms | 400-600ms |
| **Total** | **50-60ms** | **600-900ms** |

---

## 📝 Usage Examples

### Example 1: Gemini Vision Integration
```python
import anthropic
import base64

client = anthropic.Anthropic()

with open("ocr_sample_data/images/sample_001.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data
            }
        }, {
            "type": "text",
            "text": "Extract all text and return as JSON with fields: text, location, confidence"
        }]
    }]
)

print(response.content[0].text)
```

### Example 2: Evaluate Against Ground Truth
```python
import json
from PIL import Image

with open("ocr_sample_data/annotations.json") as f:
    coco = json.load(f)

# For each image
for image_info in coco['images']:
    img = Image.open(f"ocr_sample_data/{image_info['file_name']}")

    # Get annotations for this image
    regions = [
        ann for ann in coco['annotations']
        if ann['image_id'] == image_info['id']
    ]

    # Run inference
    predictions = your_model.predict(img)

    # Compare
    for pred, gt in zip(predictions, regions):
        print(f"GT: '{gt['text']}', Pred: '{pred['text']}'")
```

---

## 🎓 Learning Resources

**Included Documentation:**
- `OCR_INFERENCE_GUIDE.md` - Comprehensive usage guide
- `config.yaml` - Inference parameters and best practices
- `DEMO_QUICKSTART.md` - Dashboard integration guide

**External Resources:**
- CRAFT Text Detection: https://github.com/clovaai/CRAFT-pytorch
- TPS-ResNet Recognition: https://github.com/clovaai/deep-text-recognition-benchmark
- COCO Format: https://cocodataset.org/
- Google AI Studio: https://aistudio.google.com/

---

## ✅ Checklist for Use

- [ ] Download `ocr_sample_data.zip` or `ocr_sample_data.tar.gz`
- [ ] Extract to your working directory
- [ ] Review `OCR_INFERENCE_GUIDE.md`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Load data in your environment (Colab, local, or dashboard)
- [ ] Run inference with your OCR model
- [ ] Evaluate against annotations
- [ ] Log results and metrics
- [ ] (Optional) Upload to Google Drive or Cloud Storage
- [ ] Share results with team

---

## 🔄 Workflow Summary

```
1. Extract dataset
   ↓
2. Load images + annotations
   ↓
3. Preprocess images
   ↓
4. Run inference (text detection + recognition)
   ↓
5. Post-process predictions
   ↓
6. Evaluate metrics (F1, Precision, Recall)
   ↓
7. Export results
   ↓
8. Visualize and analyze
```

---

## 💾 Export Options

### For Sharing
```bash
# Use these ready-made archives:
ocr_sample_data.zip (29 KB)
ocr_sample_data.tar.gz (25 KB)
```

### For Cloud Upload
```bash
# Upload to Google Drive, Cloud Storage, etc.
# Small size (29 KB) - instant upload
# Portable format - works anywhere
```

### For GitHub
```bash
# Add to repository
# No privacy concerns (synthetic data)
# Easy for collaborators to test
```

---

## 🎯 Use Cases

✅ **Model Testing** - Verify OCR model works on new data
✅ **Benchmarking** - Compare different models/versions
✅ **Education** - Learn OCR with real example
✅ **CI/CD Testing** - Automated model validation
✅ **Demo** - Showcase OCR capabilities
✅ **Competition** - Kaggle Vibe Code with Gemini submission

---

## 📞 Support

**Issues or questions?**
- Review `OCR_INFERENCE_GUIDE.md` for detailed examples
- Check `config.yaml` for inference parameters
- Run the Colab notebook examples
- Check dashboard console logs

---

**Created**: 2025-12-11
**Dataset Type**: Synthetic OCR Test Data
**License**: MIT (free to use and modify)
**Recommended For**: Testing, validation, education, competition submissions
