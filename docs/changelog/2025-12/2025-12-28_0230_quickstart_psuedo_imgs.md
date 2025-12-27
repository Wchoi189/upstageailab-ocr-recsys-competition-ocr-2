# Sample Batch Validation - Quick Start Guide

## Objective
Test pseudo-labeling pipeline with 3 enhancement strategies on a 50-image sample before processing the full 4,089-image baseline dataset.

---

## Step 1: Extract Sample Batch

Extract 50 diverse images from baseline training dataset:

```bash
uv run python runners/extract_sample_batch.py \
  --parquet data/processed/baseline_train.parquet \
  --output data/samples/validation_batch_50 \
  --count 50
```

**Expected Output:**
- `data/samples/validation_batch_50/` with 50 images
- `sample_manifest.csv` with stratified sampling info

---

## Step 2: Generate Pseudo-Labels (3 Strategies)

### Strategy A: Baseline (No Enhancement)
```bash
# Note: The generate_pseudo_labels.py script is in data/ which is gitignored
# You may need to copy it or run it from its current location
uv run python data/generate_pseudo_labels.py \
  --image_dir data/samples/validation_batch_50 \
  --output data/samples/pseudo_labels_baseline.parquet \
  --name validation_baseline \
  --limit 50
```

### Strategy B: Sepia (Moderate Boost - brightness_scale=0.85)
```bash
uv run python data/generate_pseudo_labels.py \
  --image_dir data/samples/validation_batch_50 \
  --output data/samples/pseudo_labels_sepia_085.parquet \
  --name validation_sepia_085 \
  --limit 50 \
  --enhance
```

### Strategy C: Sepia (Full Boost - brightness_scale=1.0)
```bash
# Note: You'll need to modify the script to accept brightness_scale parameter
# or temporarily change the default in sepia_enhancement.py to 1.0
uv run python data/generate_pseudo_labels.py \
  --image_dir data/samples/validation_batch_50 \
  --output data/samples/pseudo_labels_sepia_100.parquet \
  --name validation_sepia_100 \
  --limit 50 \
  --enhance
```

**Expected Time:** ~3-4 minutes total (50 images ÷ 50 requests/min)

---

## Step 3: View API Usage Report

Check API usage statistics:

```bash
uv run python scripts/view_api_usage.py
```

**Expected Output:**
```
=== Upstage API Usage Report ===
Total Calls: 150 (50 × 3 strategies)
  ✓ Successful: ~147-150 (98-100%)
  ✗ Failed: 0-3
  ⏸ Rate Limited: 0

Average Response Time: ~1200-1500 ms
...
```

---

## Step 4: Compare Results

### Quick Analysis Script

Create `runners/compare_pseudo_labels.py`:

```python
#!/usr/bin/env python3
"""Compare pseudo-label results across strategies."""

import pandas as pd

# Load results
baseline = pd.read_parquet("data/samples/pseudo_labels_baseline.parquet")
sepia_085 = pd.read_parquet("data/samples/pseudo_labels_sepia_085.parquet")
sepia_100 = pd.read_parquet("data/samples/pseudo_labels_sepia_100.parquet")

print("=== Pseudo-Label Comparison ===\n")

for name, df in [("Baseline", baseline), ("Sepia 0.85", sepia_085), ("Sepia 1.0", sepia_100)]:
    avg_polygons = df['polygons'].apply(len).mean()
    avg_texts = df['texts'].apply(len).mean()
    empty_count = (df['polygons'].apply(len) == 0).sum()
    
    print(f"{name}:")
    print(f"  Avg Polygons: {avg_polygons:.1f}")
    print(f"  Avg Texts: {avg_texts:.1f}")
    print(f"  Empty Results: {empty_count}/50")
    print()
```

Run comparison:
```bash
uv run python runners/compare_pseudo_labels.py
```

---

## Step 5: Visual QA (Sample 10 Images)

Manually inspect 10 random samples:

```python
#!/usr/bin/env python3
"""Visualize pseudo-labels for quality check."""

import random
import cv2
import pandas as pd
import numpy as np

# Load one strategy's results
df = pd.read_parquet("data/samples/pseudo_labels_sepia_085.parquet")

# Sample 10 random images
samples = df.sample(10, random_state=42)

for _, row in samples.iterrows():
    img = cv2.imread(row['image_path'])
    
    # Draw polygons
    for poly in row['polygons']:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    
    # Save visualization
    out_path = f"data/samples/viz_{row['image_filename']}"
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")
```

---

## Validation Checklist

- [ ] **Empty Results**: ≤ 2% of images (confirm API never returns empty)
- [ ] **Word Count Increase**: Sepia shows ≥5% improvement over baseline (on average)
- [ ] **API Success Rate**: ≥95% successful calls
- [ ] **Coordinate Accuracy**: Visual QA shows polygons align with text
- [ ] **Brightness Quality**: Sepia images don't appear over-bright
- [ ] **API Tracking**: Usage properly logged in `data/ops/upstage_api_usage.json`

---

## Decision Criteria

### ✅ Proceed with Full Dataset if:
- Empty results ≤ 1 image (acceptable API/network issues)
- Sepia enhancement shows measurable benefit
- API success rate ≥ 95%
- No visual quality degradation

### ⚠️ Adjust if:
- Empty results > 5% → Investigate API/network issues
- No enhancement benefit → Skip enhancement, use baseline only
- Over-bright images → Reduce brightness_scale to 0.74

### ❌ Stop if:
- Empty results > 10% → Critical API issue
- API success rate < 90% → Rate limiting or quota issues
- Coordinate mapping errors detected → Fix inverse matrix logic

---

## Next Steps After Validation

If validation passes:

1. **Full Dataset Processing** (baseline_train: 4,089 images)
   - Batch size: 500 images per checkpoint
   - Total time: ~82 minutes
   - Storage: ~200 MB

2. **Quality Assurance**
   - Visual QA on 50 random full dataset samples
   - Validate polygon accuracy vs original annotations

3. **KIE Model Training**
   - Select KIE model (LayoutLMv3, PICK, DocFormer)
   - Fine-tune on pseudo-labeled dataset

---

## Troubleshooting

### Issue: Script can't find generate_pseudo_labels.py
**Solution**: The script is in [data/generate_pseudo_labels.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/data/generate_pseudo_labels.py) (gitignored). You may need to:
- Temporarily ungitignore it, or
- Run from absolute path, or
- Copy to `runners/` directory

### Issue: Empty API responses
**Check**: 
1. API key in `.env.local`
2. Network connectivity
3. API quota/rate limits
4. Response structure in tracker logs

### Issue: Brightness_scale parameter not available
**Solution**: Modify [generate_pseudo_labels.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/data/generate_pseudo_labels.py) to accept `--brightness-scale` argument and pass to [enhance_sepia()](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/utils/sepia_enhancement.py#18-65)

---

## Files Generated

```
data/samples/
├── validation_batch_50/
│   ├── *.jpg (50 images)
│   └── sample_manifest.csv
├── pseudo_labels_baseline.parquet
├── pseudo_labels_sepia_085.parquet
├── pseudo_labels_sepia_100.parquet
└── viz_*.jpg (10 visualization samples)

data/ops/
└── upstage_api_usage.json (updated with 150 calls)
```
