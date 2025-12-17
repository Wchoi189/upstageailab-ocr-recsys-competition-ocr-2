---
type: architecture
component: data
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Data Architecture

**Purpose**: ICDAR-format dataset with JSON annotations; quadrilateral polygons per word; CLEval character-level evaluation.

---

## Dataset Structure

```
data/datasets/
├── images/
│   ├── train/      # Training images
│   ├── val/        # Validation images
│   └── test/       # Test images (no ground truth)
└── jsons/
    ├── train.json  # Training annotations
    ├── val.json    # Validation annotations
    └── test.json   # Test annotations (no labels)
```

---

## Annotation Format

| Field | Type | Description |
|-------|------|-------------|
| `images` | Object | Map of image filename → words |
| `words` | Object | Map of word_id → polygon |
| `points` | Array | 4-point quadrilateral `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]` |

**Example**:
```json
{
  "images": {
    "image1.jpg": {
      "words": {
        "word_1": {"points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]},
        "word_2": {"points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}
      }
    }
  }
}
```

---

## Data Processing Pipeline

| Step | Component | Purpose |
|------|-----------|---------|
| 1. **Load** | `OCRDataset` (torch.utils.data.Dataset) | Load images, parse JSON annotations, validate |
| 2. **Transform** | Albumentations pipeline | Apply augmentations (rotation, scaling), normalize |
| 3. **Collate** | `db_collate_fn` | Batch samples for DBNet model |
| 4. **DataLoader** | PyTorch DataLoader | Manage batching, shuffling, parallel loading |

---

## Evaluation Metric

**Primary**: CLEval (Character-Level Evaluation)

| Metric | Calculation | Purpose |
|--------|-------------|---------|
| **Precision** | True positives / (True positives + False positives) | Character-level match accuracy |
| **Recall** | True positives / (True positives + False negatives) | Character-level coverage |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean |

**Granularity**: Character-level (more precise than bounding box IoU)

---

## Dependencies

| Component | Imports | Internal Dependencies |
|-----------|---------|----------------------|
| **OCRDataset** | PyTorch, Albumentations | JSON annotations |
| **Transform Pipeline** | Albumentations | Augmentation configs (`configs/data/transforms/`) |
| **DataLoader** | PyTorch | `db_collate_fn` |

---

## Constraints

- **Annotation Format**: ICDAR JSON structure required
- **Polygon Shape**: Must be 4-point quadrilaterals
- **Image Filtering**: Images without annotations are skipped
- **Evaluation**: CLEval requires character-level annotations (not available in test set)

---

## Backward Compatibility

**Status**: Maintained for ICDAR JSON format

**Breaking Changes**: None

**Compatibility Matrix**:

| Interface | v1.x | v2.0 | Notes |
|-----------|------|------|-------|
| JSON Format | ✅ Compatible | ✅ Compatible | ICDAR structure unchanged |
| OCRDataset API | ✅ Compatible | ✅ Compatible | Dataset interface stable |
| CLEval Metric | ✅ Compatible | ✅ Compatible | Evaluation unchanged |

---

## References

- [System Architecture](system-architecture.md)
- [Data Contracts](../pipeline/data-contracts.md)
- [Config Architecture](config-architecture.md)
