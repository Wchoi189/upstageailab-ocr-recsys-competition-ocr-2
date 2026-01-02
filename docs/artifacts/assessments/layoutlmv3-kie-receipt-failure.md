---
title: "LayoutLMv3 KIE Training on Receipt Data"
author: "AI Agent"
date: "2026-01-03"
status: "completed"
tags: ["kie", "layoutlmv3", "receipt", "document-parse", "failure-analysis"]
severity: "blocker"
outcome: "abandoned"
wandb_run: "https://wandb.ai/ocr-team2/ocr-kie/runs/e7l6f9k4"
---

## 1. Summary

Attempted to train LayoutLMv3 for Key Information Extraction (KIE) on Korean receipts by merging Upstage KIE API entity labels with Document Parse API bounding boxes.

| Metric | Value |
|--------|-------|
| Final Epoch | 10 |
| val_F1 | 0.623 |
| val_loss | 1.170 |
| train_loss | 0.441 |
| Outcome | **Abandoned** |

> [!CAUTION]
> **Root Cause:** Document Parse is designed for structured documents (forms, reports), not receipts. It returns single-table bounding boxes covering the entire receipt instead of individual text blocks.

## 2. Assessment

### 2.1 Problem Statement

- Upstage KIE API provides entity labels but no bounding boxes
- Upstage Document Parse API provides bounding boxes but treats receipts as tables
- Merged dataset contained HTML contamination and oversized bboxes

### 2.2 Evidence

**Sample 0 (Validation Set):**
```python
Text: "<br><table id='1' style='font-size:18px'><thead>..."
Polygon: x=[0.000-0.965], y=[0.014-0.999]  # 96.5% × 98.5% of image
```

**Dataset Quality:**
- ~40% of validation samples had <5 text blocks (expected: 20-50+)
- HTML artifacts in text fields (uncleaned DP output)
- 1-2 giant bboxes per image vs expected 20-50 small ones

### 2.3 Timeline

| Date | Event |
|------|-------|
| 2026-01-02 | Initial training: val_F1=0.0 |
| 2026-01-02 | Fixed label mismatch (BIO→simple) |
| 2026-01-03 | Epoch 10: val_F1=0.623 |
| 2026-01-03 | Manual DP console test confirmed issue |
| 2026-01-03 | **Decision: Abandon approach** |

## 3. Recommendations

### Immediate: Pivot to Text Recognition

Focus on AI Hub 공공행정문서 dataset:
- 5467 optimized images ready
- Clean word-level bboxes + text labels
- No HTML contamination

### For Receipts Specifically

1. Use OCR + LLM extraction (no layout needed)
2. Train text-only NER on Upstage KIE outputs
3. Skip layout-based models entirely

### Long-term (If Layout KIE Needed)

1. Pretrain LayoutLMv3 on AI Hub (Korean document layouts)
2. Fine-tune on manually labeled receipt data
3. Use text-detection outputs (DBNet/CRAFT) instead of Document Parse

## 4. Files Modified

| File | Change |
|------|--------|
| `configs/train_kie.yaml` | num_labels=32, simple labels, warmup |
| `runners/train_kie.py` | Filter warmup_steps from trainer |
| `configs/train_kie_baseline_optimized_v2.yaml` | Updated paths and labels |

## 5. Lessons Learned

1. **Validate API outputs before building pipelines** — A 5-minute manual test would have saved days
2. **Document Parse ≠ Universal Layout Extractor** — Optimized for forms, not receipts
3. **Receipts don't need layout analysis** — They're inherently linear
