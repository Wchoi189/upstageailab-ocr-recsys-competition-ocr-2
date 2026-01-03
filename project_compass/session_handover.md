# Session Handover: KIE → Text Recognition Pivot

**Date:** 2026-01-03
**Status:** Strategic Pivot Complete, Ready for Text Recognition
**Previous Phase:** Key Information Extraction (KIE) on receipts

---

## Major Accomplishments

### 1. Debugged KIE Training (val_F1: 0.0 → 0.623)

Fixed critical configuration issues:
- **Label mismatch**: Config expected 65 BIO labels, data had 32 simple labels
- **Solution**: Updated `num_labels: 65 → 32`, revised label list, added `warmup_steps: 100`
- **Result**: Training progressed to val_F1=0.623 by Epoch 10

### 2. Discovered Fundamental Flaw in Document Parse for Receipts

> [!CAUTION]
> **Critical Finding**: Document Parse API treats receipts as tables, not text blocks.
> - Returns 1-2 giant bounding boxes covering 96.5%+ of image
> - Text fields contain raw HTML (`<table>`, `<thead>`, `<br>` tags)
> - Makes LayoutLMv3 training fundamentally broken for receipt KIE

**Evidence from manual validation:**
- Val Sample 0: Single bbox spanning 96.5% x 98.5% of image area
- HTML-contaminated text rather than clean OCR tokens
- Confirmed in Upstage Console: DP optimized for forms/reports, not receipts

### 3. Strategic Decision: Abandon KIE+DP, Pivot to Text Recognition

**Rationale:**
- Final metrics (Epoch 10): `val_loss=1.170, val_f1=0.623, train_loss=0.441`
- F1 likely inflated by easy "O" predictions
- Would plateau below 0.70 even with extended training
- **Better path forward:** Build robust text recognition on AI Hub dataset

---

## Current State of Assets

### ✅ AI Hub 공공행정문서 (READY FOR TEXT RECOGNITION)

| Item | Status |
|------|--------|
| Images | 5467 optimized @ 1024px height |
| Location | `data/optimized_images_v2/aihub_validation/` |
| Parquet | `data/processed/archive/aihub_validation_optimized_v2.parquet` |
| Data Quality | Clean word-level bboxes, no HTML contamination |
| Average Size | 2459x3443 (raw) → ~726x1024 (optimized) |

**⚠️ Action Required:** Verify `image_path` column points to optimized images before training

### 🔴 KIE+DP Merged Datasets (DEPRECATED)

| Dataset | Location | Notes |
|---------|----------|-------|
| Train | `data/processed/aligned/baseline_kie_dp_train.parquet` (3125 samples) | HTML contaminated |
| Val | `data/processed/aligned/baseline_kie_dp_val.parquet` (402 samples) | Broken bboxes |
| **Status** | **DO NOT USE** | Preserved for reference only |

### 📦 Competition Baseline Data

| Dataset | Location | Status |
|---------|----------|--------|
| Text Detection | `data/raw/competition/baseline_text_detection/` | Available |
| KIE (raw) | `data/export/baseline_kie/` | Entities only, no bboxes |
| DP (raw) | `data/export/baseline_dp/` | Bboxes, HTML contaminated |

---

## Next Session Focus: Text Recognition Pipeline

### Recommended Tasks

1. **Verify AI Hub Parquet Paths**
   - Check `image_path` column references optimized images
   - Update paths if necessary

2. **Create Word-Crop Dataset**
   - Extract cropped images from bounding boxes
   - Use `texts` column as ground truth labels
   - Estimate ~100k+ word crops from 5467 images

3. **Train Text Recognizer**
   - Start with PARSeq (SOTA) or CRNN (simpler baseline)
   - Validate on held-out AI Hub samples
   - Target Korean + alphanumeric recognition

4. **Test Existing OCR Pipeline**
   - Check `ocr/models/` and `runners/` for recognition scripts
   - May need new crop extraction script

### Known Blockers

| Issue | Status | Resolution Path |
|-------|--------|-----------------|
| Image path verification | ⚠️ Needs check | Update parquet or symlinks |
| Text-recognition pipeline status | 🔴 Unknown | Audit existing code |
| Crop extraction script | ⚠️ May not exist | Create new script if needed |

---

## Key Lessons Learned

> [!IMPORTANT]
> 1. **Document Parse ≠ Universal OCR** — Optimized for structured documents, not receipts
> 2. **Receipts don't require layout analysis** — Linear text structure; text-only NER may suffice
> 3. **Always validate API outputs** — 5 minutes of manual testing prevents days of wasted effort

---

## Strategic Recommendations

### For Receipt KIE (Long-term)
1. Use OCR + LLM extraction (no layout modeling needed)
2. Train text-only NER on Upstage KIE outputs
3. Alternative: Pretrain LayoutLMv3 on AI Hub, fine-tune on manually labeled receipts

### For Current Work (AI Hub)
1. Build robust text-detection + recognition stack
2. Consider pretraining LayoutLMv3 on Korean document layouts (MLM + layout objectives)
3. Use text-detection models (DBNet/CRAFT) for layout understanding instead of Document Parse

---

## Files Modified (2026-01-03 Session)

| File | Change |
|------|--------|
| [train_kie.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train_kie.yaml) | Fixed num_labels (65→32), warmup_steps added |
| [train_kie.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train_kie.py) | Filtered warmup_steps from trainer args |
| [train_kie_baseline_optimized_v2.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train_kie_baseline_optimized_v2.yaml) | Updated dataset paths and labels |

---

## Reference Artifacts

- **WandB Run:** [kie-layoutlmv3-base Epoch 10](https://wandb.ai/ocr-team2/ocr-kie/runs/e7l6f9k4)
- **Assessment:** [KIE+DP Failure Analysis](file:///home/vscode/.gemini/antigravity/brain/ff1e2aa9-0a2a-4b92-baee-4c9e31fea213/assessment_kie_dp_failure.md.resolved)
- **Detailed Walkthrough:** [Session Handover Details](file:///home/vscode/.gemini/antigravity/brain/ff1e2aa9-0a2a-4b92-baee-4c9e31fea213/walkthrough.md.resolved)
- **Alignment Script:** [tools/align_kie_dp.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/tools/align_kie_dp.py)
