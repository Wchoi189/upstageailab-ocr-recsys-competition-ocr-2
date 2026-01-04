# Session Handover: ParSeq Dataset Implementation
**Date:** 2026-01-05
**Status:** IMPLEMENTATION COMPLETE | Training Integration Pending
**Previous Phase:** Dataset Components Implemented

---

## Quick Summary

**Completed:** All dataset components for PARSeq text recognition training.

**LMDB Data Format:**
```
image-{idx:09d} → JPEG bytes (cropped text line)
label-{idx:09d} → UTF-8 string (Korean text)
num-samples → total count (616K+)
```

**PARSeq Decoder Contract:**
- `vocab_size=100`, `max_len=25`
- Token IDs: PAD=0, BOS=1, EOS=2
- Input: `text_tokens [B, T]`

---

## Components to Implement

| Component              | File                                     | Priority |
| ---------------------- | ---------------------------------------- | -------- |
| LMDBRecognitionDataset | `ocr/datasets/lmdb_dataset.py`           | 1        |
| KoreanOCRTokenizer     | `ocr/data/tokenizer.py`                  | 1        |
| recognition_collate_fn | `ocr/datasets/recognition_collate_fn.py` | 2        |
| Hydra config           | `configs/data/recognition.yaml`          | 2        |
| Dependency             | `pyproject.toml` + `uv add lmdb`         | 0        |

---

## Key Decision (Resolved)

**Charset:** LMDB-driven (extract all unique chars from labels) + safety set
**vocab_size:** `3 (specials) + len(charset)` → dynamically computed
**out_channels:** Must match vocab_size (remove hardcoded 98)

---

## Implementation Plan Location

Full plan: [implementation_plan.md](file:///home/vscode/.gemini/antigravity/brain/7904a812-988d-42df-8978-334974c64f48/implementation_plan.md)

---

## Verification Command

```bash
uv add lmdb && uv run python runners/train.py --config-name train_parseq trainer.fast_dev_run=true
```

---

## Resolved (Previous Sessions)

- ✅ BUG_001–004: Hydra, Timm, Config, CrossEntropyLoss
