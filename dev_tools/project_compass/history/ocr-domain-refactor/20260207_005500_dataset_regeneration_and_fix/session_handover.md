# Session Handover - Recognition Training Debugging

**Date:** 2026-02-07
**Session:** Vocab Injection Fix & Code Audit

---

## Summary

Successfully resolved the critical **CUDA Device-Side Assert** (Token Index Out of Bounds) bug.
Identified and documented "Zero Learning" risks and technical debt in the recognition pipeline.

---

## ✅ Resolved Issues

### 1. Token Index Out of Bounds (RESOLVED)

**Problem:** `OCRProjectOrchestrator` failed to inject the correct `vocab_size` (1027) into `PARSeq` configuration, causing it to default to 1000. Token IDs > 999 caused embedding layer index errors on GPU.

**Fix Applied:**
- Updated [`ocr/pipelines/orchestrator.py`](file:///workspaces/ocr/pipelines/orchestrator.py):
  - Robust injection logic that handles both flat (`out_features`) and nested (`params`) configuration structures.
  - Updates global `model.vocab_size` for interpolation.

**Verification:**
- Training `rec_baseline_v1` is stable beyond 300 iterations.
- Device-side assertions are gone.

**Artifacts:**
- Bug Report: [`/workspaces/docs/artifacts/bug_reports/2026-02-07_0419_bug_20260207_vocab-injection.md`](file:///workspaces/docs/artifacts/bug_reports/2026-02-07_0419_bug_20260207_vocab-injection.md)
- Walkthrough: [`/home/vscode/.gemini/antigravity/brain/bd0b1946-e525-46ad-bf00-3d9cbd3f98bc/walkthrough.md`](file:///home/vscode/.gemini/antigravity/brain/bd0b1946-e525-46ad-bf00-3d9cbd3f98bc/walkthrough.md)

---

## 🔍 Code Audit & Roadmap

A comprehensive audit of the recognition pipeline was conducted. High-risk areas include hardcoded architectural defaults, fragility in configuration access, and production code clutter.

**See Full Roadmap:** [`/workspaces/__DEBUG__/training_failures/roadmap.md`](file:///workspaces/__DEBUG__/training_failures/roadmap.md)

**Key Recommendations:**
1. **Remove defaults** in `PARSeqDecoder` `__init__` to enforce explicit configuration.
2. **Clean up debug prints** in `module.py` and `architecture.py`.
3. **Refactor inference** to unify `forward` (greedy) and `beam_search_inference` logic.

---

## ⚠️ Outstanding Warnings

### Domain Mismatch
```
⚠️ Mismatch detected! Domain=detection but Model=PARSeq.
⚠️ Forcing domain='recognition' to prevent runtime errors.
```
**Status:** Benign but noisy.
**Cause:** `OCRProjectOrchestrator` detects `detection` because `experiment=rec_baseline_v1` overrides don't fully replace the root `domain` key in certain Hydra composition contexts.
**Recommendations:** Fix in Phase 1 of Roadmap.

---

## Deferred Issues (From Previous Session)

### Multiprocessing/CUDA Workers
**Status:** Works with `num_workers=0`. High priority for optimization (Phase 3).

---

## Current Working Command

```bash
uv run python scripts/runners/train.py \
  experiment=rec_baseline_v1 \
  trainer.precision=32 \
  dataloaders.train_dataloader.num_workers=0 \
  dataloaders.val_dataloader.num_workers=0
```
**Status:** STABLE.

---

## References

- Debug Workspace: [`/workspaces/__DEBUG__/training_failures/`](file:///workspaces/__DEBUG__/training_failures/)
- Findings Log: [`/workspaces/__DEBUG__/training_failures/findings.md`](file:///workspaces/__DEBUG__/training_failures/findings.md)
