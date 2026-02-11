# Recognition Pipeline Roadmap & Audit

> **Current Status:** 🔴 **CRITICAL PHASE** (0% Accuracy Debugging)
> **Refactoring Phase:** ✅ Complete

## 🔴 CRITICAL PHASE: 0% Accuracy Debugging (Blocked)
**Goal**: Resolve model failing to learn (0% Acc / 100% CER) on RTX 3090.
**Context**: Run 41 Failed (Synthetic Data). Model collapses to Unigram.
**Handover**: See `session_handover.md` and `debug_report_2026-02-07_2228.md`.

- [x] **Data Pipeline Audit**: Verified (Input Images & Tokens are valid).
- [x] **Model Initialization**: Verified (Gradients exist but vanish or explode).
- [x] **Quick Iteration**: Runs 22-41 completed.
- [ ] **Next Step**: Try CRNN Architecture (Simplification).
- [ ] **Next Step**: Try Pretrained Weights (Bypass Initialization).

---

## ✅ Phase 1: Refactoring & Cleanup (Completed)

### A. Orchestrator Simplification
- [x] **Strategy Pattern**: Extracted `_inject_vocab_size` into `RecognitionConfigStrategy`.
- [x] **Domain Resolution**: Logic updated to handle both String and DictConfig.

### B. Module Cleanup
- [x] **Sanitized Config Access**: Replaced fragile `getattr(getattr(...))` with robust `cfg.get()`.
- [x] **Debug Prints**: Removed spammy print statements.

### C. Compliance
- [x] **Pre-commit**: Fixed `debug-statements`, path usage, and artifact validation.
- [x] **Environment**: Fixed `.bashrc` `VIRTUAL_ENV` warning.

---

## ⏸️ Phase 2: Deferred / Future Work

### Architecture
- [ ] **Remove Defaults**: `PARSeqDecoder` still has hardcoded defaults. (Low Priority)
- [ ] **Inference Strategy**: Refactor `forward` vs `generate` into a strategy pattern. (Deferred)

### Infrastructure
- [ ] **Hydra Inspection Tool**: Build tool to print exact resolved config structure. (Deferred)
- [ ] **Multiprocessing**: Investigate `num_workers > 0` CUDA context crash. (Paused)
