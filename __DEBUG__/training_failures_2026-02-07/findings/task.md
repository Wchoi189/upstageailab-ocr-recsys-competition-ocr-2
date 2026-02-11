# Recognition Pipeline Debugging Task List

> **Status:** 🔴 **CRITICAL** - Cross-Attention Failure Diagnosis (Session 2026-02-07)
> **Context:** Model collapses to unigram predictions (0% accuracy, 100% CER) despite valid data

---

## Phase 0: Analysis & Planning ✅
- [x] **Root Cause Analysis**
    - [x] Review session handover and findings from previous debugging (Runs 22-41)
    - [x] Analyze external insights on cross-attention mechanism
    - [x] Identify three critical issues:
        1. Missing `memory_key_padding_mask` in decoder
        2. BOS/EOS token handling concerns
        3. Visual-text embedding scale disparity
    - [x] Create comprehensive analysis document (`cross_attention_analysis.md`)
    - [x] Create phased implementation plan (`implementation_plan_cross_attention_fix.md`)
    - [x] Implementation plan approved by user

---

## Phase 1: Diagnostic 🔄 IN PROGRESS
**Goal:** Verify assumptions before making fixes

- [/] **BOS/EOS Token Verification**
    - [/] Create `inspect_tokens.py` script
    - [ ] Run diagnostic to verify:
        - Tokens start with BOS (id=1)
        - Tokens end with EOS (id=2)
        - Tokenizer decode/encode roundtrip works
    - [ ] Document findings
- [ ] **Visual Feature Padding Analysis**
    - [ ] Create `check_visual_padding.py` script
    - [ ] Run diagnostic to check:
        - Feature map dimensions [B, C, H, W]
        - Sequence length S = H × W
        - Whether padding exists in visual features
    - [ ] Document findings

---

## Phase 2: Critical Fix - Memory Masking ❌ FAILED

- [x] **Decoder Modifications**
    - [x] Add `memory_key_padding_mask` parameter to `decoder.forward()`
    - [x] Generate default mask (all False) if not provided
    - [x] Pass mask to `nn.TransformerDecoder`
- [x] **Verification**
    - [x] Run overfitting test (Run 42) with micro-training config
    - [x] **Results:** Model degraded further (repeated chars → empty strings)
    - [x] **Metrics:** Still 0% acc, 100% CER
    - [x] **Gradients:** Still vanishing (0.0000 by step 10)
    - [x] **Root Cause:** 8:1 scale disparity between visual (~1.0) and positional (~8.0)

---

## Phase 3: Visual Normalization ✅ COMPLETE
**Motivation:** Run 42 revealed 8:1 scale disparity is the PRIMARY issue, not memory masking

- [x] **Architecture Modifications**
    - [x] Add `self.visual_norm = nn.LayerNorm(256)` in `__init__`
    - [x] Apply normalization before positional encoding
    - [x] Normalize channel dimension: [B, C, H, W] → [B, H, W, C] → normalize → back
- [/] **Verification**
    - [/] Run overfitting test (Run 43) with visual normalization
    - [ ] Check for balanced visual/positional signal magnitudes
    - [ ] Verify non-zero accuracy and gradient flow

---

## Legacy Refactoring Work (Deferred)

### Completed ✅
- [x] Orchestrator refactor (Strategy Pattern)
- [x] Module cleanup (robust config access)
- [x] Pre-commit compliance fixes

### Deferred ⏸️
- [ ] Architecture defaults audit
- [ ] Inference strategy pattern refactor

---

## Artifacts & Documentation

**Analysis:**
- `cross_attention_analysis.md` - Root cause technical analysis
- `debug_report_2026-02-07_2228.md` - Previous session findings
- `findings.md` - Exhaustive evidence log
- `session_handover.md` - Previous session summary

**Planning:**
- `implementation_plan_cross_attention_fix.md` - Fix strategy
- `roadmap.md` - High-level project status

**Scripts:** (in `__DEBUG__/training_failures/scripts/`)
- `inspect_tokens.py` (NEW) - Phase 1 diagnostic
- `check_visual_padding.py` (NEW) - Phase 1 diagnostic
- Various analysis scripts from previous runs
