# Session Handover: AST Debugging Toolkit & BUG_003 Root Cause
**Date:** 2026-01-04
**Status:** Ready to Fix BUG_003 (Root Cause Identified)
**Previous Phase:** Text Recognition Training (PARSeq) - Blocked

---

## Session Summary
Built `agent-debug-toolkit`, a standalone AST-based debugging package to help agents analyze complex Hydra/OmegaConf configurations. Using this toolkit, we identified the **exact root cause** of BUG_003: the merge precedence order in `OCRModel._prepare_component_configs` allows `top_level_overrides` (P4) to override `direct_overrides` from architecture (P3).

---

## Current Objectives
1.  **Fix BUG_003**: Apply the precedence insight to fix `OCRModel._prepare_component_configs`
2.  **Launch Training**: Successfully start PARSeq training on `aihub_lmdb_validation`
3.  **(Optional)** MCP Phase 2: Integrate `agent-debug-toolkit` as an MCP server

---

## Key Assets (NEW)
| Asset                   | Location                                                                           | Details                            |
| ----------------------- | ---------------------------------------------------------------------------------- | ---------------------------------- |
| **Debug Toolkit**       | `agent-debug-toolkit/`                                                             | AST-based config debugging package |
| **Toolkit Walkthrough** | `docs/artifacts/walkthroughs/2026-01-04_1819_walkthrough_AST-Debugging-Toolkit.md` | Usage docs & validation results    |
| **Toolkit README**      | `agent-debug-toolkit/README.md`                                                    | CLI commands & installation        |

## Key Assets (Existing)
| Asset              | Location                                                                       | Details                                        |
| ------------------ | ------------------------------------------------------------------------------ | ---------------------------------------------- |
| **Architecture**   | `ocr/models/architectures/parseq.py`                                           | PARSeq Implementation (Encoder, Decoder, Head) |
| **Config (Train)** | `configs/train_parseq.yaml`                                                    | Training Experiment Config                     |
| **Bug Report 3**   | `docs/artifacts/bug_reports/2026-01-04_1730_BUG_003_config-precedence-leak.md` | Config Precedence Issue (ROOT CAUSE FOUND)     |

---

## Blockers & Bugs

### � Actionable: Configuration Precedence Leak (ROOT CAUSE IDENTIFIED)
**Error**: `ValueError: FPNDecoder requires at least two feature maps from the encoder.`

**Root Cause** (via `adt trace-merges ocr/models/architecture.py`):
```
| Priority | Line | Operation | Winner on Conflict      |
| -------- | ---- | --------- | ----------------------- |
| P2       | 113  | merge     | arch_overrides          |
| P3       | 128  | merge     | direct_overrides        | ← PARSeqDecoder (architecture)    |
| P4       | 140  | merge     | top_level_overrides     | ← FPNDecoder (legacy cfg.decoder) |
| P5       | 144  | merge     | cfg.component_overrides |
```

**Problem**: `top_level_overrides` (containing legacy `cfg.decoder = FPNDecoder`) merges AFTER `direct_overrides` (containing architecture's `PARSeqDecoder`), giving legacy higher precedence.

**Fix Options**:
1. Swap merge order: merge `top_level_overrides` BEFORE `direct_overrides`
2. Filter `top_level_overrides` to exclude components already in architecture
3. Update `train_parseq.yaml` to set `decoder: null` to explicitly clear legacy

### ✅ Resolved: Hydra Composition Crash
### ✅ Resolved: Timm ViT Compatibility

---

## Recommended Next Steps
1.  **Fix precedence** in `OCRModel._prepare_component_configs` (lines 128-140) - likely Option 1 or 2
2.  **Verify fix** with `uv run adt trace-merges ocr/models/architecture.py` to confirm new order
3.  **Launch training** with `uv run python runners/train.py --config-name train_parseq`

---

## Available Debugging Tools
```bash
# Analyze config access patterns (filter by component)
uv run adt analyze-config ocr/models/architecture.py --component decoder

# Trace OmegaConf.merge precedence (KEY TOOL for BUG_003)
uv run adt trace-merges ocr/models/architecture.py --output markdown

# Find component instantiation sites
uv run adt find-instantiations ocr/models/ --component decoder
```
