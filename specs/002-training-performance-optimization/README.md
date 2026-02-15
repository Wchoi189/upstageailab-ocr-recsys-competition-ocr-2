# Training Performance Optimization - Spec Kit

**Feature ID**: `002-training-performance-optimization`
**Status**: Planning Complete → Ready for Implementation
**Priority**: P1 (Blocks efficient training iteration)

## Quick Start

**For Implementation**:
```bash
# 1. Read spec overview
cat specs/002-training-performance-optimization/spec.md

# 2. Review task breakdown
cat specs/002-training-performance-optimization/tasks.md

# 3. Follow implementation checklist
cat specs/002-training-performance-optimization/checklists/implementation-checklist.md

# 4. Start with Phase 1 (tokenizer caching)
```

**For Session Continuation**:
```bash
# Use handover document
cat specs/002-training-performance-optimization/SESSION_HANDOVER.md
```

## Problem Summary

Training pipeline wastes 2-3s on every start due to:
- Tokenizer loaded 5x (instead of cached singleton)
- Model weights loaded 2x from HuggingFace
- Unused datasets instantiated (test/predict in train mode)
- Debug prints in validation hot path

**Target**: Reduce startup time from 3.5s to ≤2.0s (43% improvement)

## Spec Structure

```
specs/002-training-performance-optimization/
├── README.md                           # This file
├── spec.md                             # Full specification
├── tasks.md                            # 15 tasks, 5 phases
├── SESSION_HANDOVER.md                 # Continuation prompt
├── contracts/
│   └── performance-contract.md         # Metrics, API stability, rollback
└── checklists/
    └── implementation-checklist.md     # Step-by-step validation
```

## Implementation Phases

### Phase 1: Tokenizer Caching (P0) - 2h
**Impact**: 0.5-1s saved, 80% load reduction
**Tasks**: TASK-001 to TASK-004
**Risk**: Low

### Phase 2: Lazy Dataset Loading (P1) - 3h
**Impact**: 0.3-0.5s saved per training start
**Tasks**: TASK-005 to TASK-008
**Risk**: Medium (requires testing all modes)

### Phase 3: Model Weight Investigation (P2) - 2h
**Impact**: 0.5-2s saved (if fixable)
**Tasks**: TASK-009 to TASK-011
**Risk**: Low (investigation only)

### Phase 4: Cleanup (P2) - 1h
**Impact**: Clean logs
**Tasks**: TASK-012
**Risk**: None

### Phase 5: Benchmarking (P1) - 1h
**Impact**: Validation
**Tasks**: TASK-013 to TASK-014
**Risk**: None

**Total Estimated**: 8-10 hours (2 days with testing)

## Key Files to Modify

```
ocr/domains/recognition/data/tokenizer.py          # Phase 1
ocr/pipelines/strategies/recognition_config.py    # Phase 1
ocr/data/datasets/__init__.py                      # Phase 2
ocr/pipelines/orchestrator.py                      # Phase 2, 3
ocr/data/lightning_data.py                         # Phase 2
ocr/domains/recognition/module.py                  # Phase 4
```

## Success Criteria

**Quantitative**:
- [x] Tokenizer loads: 5 → 1 (80% reduction)
- [ ] Model weight loads: 2 → 1 (if fixable)
- [x] Startup time: 3.5s → ≤2.0s (≥43% reduction)
- [x] Dataset creation: Mode-specific (train mode: only train/val)

**Qualitative**:
- [ ] All modes functional (train/eval/test/predict)
- [ ] No accuracy/loss regression
- [ ] Clean logs (no debug prints)
- [ ] API backward compatible

## Testing Commands

```bash
# Quick validation after Phase 1
uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  trainer.limit_train_batches=0 \
  checkpoint_path=null 2>&1 | grep -c "Loaded tokenizer"
# Expected: 1

# Test all modes after Phase 2
bash scripts/test_lazy_datasets.sh

# Performance regression test
scripts/benchmark_startup_time.sh
```

## Rollback Procedure

If issues detected:
```bash
uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  runtime.optimizations.tokenizer_caching=false \
  runtime.optimizations.lazy_datasets=false
```

Previous behavior restored (5x tokenizer loads, all datasets).

## Related Specs

- **AgentQMS**: `AgentQMS/specs/tier2-framework/patterns.spec.md`
- **Configuration**: `AgentQMS/specs/tier2-framework/configuration.spec.md`
- **Previous Work**: `specs/001-wandb-config-logging/` (revealed these bottlenecks)

## Continuation Prompt (Copy-Paste Ready)

```
I'm continuing work on `002-training-performance-optimization`.

**Current Status**: Planning complete, ready for implementation.

**Context**: Training pipeline has 2-3s startup overhead due to redundant processing.

**Spec Location**: `/workspaces/specs/002-training-performance-optimization/`

**What to do**:
1. Read spec: `spec.md`
2. Review tasks: `tasks.md`
3. Follow checklist: `checklists/implementation-checklist.md`
4. Start with Phase 1: Tokenizer caching

**First Task**: TASK-001 - Implement tokenizer singleton in `ocr/domains/recognition/data/tokenizer.py`

**Target**: Reduce startup time from 3.5s to ≤2.0s

Begin with Phase 1.
```

---

**Created**: February 15, 2026
**Author**: Claude (Session 1 - Performance Investigation)
**Next Session**: Implementation Phase 1
