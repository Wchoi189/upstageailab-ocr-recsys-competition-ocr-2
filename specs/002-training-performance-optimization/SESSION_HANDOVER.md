# Session Handover: Training Performance Optimization

**Spec**: `002-training-performance-optimization`
**Handover Date**: February 15, 2026
**Status**: Planning Complete → Ready for Implementation

---

## Context Summary

During WandB config logging investigation, identified multiple performance bottlenecks in training pipeline initialization:

1. **Tokenizer loaded 5x per training start** (~0.5-1s overhead)
2. **Pretrained model weights loaded 2x** from HuggingFace
3. **Unused datasets instantiated** (test/predict in train mode)
4. **Debug prints in validation hot path**

**Total Estimated Impact**: 2-3s wasted on every training start.

---

## Specification Status

### Completed Artifacts

✅ **Spec Document**: `specs/002-training-performance-optimization/spec.md`
- Problem statement with evidence from logs
- Root cause analysis for each bottleneck
- Optimization plan with priorities (P0-P3)
- Success criteria and testing strategy

✅ **Task Breakdown**: `specs/002-training-performance-optimization/tasks.md`
- 15 tasks across 5 phases
- Estimated time per task
- Critical path identified
- Validation steps for each task

✅ **Performance Contract**: `specs/002-training-performance-optimization/contracts/performance-contract.md`
- Baseline metrics documented
- Target metrics defined
- API stability guarantees
- Rollback procedure
- Acceptance criteria

✅ **Implementation Checklist**: `specs/002-training-performance-optimization/checklists/implementation-checklist.md`
- Pre-implementation setup
- Per-task validation steps
- Final validation requirements
- Post-merge monitoring

---

## Key Files to Modify

### Phase 1: Tokenizer Caching (P0 - Highest Impact)
```
ocr/domains/recognition/data/tokenizer.py          # Add singleton cache
ocr/pipelines/strategies/recognition_config.py    # Update vocab injection
configs/data/datasets/recognition.yaml             # Update dataset configs
tests/ocr/domains/recognition/test_tokenizer_cache.py  # New unit tests
```

### Phase 2: Lazy Dataset Loading (P1)
```
ocr/data/datasets/__init__.py                      # Add splits parameter
ocr/pipelines/orchestrator.py                      # Mode-specific dataset loading
ocr/data/lightning_data.py                         # Handle missing splits
scripts/test_lazy_datasets.sh                      # New test script
```

### Phase 3: Investigation (P2)
```
ocr/pipelines/orchestrator.py                      # Add profiling
ocr/pipelines/strategies/recognition_config.py    # Check vocab injection
specs/002-training-performance-optimization/findings.md  # Document results
```

---

## Evidence from Investigation

### Tokenizer Duplication (Log Extract)
```
[2026-02-15 15:46:03,206][ocr.domains.recognition.data.tokenizer][INFO] - Loaded tokenizer: 1023 chars, vocab_size=1027, max_len=25
[2026-02-15 15:46:05,574][ocr.domains.recognition.data.tokenizer][INFO] - Loaded tokenizer: 1023 chars, vocab_size=1027, max_len=25
[2026-02-15 15:46:05,580][ocr.domains.recognition.data.tokenizer][INFO] - Loaded tokenizer: 1023 chars, vocab_size=1027, max_len=25
[2026-02-15 15:46:05,584][ocr.domains.recognition.data.tokenizer][INFO] - Loaded tokenizer: 1023 chars, vocab_size=1027, max_len=25
[2026-02-15 15:46:05,588][ocr.domains.recognition.data.tokenizer][INFO] - Loaded tokenizer: 1023 chars, vocab_size=1027, max_len=25
```

### Model Weight Duplication (Log Extract)
```
[2026-02-15 15:46:04,843][timm.models._builder][INFO] - Loading pretrained weights from Hugging Face hub (timm/resnet18.a1_in1k)
[2026-02-15 15:46:05,274][timm.models._builder][INFO] - Loading pretrained weights from Hugging Face hub (timm/resnet18.a1_in1k)
```

### Debug Prints in Hot Path (Code Reference)
```python
# ocr/domains/recognition/module.py:114-128
if batch_idx == 0:
    print(f"\n[Validation Debug] Samples:")
    print(f"  Pred Type: {type(inference_out)}")
    # ... more prints
```

---

## Implementation Priority

**Critical Path** (Implement First):
1. Phase 1: Tokenizer Caching (80% of low-hanging fruit)
2. Phase 2: Lazy Dataset Loading (mode-specific benefits)
3. Phase 5: Benchmarking (validate improvements)

**Secondary** (Implement After Critical Path):
3. Phase 3: Model Weight Investigation (requires profiling)
4. Phase 4: Debug Logging Cleanup (polish)

**Optional** (Defer to Future):
- Config serialization caching (low impact unless `log_config=true` becomes common)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Tokenizer state mutation breaks training | Make tokenizer immutable after `__init__` |
| Dataset factory API break | Add backward-compatible `splits=None` default |
| Mode-specific bugs | Test all 4 modes (train/eval/test/predict) |
| Cache invalidation issues | Use immutable tuple keys (charset_path, max_len) |

---

## Testing Strategy

### Unit Tests (Mandatory)
```bash
pytest tests/ocr/domains/recognition/test_tokenizer_cache.py -v
pytest tests/ocr/data/test_dataset_factory.py -v
```

### Integration Tests (Mandatory)
```bash
bash scripts/test_lazy_datasets.sh  # Test all modes
```

### Performance Regression Test (Mandatory)
```bash
scripts/benchmark_startup_time.sh  # Fail if >2.0s startup
```

---

## Success Metrics

**Baseline (Pre-Optimization)**:
- Tokenizer loads: 5
- Model weight loads: 2
- Startup time: ~3.5s
- Datasets: All 4 splits created regardless of mode

**Target (Post-Optimization)**:
- Tokenizer loads: 1 (80% reduction) ✓
- Model weight loads: 1 (if fixable) 🎯
- Startup time: ≤2.0s (≥43% reduction) ✓
- Datasets: Only required splits per mode ✓

---

## Continuation Prompt

```
# CONTINUATION PROMPT FOR NEXT SESSION

I'm continuing work on `002-training-performance-optimization`.

**Current Status**: Planning complete, ready for implementation.

**Context**:
- Training pipeline has 2-3s startup overhead due to redundant processing
- Tokenizer loaded 5x per start (should be cached singleton)
- Pretrained model weights loaded 2x from HuggingFace (needs investigation)
- Unused datasets instantiated in wrong modes (need lazy loading)

**Spec Location**: `/workspaces/specs/002-training-performance-optimization/`

**What to do**:
1. Read the spec: `spec.md` (problem statement, root causes, optimization plan)
2. Review tasks: `tasks.md` (15 tasks, 5 phases, estimated 2-3 days)
3. Follow checklist: `checklists/implementation-checklist.md`
4. Start with Phase 1: Tokenizer caching (highest impact/effort ratio)

**First Task**: TASK-001 - Implement tokenizer singleton in `ocr/domains/recognition/data/tokenizer.py`

**Target**: Reduce training startup time from 3.5s to ≤2.0s (43% improvement)

**Validation**: Run `uv run python scripts/runners/train.py experiment=parseq_flash_fast trainer.limit_train_batches=0` and verify tokenizer loaded only 1x (check logs).

Begin with Phase 1.
```

---

## Related Documentation

**AgentQMS Specs**:
- `/workspaces/AgentQMS/specs/tier2-framework/patterns.spec.md` - Hydra patterns
- `/workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md` - Config standards

**Recent Work**:
- `001-wandb-config-logging` - Just completed, revealed these bottlenecks
- WandB serialization fix in `orchestrator.py:186-230` - Performance considerations documented

**Related Files**:
- Orchestrator: `/workspaces/ocr/pipelines/orchestrator.py`
- Tokenizer: `/workspaces/ocr/domains/recognition/data/tokenizer.py`
- Dataset factory: `/workspaces/ocr/data/datasets/__init__.py`

---

## Questions for Next Session

1. **Tokenizer Immutability**: Can tokenizer be made immutable after init? Check if any code mutates tokenizer state.
2. **Hydra Instantiate**: Does `hydra.utils.instantiate()` support factory methods? May need wrapper for `get_or_create()`.
3. **Model Weight Duplication**: Is timm caching weights internally? Or are we actually creating encoder twice?
4. **Dataset Split Validation**: Should DataModule raise error or return None for missing optional splits?

---

## Handover Checklist

- [x] Spec document written (`spec.md`)
- [x] Task breakdown complete (`tasks.md`)
- [x] Performance contract defined (`contracts/performance-contract.md`)
- [x] Implementation checklist created (`checklists/implementation-checklist.md`)
- [x] Evidence collected (log extracts, code references)
- [x] Success criteria defined (baseline → target metrics)
- [x] Continuation prompt provided
- [x] Related documentation linked
- [x] Risks documented with mitigations
- [x] Testing strategy outlined

**Next Agent**: Ready to implement. Start with continuation prompt above.
