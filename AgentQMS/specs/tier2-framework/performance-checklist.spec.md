---
ads_version: '2.0'
id: 'FW-025'
type: 'quick_reference'
tier: 2
priority: 'high'
updated: '2026-02-16'
spec_version: '1.0.0'
description: 'Quick reference checklist for common initialization performance anti-patterns'
---

# Performance Audit Checklist

> Quick reference for detecting common initialization inefficiencies - Use for 5-10 minute spot checks

## Specification

```yaml
agent: all
name: Performance Quick Checklist
version: 1.0
last_updated: '2026-02-16'
spec_version: '1.0.0'
when_to_use:
  quick_check: Before committing initialization code changes
  spot_audit: Periodic performance sanity checks
  pre_commit: Automated CI/CD performance gates
  debugging: When investigating slow startup
auto_load: false
check_time: 5-10 minutes
triggers:
  - performance
  - checklist
  - quick
  - sanity
related_specs:
  full_methodology: AgentQMS/specs/tier2-framework/performance-audit.spec.md (FW-024)
  patterns: AgentQMS/specs/tier2-framework/patterns.spec.md (FW-018)
```

## Quick Detection Commands

```bash
# Count all resource loads (< 1 min)
uv run python scripts/runners/train.py experiment=<name> \
  trainer.limit_train_batches=0 2>&1 | tee /tmp/audit.log

grep -c "Loading pretrained weights" /tmp/audit.log  # Should be 1 per model
grep -c "Loaded tokenizer" /tmp/audit.log            # Should be 1 total
grep -c "Creating datasets" /tmp/audit.log           # Should match mode
```

**🚨 Red Flag**: Any count > 1 for singleton resources → Investigation needed

## Checklist Items

### ☑️ Hydra Instantiation (5 min)

```bash
# Search for redundant cfg parameter
grep -rn "hydra.utils.instantiate.*cfg=" ocr/core/

# Check for nested _target_ conflicts
grep -rn "_target_.*Encoder\|_target_.*Tokenizer" configs/ | wc -l

# Verify single definition per component
python -c "
import yaml
from pathlib import Path

for f in Path('configs/model/architectures/').glob('*.yaml'):
    with open(f) as file:
        cfg = yaml.safe_load(file)
        if 'encoder' in cfg and '_target_' in cfg.get('encoder', {}):
            print(f'{f.name}: encoder defined in architecture')
"
```

**Red Flags**:
- [ ] `instantiate(config, cfg=config)` pattern found
- [ ] Same component defined at multiple config levels
- [ ] Config passed as parameter when unnecessary

**Expected**: Zero matches for redundant patterns

---

### ☑️ Singleton Resources (5 min)

```bash
# Find tokenizer instantiation without singleton
grep -rn "class.*Tokenizer" ocr/ --include="*.py" -A10 | \
  grep -v "get_or_create\|_instance\|@lru_cache"

# Check for instantiation in loops
grep -rn "for.*Tokenizer\|while.*Tokenizer" ocr/ --include="*.py"

# Verify caching patterns exist
grep -rn "get_or_create\|_instance.*=.*None\|@lru_cache" ocr/
```

**Checklist**:
- [ ] Tokenizers use `get_or_create()` or `_instance` pattern
- [ ] No tokenizer instantiation in loops
- [ ] Expensive operations have `@lru_cache` decorator
- [ ] Global resources initialized once

**Expected**: All tokenizers and shared resources use singleton pattern

---

### ☑️ Mode-Aware Loading (3 min)

```bash
# Check for hardcoded split lists
grep -rn 'splits.*=.*\["train".*"val".*"test"\]' ocr/

# Verify mode-specific loading exists
grep -rn "_get_required_splits\|mode_to_splits" ocr/pipelines/

# Validate runtime behavior
grep "Creating datasets for splits:" /tmp/audit.log
# Train mode: ['train', 'val']
# Eval mode:  ['val']
# Test mode:  ['test']
```

**Checklist**:
- [ ] No hardcoded `["train", "val", "test"]` lists
- [ ] `_get_required_splits()` method exists
- [ ] Log shows only required splits per mode
- [ ] Predict mode loads only predict split

**Expected**: Only required splits loaded per execution mode

---

### ☑️ Component Duplication (10 min)

```bash
# Add temporary profiling to suspected component
# (See full methodology FW-024 for details)

# Quick test: Count log messages
grep "Loading pretrained\|Tokenizer\|Creating" /tmp/audit.log | \
  sort | uniq -c | grep -E "^\s*[2-9]"

# Any count > 1 indicates duplication
```

**Checklist**:
- [ ] Each component instantiated exactly once
- [ ] No duplicate log messages within 3 seconds
- [ ] Stack traces (if added) show single call path
- [ ] Timing gaps < 0.5s between different operations

**Expected**: All components instantiated once, no temporal duplicates

---

### ☑️ Config Structure (5 min)

```bash
# Check architecture purity
grep -r "optimizer:\|loss:" configs/model/architectures/

# Verify dataset configs don't include transforms
grep -r "transform:" configs/data/datasets/

# Confirm domain isolation
grep -rn "detection: null\|kie: null" configs/domain/
```

**Checklist**:
- [ ] Model architectures contain NO optimizer/loss keys
- [ ] Dataset configs define ONLY paths, NOT transforms
- [ ] Domain configs nullify unused cross-domain keys
- [ ] No circular dependencies in defaults lists

**Expected**: Zero violations, clean config hierarchy

---

## Red Flag Quick Reference

| Symptom | Likely Cause | Priority | Quick Fix |
|---------|--------------|----------|-----------|
| "Loading pretrained" 2x | Redundant cfg param | 🔴 High | Remove cfg from instantiate() |
| "Loaded tokenizer" 5x | No singleton | 🔴 High | Add get_or_create() |
| All splits in eval mode | No mode check | 🟡 Medium | Add _get_required_splits() |
| 3+ second init delay | Component duplication | 🔴 High | Profile with traceback |
| Config key not found | Namespace collision | 🟡 Medium | Check @package directive |
| Memory spike on init | Eager loading | 🟡 Medium | Implement lazy loading |

---

## Performance Targets

After optimization, these metrics should hold:

```bash
# Singleton resources (MUST be 1)
grep -c "Loading pretrained weights" <log> == 1  # ✅
grep -c "Loaded tokenizer" <log> == 1            # ✅

# Mode-specific splits (MUST match mode)
# Train mode
grep "Creating datasets for splits:" <log>  # ['train', 'val'] ✅

# Eval mode
grep "Creating datasets for splits:" <log>  # ['val'] ✅

# Test mode
grep "Creating datasets for splits:" <log>  # ['test'] ✅

# Init timing (Target: < 10s for simple models)
grep "Building model" <log>  # Record timestamp
grep "✓ Model created" <log> # Should be < 5s apart
```

---

## Investigation Template

When checklist identifies an issue:

```markdown
## Issue: [Duplicate Tokenizer Loading]

1. **Quantify**: grep -c "Loaded tokenizer" → 5 (expected: 1)
2. **Locate**: grep -n "Tokenizer(" → Found in 3 files
3. **Root Cause**: No singleton pattern, instantiated per batch
4. **Fix**: Add get_or_create() class method
5. **Verify**: grep -c "Loaded tokenizer" → 1 ✅
6. **Document**: Added to findings.md with before/after
```

---

## Quick Wins (< 30 min each)

### 1. Tokenizer Singleton
```bash
# Impact: 5→1 loads, ~1s saved
# Location: ocr/domains/recognition/data/tokenizer.py
# Pattern: Add get_or_create() method
```

### 2. Mode-Aware Splits
```bash
# Impact: Load only needed, ~2s saved
# Location: ocr/pipelines/orchestrator.py
# Pattern: Add _get_required_splits()
```

### 3. Remove Redundant cfg
```bash
# Impact: 2→1 weight loads, ~1.4s saved
# Location: ocr/core/models/__init__.py
# Pattern: Remove cfg parameter from instantiate()
```

### 4. Config Cleanup
```bash
# Impact: Memory saved, cleaner codebase
# Location: configs/model/architectures/*.yaml
# Pattern: Remove optimizer/loss keys
```

**Total Potential**: 3-5 seconds per run + cleaner architecture

---

## Example Investigation Flow

```bash
# 1. Quick check (30 seconds)
$ uv run python scripts/runners/train.py experiment=parseq_flash \
    trainer.limit_train_batches=0 2>&1 | tee /tmp/check.log
$ grep -c "Loading pretrained weights" /tmp/check.log
2  # ❌ Should be 1!

# 2. Locate issue (2 minutes)
$ grep -n "Loading pretrained" /tmp/check.log
12:[...23:49:26...] Loading pretrained weights...
15:[...23:49:28...] Loading pretrained weights...
# → 2 seconds apart, likely duplicate instantiation

# 3. Profile (5 minutes)
# Add traceback.format_stack() to TimmBackbone.__init__
$ uv run python scripts/runners/train.py ... | grep -A20 "TimmBackbone"
# → Shows two different call paths

# 4. Identify root cause (1 minute)
$ grep -n "instantiate.*cfg=" ocr/core/models/__init__.py
11:    return hydra.utils.instantiate(architectures, cfg=config)
# → Found it!

# 5. Fix (30 seconds)
- return hydra.utils.instantiate(architectures, cfg=config)
+ return hydra.utils.instantiate(architectures)

# 6. Verify (30 seconds)
$ grep -c "Loading pretrained weights" /tmp/fixed.log
1  # ✅ Fixed!

# Total time: ~10 minutes
```

---

## Automated Check Script

Save as `scripts/audit/performance_sanity_check.sh`:

```bash
#!/bin/bash
# Quick performance sanity check

echo "🔍 Performance Sanity Check"

# Run minimal training
uv run python scripts/runners/train.py experiment=parseq_flash \
  trainer.limit_train_batches=0 2>&1 > /tmp/perf_check.log

# Check singleton resources
weights=$(grep -c "Loading pretrained weights" /tmp/perf_check.log)
tokenizer=$(grep -c "Loaded tokenizer" /tmp/perf_check.log)

echo "Model weights: $weights (expected: 1)"
echo "Tokenizer: $tokenizer (expected: 1)"

# Check mode-aware loading
splits=$(grep "Creating datasets for splits:" /tmp/perf_check.log | grep -o '\[.*\]')
echo "Dataset splits: $splits"

# Exit with error if issues found
if [ "$weights" -gt 1 ] || [ "$tokenizer" -gt 1 ]; then
    echo "❌ Performance issues detected!"
    exit 1
fi

echo "✅ Performance check passed"
exit 0
```

**Usage**:
```bash
chmod +x scripts/audit/performance_sanity_check.sh
./scripts/audit/performance_sanity_check.sh
```

---

## Integration

```yaml
companion_standards:
  full_methodology: AgentQMS/specs/tier2-framework/performance-audit.spec.md (FW-024)
  patterns: AgentQMS/specs/tier2-framework/patterns.spec.md (FW-018)
usage_scenarios:
  pre_commit: Run checklist before committing init code changes
  code_review: Verify no performance regressions
  ci_cd: Automated performance gate in pipeline
  debugging: Quick diagnostic when startup is slow
triggers:
  - performance
  - checklist
  - quick
  - sanity
  - pre-commit
compliance:
  agentqms_validated: '2026-02-16'
  example_implementation: specs/002-training-performance-optimization
  ai_optimized: true
  check_duration: 5-10 minutes
```

