---
ads_version: '2.0'
id: 'FW-024'
type: 'methodology'
tier: 2
priority: 'medium'
updated: '2026-02-16'
spec_version: '1.0.0'
description: 'Performance audit methodology for detecting initialization bottlenecks and resource duplication'
---

# Performance Audit Methodology

> Systematic approach for detecting duplicate loading, redundant instantiation, and initialization inefficiencies

## Specification

```yaml
agent: all
name: Performance Audit Methodology
version: 1.0
last_updated: '2026-02-16'
spec_version: '1.0.0'
when_to_use:
  slow_startup: Apply when initialization takes > 5 seconds
  duplicate_logs: Multiple identical loading messages
  memory_overhead: Unexpected memory usage during startup
  resource_investigation: Auditing component initialization
auto_load: false
triggers:
  - performance
  - slow
  - duplicate
  - loading
  - initialization
  - bottleneck
related_specs:
  patterns: AgentQMS/specs/tier2-framework/patterns.spec.md (FW-018)
  checklist: AgentQMS/specs/tier2-framework/performance-checklist.spec.md (FW-025)
investigation_time: 30-60 minutes
expected_improvement: 20-40% startup time reduction
```

## Investigation Steps

### Step 1: Duplicate Loading Detection (10 min)

**Objective**: Quantify resource loading frequency

**Commands**:
```bash
# Run training with minimal batches
uv run python scripts/runners/train.py experiment=<name> \
  trainer.limit_train_batches=0 2>&1 | tee /tmp/init_profile.log

# Count specific resource loads
grep -c "Loading pretrained weights" /tmp/init_profile.log
grep -c "Loaded tokenizer" /tmp/init_profile.log
grep -c "Creating datasets" /tmp/init_profile.log
```

**Expected Results**:
- Model weights: 1 per model component
- Tokenizer: 1 total (singleton)
- Datasets: Count matches mode (train: 2, eval: 1, test: 1)

**Red Flags**:
- Any count > 1 for singleton resources
- Dataset count doesn't match mode
- Same log message with different timestamps

### Step 2: Stack Trace Profiling (15 min)

**Objective**: Identify duplicate instantiation call paths

**Method**: Add temporary profiling to suspected components:

```python
import traceback
import logging

logger = logging.getLogger(__name__)

class SuspectedComponent:
    def __init__(self, ...):
        logger.info("=" * 60)
        logger.info(f"{self.__class__.__name__}.__init__ called")
        logger.info("Stack trace:")
        logger.info("".join(traceback.format_stack()))
        logger.info("=" * 60)
        # ... rest of initialization
```

**Analysis**:
- Compare stack traces for duplicates
- Look for different call paths (direct vs nested)
- Identify common ancestor (model factory, config loader, etc.)
- Check for `instantiate_node` or `cfg[key] =` in traces

**Common Patterns**:
```
# Pattern 1: Double instantiation via redundant parameter
First:  get_model_by_cfg → instantiate(architectures) → Component.__init__
Second: instantiate_node(cfg[key]) → Component.__init__

# Pattern 2: Missing singleton
First:  setup_tokenizer → Tokenizer.__init__
Second: preprocess_batch → Tokenizer.__init__

# Pattern 3: No mode filtering
First:  load_train_split → Dataset.__init__
Second: load_val_split → Dataset.__init__
Third:  load_test_split → Dataset.__init__  # ❌ Not needed in train mode
```

### Step 3: Hydra Instantiation Audit (5 min)

**Objective**: Find potential double instantiation patterns

**Search Commands**:
```bash
# Find hydra.utils.instantiate calls with extra parameters
grep -rn "hydra.utils.instantiate.*cfg=" ocr/

# Check for passing config objects
grep -rn "instantiate.*config\)" ocr/core/models/

# Verify no redundant component definitions
grep -rn "_target_.*Encoder\|_target_.*Tokenizer" configs/
```

**Anti-Pattern Detection**:
```python
# ❌ BAD: Redundant cfg parameter
return hydra.utils.instantiate(architectures, cfg=config)
# If both architectures and config contain encoder._target_ → double load

# ✅ GOOD: Single source of truth
return hydra.utils.instantiate(architectures)
```

### Step 4: Singleton Pattern Check (10 min)

**Objective**: Verify singleton pattern for shared resources

**Audit Checklist**:
```bash
# Find tokenizer instantiation points
grep -rn "Tokenizer(" ocr/ --include="*.py"

# Check for caching patterns
grep -rn "@lru_cache\|_instance\|get_or_create" ocr/

# Verify no instantiation in loops
grep -rn "for.*Tokenizer\|while.*Tokenizer" ocr/
```

**Required Patterns**:
```python
# ✅ Singleton with class variable
class Tokenizer:
    _instance = None

    @classmethod
    def get_or_create(cls, vocab_path):
        if cls._instance is None:
            cls._instance = cls(vocab_path)
        return cls._instance

# ✅ LRU cache for expensive operations
from functools import lru_cache

@lru_cache(maxsize=1)
def load_tokenizer(vocab_path: str):
    return Tokenizer(vocab_path)
```

### Step 5: Mode-Aware Loading Check (5 min)

**Objective**: Verify resources loaded only when needed

**Verification**:
```bash
# Check dataset split loading
grep "Creating datasets" /tmp/init_profile.log

# Expected output by mode:
# train:   Creating datasets for splits: ['train', 'val']
# eval:    Creating datasets for splits: ['val']
# test:    Creating datasets for splits: ['test']
# predict: Creating datasets for splits: ['predict']
```

**Code Audit**:
```bash
# Find hardcoded split lists
grep -rn "splits.*=.*\[.*train.*val.*test" ocr/

# Check for mode-aware logic
grep -rn "def.*get.*splits\|_get_required_splits" ocr/pipelines/
```

**Required Pattern**:
```python
def _get_required_splits(self, mode: str) -> list[str]:
    """Return only required splits for the current mode."""
    mode_to_splits = {
        "train": ["train", "val"],
        "eval": ["val"],
        "test": ["test"],
        "predict": ["predict"]
    }
    return mode_to_splits.get(mode, ["train", "val"])
```

### Step 6: Timing Analysis (5 min)

**Objective**: Quantify time wasted on duplicates

**Method**:
```bash
# Extract timestamps for operations
grep "Loading pretrained\|Tokenizer\|datasets" /tmp/init_profile.log | \
  awk '{print $1, $2, substr($0, index($0, $3))}' > /tmp/timeline.txt

# Calculate time gaps
python -c "
import re
from datetime import datetime

with open('/tmp/timeline.txt') as f:
    lines = f.readlines()

timestamps = []
for line in lines:
    match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\]', line)
    if match:
        ts = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f')
        timestamps.append((ts, line.strip()))

# Report gaps > 0.5s
for i in range(1, len(timestamps)):
    gap = (timestamps[i][0] - timestamps[i-1][0]).total_seconds()
    if gap < 3:  # Same operation if within 3s
        print(f'Gap: {gap:.2f}s')
        print(f'  {timestamps[i-1][1]}')
        print(f'  {timestamps[i][1]}')
"
```

**Red Flags**:
- Duplicate operations 1-3 seconds apart
- Total init time > 10 seconds for simple models
- Multiple tokenizer loads across 5+ seconds

## Common Anti-Patterns

### Anti-Pattern 1: Redundant cfg Parameter

**Detection**: `grep -rn "instantiate.*cfg=" ocr/`

**Problem**:
```python
# Both architectures and config contain encoder definition
return hydra.utils.instantiate(architectures, cfg=config)
# → Hydra instantiates encoder twice
```

**Fix**:
```python
# Remove redundant parameter
return hydra.utils.instantiate(architectures)
```

**Verification**: `grep -c "Loading pretrained" <log>` should return 1

### Anti-Pattern 2: Missing Singleton

**Detection**: Count > 1 tokenizer loads

**Problem**:
```python
class Tokenizer:
    def __init__(self, vocab_path):
        self.vocab = self._load_vocab(vocab_path)  # Expensive!

# Called multiple times
tokenizer1 = Tokenizer(path)
tokenizer2 = Tokenizer(path)  # ❌ Duplicate load
```

**Fix**:
```python
class Tokenizer:
    _instance = None

    @classmethod
    def get_or_create(cls, vocab_path):
        if cls._instance is None:
            cls._instance = cls(vocab_path)
        return cls._instance
```

**Verification**: `grep -c "Loaded tokenizer" <log>` should return 1

### Anti-Pattern 3: Eager Loading

**Detection**: All splits load regardless of mode

**Problem**:
```python
# Always loads all splits
splits = ["train", "val", "test", "predict"]
datasets = {split: load_dataset(split) for split in splits}
```

**Fix**:
```python
# Mode-aware loading
required_splits = self._get_required_splits(self.mode)
datasets = {split: load_dataset(split) for split in required_splits}
```

**Verification**: Check log shows only required splits

### Anti-Pattern 4: Nested Instantiation Conflict

**Detection**: Stack traces show `cfg[key] = instantiate_node`

**Problem**:
```yaml
# Config defines encoder at multiple levels
architectures:
  _target_: Model
  encoder:
    _target_: Encoder  # First definition

encoder:
  _target_: Encoder    # Second definition (redundant)
```

**Fix**: Remove redundant definition, keep only one location

## Deliverables

### 1. Findings Document

Template: `specs/XXX-performance-optimization/findings.md`

**Required Sections**:
- Phase description (Tokenizer / Model / Dataset / etc.)
- Root cause analysis
- Before/after metrics (load counts, timing)
- Fix description with file paths
- Verification commands
- Commit reference

### 2. Verification Report

**Before Fix**:
```bash
# Count loads
grep -c "Loading pretrained weights" /tmp/before.log  # 2
grep -c "Loaded tokenizer" /tmp/before.log            # 5
grep "Creating datasets" /tmp/before.log              # ['train','val','test']

# Timing
Initialization time: 12.3 seconds
```

**After Fix**:
```bash
# Count loads
grep -c "Loading pretrained weights" /tmp/after.log   # 1 ✅
grep -c "Loaded tokenizer" /tmp/after.log             # 1 ✅
grep "Creating datasets" /tmp/after.log               # ['train','val'] ✅

# Timing
Initialization time: 8.7 seconds  # 29% improvement ✅
```

### 3. Commit

**Format**:
```
perf: eliminate [resource] duplication

Fixed [description of root cause].

Result: [Before → After metrics]

Related: spec XXX-performance-optimization Phase N
```

**Example**:
```
perf: eliminate model weight duplication (Phase 3 complete)

Fixed double instantiation of encoder in model factory by removing
redundant cfg parameter from hydra.utils.instantiate() call.

Root cause: Both architectures and cfg contained encoder definitions,
causing Hydra to instantiate the encoder twice.

Result: Model weight loads reduced from 2→1

Related: spec 002-training-performance-optimization Phase 3
```

## Quick Reference: Common Grep Patterns

```bash
# Count resource loads
grep -c "Loading pretrained weights" <log>
grep -c "Loaded tokenizer" <log>
grep -c "Creating datasets" <log>

# Find instantiation patterns
grep -rn "hydra.utils.instantiate" ocr/core/
grep -rn "_target_.*Encoder\|_target_.*Tokenizer" configs/

# Check for singletons
grep -rn "get_or_create\|@lru_cache\|_instance" ocr/

# Find eager loading
grep -rn 'splits.*=.*\["train".*"val".*"test"\]' ocr/

# Architecture purity check
grep -r "optimizer:\|loss:" configs/model/architectures/
```

## Success Criteria

✅ **Complete Audit**:
- All singleton resources load exactly once
- Mode-specific resources match mode requirements
- No duplicate stack traces for same component
- Init time reduced by 20-40%
- All fixes documented and verified

✅ **Quality Standards**:
- Before/after metrics captured
- Root cause clearly identified
- Fix is minimal and targeted
- Verification commands provided
- Knowledge documented for future

## Integration

```yaml
companion_standards:
  patterns: AgentQMS/specs/tier2-framework/patterns.spec.md (FW-018)
  checklist: AgentQMS/specs/tier2-framework/performance-checklist.spec.md (FW-025)
triggers:
  - performance
  - slow
  - duplicate
  - loading
  - initialization
compliance:
  agentqms_validated: '2026-02-16'
  example_spec: specs/002-training-performance-optimization
  ai_optimized: true
```

