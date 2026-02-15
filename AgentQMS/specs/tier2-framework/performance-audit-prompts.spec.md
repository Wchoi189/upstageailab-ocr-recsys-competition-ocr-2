---
ads_version: '2.0'
id: 'FW-026'
type: 'prompt_templates'
tier: 2
priority: 'low'
updated: '2026-02-16'
spec_version: '1.0.0'
description: 'Example prompts for triggering performance audits with AI agents'
---

# Performance Audit Prompt Templates

> Ready-to-use prompt templates for triggering systematic performance investigations

## Specification

```yaml
agent: all
name: Performance Audit Prompts
version: 1.0
last_updated: '2026-02-16'
spec_version: '1.0.0'
when_to_use:
  ai_agent_invocation: When requesting AI to perform performance audit
  systematic_investigation: Structured performance analysis
  issue_reproduction: Investigating reported performance problems
auto_load: false
related_specs:
  methodology: AgentQMS/specs/tier2-framework/performance-audit.spec.md (FW-024)
  checklist: AgentQMS/specs/tier2-framework/performance-checklist.spec.md (FW-025)
  patterns: AgentQMS/specs/tier2-framework/patterns.spec.md (FW-018)
```

## General Performance Audit

**Use Case**: Comprehensive performance investigation

```
Perform an initialization performance audit following the methodology in
AgentQMS/specs/tier2-framework/performance-audit.spec.md (FW-024).

Focus on:
1. Duplicate resource loading (weights, tokenizers, datasets)
2. Hydra instantiation patterns (check for redundant cfg parameters)
3. Mode-aware loading (verify splits match mode)

Deliverables:
- Findings document in specs/XXX-performance-audit/findings.md
- Before/after metrics for each issue found
- Commits with perf: prefix
- Verification commands showing improvements

Target: 20-40% reduction in initialization time
```

---

## Targeted Component Audit

**Use Case**: Investigating specific component performance

```
Investigate [tokenizer/model/encoder/dataset] loading efficiency:

Steps:
1. Count load operations (should be 1 for singletons)
2. If count > 1, add stack trace profiling to __init__
3. Compare call paths to identify duplication source
4. Check for singleton patterns (@lru_cache, get_or_create, _instance)
5. Verify mode-aware loading where applicable

Reference: AgentQMS/specs/tier2-framework/performance-checklist.spec.md (FW-025)

Expected deliverable:
- Root cause identified
- Fix implemented (singleton pattern or remove duplication)
- Verification: grep -c shows count == 1
```

---

## Hydra Configuration Audit

**Use Case**: Auditing for Hydra double instantiation

```
Audit for Hydra double instantiation anti-patterns:

Search targets:
- ocr/core/models/ (model factories)
- ocr/pipelines/ (orchestration)
- configs/ (nested _target_ conflicts)

Specific checks:
1. grep -rn "hydra.utils.instantiate.*cfg=" ocr/
2. Check for redundant component definitions in configs
3. Verify no nested _target_ conflicts (same component at multiple levels)
4. Reference patterns.spec.md (FW-018) failure_modes.double_component_instantiation

For any matches:
- Add stack trace profiling
- Verify both config sources contain same component
- Remove redundant parameter or config definition
- Verify fix: grep -c "Loading pretrained" should return 1
```

---

## Quick Sanity Check

**Use Case**: Fast spot check (< 5 min)

```
Quick performance sanity check:

1. Run training with trainer.limit_train_batches=0
2. Count singleton resources:
   - grep -c "Loading pretrained weights" (expected: 1)
   - grep -c "Loaded tokenizer" (expected: 1)
3. Verify mode-specific splits:
   - grep "Creating datasets for splits:"
   - Train mode: ['train', 'val']
   - Eval mode: ['val']
   - Test mode: ['test']

Report:
- ✅ if all counts match expectations
- ❌ with specific violations if any count > 1 or wrong splits

If violations found, reference FW-024 for full investigation.
```

---

## Fix Verification

**Use Case**: Confirming performance fix effectiveness

```
Verify that performance fix for [issue] is working:

Before metrics (from issue report):
- [specific count or timing, e.g., "Loading pretrained: 2"]

Test procedure:
1. Run: uv run python scripts/runners/train.py experiment=<name>
   trainer.limit_train_batches=0 2>&1 | tee /tmp/verify.log
2. Count: grep -c "[specific message]" /tmp/verify.log
3. Compare timing: [before time] vs [after time]
4. Run tests: pytest [relevant test path]

Expected results:
- Count reduced to expected value (e.g., 2→1)
- No functionality regression (all tests pass)
- Timing improvement documented

Use verification commands from performance-checklist.spec.md (FW-025)
```

---

## Pattern Recognition

**Use Case**: Proactive anti-pattern detection

```
Search codebase for common performance anti-patterns:

1. Missing singletons:
   grep -rn "class.*Tokenizer" ocr/ --include="*.py" -A10 |
     grep -v "get_or_create\|_instance\|@lru_cache"

2. Eager loading:
   grep -rn 'splits.*=.*\["train".*"val".*"test"\]' ocr/

3. Redundant instantiation:
   grep -rn "hydra.utils.instantiate.*cfg=" ocr/

4. Architecture impurity:
   grep -r "optimizer:\|loss:" configs/model/architectures/

For each match:
- Report file path and line number
- Classify issue type (singleton / eager / redundant / impure)
- Estimate impact (high / medium / low)
- Suggest fix pattern from FW-018

Priority: Fix high-impact issues first (duplicated weights > eager loading)
```

---

## Post-Refactor Audit

**Use Case**: Ensuring refactoring didn't introduce performance regressions

```
After refactoring [component], run full performance audit:

Comparison metrics:
1. Initialization timing:
   - Before: [baseline time]
   - After: [current time]
   - Delta: [difference and %]

2. Resource load counts:
   - Model weights: before vs after
   - Tokenizer: before vs after
   - Datasets: before vs after

3. Run performance-checklist.spec.md (FW-025) items:
   - ☑️ Hydra instantiation (5 min)
   - ☑️ Singleton resources (5 min)
   - ☑️ Mode-aware loading (3 min)
   - ☑️ Component duplication (10 min)
   - ☑️ Config structure (5 min)

Document:
- Any regressions found (with fixes)
- Improvements achieved
- Updated performance baseline
```

---

## Memory Profiling

**Use Case**: Investigating memory overhead during initialization

```
Profile memory usage during initialization:

1. Add memory tracking:
   ```python
   import tracemalloc
   tracemalloc.start()

   # ... initialization code ...

   current, peak = tracemalloc.get_traced_memory()
   print(f"Current: {current / 1024 / 1024:.2f} MB")
   print(f"Peak: {peak / 1024 / 1024:.2f} MB")
   tracemalloc.stop()
   ```

2. Identify memory spikes:
   - Compare before/after each major component
   - Look for unexpected large allocations
   - Check for duplicate object retention

3. Cross-reference with:
   - Resource loading counts (FW-025 checklist)
   - Component instantiation patterns (FW-024 methodology)

Expected issues:
- Duplicate model weights in memory (2x expected size)
- Multiple tokenizer vocabularies (5x expected)
- All dataset splits loaded (3-4x expected)

Fix patterns:
- Singleton for shared resources
- Lazy loading for mode-specific resources
- Remove redundant instantiation
```

---

## Continuous Integration Check

**Use Case**: Automated performance gate in CI/CD

```
Implement CI performance check:

Script: scripts/audit/ci_performance_check.sh

Requirements:
1. Run training with minimal batches (< 1 min)
2. Extract and validate metrics:
   - Model weights: must be 1
   - Tokenizer: must be 1
   - Datasets: must match mode
3. Compare against baseline (if exists)
4. Exit 1 if violations found, 0 if pass

Thresholds:
- Initialization time: < 10 seconds (simple model)
- Memory usage: < baseline * 1.1 (10% tolerance)
- Resource counts: exact matches only

On failure:
- Print specific violations
- Reference FW-024 for investigation
- Block merge until fixed

On success:
- Update baseline if improved
- Log metrics for trending
```

---

## Investigation Documentation

**Use Case**: Standardizing investigation write-ups

```
Document performance investigation:

Template: specs/XXX-performance-optimization/findings.md

Required sections:

## Phase N: [Issue Description]

### Root Cause
- Technical explanation
- Config/code snippet showing problem
- Why it causes performance impact

### Investigation Method
- Detection commands used
- Profiling techniques applied
- Evidence collected (logs, traces, metrics)

### Fix
- File(s) modified with line numbers
- Before/after code comparison
- Rationale for approach

### Results
- Before metrics: [counts, timing]
- After metrics: [counts, timing]
- Improvement: [absolute and %]

### Verification
- Commands to reproduce test
- Expected output
- How to detect regression

### Commit
- Commit hash
- Commit message
- Link to diff

Reference existing example: specs/002-training-performance-optimization/findings.md
```

---

## Prompt Usage Guidelines

### When to Use Each Prompt

| Scenario | Prompt | Time | Priority |
|----------|--------|------|----------|
| Daily commit check | Quick Sanity Check | 5 min | High |
| Reported slow startup | General Performance Audit | 60 min | High |
| After refactoring | Post-Refactor Audit | 30 min | High |
| Component investigation | Targeted Component Audit | 20 min | Medium |
| Code review | Pattern Recognition | 15 min | Medium |
| CI/CD pipeline | Continuous Integration Check | 5 min | High |
| Fix validation | Fix Verification | 10 min | High |
| Memory issues | Memory Profiling | 30 min | Medium |

### Prompt Customization

Replace placeholders:
- `[issue]`: Specific issue description (e.g., "duplicate tokenizer loading")
- `<name>`: Experiment name (e.g., "parseq_flash")
- `[component]`: Component name (e.g., "tokenizer", "model", "encoder")
- `[specific message]`: Log message to search (e.g., "Loading pretrained weights")
- `[baseline time]`: Previous timing measurement

### Expected Outputs

All prompts should produce:
1. **Concrete metrics**: Numbers, not descriptions
2. **Before/after comparison**: Quantifiable improvement
3. **Verification commands**: Reproducible test
4. **Documentation**: Updated findings.md

### Integration with Tools

```bash
# Use with Claude Code
claude code "$(cat AgentQMS/specs/tier2-framework/performance-audit-prompts.spec.md | grep -A20 'General Performance Audit')"

# Use with custom scripts
./scripts/audit/performance_check.sh --type=quick
./scripts/audit/performance_check.sh --type=full --document

# CI/CD integration
pre-commit run performance-check
```

---

## Success Metrics

After applying prompts, validate:

```yaml
resource_loading:
  model_weights: 1  # No duplicates
  tokenizer: 1      # Singleton
  datasets: mode_specific  # Train: 2, Eval: 1, Test: 1

timing:
  initialization: < 10s  # For simple models
  improvement: 20-40%    # From baseline

quality:
  documentation: complete  # All sections filled
  verification: passing    # Commands work
  tests: passing          # No regressions

knowledge:
  anti_patterns: documented  # Added to FW-018
  methodology: refined      # Updated FW-024
  prompts: updated          # This file
```

---

## Integration

```yaml
companion_standards:
  methodology: AgentQMS/specs/tier2-framework/performance-audit.spec.md (FW-024)
  checklist: AgentQMS/specs/tier2-framework/performance-checklist.spec.md (FW-025)
  patterns: AgentQMS/specs/tier2-framework/patterns.spec.md (FW-018)
usage:
  ai_agents: Copy and customize prompts
  human_investigators: Reference for systematic approach
  ci_cd: Automated checks
triggers:
  - prompt
  - template
  - example
  - audit
compliance:
  agentqms_validated: '2026-02-16'
  example_usage: specs/002-training-performance-optimization
  ai_optimized: true
```

