---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
title: "Hydra Configuration Architecture Restructuring Implementation Plan"
date: "2026-01-08 03:59 (KST)"
version: "1.0"
tags: "hydra, configuration, refactoring, architecture, agent-debug-toolkit"
description: "Executable plan to restructure Hydra configuration system from 107 files/37 directories to ~50 files/15 directories with domain-first organization"
source_walkthrough: "docs/artifacts/walkthroughs/2026-01-05_0441_Hydra-Config-Architecture-Restructuring.md"
complexity: "high"
estimated_effort: "8-12 hours"
requires_stakeholder_approval: true
---

# Implementation Plan - Hydra Configuration Architecture Restructuring

## Executive Summary

This plan restructures the Hydra configuration system from **107 YAML files across 37 directories** down to approximately **50 files in 15 directories**. The refactor introduces domain-first organization, eliminates duplication, and simplifies entry points while maintaining backward compatibility where possible.

**Source Walkthrough**: [2026-01-05_0441_Hydra-Config-Architecture-Restructuring.md](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/artifacts/walkthroughs/2026-01-05_0441_Hydra-Config-Architecture-Restructuring.md)

---

## Prerequisites

### Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify uv is available
uv --version

# Sync dependencies
uv sync

# Install agent-debug-toolkit if not already installed
uv pip install -e agent-debug-toolkit[all]
```

### Required Tools

- **AgentQMS CLI**: `bin/aqms` (artifact validation, compliance)
- **Agent Debug Toolkit (ADT)**: `adt` command (AST-based code analysis)
- **uv**: Package manager (NEVER use pip directly)

---

## Phase 0: Discovery & Analysis

### Task 0.1: Analyze Current Configuration Dependencies

**Objective**: Map all Hydra entry points and config access patterns before making changes.

```bash
# Find all Hydra decorators (@hydra.main)
uv run adt find-hydra . --output json > analysis/hydra_entry_points.json

# Alternative: markdown format for human review
uv run adt find-hydra . --output markdown > analysis/hydra_entry_points.md

# Find all component instantiation patterns
uv run adt find-instantiations ocr/ --output json > analysis/component_factories.json

# Analyze config access patterns in key modules
uv run adt analyze-config ocr/models/ --output markdown > analysis/model_config_access.md
uv run adt analyze-config runners/ --output markdown > analysis/runner_config_access.md
```

**Expected Outputs**:
- `analysis/hydra_entry_points.json` - All `@hydra.main` decorators with config names
- `analysis/component_factories.json` - Factory patterns (get_*_by_cfg)
- `analysis/model_config_access.md` - Config access patterns

**Verification**: Review outputs to identify hardcoded config paths that need updating.

---

### Task 0.2: Generate Config Flow Documentation

**Objective**: Understand how configs flow through critical files.

```bash
# Explain config flow in training scripts
uv run adt explain-config-flow runners/train.py > analysis/train_config_flow.md
uv run adt explain-config-flow runners/train_kie.py > analysis/train_kie_config_flow.md

# Trace OmegaConf.merge operations
uv run adt trace-merges ocr/models/architecture.py --output markdown > analysis/merge_precedence.md
```

**Expected Outputs**:
- High-level summaries of config flow
- OmegaConf.merge precedence order

**Verification**: Identify which configs override which values.

---

### Task 0.3: Create Baseline Snapshot

```bash
# Create directory structure snapshot
tree configs/ -L 3 > analysis/configs_structure_before.txt

# Count files and directories
find configs/ -name "*.yaml" | wc -l > analysis/file_count_before.txt
find configs/ -type d | wc -l >> analysis/file_count_before.txt

# Test current configuration composition
uv run python -c "
from hydra import compose, initialize
import sys

configs_to_test = ['train', 'test', 'predict', 'synthetic', 'train_kie']
failed = []

with initialize(version_base=None, config_path='configs'):
    for cfg_name in configs_to_test:
        try:
            cfg = compose(config_name=cfg_name)
            print(f'✅ {cfg_name} composes successfully')
        except Exception as e:
            print(f'❌ {cfg_name} failed: {e}')
            failed.append(cfg_name)

if failed:
    print(f'\n⚠️  Failed configs: {failed}')
    sys.exit(1)
" > analysis/baseline_composition_test.log
```

**Expected Outputs**:
- `analysis/configs_structure_before.txt` - Current directory tree
- `analysis/file_count_before.txt` - File counts (should be ~107 files, 37 dirs)
- `analysis/baseline_composition_test.log` - Composition test results

---

## Phase 1: Foundation (Non-Breaking Changes)

### Task 1.1: Create New Directory Structure

```bash
# Create new directories
mkdir -p configs/_foundation
mkdir -p configs/domain
mkdir -p configs/training/callbacks
mkdir -p configs/training/logger
mkdir -p configs/training/profiling
mkdir -p configs/__EXTENDED__/kie_variants
mkdir -p configs/__EXTENDED__/benchmarks
mkdir -p configs/__EXTENDED__/examples
mkdir -p configs/__EXTENDED__/experiments

# Verify creation
ls -la configs/
```

**Verification**: All new directories exist.

---

### Task 1.2: Rename `_base/` to `_foundation/`

```bash
# Rename directory
mv configs/_base configs/_foundation

# Update internal references in foundation files
uv run python scripts/utils/update_config_references.py \
  --pattern "/_base/" \
  --replacement "/_foundation/" \
  --path "configs/_foundation/"
```

**Manual Verification Required**:
- Open each file in `configs/_foundation/`
- Search for any remaining `_base` references
- Update to `_foundation`

---

### Task 1.3: Create `_foundation/defaults.yaml`

```bash
# Copy base.yaml to new location
cp configs/base.yaml configs/_foundation/defaults.yaml
```

**Manual Edit Required**: Open [configs/_foundation/defaults.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/_foundation/defaults.yaml)

Replace:
```yaml
defaults:
  - _base/core
  - _base/data
  # ... other _base references
```

With:
```yaml
defaults:
  - _foundation/core
  - _foundation/data
  # ... other _foundation references
```

---

### Task 1.4: Update README.md

**Manual Edit Required**: Open [configs/README.md](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/README.md)

Add deprecation notices at the top:

```markdown
# Configuration System

> [!WARNING]
> **Migration In Progress**: The configuration system is being restructured.
> - `_base/` → `_foundation/` (renamed)
> - `base.yaml` → `_foundation/defaults.yaml` (relocated)
> - Multiple KIE configs → unified `domain/kie.yaml` (consolidated)

## New Structure (Post-Migration)

```
configs/
├── train.yaml              # Universal training entry
├── eval.yaml               # Testing/evaluation
├── predict.yaml            # Inference
├── _foundation/            # Core composition fragments
├── domain/                 # Multi-domain configs (detection, recognition, kie, layout)
├── model/                  # Architecture components
├── data/                   # Dataset configs
├── training/               # Training infrastructure (callbacks, logger, profiling)
└── __EXTENDED__/           # Edge cases, experiments
```
```

---

## Phase 2: Domain Unification

### Task 2.1: Create Domain Configs

**Create** [configs/domain/detection.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/domain/detection.yaml):

```yaml
# @package _global_

defaults:
  - /model/architectures/detection/dbnet
  - /data/detection
  - _self_

task: detection

model:
  type: detection
  architecture: ${model.architectures.detection.dbnet}

data:
  task_type: detection
```

**Create** [configs/domain/recognition.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/domain/recognition.yaml):

```yaml
# @package _global_

defaults:
  - /model/architectures/recognition/parseq
  - /data/recognition
  - _self_

task: recognition

model:
  type: recognition
  architecture: ${model.architectures.recognition.parseq}

tokenizer:
  charset: ${data.charset}

data:
  task_type: recognition
```

**Create** [configs/domain/kie.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/domain/kie.yaml):

```yaml
# @package _global_

defaults:
  - /model/architectures/kie/baseline
  - /data/kie
  - _self_

task: kie

model:
  type: kie
  use_ocr_features: true
  use_layout_features: true

data:
  task_type: kie
  include_relations: true
```

**Create** [configs/domain/layout.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/domain/layout.yaml):

```yaml
# @package _global_

defaults:
  - /model/architectures/layout/detectron2
  - /data/layout
  - _self_

task: layout_analysis

model:
  type: layout

data:
  task_type: layout
```

---

### Task 2.2: Verify Domain Configs Compose

```bash
# Test each domain config
uv run python -c "
from hydra import compose, initialize
from omegaconf import OmegaConf

with initialize(version_base=None, config_path='configs'):
    for domain in ['detection', 'recognition', 'kie', 'layout']:
        try:
            cfg = compose(config_name='train', overrides=[f'+domain={domain}'])
            print(f'✅ domain={domain} composes')
            print(f'   Task: {cfg.task}')
        except Exception as e:
            print(f'❌ domain={domain} failed: {e}')
"
```

**Expected Output**: All domains compose successfully.

---

## Phase 3: Entry Point Simplification

### Task 3.1: Update train.yaml

**Manual Edit Required**: Open [configs/train.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train.yaml)

Add domain composition at the top of defaults:

```yaml
defaults:
  - _foundation/defaults
  - domain: detection  # Default domain
  - override hydra/launcher: local
  - _self_

# Enable domain switching via CLI:
# python runners/train.py +domain=recognition
# python runners/train.py +domain=kie
```

---

### Task 3.2: Rename test.yaml → eval.yaml

```bash
# Rename file
mv configs/test.yaml configs/eval.yaml

# Update any references in scripts (use ADT to find them)
uv run adt intelligent-search "config_name.*test" --root runners/
```

**Manual Updates Required**:
- Update any `@hydra.main(config_name="test")` to `config_name="eval"`
- Search for hardcoded "test.yaml" strings

---

### Task 3.3: Archive KIE Variant Configs

```bash
# Move KIE variants to extended
mv configs/train_kie.yaml configs/__EXTENDED__/kie_variants/
mv configs/train_kie_aihub.yaml configs/__EXTENDED__/kie_variants/
mv configs/train_kie_aihub_only.yaml configs/__EXTENDED__/kie_variants/
mv configs/train_kie_aihub_production.yaml configs/__EXTENDED__/kie_variants/
mv configs/train_kie_baseline_optimized_v2.yaml configs/__EXTENDED__/kie_variants/
mv configs/train_kie_merged_3090_10ep.yaml configs/__EXTENDED__/kie_variants/
mv configs/train_parseq.yaml configs/__EXTENDED__/kie_variants/

# Create index file explaining the archive
cat > configs/__EXTENDED__/kie_variants/README.md << 'EOF'
# KIE Configuration Variants (Archived)

These configs have been archived in favor of the unified domain-based approach.

## Migration Guide

**Old Pattern**:
```bash
python runners/train.py --config-name train_kie_aihub
```

**New Pattern**:
```bash
python runners/train.py +domain=kie data=kie/aihub
```

## Archived Configs

- `train_kie.yaml` → Use: `+domain=kie`
- `train_kie_aihub.yaml` → Use: `+domain=kie data=kie/aihub`
- `train_parseq.yaml` → Use: `+domain=recognition model/architectures=recognition/parseq`

EOF
```

---

## Phase 4: Infrastructure Reorganization

### Task 4.1: Move Training Infrastructure

```bash
# Move callbacks
mv configs/callbacks configs/training/

# Move logger
mv configs/logger configs/training/

# Move performance configs
mv configs/performance_test.yaml configs/training/profiling/performance.yaml
mv configs/cache_performance_test.yaml configs/training/profiling/cache_performance.yaml
```

---

### Task 4.2: Merge UI Directories

```bash
# Check if ui_meta has unique files
diff -r configs/ui/ configs/ui_meta/

# If no conflicts, merge
cp -r configs/ui_meta/* configs/ui/

# Remove ui_meta
rm -rf configs/ui_meta/
```

**Manual Verification**: Check that no configs were lost.

---

### Task 4.3: Move Examples and Benchmarks

```bash
# Move to extended
mv configs/examples configs/__EXTENDED__/
mv configs/benchmark configs/__EXTENDED__/benchmarks/

# Create README for extended configs
cat > configs/__EXTENDED__/README.md << 'EOF'
# Extended Configurations

This directory contains edge cases, experiments, and archived configurations that are not part of the core system.

## Structure

- `benchmarks/` - Performance benchmarking configs
- `examples/` - Example configurations for testing
- `experiments/` - Experimental features and ablations
- `kie_variants/` - Legacy KIE configuration variants (archived)

## Usage

These configs are still functional but not actively maintained. Use for reference or special cases only.
EOF
```

---

## Phase 5: Script Updates

### Task 5.1: Find All Hydra Entry Points

```bash
# Use ADT to find all @hydra.main decorators
uv run adt find-hydra . --output json > analysis/hydra_decorators.json

# Pretty print for review
uv run python -c "
import json
with open('analysis/hydra_decorators.json') as f:
    data = json.load(f)
    for entry in data.get('hydra_main_decorators', []):
        print(f\"{entry['file']}:{entry['line']} -> config_name='{entry['config_name']}'\")
"
```

**Expected Output**: List of all files with `@hydra.main` and their config names.

---

### Task 5.2: Update Runner Scripts

**For each file in `runners/` that uses a moved config:**

Example for [runners/train_kie.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train_kie.py):

**Before**:
```python
@hydra.main(config_path="../configs", config_name="train_kie")
```

**After**:
```python
@hydra.main(config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Add domain override at runtime if not provided
    if 'domain' not in cfg or cfg.domain != 'kie':
        from omegaconf import OmegaConf
        domain_cfg = OmegaConf.load("configs/domain/kie.yaml")
        cfg = OmegaConf.merge(cfg, domain_cfg)
```

**Alternative Approach** (simpler): Update CLI documentation instead:

```bash
# Old: python runners/train_kie.py
# New: python runners/train.py +domain=kie
```

---

### Task 5.3: Update Test/Evaluation Scripts

```bash
# Find all references to "test" config
grep -r "config_name.*test" runners/ scripts/

# Update runners/test.py
sed -i 's/config_name="test"/config_name="eval"/' runners/test.py
```

**Manual Verification**: Check that the sed command worked correctly.

---

### Task 5.4: Update Performance Benchmark Scripts

**Update** [scripts/performance/benchmark_optimizations.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/performance/benchmark_optimizations.py):

Find:
```python
@hydra.main(config_path="../../configs", config_name="performance_test")
```

Replace with:
```python
@hydra.main(config_path="../../configs", config_name="training/profiling/performance")
```

**Update** [scripts/performance/decoder_benchmark.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/performance/decoder_benchmark.py):

Find:
```python
@hydra.main(config_path="../../configs", config_name="benchmark/decoder")
```

Replace with:
```python
@hydra.main(config_path="../../configs", config_name="__EXTENDED__/benchmarks/decoder")
```

---

## Phase 6: Verification & Validation

### Task 6.1: Hydra Composition Tests

```bash
# Test all entry points
uv run python -c "
from hydra import compose, initialize
import sys

configs_to_test = [
    'train',
    'eval',
    'predict',
    'synthetic',
]

domain_tests = [
    ('train', '+domain=detection'),
    ('train', '+domain=recognition'),
    ('train', '+domain=kie'),
    ('train', '+domain=layout'),
]

failed = []

with initialize(version_base=None, config_path='configs'):
    print('=== Testing Base Configs ===')
    for cfg_name in configs_to_test:
        try:
            cfg = compose(config_name=cfg_name)
            print(f'✅ {cfg_name}')
        except Exception as e:
            print(f'❌ {cfg_name}: {e}')
            failed.append(cfg_name)

    print('\n=== Testing Domain Compositions ===')
    for cfg_name, override in domain_tests:
        try:
            cfg = compose(config_name=cfg_name, overrides=[override])
            print(f'✅ {cfg_name} {override}')
        except Exception as e:
            print(f'❌ {cfg_name} {override}: {e}')
            failed.append(f'{cfg_name} {override}')

if failed:
    print(f'\n⚠️  Failed: {failed}')
    sys.exit(1)
else:
    print('\n🎉 All compositions successful!')
" | tee analysis/composition_test_after.log
```

---

### Task 6.2: Smoke Test Training

```bash
# Detection (default)
uv run python runners/train.py trainer.fast_dev_run=true trainer.max_epochs=1

# Recognition
uv run python runners/train.py +domain=recognition trainer.fast_dev_run=true

# KIE
uv run python runners/train.py +domain=kie trainer.fast_dev_run=true

# Layout
uv run python runners/train.py +domain=layout trainer.fast_dev_run=true
```

**Expected**: All commands complete without errors.

---

### Task 6.3: Validate with AgentQMS

```bash
# Run AgentQMS validation
cd AgentQMS/bin && make validate

# Check compliance
cd AgentQMS/bin && make compliance
```

**Expected**: No critical violations.

---

### Task 6.4: Compare Before/After Metrics

```bash
# Count files after migration
tree configs/ -L 3 > analysis/configs_structure_after.txt
find configs/ -name "*.yaml" | wc -l > analysis/file_count_after.txt
find configs/ -type d | wc -l >> analysis/file_count_after.txt

# Generate comparison report
uv run python -c "
before_files = int(open('analysis/file_count_before.txt').readline().strip())
before_dirs = int(open('analysis/file_count_before.txt').readlines()[1].strip())
after_files = int(open('analysis/file_count_after.txt').readline().strip())
after_dirs = int(open('analysis/file_count_after.txt').readlines()[1].strip())

print('=== Migration Metrics ===')
print(f'YAML Files: {before_files} → {after_files} ({after_files - before_files:+d})')
print(f'Directories: {before_dirs} → {after_dirs} ({after_dirs - before_dirs:+d})')
print(f'File Reduction: {100 * (before_files - after_files) / before_files:.1f}%')
print(f'Directory Reduction: {100 * (before_dirs - after_dirs) / before_dirs:.1f}%')

# Check targets
targets_met = []
if after_files < 60:
    targets_met.append('✅ Files < 60')
else:
    targets_met.append(f'❌ Files {after_files} >= 60')

if after_dirs < 20:
    targets_met.append('✅ Directories < 20')
else:
    targets_met.append(f'❌ Directories {after_dirs} >= 20')

print('\n=== Target Verification ===')
for target in targets_met:
    print(target)
" | tee analysis/metrics_comparison.txt
```

---

## Phase 7: Documentation & Cleanup

### Task 7.1: Update Main Documentation

**Update** [README.md](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/README.md) or relevant docs:

Add migration notes:

```markdown
## Configuration System

The project uses Hydra for configuration management with a domain-first architecture.

### Training Different Domains

```bash
# Text Detection (default)
python runners/train.py

# Text Recognition
python runners/train.py +domain=recognition

# Key Information Extraction
python runners/train.py +domain=kie

# Layout Analysis
python runners/train.py +domain=layout
```

### Configuration Structure

- `configs/train.yaml` - Universal training entry point
- `configs/domain/` - Domain-specific configs (detection, recognition, kie, layout)
- `configs/model/` - Model architecture components
- `configs/data/` - Dataset configurations
- `configs/training/` - Training infrastructure (callbacks, logging, profiling)
```

---

### Task 7.2: Create Migration Cheatsheet

**Create** `configs/MIGRATION_GUIDE.md`:

```markdown
# Configuration Migration Guide

## Quick Reference

| Old Command                      | New Command                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `--config-name train_kie`        | `+domain=kie`                                                |
| `--config-name train_parseq`     | `+domain=recognition model/architectures=recognition/parseq` |
| `--config-name test`             | `--config-name eval`                                         |
| `--config-name performance_test` | `--config-name training/profiling/performance`               |

## Finding Archived Configs

Configs not deleted but moved to `__EXTENDED__/`:

- KIE variants: `configs/__EXTENDED__/kie_variants/`
- Benchmarks: `configs/__EXTENDED__/benchmarks/`
- Examples: `configs/__EXTENDED__/examples/`

## Breaking Changes

1. **Config Names**: Some entry points renamed (test → eval)
2. **Directory Paths**: `_base/` → `_foundation/`, callbacks moved to `training/`
3. **Domain Switching**: Use `+domain=X` instead of separate config files

## Troubleshooting

### "Config not found" error

Check if config moved to `__EXTENDED__/`. Use full path:
```bash
python script.py --config-name __EXTENDED__/benchmarks/decoder
```

### "Override not applied" error

Ensure domain override comes before other overrides:
```bash
# Correct
python runners/train.py +domain=kie model.lr=0.001

# Wrong
python runners/train.py model.lr=0.001 +domain=kie
```
```

---

### Task 7.3: Remove Old base.yaml

```bash
# Only after verifying _foundation/defaults.yaml works
rm configs/base.yaml

# Update any remaining references
grep -r "base.yaml" configs/ docs/
```

---

## Phase 8: CI/CD Updates

### Task 8.1: Update GitHub Actions

**If CI configs exist**, update them:

```bash
# Find workflow files
find .github/workflows -name "*.yml"
```

**Update any references to**:
- `configs/test.yaml` → `configs/eval.yaml`
- `configs/train_kie.yaml` → Use `configs/train.yaml +domain=kie`

---

### Task 8.2: Update Docker Configs

```bash
# Check Docker configurations
grep -r "train_kie\|test.yaml\|base.yaml" docker/
```

**Update** any found references to use new config names.

---

## Rollback Plan

### If Something Breaks

```bash
# Restore from git (assuming work is in a branch)
git checkout main -- configs/

# Or restore specific files
git checkout main -- configs/train_kie.yaml
```

### Git Branch Strategy

**Before starting**:
```bash
git checkout -b refactor/hydra-config-restructure
git push -u origin refactor/hydra-config-restructure
```

**Commit strategy**:
- Phase 1: Foundation changes (commit)
- Phase 2: Domain configs (commit)
- Phase 3: Entry points (commit)
- Phase 4: Infrastructure (commit)
- Phase 5: Script updates (commit)

**Each commit should pass**: `uv run python -c "from hydra import compose, initialize; ..."`

---

## Success Criteria

| Metric                | Target    | Verification Command                    |
| --------------------- | --------- | --------------------------------------- |
| Total YAML files      | < 60      | `find configs/ -name "*.yaml" \| wc -l` |
| Directory count       | < 20      | `find configs/ -type d \| wc -l`        |
| Root-level configs    | ≤ 5       | `ls configs/*.yaml \| wc -l`            |
| All compositions pass | 100%      | Run verification script                 |
| Smoke tests pass      | 100%      | Run training with `fast_dev_run=true`   |
| AgentQMS validation   | No errors | `cd AgentQMS/bin && make validate`      |

---

## Stakeholder Sign-Off Checklist

Before executing, confirm:

- [ ] Proposed structure approved by team
- [ ] Migration strategy reviewed
- [ ] Breaking changes acknowledged
- [ ] Decision on unused configs (archive vs. delete) finalized
- [ ] CI/CD pipeline updates planned
- [ ] Documentation updates assigned
- [ ] Rollback plan understood

---

## Agent Debug Toolkit (ADT) Command Reference

### Discovery Commands

```bash
# Find all Hydra entry points
uv run adt find-hydra <path> [--output json|markdown]

# Find config access patterns
uv run adt analyze-config <path> [--component <name>] [--output json|markdown]

# Find component instantiation patterns
uv run adt find-instantiations <path> [--component <type>] [--output json|markdown]

# Trace OmegaConf.merge order
uv run adt trace-merges <file> [--explain] [--output json|markdown]

# Explain config flow
uv run adt explain-config-flow <file>

# Run full analysis
uv run adt full-analysis <path>

# Generate context tree
uv run adt context-tree <path> [--depth N] [--output json|markdown]

# Intelligent symbol search
uv run adt intelligent-search <query> [--root <path>] [--fuzzy] [--threshold 0.6]
```

### When to Use ADT

1. **Before moving files**: Find all references with `intelligent-search`
2. **When updating decorators**: Use `find-hydra` to locate all `@hydra.main`
3. **When debugging composition**: Use `trace-merges` to see precedence
4. **When analyzing impact**: Use `analyze-config` to see access patterns
5. **When documenting**: Use `explain-config-flow` for high-level summaries

---

## Resources

- **AgentQMS Standards**: `AgentQMS/standards/INDEX.yaml`
- **Tool Catalog**: `AgentQMS/standards/tier2-framework/tool-catalog.yaml`
- **AGENTS.yaml**: `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/AGENTS.yaml`
- **ADT Documentation**: `agent-debug-toolkit/README.md`
- **Source Walkthrough**: [docs/artifacts/walkthroughs/2026-01-05_0441_Hydra-Config-Architecture-Restructuring.md](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/artifacts/walkthroughs/2026-01-05_0441_Hydra-Config-Architecture-Restructuring.md)

---

## Notes for Web Workers

This plan is designed to be executed by autonomous agents. Each task:

1. **Has clear objectives**: What to accomplish
2. **Provides commands**: Exact bash/Python commands to run
3. **Specifies verification**: How to confirm success
4. **Includes context**: Why the task matters

**Execution Strategy**:
- Run tasks sequentially (phases have dependencies)
- Commit after each phase
- Run verification commands before proceeding
- Use ADT tools for discovery and validation
- Flag any failures for human review

**Safety**:
- All file moves preserve originals in git history
- `__EXTENDED__/` keeps archived configs accessible
- Rollback plan available if needed
- Test composition after every phase
