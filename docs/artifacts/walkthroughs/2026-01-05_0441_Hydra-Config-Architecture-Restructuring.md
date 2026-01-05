# Hydra Configuration Architecture Restructuring

Comprehensive redesign of the Hydra configuration system to resolve organizational chaos, schema ambiguity, and multi-domain confusion across Text Detection, Recognition, Layout Analysis, and KIE.

---

## User Review Required

> [!IMPORTANT]
> **Decision Point**: This plan proposes collapsing 107 YAML files across 37 directories down to approximately 50 files in 15 directories. Some existing configs will be relocated to `__EXTENDED__/` rather than deleted.

> [!WARNING]
> **Breaking Change**: Entry point config names will change (e.g., [train_kie.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train_kie.yaml) → Domain-based composition via `+domain=kie`).

**Questions for stakeholder:**
1. Should unused configs ([benchmark/decoder.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/benchmark/decoder.yaml), `examples/*.yaml`) be archived or deleted outright?
2. Is the `ui_meta/` directory actively used, or can it merge with `ui/`?
3. For the 8 `train_*.yaml` variants at root level—how many must remain as distinct entry points vs. become overrides?

---

## Current State Assessment

### Directory Structure Analysis

```
configs/                          # 107 files, 37 dirs
├── README.md                     # Outdated (pre-text-recognition)
├── base.yaml                     # Composition defaults
├── _base/                        # 6 foundation fragments
│   ├── core.yaml, data.yaml, model.yaml, trainer.yaml...
├── train.yaml, test.yaml, predict.yaml, synthetic.yaml  # Core entry points
├── train_kie.yaml, train_kie_aihub.yaml, train_kie_aihub_only.yaml...  # 8 KIE variants!
├── train_parseq.yaml             # Text recognition
├── performance_test.yaml, cache_performance_test.yaml   # Misplaced
├── predict_shadow_removal.yaml   # Feature-specific (misplaced)
├── model/                        # 23 items (well-structured)
│   ├── architectures/{craft,dbnet,dbnetpp,parseq}.yaml
│   ├── decoder/, encoder/, head/, loss/, optimizers/, presets/
├── data/                         # 21 items
│   ├── datasets/, transforms/, dataloaders/, performance_preset/
├── callbacks/, logger/, trainer/, evaluation/, paths/, debug/, hydra/
├── ui/ + ui_meta/                # OVERLAPPING directories
├── extraction/, layout/, recognition/  # Multi-domain (sparse)
├── examples/, benchmark/         # Unclear purpose
└── __LEGACY__/                   # Already archived
```

### Problems Identified

| Category | Problem | Impact |
|----------|---------|--------|
| **Schema Ambiguity** | [base.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/base.yaml) (composition root) vs `_base/` (fragment dir) | AI agents confused on override patterns |
| **Root Sprawl** | 17 YAML files at root level | Hard to find canonical entry points |
| **Multi-Domain Chaos** | KIE, Recognition, Detection scattered | No unified domain structure |
| **Duplication** | `ui/` vs `ui_meta/`; 8x KIE configs | Maintenance burden |
| **Orphaned Configs** | `benchmark/`, `examples/` | Unclear if actively used |
| **Misplaced Files** | [performance_test.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/performance_test.yaml) at root | Should be in `benchmark/` |

### Script-Config Dependencies

| Script Entry Point | Config Name | Config Path |
|-------------------|-------------|-------------|
| [runners/train.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train.py) | `train` | [configs/train.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train.yaml) |
| [runners/train_kie.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train_kie.py) | `train_kie` | [configs/train_kie.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train_kie.yaml) |
| [runners/test.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/test.py) | `test` | [configs/test.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/test.yaml) |
| [runners/predict.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/predict.py) | `predict` | [configs/predict.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/predict.yaml) |
| [runners/generate_synthetic.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/generate_synthetic.py) | `synthetic` | [configs/synthetic.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/synthetic.yaml) |
| [scripts/performance/benchmark_optimizations.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/performance/benchmark_optimizations.py) | `performance_test` | [configs/performance_test.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/performance_test.yaml) |
| [scripts/performance/decoder_benchmark.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/performance/decoder_benchmark.py) | `benchmark/decoder` | [configs/benchmark/decoder.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/benchmark/decoder.yaml) |

---

## Proposed Architecture

### Design Principles

1. **Domain-First Organization**: Group by OCR domain (detection, recognition, kie, layout)
2. **Flat Entry Points**: Max 5 root-level entry configs
3. **Composition over Duplication**: Use Hydra `+domain=X` patterns
4. **Clear Naming**: `_foundation/` for shared fragments, not `_base/`
5. **Archive before Delete**: Move to `__EXTENDED__/` not instant deletion

### New Directory Structure

```
configs/
├── README.md                     # [UPDATE] Comprehensive guide
├── train.yaml                    # Universal training entry point
├── eval.yaml                     # Evaluation/testing (renamed from test.yaml)
├── predict.yaml                  # Inference entry point
├── _foundation/                  # [RENAME from _base/]
│   ├── defaults.yaml             # Global composition defaults (was base.yaml)
│   ├── core.yaml
│   ├── paths.yaml
│   └── trainer.yaml
│
├── domain/                       # [NEW] Multi-domain configs
│   ├── detection.yaml            # Text detection (DBNet, CRAFT)
│   ├── recognition.yaml          # Text recognition (PARSeq)
│   ├── kie.yaml                  # Key Information Extraction
│   └── layout.yaml               # Layout analysis
│
├── model/                        # Component configs (keep)
│   ├── architectures/
│   │   ├── detection/{dbnet,dbnetpp,craft}.yaml
│   │   └── recognition/{parseq}.yaml
│   ├── encoder/, decoder/, head/, loss/
│   ├── optimizers/
│   └── presets/
│
├── data/                         # Data configs (flatten)
│   ├── detection.yaml            # Consolidated detection datasets
│   ├── recognition.yaml          # LMDB-based recognition
│   ├── transforms/               # Keep nested
│   └── dataloaders/              # Keep nested
│
├── training/                     # [NEW] Training infrastructure
│   ├── callbacks/                # Moved from root
│   ├── logger/                   # Moved from root
│   └── profiling/                # Moved performance_test, cache_test
│
├── ui/                           # Merge ui + ui_meta
│   ├── inference.yaml
│   ├── unified_app.yaml
│   └── preprocessing.yaml
│
├── hydra/                        # Keep
│   └── default.yaml
│
├── __EXTENDED__/                 # [NEW] Edge cases, experiments
│   ├── examples/                 # Relocated
│   ├── benchmarks/               # Relocated
│   └── experiments/              # For ablations, hardware-specific
│
└── __LEGACY__/                   # Keep (already exists)
```

### Migration Mapping

| Current Location | New Location | Action |
|-----------------|--------------|--------|
| [base.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/base.yaml) | `_foundation/defaults.yaml` | Rename |
| `_base/*.yaml` | `_foundation/*.yaml` | Rename |
| [train.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train.yaml) | [train.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train.yaml) | Keep |
| [train_kie.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train_kie.yaml) | Use `train.yaml +domain=kie` | Archive original |
| [train_parseq.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train_parseq.yaml) | Use `train.yaml +domain=recognition model/architectures=recognition/parseq` | Archive |
| `train_kie_*.yaml` (7 files) | `__EXTENDED__/kie_variants/` | Relocate |
| [test.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/test.yaml) | `eval.yaml` | Rename |
| `callbacks/`, `logger/` | `training/callbacks/`, `training/logger/` | Move |
| [performance_test.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/performance_test.yaml) | `training/profiling/performance.yaml` | Move |
| `benchmark/` | `__EXTENDED__/benchmarks/` | Move |
| `examples/` | `__EXTENDED__/examples/` | Move |
| `ui/` + `ui_meta/` | `ui/` | Merge |
| `extraction/`, `layout/`, `recognition/` | `domain/` | Consolidate |

---

## Implementation Plan

### Phase 1: Foundation (Non-Breaking)

#### [MODIFY] [README.md](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/README.md)
- Update to document new structure post-migration
- Add deprecation notices for moved configs

#### [NEW] `_foundation/defaults.yaml`
- Copy content from [base.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/base.yaml)
- Update internal references

#### [RENAME] `_base/` → `_foundation/`
- Update all references in existing configs

---

### Phase 2: Domain Unification

#### [NEW] `domain/detection.yaml`
```yaml
defaults:
  - /model/architectures/detection/dbnet
  - /data/detection
  
task: detection
```

#### [NEW] `domain/recognition.yaml`
```yaml
defaults:
  - /model/architectures/recognition/parseq
  - /data/recognition

task: recognition
tokenizer:
  charset: ${data.charset}
```

#### [NEW] `domain/kie.yaml`
- Consolidate the 8 KIE variants into composable overrides

---

### Phase 3: Entry Point Simplification

#### [MODIFY] [train.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train.yaml)
- Add domain composition: `domain: detection` as default
- Support `+domain=recognition` override

#### [MOVE] `train_kie*.yaml` → `__EXTENDED__/kie_variants/`

---

### Phase 4: Infrastructure Reorganization

#### [MOVE] `callbacks/` → `training/callbacks/`
#### [MOVE] `logger/` → `training/logger/`
#### [MOVE] [performance_test.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/performance_test.yaml), [cache_performance_test.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/cache_performance_test.yaml) → `training/profiling/`
#### [MERGE] `ui/` + `ui_meta/` → `ui/`
#### [MOVE] `examples/`, `benchmark/` → `__EXTENDED__/`

---

### Phase 5: Script Updates

Update Hydra decorators in all affected scripts:

#### [MODIFY] [runners/train_kie.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train_kie.py)
```python
# Before
@hydra.main(config_path="../configs", config_name="train_kie")

# After
@hydra.main(config_path="../configs", config_name="train")
# With default override: +domain=kie
```

---

## Verification Plan

### Automated Tests

1. **Hydra Composition Validation**
```bash
# Verify all entry points compose without errors
uv run python -c "
from hydra import compose, initialize
with initialize(version_base=None, config_path='configs'):
    for cfg in ['train', 'eval', 'predict']:
        compose(config_name=cfg)
        print(f'✅ {cfg} composes')
"
```

2. **Existing Override Tests**
```bash
uv run python tests/unit/test_hydra_overrides.py
```

3. **Training Smoke Test**
```bash
# Detection
uv run python runners/train.py trainer.fast_dev_run=true

# Recognition (new pattern)
uv run python runners/train.py +domain=recognition trainer.fast_dev_run=true
```

### Manual Verification

1. **Directory Structure**
   - Confirm new structure matches proposed layout
   - Verify no broken config references

2. **Script Dependencies**
   - Run each script entry point with `--cfg job` to validate composition
   - Check that relocated configs are still discoverable

3. **Documentation Accuracy**
   - Ensure [configs/README.md](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/README.md) matches reality

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Broken script dependencies | Medium | High | Update decorators incrementally, test each |
| Missing config references | Medium | Medium | Search for hardcoded paths before moving |
| User confusion during transition | Low | Medium | Keep `__EXTENDED__/` accessible, add deprecation warnings |
| CI pipeline failures | Low | High | Update CI config paths first |

---

## Success Metrics

| Metric | Before | After | Target Met? |
|--------|--------|-------|-------------|
| Total YAML files | 107 | ~50 | ✓ if <60 |
| Directory count | 37 | ~15 | ✓ if <20 |
| Root-level configs | 17 | 4 | ✓ |
| Orphaned configs | ~15 | 0 | ✓ |

---

## Stakeholder Sign-off

**Approval Required Before Proceeding:**

- [ ] Proposed structure approved
- [ ] Migration strategy approved  
- [ ] Breaking changes acknowledged
- [ ] Answers to decision questions provided

