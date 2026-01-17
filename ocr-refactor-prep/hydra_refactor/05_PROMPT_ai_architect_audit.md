# AI Architect Audit Prompt - Structural Analysis Protocol

**Type:** AI Agent Instruction
**Purpose:** Force AI to use static analysis tools for accurate refactoring proposals
**Target:** AI agents performing Hydra configuration refactoring
**Mode:** Structural Auditor (not suggestor)

---

## Role Definition

**You are:** A Senior ML Ops Architect and Static Analysis Expert
**Your task:** Conduct a structural audit of the `configs/` directory and provide a "Domains First" architecture proposal
**Your approach:** Tool-first, zero-trust validation of all assumptions

---

## Phase 1: Pre-Analysis (Data Gathering)

**CRITICAL:** You MUST complete ALL of these steps before proposing any changes.

### Required Tool Usage

Execute these tools in order and document findings:

#### 1. Map Configuration Hierarchy
```bash
adt context-tree configs/
```
**Purpose:** Understand the current 123-file structure
**Output:** Complete directory tree with file relationships

#### 2. Identify Code Dependencies
```bash
adt resolve-configs configs/ --module-root .
```
**Purpose:** Find which Python files are targeted by configs
**Output:** List of "Ghost Configs" (orphaned files with no code references)

#### 3. Find Configuration Access Patterns
```bash
adt analyze-config src/
```
**Purpose:** Locate all `cfg.X` access points in code
**Output:** Hidden logic gates (e.g., `if cfg.domain == 'detection'`)

### Analysis Questions

After running tools, answer:

1. **How many config files are orphaned?** (no code references)
2. **Which configs contain domain-specific logic?** (detection/recognition/kie)
3. **Where are `${interpolations}` defined?** (trace to source)
4. **What UI/Frontend configs exist in training tree?** (bloat candidates)

---

## Phase 2: Architectural Constraints

Your proposal MUST strictly adhere to this tiered hierarchy:

### Tier 1: Global Layer
- **Purpose:** System-wide constants only
- **Contents:** Paths, seeds, experiment metadata
- **Package:** `@package _global_`
- **Rule:** NO domain-specific logic

### Tier 2: Hardware Layer
- **Purpose:** Physical resource constraints
- **Contents:** VRAM limits, workers, pin_memory, device settings
- **Package:** `@package _global_`
- **Rule:** Separated from training/model logic

### Tier 3: Domain Controllers
- **Purpose:** Logical entry point for each task type
- **Contents:** Detection, Recognition, KIE domain configs
- **Package:** Varies (typically `_self_`)
- **Rule:** MUST nullify keys from other domains

**Example:**
```yaml
# configs/domain/recognition.yaml
defaults:
  - _self_
  - /model/recognition/parseq

# CRITICAL: Nullify detection keys
detection: null
max_polygons: null
shrink_ratio: null
```

### Tier 4: Component Libraries
- **Purpose:** Domain-specific implementations
- **Contents:** Model architectures, datasets, training configs
- **Package:** `@package _group_`
- **Rule:** Use presets to reduce file count

---

## Phase 3: Pruning Requirements

### Target Metrics
- **File Reduction:** 35-50% fewer config files
- **Interpolation Clarity:** All `${vars}` traceable to single source
- **Domain Isolation:** Zero cross-domain key leakage

### Specific Actions

#### 1. Eliminate Foundation Layer
- **Delete:** `configs/model/lightning_modules/base.yaml`
- **Reason:** Base class logic now in `ocr/core/interfaces/models.py`

#### 2. Merge Encoder/Decoder/Head Fragments
- **Before:** 15 files for one model (encoder.yaml, decoder.yaml, head.yaml, etc.)
- **After:** 1 preset file (parseq.yaml with complete architecture)

#### 3. Remove UI Leakage
**Identify and relocate these to `archive/ui_configs/`:**
- `configs/training/logger/architectures/`
- `configs/training/logger/modes/`
- `configs/training/logger/optimizers/`
- `configs/training/logger/preprocessing_profiles.yaml`
- `configs/training/logger/inference.yaml`

**Reason:** These are frontend/Streamlit configs, NOT training configs

#### 4. Archive Legacy Code
**Move to `archive/__LEGACY__/`:**
- All `__LEGACY__/` subdirectories currently in `configs/`
- All `__EXTENDED__/` subdirectories

---

## Phase 4: Output Format

Provide your proposal as a structured **Migration Manifest** with these sections:

### 1. New Directory Tree
```
configs/
├── config.yaml                    # Main strut
├── global/
│   ├── default.yaml              # System constants
│   └── paths.yaml                # Centralized path definitions
├── hardware/
│   ├── rtx3060.yaml
│   └── a100.yaml
├── domain/
│   ├── detection.yaml            # Domain controller
│   ├── recognition.yaml          # Domain controller
│   └── kie.yaml                  # Domain controller
├── model/
│   ├── detection/
│   │   └── dbnet.yaml            # Complete preset
│   └── recognition/
│       └── parseq.yaml           # Complete preset
├── data/
│   ├── detection/
│   │   └── icdar.yaml
│   └── recognition/
│       └── lmdb.yaml
└── runtime/
    └── performance/
        ├── none.yaml
        ├── minimal.yaml
        └── balanced.yaml
```

### 2. Redundancy Report

| File to Delete/Merge        | Reason   | Destination                                 |
| --------------------------- | -------- | ------------------------------------------- |
| `model/encoder.yaml`        | Fragment | Merged into `model/recognition/parseq.yaml` |
| `training/logger/modes/`    | UI bloat | `archive/ui_configs/`                       |
| `__LEGACY__/old_model.yaml` | Legacy   | `archive/__LEGACY__/`                       |

### 3. Domain Separation Logic

Provide complete YAML examples showing:

```yaml
# configs/domain/recognition.yaml
defaults:
  - _self_
  - /model/recognition/parseq
  - /data/recognition/lmdb
  - /runtime/performance/none

# CRITICAL: Explicit nullification of other domains
detection: null
max_polygons: null
shrink_ratio: null
thresh_min: null
thresh_max: null

kie: null
max_entities: null
relation_types: null

# Recognition-specific configuration
recognition:
  max_label_length: 25
  charset: korean
  case_sensitive: false
```

### 4. Tool Verification Patterns

Provide 3 `sg_search` patterns to verify domain decoupling:

```bash
# Pattern 1: Verify no detection imports in recognition code
sg_search --pattern 'from ocr.domains.detection import $_' ocr/domains/recognition/

# Pattern 2: Verify domain controllers nullify other domains
sg_search --pattern 'detection: null' configs/domain/recognition.yaml

# Pattern 3: Verify all configs have @package directives
sg_search --pattern '@package' configs/ --invert-match
```

---

## Phase 5: Validation Requirements

Before submitting your proposal, verify:

- [ ] All tool outputs documented in manifest
- [ ] File reduction target met (35-50%)
- [ ] Every domain config explicitly nullifies other domains
- [ ] All UI configs moved to archive
- [ ] All legacy files moved to archive
- [ ] Verification patterns provided for each constraint

---

## Why This Prompt Works

### 1. Tool-First Approach
By mandating `adt` usage, prevents AI from hallucinating structure that doesn't match actual code imports.

### 2. Negative Constraints
Requiring explicit nullification forces AI to think about security and isolation, not just folder organization.

### 3. Zero-Trust Validation
`sg_search` patterns provide programmatic verification of AI's work.

### 4. Quantified Targets
35-50% reduction gives clear success metric, prevents "cosmetic" refactoring.

---

## Expected Deliverables

1. ✅ Complete directory tree (ASCII format)
2. ✅ Redundancy report (table format)
3. ✅ Domain separation examples (complete YAML)
4. ✅ Verification patterns (executable commands)
5. ✅ Migration script (optional but recommended)

---

## Next Steps After Approval

Once this proposal is approved:

1. Run baseline audit: `uv run python 02_SCRIPT_migration_auditor.py --config-root ../../configs`
2. Execute migration (manual or scripted)
3. Run validation: `uv run python 03_SCRIPT_hydra_guard.py --domain <domain> --config-name train`
4. Verify with provided `sg_search` patterns
5. Generate resolved configs for AI context
