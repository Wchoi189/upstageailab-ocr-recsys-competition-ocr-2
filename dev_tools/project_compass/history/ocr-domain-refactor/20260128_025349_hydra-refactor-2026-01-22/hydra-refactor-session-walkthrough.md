# Debugging Session: Hydra Configs and Legacy Imports

**Date**: 2026-01-22
**Session Type**: Critical debugging and systematic refactoring
**Primary Focus**: Detection pipeline (det_resnet50_v1)
**Secondary Focus**: Recognition pipeline (rec_baseline_v1)

---

## Quick Navigation

- **[01_initial_analysis.md](01_initial_analysis.md)** - Symptoms, audit results, initial state
- **[02_investigation.md](02_investigation.md)** - Debugging process, techniques, lessons learned
- **[03_findings.md](03_findings.md)** - Root causes, solutions, testing recommendations
- **[artifacts/](artifacts/)** - AI-optimized knowledge base and tooling
- **[logs/](logs/)** - Raw debugging logs and execution traces

---

## Session Summary

### Critical Issues Identified

1. **Ghost Code Phenomenon** (CRITICAL)
   - Runtime environment detached from source code
   - Package installed in standard mode, not editable
   - All code modifications ignored

2. **Hydra Recursive Instantiation Trap** (HIGH)
   - `_recursive_=True` causing premature optimizer instantiation
   - Recognition pipeline completely blocked
   - Factory pattern conflicting with Hydra defaults

3. **Systematic Import Breakage** (HIGH)
   - 53 broken Python imports
   - 13-18 broken Hydra targets
   - Incomplete refactoring from domain separation

### Solutions Delivered

1. **Environment Validation**
   - Migration guard script
   - Runtime assertions
   - Pre-flight checks

2. **Automated Healing**
   - Auto-alignment script for Hydra configs
   - Master audit for systematic scanning
   - Fix manifest generation

3. **Knowledge Base**
   - 8 AI-optimized artifacts
   - Tool mastery guides
   - Implementation patterns

---

## Directory Structure

```
__DEBUG__/2026-01-22_hydra_configs_legacy_imports/
├── README.md                    # This file
├── 01_initial_analysis.md       # Symptoms and audit results
├── 02_investigation.md          # Debugging process
├── 03_findings.md               # Solutions and recommendations
│
├── artifacts/                   # AI-optimized knowledge base
│   ├── README.md                # Artifact navigation guide
│   ├── QUICK_START.md           # Quick reference
│   ├── CONVERSION_SUMMARY.md    # Artifact creation process
│   │
│   ├── ai_guidance/             # AI agent instruction patterns
│   │   └── instruction_patterns.md
│   │
│   ├── implementation_guides/   # Complete implementations
│   │   ├── migration_guard_implementation.md
│   │   └── auto_align_hydra_script.md
│   │
│   ├── refactoring_patterns/    # Design patterns
│   │   └── shim_antipatterns_guide.md
│   │
│   ├── tool_guides/             # Tool mastery
│   │   ├── yq_mastery_guide.md
│   │   └── adt_usage_patterns.md
│   │
│   └── analysis_outputs/        # Generated analysis
│       ├── debugging_pain_points.md
│       ├── broken_targets.json
│       ├── hydra_interpolation_map.md
│       └── ...
│
├── logs/                        # Raw debugging logs
│   ├── debug_log_relevant_to_pain_points.log  # Complete session log
│   ├── debug_final.log          # Final state
│   ├── debug_flow.log           # Execution flow
│   ├── debug_base_path.log      # Path resolution
│   └── ...
│
└── scripts/                     # Debugging scripts
    └── ...
```

---

## Key Metrics

| Metric                 | Initial | Final | Target |
| ---------------------- | ------- | ----- | ------ |
| Broken Python imports  | 53      | 49    | 0      |
| Broken Hydra targets   | 18      | 13    | 0      |
| Detection pipeline     | ❌       | ⏳     | ✅      |
| Recognition pipeline   | ❌       | ⏳     | ✅      |
| Environment validation | ❌       | ✅     | ✅      |
| Automated tooling      | ❌       | ✅     | ✅      |
| Documentation          | ❌       | ✅     | ✅      |

---

## Quick Start

### 1. Fix Environment

```bash
# Reinstall in editable mode
uv pip install -e .

# Verify
uv run python -c "import ocr; assert 'site-packages' not in ocr.__file__"
```

### 2. Run Pre-Flight Check

```bash
bash scripts/preflight.sh
```

### 3. Test Detection Pipeline

```bash
uv run uv run python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True
```

### 4. Run Systematic Audit

```bash
uv run python scripts/audit/master_audit.py > audit_results.txt
```

### 5. Auto-Heal Hydra Configs

```bash
# Dry run first
uv run python scripts/audit/auto_align_hydra.py --dry-run

# Apply fixes
uv run python scripts/audit/auto_align_hydra.py
```

---

## Testing Priorities

### Priority 1: Detection Pipeline (PRIMARY)

**Why**: Established implementation, critical for project

**Test**:
```bash
uv run python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True
```

**Success Criteria**:
- ✅ No import errors
- ✅ Model builds
- ✅ Dataset loads
- ✅ Training starts

### Priority 2: Recognition Pipeline (SECONDARY)

**Why**: New implementation, only single epoch completed

**Test**:
```bash
uv run python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True
```

**Success Criteria**:
- ✅ No Hydra errors
- ✅ Vocab injection works
- ✅ Model builds
- ✅ Training starts

---

## Useful Analysis Outputs

### For Understanding Issues

- **[artifacts/analysis_outputs/debugging_pain_points.md](artifacts/analysis_outputs/debugging_pain_points.md)**
  - Ghost code phenomenon explained
  - Hydra recursive instantiation trap
  - Recommendations for prevention

- **[artifacts/analysis_outputs/broken_targets.json](artifacts/analysis_outputs/broken_targets.json)**
  - Structured list of broken Hydra targets
  - Suggested fixes
  - Action items

### For Implementing Solutions

- **[artifacts/implementation_guides/migration_guard_implementation.md](artifacts/implementation_guides/migration_guard_implementation.md)**
  - Complete validation script
  - Runtime assertions
  - CI/CD integration

- **[artifacts/implementation_guides/auto_align_hydra_script.md](artifacts/implementation_guides/auto_align_hydra_script.md)**
  - Automated Hydra fixing
  - Runtime reflection approach
  - Usage examples

### For Tool Mastery

- **[artifacts/tool_guides/yq_mastery_guide.md](artifacts/tool_guides/yq_mastery_guide.md)**
  - 20+ ready-to-use yq commands
  - Hydra-specific patterns
  - Troubleshooting tips

- **[artifacts/tool_guides/adt_usage_patterns.md](artifacts/tool_guides/adt_usage_patterns.md)**
  - AST-Grep structural patterns
  - ADT integration workflows
  - Refactoring automation

---

## Lessons Learned

### 1. Environment Hygiene is Critical

**Problem**: Non-editable install caused ghost code
**Solution**: Always use `uv pip install -e .`
**Prevention**: Runtime assertions + pre-flight checks

### 2. Hydra Requires Explicit Control

**Problem**: Recursive instantiation broke factory pattern
**Solution**: Use `_recursive_=False` in factory calls
**Prevention**: AST-grep lint rules + documentation

### 3. Refactoring Needs Systematic Validation

**Problem**: 53 broken imports from incomplete refactoring
**Solution**: Automated audit + healing scripts
**Prevention**: CI/CD integration + regular audits

### 4. Debugging Needs Observable Signals

**Problem**: Silent failures wasted time
**Solution**: Logging, assertions, validation layers
**Prevention**: Fail-fast patterns + clear error messages

---

## Next Steps

### Immediate (Today)

1. ✅ Fix environment (editable install)
2. ✅ Run pre-flight check
3. ⏳ Test detection pipeline
4. ⏳ Run systematic audit

### Short-Term (This Week)

1. ⏳ Process broken imports in batches
2. ⏳ Auto-heal Hydra configs
3. ⏳ Add runtime assertions
4. ⏳ Create AST-grep lint rules

### Long-Term (This Month)

1. ⏳ CI/CD integration
2. ⏳ Complete refactoring validation
3. ⏳ Update documentation
4. ⏳ Knowledge transfer

---

## References

### Internal Documentation

- [01_initial_analysis.md](01_initial_analysis.md) - Initial state and symptoms
- [02_investigation.md](02_investigation.md) - Debugging methodology
- [03_findings.md](03_findings.md) - Solutions and recommendations
- [artifacts/README.md](artifacts/README.md) - Artifact navigation

### External Resources

- [Hydra Documentation](https://hydra.cc/)
- [yq Documentation](https://mikefarah.gitbook.io/yq/)
- [AST-Grep Documentation](https://ast-grep.github.io/)

### Project Context

- AgentQMS standards (if applicable)
- Project Compass documentation (if applicable)
- OCR pipeline documentation (if applicable)

---

## Contact and Support

For questions or issues related to this debugging session:

1. Review the documentation in order:
   - Start with `README.md` (this file)
   - Read `01_initial_analysis.md` for context
   - Check `03_findings.md` for solutions
   - Consult `artifacts/` for detailed guides

2. Use the provided tooling:
   - `scripts/audit/master_audit.py` for scanning
   - `scripts/audit/auto_align_hydra.py` for healing
   - `scripts/preflight.sh` for validation

3. Refer to artifact guides:
   - `artifacts/QUICK_START.md` for common tasks
   - `artifacts/tool_guides/` for tool usage
   - `artifacts/implementation_guides/` for scripts

---

**Session Status**: ✅ Analysis complete, tooling delivered, testing in progress
**Last Updated**: 2026-01-22
**Python Manager**: uv (recommended)



---

## Update: 2026-01-25

**Directory Migration Summary:**

**Migrated directories (now accessible via symlinks):**
`/apps`
`/archive`
`/cache`
`/data`
`/extensions`
`/hydra_outputs`
`/lightning_logs`
`/node_modules`
`/outputs`
`/packages`
`/venvs`
`/wandb`

**New project root:**
`/workspaces/project-artifacts/ocr-external-storage/`

**Note:** Update Hydra configuration in `configs/global/paths.yaml` to reflect new paths.