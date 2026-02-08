# Legacy Scripts Archive - 2025 Q4

**Archived Date:** 2026-01-25  
**Reason:** Import cleanup following Hydra refactor completion  
**Status:** Historical reference only - not maintained

## Overview

This directory contains scripts that were archived during the Hydra refactor cleanup phase (2026-01-25). These scripts had broken dependencies, used deprecated APIs, or were superseded by newer infrastructure. They are preserved for historical reference but are **not maintained** and **should not be used** in active development.

## Archive Organization

```
archive/2025_q4_legacy_scripts/
├── aws_batch/              # AWS batch processing scripts → airflow-batch-processor
├── demos/                  # Demo scripts with broken UI/preprocessing dependencies
├── debugging/              # One-time analysis scripts with legacy data paths
├── preprocessing/          # Removed preprocessing functionality scripts
└── to_update/              # Scripts needing API updates (predict.py, test.py)
    ├── benchmarks/         # Future: Benchmark scripts needing orchestrator update
    ├── etl/                # Future: ETL scripts needing module clarification
    └── integrations/       # Future: HuggingFace/MCP scripts
```

---

## Archived Scripts by Category

### AWS/Cloud Scripts (4 scripts) → `aws_batch/` and `to_update/`

**Archived:**
- `runners/batch_pseudo_labels_aws.py` → `aws_batch/`
- `scripts/cloud/prepare_test_dataset.py` → `aws_batch/`
- `runners/predict.py` → `to_update/`
- `runners/test.py` → `to_update/`

**Why Archived:**
- AWS batch functionality replaced by Airflow DAGs
- Legacy inference API (needs orchestrator update)
- One-time setup scripts no longer needed

**Modern Replacement:**
```bash
# Use Airflow batch processor instead
cd airflow-batch-processor
docker-compose up -d
# See: airflow-batch-processor/README.md
```

---

### Demo Scripts (8 scripts) → `demos/` and `preprocessing/`

**Archived:**
- `scripts/demos/demo_evaluation_viewer.py`
- `scripts/demos/demo_ui.py`
- `scripts/demos/demo_document_flattening.py`
- `scripts/demos/compare_preprocessors.py`
- `scripts/demos/test_preprocessing_systematic.py`
- `scripts/demos/offline_perspective_preprocess_train.py`
- `scripts/demos/test_perspective_on_pseudo_label.py`
- `scripts/demos/type_checking_demo/debug_canonical_size.py`

**Why Archived:**
- Broken dependencies: `ui.apps.*`, `ui.utils.*` (UI package separate/broken)
- Removed functionality: Preprocessing module no longer exists
- Legacy data paths: Hardcoded paths to old dataset structure
- Educational only: Not critical for pipeline operation

**Modern Replacement:**
For UI-based evaluation:
```bash
# Use the UI app directly (if available)
uv run python run_ui.py
```

For preprocessing:
```python
# Preprocessing now handled in data pipeline
from ocr.data.datasets import get_datasets_by_cfg
dataset = get_datasets_by_cfg(cfg.data, data_config, cfg)
```

---

### Debug/Analysis Scripts (3 scripts) → `debugging/` and `preprocessing/`

**Archived:**
- `scripts/debug/data_analyzer.py`
- `scripts/debug/generate_offline_samples.py`
- `scripts/data/debug_etl_core.py`

**Why Archived:**
- Legacy data paths (old dataset structure)
- ETL module not in main package (unclear structure)
- One-time analysis tools not maintained

**Modern Replacement:**
For data analysis:
```bash
# Use experiment manager's analysis tools
uv run python experiment_manager/etk.py analyze <experiment_id>
```

For dataset debugging:
```python
# Use orchestrator's dataset introspection
from ocr.pipelines.orchestrator import OCRProjectOrchestrator
orchestrator = OCRProjectOrchestrator(cfg)
pl_module, data_module = orchestrator.setup_modules()
# Inspect datasets via data_module
```

---

## Scripts Remaining for Future Review

The following script categories are **NOT archived yet** but should be reviewed:

### Medium Priority - Review for Update or Archive (8 scripts)

**ETL Scripts (2)** - `to_update/etl/`
- `scripts/data/etl/cli.py` - ETL core module unclear
- `scripts/data/generate_pseudo_labels.py` - Old inference API

**Benchmark Scripts (4)** - `to_update/benchmarks/`
- `scripts/performance/benchmark_optimizations.py`
- `scripts/performance/benchmark_pipeline.py`
- `scripts/performance/benchmark_recognition.py`
- `scripts/performance/decoder_benchmark.py`

**HuggingFace/MCP Scripts (2)** - `to_update/integrations/`
- `scripts/huggingface/hf_inference.py`
- `scripts/mcp/verify_server.py`

**Action:** Create GitHub issue to update or archive these scripts.

### Low Priority - Keep but Document (3 scripts)

**Keep These (with warnings):**
- `scripts/checkpoints/convert_legacy_checkpoints.py` - May need for old checkpoints
- `scripts/validation/checkpoints/validate_coordinate_consistency.py` - UI dep, may fix later
- `scripts/documentation/translate_readme.py` - deep_translator optional, has error handling

**Action:** Add deprecation warnings to these scripts.

---

## Import Impact

**Before Archival:** 25+ broken script imports  
**After Archival:** 7 broken imports (all deferred/acceptable)

**Remaining 7 Broken Imports:**
- 2 tiktoken imports (optional dependency with error handling)
- 5 UI module imports (separate UI package, future work)

---

## Recovery Instructions

If you need to restore any archived script:

```bash
# 1. Check git history for the original file
git log -- path/to/original/script.py

# 2. Restore from archive (if script can be updated)
git mv archive/2025_q4_legacy_scripts/<category>/<script>.py <original-path>/

# 3. Update imports to use new APIs
# - Use ocr.pipelines.orchestrator.OCRProjectOrchestrator
# - Use ocr.core.interfaces for base classes
# - Use ocr.core.infrastructure for utilities

# 4. Test the script
uv run python <original-path>/<script>.py

# 5. Run import audit
uv run python scripts/audit/master_audit.py
```

---

## Related Documentation

- **Hydra Refactor Tracking:** [project_compass/pulse_staging/hydra-refactor-progress-tracking.md](../../project_compass/pulse_staging/hydra-refactor-progress-tracking.md)
- **Import Cleanup Session:** [__DEBUG__/2026-01-22_hydra_configs_legacy_imports/](../../__DEBUG__/2026-01-22_hydra_configs_legacy_imports/)
- **Airflow Batch Processor:** [airflow-batch-processor/README.md](../../airflow-batch-processor/README.md)
- **Orchestrator Documentation:** [ocr/pipelines/orchestrator.py](../../ocr/pipelines/orchestrator.py)

---

## Archive Policy

**Do NOT:**
- ❌ Import from archived scripts
- ❌ Reference archived scripts in active code
- ❌ Use archived scripts as examples for new code

**DO:**
- ✅ Use as historical reference for understanding old approaches
- ✅ Extract useful patterns/logic for modernization
- ✅ Document lessons learned from deprecated approaches

**Maintenance:** These scripts are **frozen as-of 2026-01-25** and will not receive updates. If similar functionality is needed, implement from scratch using modern APIs.

---

## Questions?

If you need help finding modern replacements for archived functionality:

1. Check the **Modern Replacement** section for each category above
2. Review the [OCR Project Orchestrator](../../ocr/pipelines/orchestrator.py)
3. Consult [AGENTS.md](../../AGENTS.md) for AI-facing documentation
4. Search for similar functionality: `uv run rg "<functionality>" --type py`

**Last Updated:** 2026-01-25  
**Archived By:** Hydra Refactor Cleanup (Phase 5)
