# Scripts Directory Reorganization Plan

**Date:** 2025-11-09
**Status:** Proposal

## Overview

This document proposes a reorganization of loose scripts in the `scripts/` directory to improve maintainability, discoverability, and consistency with existing organizational patterns.

## Current State

### Loose Scripts in `scripts/` Root (17 files)

**Validation Scripts:**
- `validate_metadata.py` - Validate checkpoint metadata
- `validate_ui_schema.py` - Validate UI inference schema
- `validate_templates.py` - Validate templates
- `validate_links.py` - Validate documentation links
- `validate_coordinate_consistency.py` - Validate coordinate consistency

**Checkpoint Management:**
- `migrate_checkpoints.py` - Migrate checkpoints to new naming scheme
- `convert_legacy_checkpoints.py` - Convert legacy checkpoint formats
- `generate_checkpoint_metadata.py` - Generate checkpoint metadata

**Documentation & Diagrams:**
- `check_freshness.py` - Check documentation freshness
- `generate_diagrams.py` - Generate diagrams
- `manage_diagrams.sh` - Manage diagrams
- `ci_update_diagrams.sh` - CI diagram updates
- `standardize_content.py` - Standardize content

**Data Processing:**
- `preprocess_data.py` - Preprocess data

**Performance:**
- `benchmark_performance.py` - Benchmark performance

**Utilities:**
- `cache_manager.py` - Cache management
- `process_manager.py` - Process management

## Proposed Organization

### 1. **`validation/`** - Validation Scripts
**Purpose:** All validation and consistency checking scripts

**Move:**
- `validate_metadata.py` → `validation/checkpoint_metadata.py`
- `validate_ui_schema.py` → `validation/ui_schema.py`
- `validate_templates.py` → `validation/templates.py`
- `validate_links.py` → `validation/links.py`
- `validate_coordinate_consistency.py` → `validation/coordinate_consistency.py`

**Rationale:** Centralizes all validation logic, making it easier to find and maintain validation tools.

### 2. **`checkpoints/`** - Checkpoint Management
**Purpose:** Checkpoint migration, conversion, and metadata generation

**Move:**
- `migrate_checkpoints.py` → `checkpoints/migrate.py`
- `convert_legacy_checkpoints.py` → `checkpoints/convert_legacy.py`
- `generate_checkpoint_metadata.py` → `checkpoints/generate_metadata.py`

**Rationale:** Groups all checkpoint-related operations together. Note: `migrate_checkpoints.py` is currently documented in README as a root script, so we may want to keep a symlink or update documentation.

### 3. **`documentation/`** - Documentation Tools
**Purpose:** Documentation generation, validation, and maintenance

**Move:**
- `check_freshness.py` → `documentation/check_freshness.py`
- `generate_diagrams.py` → `documentation/generate_diagrams.py`
- `manage_diagrams.sh` → `documentation/manage_diagrams.sh`
- `ci_update_diagrams.sh` → `documentation/ci_update_diagrams.sh`
- `standardize_content.py` → `documentation/standardize_content.py`

**Rationale:** Consolidates documentation-related tools. Note: Some of these might overlap with `agent_tools/documentation/` - consider merging or clearly distinguishing purposes.

### 4. **`data/`** - Data Processing
**Purpose:** Data preprocessing and transformation

**Move:**
- `preprocess_data.py` → `data/preprocess.py`

**Rationale:** Groups with existing `data_processing/` directory. Consider merging `data/` and `data_processing/` or clearly distinguishing their purposes.

**Alternative:** Move to existing `data_processing/` directory.

### 5. **`performance/`** - Performance Analysis
**Purpose:** Performance benchmarking and analysis

**Move:**
- `benchmark_performance.py` → `performance/benchmark.py`

**Rationale:** Consolidates with existing `performance/` and `performance_benchmarking/` directories. Consider merging these two directories.

### 6. **`utilities/`** - General Utilities
**Purpose:** General-purpose utility scripts

**Move:**
- `cache_manager.py` → `utilities/cache_manager.py`
- `process_manager.py` → `utilities/process_manager.py`

**Rationale:** Groups general utilities together. Note: `process_manager.py` might belong in `monitoring/` if it's primarily for process monitoring.

## Directory Structure After Reorganization

```
scripts/
├── agent_tools/          # AI agent tools (existing, well-organized)
├── analysis_validation/  # Analysis and validation (existing)
├── bug_tools/           # Bug-related tools (existing)
├── checkpoints/         # ✨ NEW - Checkpoint management
│   ├── migrate.py
│   ├── convert_legacy.py
│   └── generate_metadata.py
├── data_processing/     # Data processing (existing)
├── documentation/       # ✨ NEW - Documentation tools
│   ├── check_freshness.py
│   ├── generate_diagrams.py
│   ├── manage_diagrams.sh
│   ├── ci_update_diagrams.sh
│   └── standardize_content.py
├── migration_refactoring/ # Migration and refactoring (existing)
├── monitoring/          # Monitoring (existing)
├── performance/         # Performance analysis (existing)
├── performance_benchmarking/ # Performance benchmarking (existing)
├── seroost/            # Seroost indexing (existing)
├── setup/              # Setup scripts (existing)
├── temp/               # Temporary files (existing)
├── utilities/           # ✨ NEW - General utilities
│   ├── cache_manager.py
│   └── process_manager.py
├── validation/          # ✨ NEW - Validation scripts
│   ├── checkpoint_metadata.py
│   ├── ui_schema.py
│   ├── templates.py
│   ├── links.py
│   └── coordinate_consistency.py
└── README.md
```

## Additional Recommendations

### 1. Merge Similar Directories

**Option A: Merge `performance/` and `performance_benchmarking/`**
- Both deal with performance analysis
- Consider: `performance/` for analysis, `performance/benchmarking/` for benchmarks

**Option B: Merge `data/` and `data_processing/`**
- Both deal with data processing
- Consider: Keep `data_processing/` and move `preprocess_data.py` there

### 2. Distinguish `documentation/` from `agent_tools/documentation/`

**Current:**
- `agent_tools/documentation/` - Agent-specific documentation tools
- `documentation/` (proposed) - General documentation tools

**Recommendation:**
- Keep separation if purposes are distinct
- Or merge into `agent_tools/documentation/` if they're all agent-related

### 3. Consider `checkpoints/` vs `migration_refactoring/`

**Current:**
- `migration_refactoring/` - General migration tools
- `checkpoints/` (proposed) - Checkpoint-specific tools

**Recommendation:**
- Keep `checkpoints/` separate if checkpoint operations are frequent
- Or move to `migration_refactoring/checkpoints/` if they're one-time migrations

### 4. Update Documentation

**Files to Update:**
- `scripts/README.md` - Update paths and organization
- `Makefile` - Update script paths if referenced
- Any documentation referencing old script paths

### 5. Create Symlinks for Backward Compatibility (Optional)

For scripts that are frequently used or referenced in documentation:
```bash
# Example: Keep backward compatibility for migrate_checkpoints.py
ln -s checkpoints/migrate.py migrate_checkpoints.py
```

## Migration Steps

1. **Create new directories:**
   ```bash
   mkdir -p scripts/{validation,checkpoints,documentation,utilities}
   ```

2. **Move scripts:**
   ```bash
   # Validation scripts
   git mv scripts/validate_*.py scripts/validation/
   git mv scripts/validate_coordinate_consistency.py scripts/validation/coordinate_consistency.py

   # Checkpoint scripts
   git mv scripts/migrate_checkpoints.py scripts/checkpoints/migrate.py
   git mv scripts/convert_legacy_checkpoints.py scripts/checkpoints/convert_legacy.py
   git mv scripts/generate_checkpoint_metadata.py scripts/checkpoints/generate_metadata.py

   # Documentation scripts
   git mv scripts/check_freshness.py scripts/documentation/
   git mv scripts/generate_diagrams.py scripts/documentation/
   git mv scripts/manage_diagrams.sh scripts/documentation/
   git mv scripts/ci_update_diagrams.sh scripts/documentation/
   git mv scripts/standardize_content.py scripts/documentation/

   # Utilities
   git mv scripts/cache_manager.py scripts/utilities/
   git mv scripts/process_manager.py scripts/utilities/

   # Data processing
   git mv scripts/preprocess_data.py scripts/data_processing/

   # Performance
   git mv scripts/benchmark_performance.py scripts/performance/benchmark.py
   ```

3. **Update imports and references:**
   - Search for references to old paths
   - Update documentation
   - Update Makefile targets
   - Update CI/CD scripts

4. **Test:**
   - Verify all scripts still work
   - Check that documentation is accurate
   - Ensure CI/CD still works

5. **Update README:**
   - Document new organization
   - Update usage examples
   - Add migration notes

## Benefits

1. **Better Organization:** Related scripts grouped together
2. **Easier Discovery:** Clear categories make finding scripts easier
3. **Reduced Clutter:** Root directory cleaner
4. **Consistency:** Follows existing organizational patterns
5. **Maintainability:** Easier to maintain and extend

## Considerations

1. **Backward Compatibility:** Some scripts may be referenced in documentation or CI/CD
2. **Breaking Changes:** Path changes will break existing references
3. **Documentation Updates:** Need to update all references
4. **Testing:** Need to verify all scripts still work after move

## Next Steps

1. Review and approve this plan
2. Create implementation plan artifact
3. Execute migration
4. Update documentation
5. Test and verify

---

*This reorganization plan follows the existing organizational patterns in `agent_tools/` and improves the overall structure of the scripts directory.*

