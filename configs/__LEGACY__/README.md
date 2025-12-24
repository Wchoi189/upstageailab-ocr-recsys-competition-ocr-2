# Legacy Configuration Archive

This directory contains Hydra configurations that have been superseded by the new architecture using `configs/_base/` foundation. These configs are preserved for reference, backward compatibility, and migration purposes.

---

## Why These Configs Are Here

**Status**: Deprecated but accessible
**Reason**: Superseded by new architecture patterns
**Maintained**: For backward compatibility and historical reference

The configs in this directory:
- ✅ Are **still accessible** via Hydra (Hydra searches subdirectories)
- ✅ Can **still be used** if needed for specific workflows
- ❌ Are **not recommended** for new projects or workflows
- ⚠️ May be **removed** in future versions after extended deprecation period

---

## Configs in This Archive

### `model/optimizer.yaml`

**Status**: Superseded
**Superseded by**: [`configs/model/optimizers/adam.yaml`](../model/optimizers/adam.yaml)

**Old Pattern** (single-file config):
```yaml
# @package model.optimizer
_target_: torch.optim.Adam
lr: 0.001
weight_decay: 0.0001
```

**New Pattern** (directory-based):
```yaml
# configs/model/optimizers/adam.yaml
_target_: torch.optim.Adam
lr: 0.001
weight_decay: 0.0001
```

**Migration**:
```bash
# Old (legacy):
uv run python runners/train.py +model/optimizer=__LEGACY__/optimizer

# New (recommended):
uv run python runners/train.py model/optimizers=adam
```

---

### `data/preprocessing.yaml`

**Status**: Superseded
**Superseded by**: [`configs/data/base.yaml`](../data/base.yaml) with preprocessing overrides

**Old Pattern** (standalone preprocessing config):
- Self-contained data configuration for preprocessing workflows
- Not composable with other data configs

**New Pattern** (composable data config):
- Uses `configs/data/base.yaml` foundation
- Overrides specific preprocessing parameters
- Composable with transforms and dataloaders

**Migration**:
```bash
# Old (legacy):
uv run python runners/train.py data=__LEGACY__/preprocessing

# New (recommended):
uv run python runners/train.py data=default \
  datasets.train_dataset.config.enable_preprocessing=true
```

**Note**: This config was only found in archived code and is likely not actively used.

---

## How to Use Legacy Configs

### Option 1: Direct Override (If Needed)

```bash
# Access legacy configs using __LEGACY__/ prefix
uv run python runners/train.py +model/optimizer=__LEGACY__/optimizer
uv run python runners/train.py data=__LEGACY__/preprocessing
```

### Option 2: Migrate to New Architecture (Recommended)

See migration guide below for each config.

---

## Migration Guide

### Migrating from `model/optimizer.yaml`

**Before**:
```bash
# Old command (legacy)
uv run python runners/train.py +model/optimizer=__LEGACY__/optimizer
```

**After**:
```bash
# New command (recommended)
uv run python runners/train.py model/optimizers=adam

# Or with custom parameters:
uv run python runners/train.py model/optimizers=adam \
  model.optimizer.lr=0.001 \
  model.optimizer.weight_decay=0.0001
```

**Why Migrate?**:
- ✅ Consistent with new architecture patterns
- ✅ Better organized (directory-based, not single-file)
- ✅ Easier to extend (add new optimizers without conflicts)
- ✅ More maintainable

---

### Migrating from `data/preprocessing.yaml`

**Before**:
```bash
# Old command (legacy)
uv run python runners/train.py data=__LEGACY__/preprocessing
```

**After**:
```bash
# New command (recommended)
uv run python runners/train.py data=default

# With preprocessing enabled:
uv run python runners/train.py data=default \
  datasets.train_dataset.config.enable_preprocessing=true
```

**Why Migrate?**:
- ✅ Uses composable architecture (`data/base.yaml` + transforms + dataloaders)
- ✅ More flexible (can mix preprocessing with other data configs)
- ✅ Better performance (uses performance presets)
- ✅ Aligned with project standards

---

## Removal Timeline

### Phase 1: Containerization (Current - 2025-12-24)
- ✅ Configs moved to `__LEGACY__/` directory
- ✅ Still accessible via Hydra
- ✅ Deprecation documented

### Phase 2: Deprecation Warnings (Future - TBD)
- ⏳ Add warnings when legacy configs are used
- ⏳ Notify users in logs
- ⏳ Update documentation with migration timeline

### Phase 3: Archive (Future - 6-12 months)
- ⏳ Move to `archive/configs/` (not Hydra-accessible)
- ⏳ Document in CHANGELOG
- ⏳ Provide migration guide

### Phase 4: Removal (Distant Future - 12-24 months)
- ⏳ Remove entirely
- ⏳ Breaking change announcement
- ⏳ Update to major version

**Current Recommendation**: Migrate to new architecture at your convenience. No immediate action required.

---

## Differences from New Architecture

### Legacy Pattern Characteristics:
- ❌ No composition with `_base/` configs
- ❌ Single-file or standalone configs
- ❌ Less flexible (hard-coded settings)
- ❌ Harder to extend and maintain

### New Architecture Benefits:
- ✅ Composable with `_base/` foundation
- ✅ Modular (mix and match components)
- ✅ More flexible (override at multiple levels)
- ✅ Better organized (directory-based)
- ✅ Easier to extend (add new variants)

---

## Configuration Options Lost (If Removed)

### `model/optimizer.yaml`
**Lost**: Single-file optimizer config with `@package model.optimizer`

**Alternative**: Use `model/optimizers/adam.yaml` (functionally identical, different package directive)

**Impact**: None - new pattern provides same functionality

---

### `data/preprocessing.yaml`
**Lost**: Standalone preprocessing data config

**Alternative**: Use `data/base.yaml` with preprocessing parameters

**Impact**: Minimal - preprocessing can be enabled via overrides

---

## Testing Legacy Configs

To verify legacy configs still work:

```bash
# Test legacy configs are accessible
ls configs/__LEGACY__/model/
ls configs/__LEGACY__/data/

# Test legacy config loading (dry run)
uv run python runners/train.py +model/optimizer=__LEGACY__/optimizer --cfg job

# Run override tests
uv run python tests/unit/test_hydra_overrides.py
```

---

## FAQs

### Q: Can I still use legacy configs?
**A**: Yes, they are still accessible via Hydra. However, migration to new architecture is recommended.

### Q: Will legacy configs break in the future?
**A**: Not immediately. They will be deprecated with warnings first, then archived, then removed only after extended notice.

### Q: How do I know if I'm using a legacy config?
**A**: Check if your command references `__LEGACY__/` or if you see deprecation warnings in logs (future feature).

### Q: Should I migrate now?
**A**: Recommended but not required. Migrate when convenient or when starting new workflows.

### Q: What if I need features from legacy configs?
**A**: Contact the team. We can add missing features to new architecture or preserve specific legacy configs longer.

---

## Related Documentation

- **Main Config Guide**: [`configs/README.md`](../README.md)
- **Override Patterns**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_OVERRIDE_PATTERNS.md`
- **Configuration Audit**: `docs/artifacts/audits/2025-12-24_2000_hydra_config_audit/HYDRA_CONFIG_AUDIT_ASSESSMENT.md`
- **Previous Cleanup**: `archive/archive_docs/docs/completed_plans/2025-11/2025-11-11_1439_implementation_plan_legacy-cleanup-config-consolidation.md`

---

## Support

If you encounter issues with legacy configs or need help migrating:
1. Review the migration guide above
2. Check [`configs/README.md`](../README.md) for override patterns
3. Run tests: `uv run python tests/unit/test_hydra_overrides.py`
4. Contact the team for assistance

---

**Last Updated**: 2025-12-24
**Status**: Active (Deprecated but Accessible)
**Next Review**: 2026-06-24 (6 months)
