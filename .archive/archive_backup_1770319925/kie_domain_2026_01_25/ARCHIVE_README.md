# KIE Domain Archive - 2026-01-25

## Archive Reason

The Key Information Extraction (KIE) domain was archived during the legacy purge audit as part of enforcing V5.0 "Domains First" architecture standards.

## Decision Rationale

**KIE Usage Analysis:**
- ❌ No active experiment configurations found
- ❌ No imports in production code (detection/recognition pipelines)
- ✅ One test file: `test_receipt_extraction.py` (moved to archive)
- ✅ Domain config existed: `configs/domain/kie.yaml` (moved to archive)
- ⚠️  Test reference in `test_etk_compass.py` (only for test coverage)

**Architectural Issues:**
- Used custom trainer pattern instead of V5 Lightning modules
- Implemented legacy `model.get_optimizers()` method
- Violated separation of concerns (models handling optimizer configuration)
- Would require ~300 lines of migration code to align with V5

**Conclusion:** Archive rather than migrate. If KIE functionality is needed in the future, re-implement using V5 architecture patterns from scratch.

## Archived Components

```
archive/kie_domain_2026_01_25/
├── kie/                          # Full domain implementation
│   ├── trainer.py               # Legacy Lightning module
│   ├── models/                  # LayoutLMv3Wrapper, LiLTWrapper
│   ├── data/                    # KIE datasets
│   ├── inference/               # Field extraction, receipt schema
│   ├── callbacks/               # Custom callbacks
│   ├── metrics/                 # KIE-specific metrics
│   └── utils/                   # Helper functions
├── kie.yaml                     # Domain configuration
├── test_receipt_extraction.py  # Unit tests
└── ARCHIVE_README.md           # This file
```

## Restoration Instructions

If KIE functionality is needed again:

### Option 1: Fresh V5 Implementation (RECOMMENDED)

1. Create `ocr/domains/kie/module.py` using V5 Lightning pattern
2. Use `config.train.optimizer` for optimizer configuration
3. Follow detection/recognition domain structure
4. Reference: `ocr/domains/detection/module.py` as template

### Option 2: Restore and Migrate

1. Copy archived code back to `ocr/domains/kie/`
2. Create V5-compliant Lightning module
3. Remove `get_optimizers()` methods from models
4. Update trainer.py to use Hydra config for optimizer
5. Add experiment config in `configs/experiment/`

## Related Audit

See: [docs/artifacts/audits/legacy-purge-audit-2026-01-25.md](../../../docs/artifacts/audits/legacy-purge-audit-2026-01-25.md)

**Archive Date:** 2026-01-25  
**Archived By:** AI Agent (Legacy Purge Audit Resolution)  
**Status:** Safe to delete after 30 days if unused
