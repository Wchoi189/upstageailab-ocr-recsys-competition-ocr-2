# Architecture Migration Checklist

## 🎯 Current Status: 48 Broken Imports

### Critical Path (Do First)

- [ ] **DetectionHead Interface** (affects 2 files)
  - [ ] Check if exists: `grep -r "class DetectionHead" ocr/domains/`
  - [ ] If missing: Create in `ocr/domains/detection/interfaces.py`
  - [ ] Update imports in db_head.py and craft_head.py

- [ ] **Detection Models** (6 files) 
  - [ ] ocr/domains/detection/models/heads/db_head.py
  - [ ] ocr/domains/detection/models/heads/craft_head.py
  - [ ] ocr/domains/detection/models/architectures/craft.py
  - [ ] ocr/domains/detection/models/architectures/dbnet.py
  - [ ] ocr/domains/detection/models/architectures/dbnetpp.py
  - [ ] ocr/domains/detection/models/__init__.py

- [ ] **Verify Training Works**
  ```bash
  python runners/train.py domain=detection trainer.fast_dev_run=true
  ```

### Secondary Path

- [ ] **Core Models** (2 files)
  - [ ] ocr/core/models/architectures/__init__.py
  - [ ] ocr/core/models/architectures/shared_decoders.py

- [ ] **KIE Domain** (3 files)
  - [ ] ocr/domains/kie/data/dataset.py
  - [ ] ocr/domains/kie/data/__init__.py
  - [ ] ocr/domains/kie/models/__init__.py

- [ ] **Layout Domain** (1 file)
  - [ ] ocr/domains/layout/__init__.py

### Cleanup Path

- [ ] **Test Files** (~18 files)
  ```bash
  grep -r "from ocr.features" tests/ --files-with-matches
  ```

- [ ] **Scripts** (1 file)
  - [ ] scripts/performance/benchmark_recognition.py

- [ ] **Pre-commit Hook**
  - [ ] Update `.pre-commit-config.yaml` to allow ocr/pipelines/

### Verification

- [ ] No `ocr.features` imports: `grep -r "from ocr.features" ocr/ tests/ scripts/`
- [ ] Full test suite: `pytest tests/ -v`
- [ ] Training runs: `python runners/train.py domain=detection`

## Quick Commands

```bash
# Find all broken imports
grep -r "from ocr.features" --include="*.py" ocr/ tests/ scripts/ | wc -l

# Check specific domain
grep -r "from ocr.features" ocr/domains/detection/

# Recover file from git history
git show 89fe577^:ocr/features/detection/interfaces.py

# Test import
python3 -c "from ocr.domains.detection.models.heads.db_head import DBHead"
```
