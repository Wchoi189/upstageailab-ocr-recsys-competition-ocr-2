# Hydra Refactor Execution Guide

**Type:** Execution Workflow
**Purpose:** Step-by-step implementation of "Domains First" V5 refactor
**Prerequisites:** Completed AI Architect Audit
**Estimated Time:** 4-6 hours

---

## Phase 1: Preparation

### 1.1 Baseline Audit
```bash
uv run python 02_SCRIPT_migration_auditor.py --config-root ../../configs
```

### 1.2 Create Backup
```bash
BACKUP_DIR="configs_backup_$(date +%Y%m%d_%H%M%S)"
cp -r configs/ "$BACKUP_DIR"
```

### 1.3 Create Archive
```bash
mkdir -p archive/{__LEGACY__,__EXTENDED__,ui_configs}
```

---

## Phase 2: Structural Migration

### 2.1 Establish Global Layer
```bash
mkdir -p configs/global
cp 04_TEMPLATE_global_paths.yaml configs/global/paths.yaml
```

### 2.2 Move Legacy Files
```bash
find configs/ -type d -name "__LEGACY__" -exec mv {} archive/__LEGACY__/ \;
find configs/ -type d -name "__EXTENDED__" -exec mv {} archive/__EXTENDED__/ \;
```

### 2.3 Relocate UI Configs
```bash
mv configs/training/logger/architectures archive/ui_configs/
mv configs/training/logger/modes archive/ui_configs/
mv configs/training/logger/preprocessing_profiles.yaml archive/ui_configs/
```

### 2.4 Create Domain Controllers
```bash
mkdir -p configs/domain
cp 06_TEMPLATE_domain_controller.yaml configs/domain/recognition.yaml
```

---

## Phase 3: Validation

### 3.1 Run Auditor
```bash
uv run python 02_SCRIPT_migration_auditor.py --config-root ../../configs
```

### 3.2 Run Hydra Guard
```bash
uv run python 03_SCRIPT_hydra_guard.py --domain recognition --config-name train
```

### 3.3 Test Training
```bash
python train.py domain=recognition --dry-run
```

---

## Success Criteria

- [ ] File count reduced 35-50%
- [ ] Zero critical violations
- [ ] All domains pass validation
- [ ] Training runs successfully

---

## Rollback
```bash
rm -rf configs/
cp -r "$BACKUP_DIR" configs/
```
