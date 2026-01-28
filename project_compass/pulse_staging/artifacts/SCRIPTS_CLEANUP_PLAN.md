# Scripts Directory Cleanup Plan

**Date:** 2026-01-29
**Pulse:** import-script-audit-2026-01-29
**Status:** 📋 Planning Phase

---

## Executive Summary

**Current State:**
- 128 total scripts audited
- 55 scripts: KEEP (already clean)
- 25 scripts: REFACTOR (need updates)
- 48 scripts: REVIEW (manual decision needed)

**Issues Found:**
- 3 scripts with syntax errors
- Multiple experimental/prototype scripts
- Migration scripts that may be obsolete
- MCP tools with broken imports (AgentQMS)

**Goal:** Reduce bloat, archive obsolete code, fix syntax errors, organize remaining scripts

---

## Scripts Requiring Review (48 files)

### Category 1: MCP Tools (8 scripts) 🔧

**Status:** Active but 2 have broken imports

| Script | LOC | Issue | Recommendation |
|--------|-----|-------|----------------|
| `unified_server.py` | 441 | Has AgentQMS import errors | Fix imports or remove AgentQMS features |
| `analyze_telemetry.py` | 87 | - | Keep (telemetry analysis) |
| `sync_configs.py` | 129 | - | Keep (config sync utility) |
| `test_telemetry.py` | 76 | - | Keep (testing) |
| `verify_ads_compliance.py` | 58 | - | Keep (validation) |
| `verify_all.py` | 36 | - | Keep (validation) |
| `verify_feedback.py` | 46 | - | Keep (validation) |
| `verify_force.py` | 28 | - | Keep (validation) |
| `verify_server.py` | ? | **Syntax error** (line 9) | Fix or archive |
| `verify_strict_compliance.py` | 24 | - | Keep (validation) |
| `verify_telemetry.py` | 89 | - | Keep (telemetry) |

**Action:** Fix syntax error in `verify_server.py`, fix AgentQMS imports in `unified_server.py`

---

### Category 2: Migration Scripts (3 scripts) 🚚

**Status:** Likely obsolete (one-time migrations)

| Script | LOC | Purpose | Recommendation |
|--------|-----|---------|----------------|
| `migration_refactoring/fix_export_paths.py` | 98 | Fix export paths | Archive (migration done) |
| `migration_refactoring/migrate_checkpoint_names.py` | 150 | Rename checkpoints | Archive (migration done) |
| `migration_refactoring/migrate_to_underscore_naming.py` | 186 | Naming convention fix | Archive (migration done) |

**Action:** Archive to `scripts/_archive/migration/` with README explaining they're historical

---

### Category 3: Experimental Prototypes (5 scripts) 🧪

**Status:** Experimental/testing code

| Script | LOC | Purpose | Recommendation |
|--------|-----|---------|----------------|
| `prototypes/test_middleware.py` | 28 | Middleware test | Archive (prototype) |
| `prototypes/multi_agent/rabbitmq_producer.py` | 75 | RabbitMQ test | Archive (prototype) |
| `prototypes/multi_agent/rabbitmq_worker.py` | 79 | RabbitMQ test | Archive (prototype) |
| `prototypes/multi_agent/test_linting_loop.py` | 60 | Agent test | Archive (prototype) |
| `prototypes/multi_agent/test_ocr_loop.py` | 83 | Agent test | Archive (prototype) |
| `prototypes/multi_agent/test_slack.py` | 36 | Slack integration | Archive (prototype) |

**Action:** Archive to `scripts/_archive/prototypes/` unless actively used

---

### Category 4: Documentation Tools (3 scripts) 📚

**Status:** Documentation automation

| Script | LOC | Issue | Recommendation |
|--------|-----|-------|----------------|
| `documentation/generate_diagrams.py` | 614 | - | Keep if used regularly |
| `documentation/standardize_content.py` | 587 | - | Keep if used regularly |
| `documentation/translate_readme.py` | ? | **Syntax error** (line 148) | Fix or archive |

**Action:** Fix syntax error, verify if actively used

---

### Category 5: Demos & Testing (6 scripts) 🎨

**Status:** Demo/analysis scripts

| Script | LOC | Purpose | Recommendation |
|--------|-----|---------|----------------|
| `demos/analyze_preprocessing_results.py` | 279 | Analysis tool | Keep if valuable |
| `demos/demo_preprocessing_samples.py` | 229 | Demo | Archive (one-time demo) |
| `demos/test_perspective_inference.py` | 176 | Testing | Archive (old test) |
| `troubleshooting/test_basic_cuda.py` | 82 | CUDA test | Keep (useful utility) |
| `troubleshooting/test_model_forward_backward.py` | 141 | Model test | Keep (useful utility) |
| `debug/view_api_usage.py` | 35 | Debug tool | Keep (useful) |
| `debug/view_parquet.py` | 79 | Debug tool | Keep (useful) |

**Action:** Archive demos, keep troubleshooting/debug utilities

---

### Category 6: Data Processing (4 scripts) 📊

**Status:** Data utilities

| Script | LOC | Purpose | Recommendation |
|--------|-----|---------|----------------|
| `data/check_training_data.py` | 138 | Data validation | Keep |
| `data/download_hf_sample.py` | 37 | HF sample download | Keep (utility) |
| `data/etl/core.py` | 268 | ETL core | Keep (important) |
| `datasets/sample_images.py` | 74 | Image sampling | Keep (utility) |

**Action:** Keep all

---

### Category 7: Validation/Monitoring (4 scripts) ✅

**Status:** System utilities

| Script | LOC | Purpose | Recommendation |
|--------|-----|---------|----------------|
| `audit_config_compliance.py` | 257 | Config audit | Keep |
| `monitoring/process_monitor.py` | 268 | Process monitoring | Keep (useful) |
| `utilities/cache_manager.py` | 321 | Cache management | Keep (important) |
| `research/perplexity_client.py` | 117 | Perplexity API | Archive (research) |

**Action:** Keep monitoring/utilities, archive research

---

### Category 8: Hooks & CI (5 scripts) 🪝

**Status:** Git hooks and CI tools

| Script | LOC | Purpose | Recommendation |
|--------|-----|---------|----------------|
| `hooks/validate_architecture.py` | 180 | Pre-commit hook | Keep |
| `hooks/validate_path_usage.py` | 70 | Pre-commit hook | Keep |
| `ci/analyze_mypy_errors.py` | 40 | CI tool | Keep |
| `ci/check_test_isolation.py` | 89 | CI tool | Keep |
| `ci/forbid_pip_install.py` | 69 | CI tool | Keep |

**Action:** Keep all (active CI/hooks)

---

### Category 9: HuggingFace Tools (3 scripts) 🤗

**Status:** HF integration

| Script | LOC | Purpose | Recommendation |
|--------|-----|---------|----------------|
| `huggingface/hf_inference.py` | 173 | HF inference | Keep |
| `huggingface/hf_inference_simple.py` | 59 | Simple HF test | Archive (redundant with above) |
| `finetune_ppocr.py` | 190 | PPOCRv3 finetune | Archive (external model) |

**Action:** Keep main HF tool, archive redundant/external

---

### Category 10: Misc Utilities (6 scripts) 🔧

| Script | LOC | Purpose | Recommendation |
|--------|-----|---------|----------------|
| `_bootstrap.py` | 58 | Bootstrap utility | Review usage |
| `bug_tools/next_bug_id.py` | 69 | Bug tracking | Keep if used |

**Action:** Review if actively used

---

## Syntax Errors to Fix (3 scripts)

### Critical Fixes

1. **`scripts/documentation/translate_readme.py`**
   - Error: `expected an indented block after 'try' statement on line 148`
   - Action: Fix or archive

2. **`scripts/mcp/verify_server.py`**
   - Error: `expected an indented block after 'try' statement on line 9`
   - Action: Fix or archive

3. **`scripts/performance/decoder_benchmark.py`** (in refactor category)
   - Error: `unindent does not match any outer indentation level (line 295)`
   - Action: Fix indentation

---

## Proposed Actions

### Phase 1: Fix Syntax Errors (15 minutes)

```bash
# Fix or archive the 3 scripts with syntax errors
# Priority: verify_server.py (MCP tool - might be needed)
# Lower priority: translate_readme.py, decoder_benchmark.py (can archive)
```

### Phase 2: Archive Obsolete Scripts (30 minutes)

**Create archive structure:**
```
scripts/_archive/
├── README.md                     # Explains archive purpose
├── migration/                    # One-time migrations (3 scripts)
├── prototypes/                   # Experimental code (6 scripts)
├── demos/                        # One-time demos (2 scripts)
└── research/                     # Research tools (1 script)
```

**Total to archive:** ~12 scripts

### Phase 3: Fix Import Errors (20 minutes)

**Fix AgentQMS imports in `unified_server.py`:**
- Option 1: Remove AgentQMS features if not needed
- Option 2: Fix imports if AgentQMS is available
- Option 3: Make AgentQMS features optional (try/except)

### Phase 4: Consolidate Redundant Scripts (15 minutes)

**Consolidate HuggingFace tools:**
- Keep: `hf_inference.py`
- Archive: `hf_inference_simple.py` (redundant)

### Phase 5: Update Documentation (10 minutes)

**Create/update READMEs:**
- `scripts/README.md` - Main scripts index
- `scripts/_archive/README.md` - Archive explanation
- Update relevant docs with new structure

---

## Success Metrics

### Before Cleanup
- 128 total scripts
- 48 needing review
- 3 syntax errors
- 4 broken imports
- Unclear organization

### After Cleanup (Target)
- ~116 active scripts (12 archived)
- 0 syntax errors
- 2 broken imports (UI modules - expected)
- Clear organization with archive

### Context Reduction
- Reduce scripts directory cognitive load
- Clear separation: active vs archived
- Easier navigation for future development

---

## Risk Assessment

### Low Risk ✅
- Archiving migration scripts (one-time use)
- Archiving prototypes (experimental)
- Archiving demos (presentation-only)
- Fixing syntax errors (non-functional files)

### Medium Risk ⚠️
- Archiving HuggingFace simple tool (verify not used in CI/docs)
- Modifying unified_server.py imports (test thoroughly)
- Archiving documentation tools (verify not in automated workflows)

### Mitigation
- Create git branch before cleanup
- Move to `_archive/` instead of deleting (reversible)
- Test unified_server.py after import fixes
- Document all changes in archive README

---

## Implementation Timeline

| Phase | Duration | Priority | Blocker |
|-------|----------|----------|---------|
| 1. Fix syntax errors | 15 min | High | None |
| 2. Archive obsolete | 30 min | Medium | None |
| 3. Fix imports | 20 min | Medium | None |
| 4. Consolidate redundant | 15 min | Low | None |
| 5. Documentation | 10 min | Low | None |

**Total:** ~90 minutes

---

## User Decision Required

### Question 1: Archive Strategy

> Should we:
> - **Option A:** Move to `_archive/` (can be restored)
> - **Option B:** Delete with git history (permanent but reversible via git)
> - **Option C:** Comment out and mark deprecated (keeps in place)
>
> **Recommendation:** Option A (safest, reversible)

### Question 2: AgentQMS Features

> `unified_server.py` has broken AgentQMS imports:
> - **Option A:** Remove AgentQMS features completely
> - **Option B:** Make AgentQMS features optional (try/except)
> - **Option C:** Fix AgentQMS imports (requires AgentQMS setup)
>
> **Recommendation:** Option B (graceful degradation)

### Question 3: Syntax Error Scripts

> 3 scripts have syntax errors. Should we:
> - **Option A:** Fix them (investigate and repair)
> - **Option B:** Archive them (if not critical)
>
> **Recommendation:** Option B for translate_readme/decoder_benchmark, Option A for verify_server

---

## Next Steps

1. **Get user approval** for cleanup strategy
2. **Create git branch** for cleanup work
3. **Execute phases 1-5** as approved
4. **Test critical scripts** (unified_server.py, verify_server.py)
5. **Update pulse staging** with completion report
6. **Update project compass** with final status

---

## Files to Create/Modify

### New Files
- `scripts/_archive/README.md`
- `scripts/README.md` (if doesn't exist)
- `project_compass/pulse_staging/artifacts/SCRIPTS_CLEANUP_COMPLETION.md` (after execution)

### Files to Archive (12 scripts)
- 3 migration scripts
- 6 prototype scripts
- 2 demo scripts
- 1 research script

### Files to Fix (3 scripts)
- `scripts/mcp/verify_server.py` (syntax)
- `scripts/documentation/translate_readme.py` (syntax)
- `scripts/performance/decoder_benchmark.py` (syntax)

### Files to Modify (1 script)
- `scripts/mcp/unified_server.py` (fix AgentQMS imports)

---

**Status:** Ready for user review and approval
