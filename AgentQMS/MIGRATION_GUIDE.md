# AgentQMS Migration Guide
## Upgrading to Unified Registry & QMS CLI (v0.3.0)

**Date:** 2026-01-20
**Branch:** `claude/refactor-agentqms-framework-Wx2i3`
**Breaking Changes:** None (backward compatible)

---

## Overview

AgentQMS v0.3.0 introduces significant improvements to reduce token usage and simplify the agent interface:

1. **Unified Standards Registry** - Consolidated discovery logic
2. **QMS CLI** - Single entry point for all AgentQMS tools
3. **Path-Aware Discovery** - Dynamic standard loading based on working directory

**Token Savings:** ~10,800 tokens per agent session (~87% reduction in standard loading)

---

## What Changed

### 1. Unified Standards Registry

**Before:**
- `AgentQMS/standards/INDEX.yaml` (glob-based discovery)
- `AgentQMS/standards/standards-router.yaml` (keyword-based routing)

**After:**
- `AgentQMS/standards/registry.yaml` (unified discovery with both keywords and path patterns)

**Status:** Old files archived to `AgentQMS/standards/.archive/`

---

### 2. QMS Unified CLI

**Before:** 5 separate tool scripts
```yaml
tool_mappings:
  artifact_workflow: ../tools/core/artifact_workflow.py
  validate_artifacts: ../tools/compliance/validate_artifacts.py
  monitor_artifacts: ../tools/compliance/monitor_artifacts.py
  agent_feedback: ../tools/utilities/agent_feedback.py
  documentation_quality_monitor: ../tools/compliance/documentation_quality_monitor.py
```

**After:** Single CLI with subcommands
```yaml
tool_mappings:
  qms:
    path: ../bin/qms
    description: Unified AgentQMS CLI tool
    subcommands:
      - artifact: Artifact workflow management
      - validate: Validate artifacts and compliance
      - monitor: Monitor artifact organization and compliance
      - feedback: Collect agent feedback and suggestions
      - quality: Documentation quality monitoring
      - generate-config: Generate effective.yaml with path-aware discovery
```

**Status:** Legacy tools marked as deprecated but still functional

---

### 3. Path-Aware Discovery

**New Feature:** Dynamic standard loading based on current working directory

```bash
# Generate context-aware configuration
qms generate-config --path ocr/inference

# Output includes only relevant standards (3/24 = 87.5% reduction)
context_integration:
  active_standards:
    - AgentQMS/standards/tier2-framework/coding/python-core.yaml
    - AgentQMS/standards/tier1-sst/file-placement-rules.yaml
    - AgentQMS/standards/tier2-framework/tool-catalog.yaml
```

---

## Migration Steps

### For AI Agents

**Recommended:** Update to use the new `qms` CLI

#### Old Command Format
```bash
python AgentQMS/tools/compliance/validate_artifacts.py --all
python AgentQMS/tools/compliance/monitor_artifacts.py --check
python AgentQMS/tools/core/artifact_workflow.py create --type plan --name my-feature
```

#### New Command Format (Recommended)
```bash
qms validate --all
qms monitor --check
qms artifact create --type implementation_plan --name my-feature --title "My Feature"
```

#### Path-Aware Context Loading
```bash
# Before: Load all standards (24 files, ~12,000 tokens)
cat AgentQMS/standards/INDEX.yaml

# After: Load only relevant standards (3 files, ~1,500 tokens)
qms generate-config --path ocr/inference --dry-run
```

---

### For Developers

**No changes required** - all existing scripts and imports continue to work via compatibility shims.

#### Import Compatibility (OCR Domain Refactor)

If you have tests or scripts importing from old paths, they will continue to work:

```python
# Old imports (still work via shims)
from ocr.data.datasets.db_collate_fn import DBCollateFN
from ocr.core.utils.geometry_utils import calculate_cropbox
from ocr.core.lightning.ocr_pl import OCRPLModule

# New imports (preferred)
from ocr.domains.detection.data.collate_db import DBCollateFN
from ocr.domains.detection.utils.geometry import calculate_cropbox
from ocr.core.lightning.ocr_pl import OCRPLModule
```

---

## Command Reference

### Artifact Management

| Old Command | New Command |
|------------|-------------|
| `make create-plan NAME=foo TITLE="Foo"` | `qms artifact create --type implementation_plan --name foo --title "Foo"` |
| `make validate` | `qms validate --all` |
| `make validate-file FILE=path/to/file.md` | `qms validate --file path/to/file.md` |
| `make reindex` | `qms artifact update-indexes` |

### Compliance & Monitoring

| Old Command | New Command |
|------------|-------------|
| `make compliance` | `qms monitor --check` |
| `python ../tools/compliance/monitor_artifacts.py --report` | `qms monitor --report` |
| `python ../tools/compliance/monitor_artifacts.py --alert` | `qms monitor --alert` |

### Feedback & Quality

| Old Command | New Command |
|------------|-------------|
| `python ../tools/utilities/agent_feedback.py ...` | `qms feedback report --issue-type "..." --description "..."` |
| `python ../tools/compliance/documentation_quality_monitor.py --check` | `qms quality --check` |

### Configuration Generation

| Old Command | New Command |
|------------|-------------|
| N/A (manual) | `qms generate-config --path <current_path>` |
| N/A | `qms generate-config --dry-run` (preview) |

---

## Testing the Migration

### 1. Test QMS CLI
```bash
# Verify CLI is working
qms --help
qms validate --help
qms artifact --help

# Test validation
qms validate --all
```

### 2. Test Path-Aware Discovery
```bash
# Test with different paths
qms generate-config --path ocr/inference --dry-run
qms generate-config --path configs/domain --dry-run
qms generate-config --path tests --dry-run
```

### 3. Monitor Token Usage
```bash
# Run token usage analysis
python AgentQMS/bin/monitor-token-usage.py
python AgentQMS/bin/monitor-token-usage.py --path ocr/inference
python AgentQMS/bin/monitor-token-usage.py --detailed
```

---

## Deprecation Timeline

### Phase 1: Soft Deprecation (Current)
- **Status:** All legacy tools functional with deprecation warnings
- **Timeline:** 3-6 months
- **Action:** Update AI prompts and documentation to use `qms` CLI

### Phase 2: Hard Deprecation (Future)
- **Status:** Legacy tools moved to `.archive/`
- **Timeline:** 6-12 months after Phase 1
- **Action:** Remove legacy tool paths from settings.yaml

### Phase 3: Complete Removal (Future)
- **Status:** Legacy tool files deleted
- **Timeline:** 12+ months after Phase 2
- **Action:** Update all codebase references

---

## Performance Metrics

### Token Usage Improvements

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| Tool Entries in Context | 5 separate tools | 1 unified CLI | 300 tokens |
| Standards Loaded (OCR inference) | 24 files (~12,000 tokens) | 3 files (~1,500 tokens) | 10,500 tokens (87.5%) |
| **Total per Session** | ~12,500 tokens | ~1,800 tokens | **~10,700 tokens (85.6%)** |

### Decision Complexity Reduction

- **AI Action Space:** 5 tool choices → 1 unified interface
- **Logic Fragmentation:** 2 discovery files → 1 registry
- **Context Relevance:** Static loading → Dynamic path-aware

---

## FAQs

### Q: Do I need to update my existing scripts?
**A:** No. All legacy imports and tools continue to work via compatibility shims.

### Q: When will legacy tools be removed?
**A:** Not for at least 6 months. See Deprecation Timeline above.

### Q: How do I know which standards will be loaded?
**A:** Run `qms generate-config --path <your_path> --dry-run` to preview.

### Q: Can I still use the Makefile commands?
**A:** Yes, but they internally call the `qms` CLI now. Direct `qms` usage is recommended for clarity.

### Q: What if I find a bug in the new system?
**A:** Use `qms feedback report --issue-type "bug" --description "..."` or create a GitHub issue.

---

## Rollback Plan

If you encounter issues, you can temporarily revert:

```bash
# Use legacy tools directly
python AgentQMS/tools/compliance/validate_artifacts.py --all
python AgentQMS/tools/core/artifact_workflow.py create --type plan --name foo

# Use archived discovery files (read-only)
cat AgentQMS/standards/.archive/INDEX.yaml
cat AgentQMS/standards/.archive/standards-router.yaml
```

**Note:** New features (path-aware discovery) require the new system.

---

## Support

For questions or issues:
- **Documentation:** `AgentQMS/bin/README.md`
- **CLI Help:** `qms --help`
- **Monitoring:** `python AgentQMS/bin/monitor-token-usage.py`
- **Feedback:** `qms feedback report --issue-type "question" --description "..."`

---

## Changelog

### v0.3.0 (2026-01-20)
- ✅ Unified standards registry (`registry.yaml`)
- ✅ QMS unified CLI (`qms` command)
- ✅ Path-aware standard discovery
- ✅ Token usage monitoring tool
- ✅ Backward compatibility shims
- ✅ CI workflow fixes

### v0.2.0 (Previous)
- Split discovery logic (INDEX.yaml + standards-router.yaml)
- Separate tool scripts
- Static standard loading
