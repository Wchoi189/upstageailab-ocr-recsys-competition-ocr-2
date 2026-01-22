# Task Execution Order - Debugging Session Workflow

## Session Start

1. **Environment validation**
   - Check editable install: `python -c "import ocr; print(ocr.__file__)"`
   - Run pre-flight: `bash scripts/audit/preflight.sh`

2. **Generate context outputs**
   - Master audit: `uv run python scripts/audit/master_audit.py > audit_report.txt`
   - Context tree: `uv run python -m agent_debug_toolkit.cli context-tree ocr/ --depth 3 --output json > context_tree.json`
   - AgentQMS standards: `aqms generate-config --path ocr/ > standards.yaml`

## Analysis Phase

3. **Review audit results**
   - Broken imports count
   - Broken Hydra targets
   - Prioritize by impact

4. **Identify patterns**
   - Duplicate files: `find ocr/ -name "*.py" | awk -F/ '{print $NF}' | sort | uniq -d`
   - Import chains: Review context tree
   - Config issues: Check Hydra interpolations

## Fixing Phase

5. **Fix in order**
   - **Priority 1**: Environment issues (ghost code, editable install)
   - **Priority 2**: Critical imports (training pipeline blockers)
   - **Priority 3**: Hydra targets (config alignment)
   - **Priority 4**: Non-critical imports (scripts, demos)

6. **Systematic approach**
   - Process 5-10 items at a time
   - Verify after each batch
   - Re-run audit to track progress

## Verification Phase

7. **Test pipelines**
   - Detection: `uv run python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True`
   - Recognition: `uv run python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True`

8. **Final audit**
   - Re-run master audit
   - Confirm 0 broken imports/targets
   - Document remaining issues

## Session End

9. **Documentation**
   - Update findings
   - Create walkthrough if needed
   - Export pulse (if using AgentQMS)

## Recommended Per-Session Tasks

### Quick Session (30-60 min)
- Environment validation
- Master audit
- Fix 5-10 high-priority items
- Verify one pipeline

### Standard Session (1-2 hours)
- Full context generation
- Master audit
- Fix 10-20 items in batches
- Test both pipelines
- Update documentation

### Deep Session (2-4 hours)
- All context outputs
- Systematic fixing (20-40 items)
- Duplicate file consolidation
- Full pipeline testing
- Comprehensive documentation

## Tools by Phase

**Analysis**:
- `master_audit.py`
- `context-tree`
- `analyze-dependencies`

**Fixing**:
- `sg-search` (find patterns)
- `intelligent-search` (locate symbols)
- `yq` (YAML updates)

**Verification**:
- `master_audit.py` (re-run)
- Pipeline smoke tests
- `validate_artifact` (if using AgentQMS)
