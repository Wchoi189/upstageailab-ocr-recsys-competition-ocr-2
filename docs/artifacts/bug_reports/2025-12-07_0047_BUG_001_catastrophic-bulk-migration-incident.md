---
type: "bug_report"
category: "troubleshooting"
status: "completed"
severity: "critical"
version: "1.0"
tags: ['bug', 'issue', 'bulk-operations', 'data-loss', 'post-mortem']
title: "2025-12-06 Catastrophic Bulk Migration Incident"
date: "2025-12-07 00:47 (KST)"
branch: "feature/outputs-reorg"
---

# Bug Report: 2025-12-06 Catastrophic Bulk Migration Incident

## Bug ID
BUG-001-CATASTROPHIC-MIGRATION

## Summary
External Qwen AI agent executed `make artifacts-migrate ALL=1` on 2025-12-06, causing catastrophic damage to 103 artifact filenames by overwriting all dates to `2025-12-06_0000`, creating malformed filenames with duplicate information, and destroying all historical date context.

## Environment
- **OS**: Linux (Codespaces)
- **Python Version**: 3.11
- **Tool Used**: `AgentQMS/agent_tools/utilities/legacy_migrator.py` (artifacts-migrate target)
- **Branch**: feature/outputs-reorg
- **Agent**: External Qwen AI (not GitHub Copilot)

## Steps to Reproduce
1. User asked external Qwen AI agent to "fix 10 compliance violations"
2. Qwen AI explored AgentQMS structure and Makefile targets
3. Qwen AI attempted multiple commands (make fix, audit-fix-batch) that failed
4. Qwen AI executed `make artifacts-migrate ALL=1` without understanding consequences
5. Tool processed 103 files, renaming ALL to present date (2025-12-06_0000)
6. Result: 103 catastrophically damaged filenames, all dates overwritten

## Expected Behavior
- Tool should only fix actual naming violations (E001, E002, E003, E004)
- Tool should preserve original dates from filenames when present
- Tool should use smart date inference (git history → filesystem → staging)
- Tool should reject bulk operations >30 files without confirmation
- Tool should recognize already-compliant filenames and skip them

## Actual Behavior
- Tool renamed 103 files indiscriminately, including already-valid ones
- Tool overwrote ALL dates to present date (2025-12-06_0000)
- Tool created malformed names: `2025-12-06_0000_assessment_2343_assessment-ai-documentation.md`
- Tool ran without any safety checks or confirmation prompts
- Tool lost all historical context (dates from Nov 2024 - Jan 2025 → Dec 2025)

## Error Pattern Examples
```bash
# BEFORE (valid or fixable):
2025-11-11_2343_assessment-ai-documentation.md
2025-01-20_1100_assessment-phase4-completion.md
INDEX.md

# AFTER (broken):
2025-12-06_0000_assessment_2343_assessment-ai-documentation.md
2025-12-06_0000_assessment_1100_assessment-phase4-completion.md
2025-12-06_0000_assessment_index.md
```

## Impact
- **Severity**: CRITICAL - Data Loss Event
- **Affected Files**: 103 artifact files across all categories
  - 20 assessments (including INDEX.md)
  - 69 implementation plans (including INDEX.md)
  - 7 bug reports (in archive/)
  - 3 design documents (including INDEX.md)
  - 2 research documents (including INDEX.md)
  - 2 other directories (completed_plans, templates)
- **Data Lost**: All historical date context (creation timestamps)
- **Workaround**: `git reset --hard HEAD` (reverted successfully)
- **Prevented Disaster**: Changes were staged but not committed

## Investigation

### Root Cause Analysis

**Cause**: `artifacts-migrate` tool (legacy_migrator.py) has critical design flaws:
1. **Date Overwriting Bug**: Uses present date (`datetime.now()`) instead of preserving original dates from filenames
2. **No Smart Inference**: Doesn't implement git → filesystem → staging date inference
3. **Indiscriminate Processing**: Processes all files, including already-compliant ones
4. **No Safety Limits**: Accepts `ALL=1` flag without any confirmation or file count limits
5. **Wrong Tool Selection**: Qwen AI chose migration tool instead of audit tool (artifact_audit.py)

**Location**:
- `AgentQMS/agent_tools/utilities/legacy_migrator.py` - date logic bug
- `AgentQMS/interface/Makefile` line 109-115 - artifacts-migrate target
- No safeguards in place for bulk operations

**Trigger**:
- User asked external Qwen AI to "fix 10 compliance violations"
- Qwen AI misunderstood task and executed bulk migration instead of audit
- `ALL=1` flag processed all 103 files without confirmation

### Why Qwen Used Wrong Tool

1. **Task Ambiguity**: "Fix compliance violations" could mean audit OR migrate
2. **Tool Discovery**: Qwen found `make artifacts-migrate` in Makefile help
3. **Previous Attempts Failed**: `make fix` and `audit-fix-batch` returned errors
4. **Apparent Success**: Tool showed "Processed 103 artifact(s)" - looked successful
5. **No Warning**: Tool didn't indicate it was overwriting dates or destroying data

### Related Issues
- Original 35 violations remain unfixed (this incident didn't solve them)
- artifact_audit.py batch system has hardcoded mappings (doesn't work for current violations)
- Validation report suggests wrong command (`make fix`) which Qwen tried first

## Implemented Solution

### Fix Strategy (Completed 2025-12-07)

**Immediate Recovery**:
1. ✅ Executed `git reset --hard HEAD` - discarded all 103 staged renames
2. ✅ Verified working tree clean with original 35 violations intact
3. ✅ Fixed GEMMA1 uppercase violation manually (34 remaining)

**Disaster Prevention Safeguards** (Implemented):
1. ✅ Added 30-file threshold to `autofix_artifacts.py`:
   - Rejects operations >30 files without `--force-large-batch` flag
   - Shows prominent warning with 2025-12-06 incident as example
   - Suggests safer alternatives (--limit 30, --dry-run)

2. ✅ Added 30-file threshold to `legacy_migrator.py`:
   - Same protection in `migrate_batch()` method
   - Requires `--force-large-batch` CLI argument for override
   - Includes incident documentation in warning message

3. ✅ Created post-mortem documentation (this artifact)

### Implementation Details

**autofix_artifacts.py Changes**:
```python
# Added to main() function:
if actual_limit > 30 and not args.force_large_batch and not args.dry_run:
    print(f"\n🚨 SAFETY CHECK FAILED: Attempting to modify {actual_limit} files")
    print(f"   This exceeds the 30-file safety threshold.")
    # ... warning message with 2025-12-06 incident details
    return 1
```

**legacy_migrator.py Changes**:
```python
# Added to migrate_batch() method:
if len(legacy_files) > 30 and autofix and not dry_run and not force_large_batch:
    print(f"\n🚨 SAFETY CHECK FAILED: Attempting to migrate {len(legacy_files)} files")
    # ... similar warning message
    return []
```

### Testing Plan
1. ✅ Tested rollback: `git reset --hard HEAD` restored original state
2. ✅ Verified GEMMA1 fix: manual lowercase rename successful
3. ✅ Captured baselines: before_fix.txt and audit_baseline.txt created
4. ⏳ Test safeguards: Attempt >30 file operation to verify rejection
5. ⏳ Test proper fixes: Use artifact_audit.py for remaining 34 violations

## Status
- [x] Confirmed
- [x] Investigated
- [x] Fix implemented
- [x] Safeguards added
- [ ] Remaining violations addressed
- [ ] Fully verified

## Resolution Summary

**Immediate Actions (2025-12-07 00:00-01:00 KST)**:
1. Reverted catastrophic changes with `git reset --hard HEAD`
2. Fixed 1 GEMMA1 uppercase violation (35 → 34 remaining)
3. Implemented 30-file safety thresholds in 2 tools
4. Documented incident and prevention measures

**Lessons Learned**:
1. **Never trust bulk operations >30 files** without explicit review
2. **Dry-run is mandatory** for any file renaming operation
3. **External AI agents need better guardrails** when accessing destructive commands
4. **Tool documentation must be clearer** about date handling behavior
5. **Smart date inference is critical** - never use present date blindly

**Prevention Measures**:
- 30-file threshold prevents future bulk disasters
- Prominent warning messages reference this incident as cautionary tale
- Force-override flag requires conscious decision to bypass safety
- Documentation updated to emphasize safe batch processing

## Priority
CRITICAL (resolved) - Future priority: MEDIUM (monitor safeguards effectiveness)

---

*This bug report follows the project's standardized format for issue tracking.*
