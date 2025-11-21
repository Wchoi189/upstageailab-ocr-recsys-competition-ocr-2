# Bug Report Protocol Improvements

**Status:** PROPOSAL
**Date:** 2025-11-09
**Purpose:** Document what needs to be done to make bug report generation a reliable, automated protocol

## Current Issues

### 1. Location Discrepancy
- **Protocol says:** `artifacts/bug_reports/` (project root)
- **Existing reports:** `docs/bug_reports/`
- **Impact:** Confusion about where to place bug reports

### 2. Template Integration Gap
- **Artifact workflow script** exists but may not support `bug_report` type
- **AgentQMS toolbelt** may not have `bug_report` as a registered artifact type
- **Bug ID generation** is separate from artifact creation

### 3. Manual Process
- Bug ID must be generated separately: `uv run python scripts/bug_tools/next_bug_id.py`
- Bug report must be created manually following template
- No automated integration between bug ID, template, and artifact creation

## Required Improvements

### 1. Standardize Location ✅ HIGH PRIORITY

**Option A: Move to `artifacts/bug_reports/` (Recommended)**
- Aligns with protocol
- Consistent with other artifacts (assessments, implementation_plans)
- Requires migration of existing reports

**Option B: Update Protocol to `docs/bug_reports/`**
- Keeps existing reports in place
- But breaks consistency with other artifacts

**Recommendation:** Option A - Move to `artifacts/bug_reports/`

### 2. Integrate Bug Report into AgentQMS ✅ HIGH PRIORITY

**Required Changes:**
1. **Add `bug_report` to AgentQMS manifest:**
   ```python
   # In agent_qms/manifest.yaml or equivalent
   artifact_types:
     - name: bug_report
       location: artifacts/bug_reports/
       template: bug_report
       schema: bug_report_schema
   ```

2. **Update artifact workflow to support bug reports:**
   - Integrate bug ID generation into artifact creation
   - Auto-generate bug ID when creating bug_report type
   - Use bug ID in filename generation

3. **Create bug report template in AgentQMS:**
   - Match format from `docs/bug_reports/BUG_REPORT_TEMPLATE.md`
   - Include all required sections
   - Auto-populate bug ID field

### 3. Create Unified Bug Report Helper ✅ HIGH PRIORITY

**Proposed Function:**
```python
# In scripts/agent_tools/core/bug_report_helper.py
def create_bug_report(
    title: str,
    summary: str,
    content: str,
    severity: str = "Medium",
    **kwargs
) -> str:
    """
    Create a bug report with automatic bug ID generation.

    Returns:
        Path to created bug report file
    """
    # 1. Generate bug ID
    bug_id = get_next_bug_id()

    # 2. Create artifact using AgentQMS
    artifact_path = toolbelt.create_artifact(
        artifact_type="bug_report",
        title=title,
        content=content,
        bug_id=bug_id,  # Pass bug ID to template
        severity=severity,
        **kwargs
    )

    # 3. Return path
    return artifact_path
```

**Usage:**
```python
from scripts.agent_tools.core.bug_report_helper import create_bug_report

bug_report_path = create_bug_report(
    title="Dice Loss Assertion Error",
    summary="Training crashes with AssertionError in dice loss",
    content="## Summary\n...",
    severity="High"
)
```

### 4. Update Protocol Documentation ✅ MEDIUM PRIORITY

**Update `docs/agents/protocols/governance.md`:**
- Clarify location: `artifacts/bug_reports/` (not `docs/bug_reports/`)
- Add example using unified helper
- Document migration path for existing reports

**Update `docs/agents/protocols/governance.md`:**
- Add bug report section with unified helper usage
- Link to bug fix protocol

### 5. Create Migration Script ✅ LOW PRIORITY

**Script to migrate existing reports:**
```python
# scripts/agent_tools/maintenance/migrate_bug_reports.py
def migrate_bug_reports():
    """
    Migrate bug reports from docs/bug_reports/ to artifacts/bug_reports/
    """
    source_dir = Path("docs/bug_reports/")
    target_dir = Path("artifacts/bug_reports/")

    # Move files
    # Update any references
    # Update indexes
```

## Implementation Checklist

### Phase 1: Foundation (Required for Reliability)
- [ ] **Decide on location** (recommend `artifacts/bug_reports/`)
- [ ] **Add `bug_report` to AgentQMS manifest**
- [ ] **Create bug report template in AgentQMS**
- [ ] **Integrate bug ID generation into artifact workflow**

### Phase 2: Automation (Makes it Reliable)
- [ ] **Create unified bug report helper function**
- [ ] **Update artifact workflow to auto-generate bug IDs**
- [ ] **Add bug report validation**
- [ ] **Update protocol documentation**

### Phase 3: Migration (Cleanup)
- [ ] **Create migration script**
- [ ] **Migrate existing reports to new location**
- [ ] **Update all references**
- [ ] **Update indexes**

## Recommended Immediate Actions

### For AI Agents (Now)
1. **Use existing template** from `docs/bug_reports/BUG_REPORT_TEMPLATE.md`
2. **Generate bug ID** using `uv run python scripts/bug_tools/next_bug_id.py`
3. **Create report manually** following template format
4. **Place in `docs/bug_reports/`** (current location) until migration

### For Project Maintainers (To Make Reliable)
1. **Standardize location** - Decide on `artifacts/bug_reports/` or `docs/bug_reports/`
2. **Integrate into AgentQMS** - Add bug_report as artifact type
3. **Create unified helper** - Automate bug ID + artifact creation
4. **Update protocol** - Make it clear and consistent

## Example: Reliable Protocol (After Implementation)

**For AI Agents:**
```python
from scripts.agent_tools.core.bug_report_helper import create_bug_report

# One function call does everything:
bug_report_path = create_bug_report(
    title="Dice Loss Assertion Error",
    summary="Training crashes with AssertionError in dice loss computation",
    content="""
## Summary
Training crashes with AssertionError...

## Root Cause
Numerical precision issue...

## Resolution
Added input clamping...
    """,
    severity="High",
    tags=["training", "loss", "assertion"]
)
```

**For Command Line:**
```bash
# Unified command that does everything:
uv run python scripts/agent_tools/core/artifact_workflow.py \
    create \
    --type bug_report \
    --name "dice-loss-assertion-error" \
    --title "Dice Loss Assertion Error" \
    --description "Training crashes with AssertionError..."
```

## Current Workaround

Until the protocol is fully automated, follow this process:

1. **Generate bug ID:**
   ```bash
   uv run python scripts/bug_tools/next_bug_id.py
   # Output: BUG-20251109-001
   ```

2. **Create bug report manually:**
   - Use template from `docs/bug_reports/BUG_REPORT_TEMPLATE.md`
   - Follow format from existing reports (e.g., `BUG_2025_002_MIXED_PRECISION_PERFORMANCE.md`)
   - Place in `docs/bug_reports/` (current location)

3. **Update changelog:**
   - Add entry to `docs/CHANGELOG.md` under [Unreleased] section

## Success Criteria

The protocol will be "reliable" when:
- ✅ AI agents can create bug reports with **one function call**
- ✅ Bug ID is **automatically generated** (no manual step)
- ✅ Template is **automatically applied** (no manual copy-paste)
- ✅ Location is **consistent** (no confusion about where to place)
- ✅ Validation is **automatic** (ensures all required fields)
- ✅ Integration with changelog is **automatic** (or at least documented)

---

**Next Steps:**
1. Review and approve location decision
2. Implement AgentQMS integration
3. Create unified helper function
4. Update protocol documentation
5. Migrate existing reports
