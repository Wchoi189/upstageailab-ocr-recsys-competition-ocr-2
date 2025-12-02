# **filename: docs/ai_handbook/02_protocols/development/12_bug_fix_protocol.md**
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=bug_fixes,critical_issues,documentation -->

# **Protocol: Bug Fix Documentation**

Last Updated: 2025-10-20

## **Overview**
This protocol provides comprehensive guidelines for documenting bug fixes, ensuring consistent reporting, proper categorization, and maintainable documentation structure. All bug fixes must follow this protocol to maintain project quality and debugging traceability.

## **Prerequisites**
- Access to project repository and documentation
- Understanding of the bug report template (`docs/bug_reports/BUG_REPORT_TEMPLATE.md`)
- Familiarity with the project's development workflow
- Access to bug reports directory (`docs/bug_reports/`)

## **Procedure**

### **Step 1: Assess Bug Severity**
Determine the appropriate documentation level based on bug impact:

**Critical Bugs** (require full bug report):
- System crashes or data corruption
- Security vulnerabilities
- Complete loss of functionality
- Breaking changes affecting multiple users

**Standard Bugs** (require bug report):
- Functional errors with clear reproduction steps
- Performance degradation
- UI/UX issues affecting usability
- Configuration errors

**Quick Fixes** (use `docs/quick_reference/QUICK_FIXES.md`):
- Minor patches (< 5 lines changed)
- Hotfixes applied during development
- Cosmetic fixes
- Dependency updates

### **Step 2: Generate Bug ID**
Use the standardized per-day counter format: `BUG-YYYYMMDD-###`.

- Format: `BUG-YYYYMMDD-###` where YYYYMMDD is the UTC date and ### is a zero-padded daily counter starting at 001
- Always generate IDs using the helper script to avoid collisions
- Helper (optional):
  - uv run python scripts/bug_tools/next_bug_id.py            # prints next ID and reserves it
  - uv run python scripts/bug_tools/next_bug_id.py --peek     # prints next ID without reserving
  - uv run python scripts/bug_tools/next_bug_id.py --reset    # admin only; resets today’s counter
- Examples: `BUG-20251020-001`, `BUG-20251020-002`

### **Step 3: Create Bug Report**
Create a new bug report file following the template:

**Location**: `docs/bug_reports/BUG-YYYYMMDD-###_descriptive_name.md`
**Template**: Use `docs/bug_reports/BUG_REPORT_TEMPLATE.md`

**Required Sections**:
- Bug ID, Date, Reporter, Severity, Status
- Summary, Environment, Steps to Reproduce
- Expected vs Actual Behavior
- Root Cause Analysis, Resolution
- Testing, Prevention, Files Changed
- Impact Assessment

### **Step 4: Update Related Documentation**
Update all relevant documentation files:

**Changelog Entry** (`docs/CHANGELOG.md`):
```markdown
#### Bug Fixes
- **BUG-20251020-002**: Fixed inference UI coordinate transformation bug causing annotation misalignment for EXIF-oriented images (BUG-20251020-002_inference_ui_coordinate_transformation.md)
```

**Quick Fixes Log** (`docs/quick_reference/QUICK_FIXES.md`) - if applicable:
```markdown
## 2025-10-19 10:30 BUG - Inference UI coordinate transformation

**Issue**: OCR annotations misaligned for EXIF-oriented images
**Fix**: Removed incorrect inverse transformations in InferenceEngine
**Files**: ui/utils/inference/engine.py
**Impact**: minimal
**Test**: ui
```

### **Step 5: Verify Documentation**
Ensure all documentation is consistent and properly linked:

- [ ] Bug report follows template format
- [ ] Bug ID is unique and properly formatted
- [ ] File location follows naming convention
- [ ] Changelog references bug report correctly
- [ ] All file paths are accurate
- [ ] Cross-references are working

## **File Organization**

### **Bug Reports Directory Structure**
```
docs/bug_reports/
├── BUG_REPORT_TEMPLATE.md          # Template for new reports
├── BUG-20251019-001_*\.md          # Per-day counter examples
├── BUG-20251019-002_*\.md
└── BUG-20251020-001_*\.md
```

### **Naming Convention**
- **Format**: `BUG-YYYYMMDD-###_descriptive_name.md`
- **Date**: Current date (YYYYMMDD)
- **Number**: Sequential within day (001, 002, etc.)
- **Description**: Brief, descriptive name using underscores
- **Examples**:
  - `BUG-20251020-001_inference_ui_coordinate_transformation.md`
  - `BUG-20251020-002_pydantic_validation_error.md`

### **Cross-Reference Requirements**
- Changelog entries must reference bug reports
- Bug reports should reference related issues
- Quick fixes should reference bug reports when applicable

## **Examples**

### **Critical Bug Report Structure**
```markdown
## 🐛 Bug Report Template

**Bug ID:** BUG-YYYYMMDD-###
**Date:** YYYY-MM-DD
**Reporter:** Development Team
**Severity:** Critical
**Status:** Fixed

### Summary
[Brief description of the bug and fix]

### Environment
[Technical context and affected components]

### Steps to Reproduce
[Clear reproduction steps]

### Root Cause Analysis
[Technical analysis of the problem]

### Resolution
[How the bug was fixed]

### Testing
[Validation performed]

### Files Changed
[Affected files list]
```

## Index signatures in code (required for core changes)

When a bug fix changes core behavior, add a concise, machine-greppable index signature comment near the changed function(s) or config block(s):

Recommended format:

```
# BugRef: BUG-YYYYMMDD-### — short-title
# Report: docs/bug_reports/BUG-YYYYMMDD-###_short-title.md
# Date: YYYY-MM-DD
# IndexSig: key1=value1; key2=value2  # brief technical summary of the change
```

Guidelines:
- Place directly above the function or config stanza that changed
- Keep under 3 lines excluding the IndexSig line
- Use consistent prefixes: `BugRef:`, `Report:`, `Date:`, `IndexSig:`
- For follow-up changes, append another BugRef block rather than editing history

### **Changelog Integration**
```markdown
#### Bug Fixes
- **BUG-2025-011**: Fixed inference UI coordinate transformation bug causing annotation misalignment for EXIF-oriented images (BUG-2025-011_inference_ui_coordinate_transformation.md)
```

## **Quality Assurance**
- **Template Compliance**: All bug reports must use the standard template
- **ID Uniqueness**: Bug IDs must be unique across the project
- **Location Consistency**: Bug reports must be in `docs/bug_reports/` directory
- **Cross-References**: All related documentation must be properly linked
- **Content Completeness**: All required sections must be filled out

## **Automation Support**
Use available tools for bug fix documentation:

```bash
# Validate bug report format
python scripts/agent_tools/validate_bug_report.py --file docs/bug_reports/BUG-2025-011_inference_ui_coordinate_transformation.md

# Generate changelog entry
python scripts/agent_tools/generate_changelog_entry.py --bug-id BUG-2025-011
```

## **Common Pitfalls**
- **Wrong Location**: Bug reports should not go in changelog directories
- **Inconsistent Naming**: Always use BUG-YYYY-NNN format
- **Missing Cross-References**: Ensure changelog and quick fixes reference bug reports
- **Incomplete Documentation**: Fill out all template sections
- **Duplicate IDs**: Check existing reports before assigning new IDs</content>
<parameter name="filePath">/home/vscode/workspace/upstageailab-ocr-recsys-competition-ocr-2/docs/ai_handbook/02_protocols/development/12_bug_fix_protocol.md
