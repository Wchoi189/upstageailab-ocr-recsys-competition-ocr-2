---
ads_version: "1.0"
title: "Ai Documentation Entry Points Audit And Reorganization Plan"
date: "2025-12-06 18:09 (KST)"
type: "assessment"
category: "evaluation"
status: "active"
version: "1.0"
tags: ['assessment', 'evaluation', 'documentation']
---



# AI Documentation Entry Points Audit

## Executive Summary

This audit examines AI agent documentation entry points for inefficiency, obsolete instructions, verbosity, and structural problems causing recurring compliance violations. The analysis identifies root causes and proposes solutions to improve agent effectiveness and reduce manual intervention.

## 1. Current Entry Points Inventory

### 1.1 Primary Entry Points

#### `.cursor/rules/prompts-artifacts-guidelines.mdc` (Always Applied)
- **Status**: Active, always applied
- **Purpose**: Quick reference for artifact creation and critical rules
- **Length**: 167 lines
- **Issues Identified**:
  - Duplicates content from `AgentQMS/knowledge/system.md`
  - Verbose artifact creation examples (lines 33-116)
  - References deprecated/legacy methods alongside preferred methods
  - Missing explicit links to data contracts and API references
  - No coding standards section
  - No periodic feedback mechanism mentioned

#### `AGENT_ENTRY.md` (Project Root)
- **Status**: Active, navigation guide
- **Purpose**: Single entry point navigation
- **Length**: 36 lines
- **Issues Identified**:
  - Minimal content, relies entirely on `AgentQMS/knowledge/system.md`
  - Does not surface critical information (data contracts, API refs, coding standards)
  - No validation or compliance reminders
  - No feedback mechanism

#### `AgentQMS/knowledge/system.md` (Single Source of Truth)
- **Status**: Active, version 1.1
- **Purpose**: Comprehensive AI agent instructions
- **Length**: 289 lines
- **Issues Identified**:
  - **VERBOSE**: Contains operational commands (lines 154-231) that belong in references
  - **MISSING**: No explicit section on data contracts (`docs/pipeline/data_contracts.md`)
  - **MISSING**: No explicit section on API references (`docs/backend/api/pipeline-contract.md`)
  - **MISSING**: Coding standards buried in protocols (should be prominent)
  - **MISSING**: No periodic feedback/status update mechanism
  - **ORGANIZATION**: Critical rules mixed with operational details
  - **DUPLICATION**: Artifact creation instructions duplicated across multiple files

### 1.2 Secondary Entry Points

#### `AgentQMS/knowledge/index.md`
- **Status**: Active, documentation map
- **Issues**: Does not highlight critical references (data contracts, API refs)

#### `.cursor/rules/prompts-streamlit-verification.mdc`
- **Status**: Active, conditionally applied
- **Issues**: Very specific, could be referenced rather than duplicated

## 2. Recurring Problems Analysis

### 2.1 Problem: Filename, Location, and Frontmatter Format Violations

**Root Causes**:
1. **Scattered Instructions**: Rules appear in 3+ locations with slight variations
2. **No Validation Reminders**: Agents not reminded to validate before committing
3. **Complex Naming Rules**: Timestamped vs semantic naming not clearly distinguished
4. **Frontmatter Schema Not Visible**: Schema location not prominently displayed

**Evidence**:
- Artifact creation rules duplicated in:
  - `.cursor/rules/prompts-artifacts-guidelines.mdc` (lines 29-128)
  - `AgentQMS/knowledge/system.md` (lines 16-116)
  - `AgentQMS/knowledge/protocols/governance.md` (lines 7-130)
- Naming conventions mentioned in 3 different formats across files

### 2.2 Problem: Project Documentation Rarely Referenced

**Root Causes**:
1. **No Prominent Links**: Data contracts and API references not in entry points
2. **Buried in References**: `AgentQMS/knowledge/references/` exists but not surfaced
3. **No Mandatory Checks**: No workflow step requiring documentation review
4. **Context Switching**: Agents must navigate away from entry points to find docs

**Evidence**:
- `docs/pipeline/data_contracts.md` (698+ lines) - Critical for preventing shape errors
- `docs/backend/api/pipeline-contract.md` - API contract documentation
- Neither mentioned in `.cursor/rules/` or `AGENT_ENTRY.md`
- Only found via `AgentQMS/knowledge/index.md` → `references/` → indirect

### 2.3 Problem: Lack of Regard for Artifact Generation Rules

**Root Causes**:
1. **Multiple Conflicting Examples**: Three different artifact creation methods shown
2. **Legacy Methods Prominent**: Legacy CLI shown alongside preferred toolbelt
3. **No Enforcement**: No pre-commit hooks or validation reminders
4. **Complex Workflows**: Bug report workflow requires 2-3 steps (ID generation → creation)

**Evidence**:
- `.cursor/rules/prompts-artifacts-guidelines.mdc` shows both toolbelt AND legacy CLI
- `AgentQMS/knowledge/system.md` shows both methods with equal prominence
- No clear "USE THIS, NOT THAT" hierarchy

### 2.4 Problem: Does Not Follow Coding Standards

**Root Causes**:
1. **Standards Not Prominent**: Coding standards in `AgentQMS/knowledge/protocols/development.md` (line 5-20)
2. **No Quick Reference**: Must navigate to protocols to find standards
3. **No Validation Reminders**: No mention of `ruff check` or `mypy` in entry points
4. **Standards Scattered**: Formatting, naming, type hints in different sections

**Evidence**:
- Coding standards section exists but not linked from entry points
- No mention in `.cursor/rules/` files
- No quick reference in `AGENT_ENTRY.md`

### 2.5 Problem: Agents Do Not Provide Automatic Feedback Periodically

**Root Causes**:
1. **No Mechanism Defined**: No protocol for periodic status updates
2. **No Templates**: No status update templates or formats
3. **No Triggers**: No defined intervals or event-based triggers
4. **No Tracking**: No system to track when last feedback was provided

**Evidence**:
- No mention of periodic feedback in any entry point
- Progress trackers exist for implementation plans but not for general work
- No status update protocol in `AgentQMS/knowledge/protocols/`

## 3. Structural Problems

### 3.1 Information Architecture Issues

**Problems**:
1. **Entry Points Too Verbose**: `system.md` mixes critical rules with operational commands
2. **Lack of Hierarchy**: No clear "must read" vs "reference" distinction
3. **No Progressive Disclosure**: All information shown at once, overwhelming
4. **Missing Quick Wins**: Critical info (data contracts, coding standards) buried

**Impact**: Agents skip reading or miss critical information

### 3.2 Duplication and Inconsistency

**Problems**:
1. **Artifact Rules Duplicated**: Same rules in 3+ locations
2. **Slight Variations**: Each location has slightly different wording/examples
3. **Maintenance Burden**: Changes must be made in multiple places
4. **Confusion**: Agents see conflicting information

**Impact**: Rules drift, agents follow wrong version

### 3.3 Missing Self-Regulation Mechanisms

**Problems**:
1. **No Validation Hooks**: No automated checks before commits
2. **No Compliance Checks**: No periodic validation of agent output
3. **No Feedback Loops**: No mechanism to surface violations immediately
4. **No Learning**: No system to improve based on violations

**Impact**: Violations discovered late, require manual intervention

## 4. Recommendations

### 4.1 Immediate Actions (High Priority)

#### 4.1.1 Restructure Entry Points

**Action**: Create clear hierarchy:
1. **`.cursor/rules/prompts-artifacts-guidelines.mdc`** → Ultra-concise critical rules only
2. **`AGENT_ENTRY.md`** → Enhanced with prominent links to critical docs
3. **`AgentQMS/knowledge/system.md`** → Split into:
   - `system.md` - Core rules only (50-100 lines)
   - `AgentQMS/knowledge/references/operations.md` - Operational commands
   - `AgentQMS/knowledge/references/quick-reference.md` - Quick lookup tables

**Benefits**:
- Reduced cognitive load
- Faster agent onboarding
- Clearer information hierarchy

#### 4.1.2 Add Critical Documentation Links

**Action**: Add prominent sections to entry points:
- **Data Contracts**: `docs/pipeline/data_contracts.md` - REQUIRED before modifying pipeline
- **API References**: `docs/backend/api/pipeline-contract.md` - REQUIRED before API changes
- **Coding Standards**: Quick reference in entry points, full details in protocols

**Implementation**:
```markdown
## 🚨 CRITICAL: Before Modifying Code

**MUST READ**:
- Data Contracts: `docs/pipeline/data_contracts.md` (prevents shape errors)
- API Contracts: `docs/backend/api/pipeline-contract.md` (prevents API violations)
- Coding Standards: `AgentQMS/knowledge/protocols/development.md#coding-standards`
```

#### 4.1.3 Consolidate Artifact Creation Instructions

**Action**: Single source of truth with clear hierarchy:
1. **Preferred**: AgentQMS toolbelt (Python)
2. **Alternative**: CLI script (for non-Python contexts)
3. **Deprecated**: Remove legacy methods from entry points

**Implementation**: Update `.cursor/rules/` to show ONLY preferred method, link to alternatives

### 4.2 Medium-Term Improvements

#### 4.2.1 Implement Validation Reminders

**Action**: Add validation checklist to entry points:
```markdown
## ✅ Pre-Commit Checklist

Before committing, verify:
- [ ] Artifact frontmatter validated: `python scripts/agent_tools/documentation/validate_manifest.py`
- [ ] Code formatted: `uv run ruff format .`
- [ ] Code checked: `uv run ruff check . --fix`
- [ ] Data contracts reviewed (if pipeline changes)
- [ ] API contracts reviewed (if API changes)
```

#### 4.2.2 Create Periodic Feedback Protocol

**Action**: Define feedback mechanism:
1. **Template**: Create `AgentQMS/knowledge/protocols/status-update.md`
2. **Triggers**: Every N tasks, on blockers, on completion
3. **Format**: Structured status update with progress, blockers, next steps
4. **Storage**: Status updates in `docs/sessions/` or artifacts

**Implementation**:
```markdown
## Periodic Status Updates

**When**: Every 5 tasks, on blockers, on major milestones
**Format**: See `AgentQMS/knowledge/protocols/status-update.md`
**Location**: `docs/sessions/YYYY-MM-DD_status.md`
```

#### 4.2.3 Add Pre-Commit Validation Hooks

**Action**: Create pre-commit hooks for:
- Artifact frontmatter validation
- Filename format validation
- Coding standards (ruff, mypy)
- Documentation link validation

**Implementation**: Add to `.pre-commit-config.yaml`

### 4.3 Long-Term Structural Improvements

#### 4.3.1 Create Self-Regulating Framework

**Action**: Build validation and feedback system:
1. **Automated Checks**: Pre-commit hooks, CI validation
2. **Compliance Monitoring**: Track violation patterns
3. **Auto-Correction**: Suggest fixes for common violations
4. **Learning System**: Improve rules based on violation patterns

#### 4.3.2 Implement Progressive Disclosure

**Action**: Create layered documentation:
1. **Layer 1**: Entry points (critical rules only, 50-100 lines)
2. **Layer 2**: Quick references (tables, commands, links)
3. **Layer 3**: Detailed protocols (full workflows, examples)
4. **Layer 4**: Maintainer docs (architecture, deep dives)

#### 4.3.3 Create Documentation Health Dashboard

**Action**: Build tooling to:
- Track documentation usage (which docs are accessed)
- Identify outdated sections
- Surface missing references
- Monitor compliance rates

## 5. Proposed Reorganization

### 5.1 New Structure

```
.cursor/rules/
  prompts-artifacts-guidelines.mdc  (50 lines: critical rules only)
  prompts-coding-standards.mdc      (NEW: coding standards quick ref)

AGENT_ENTRY.md                      (Enhanced: prominent critical links)

AgentQMS/knowledge/
  system.md                         (100 lines: core rules only)
  quick-reference.md                (NEW: tables, commands, links)
  protocols/
    development.md                  (Existing: full details)
    governance.md                   (Existing: full details)
    status-update.md                (NEW: feedback protocol)
  references/
    operations.md                   (NEW: operational commands)
    data-contracts.md               (NEW: links + quick ref)
    api-contracts.md                (NEW: links + quick ref)
```

### 5.2 Content Distribution

**`.cursor/rules/prompts-artifacts-guidelines.mdc`** (50 lines):
- Critical: Never use `write` for artifacts
- Preferred: AgentQMS toolbelt
- Link to full instructions

**`AGENT_ENTRY.md`** (60 lines):
- Navigation to system.md
- **NEW**: Critical documentation links (data contracts, API refs, coding standards)
- **NEW**: Pre-commit checklist
- **NEW**: Status update reminder

**`AgentQMS/knowledge/system.md`** (100 lines):
- Core rules only
- Artifact creation (preferred method)
- Links to detailed protocols
- Links to quick references

**`AgentQMS/knowledge/quick-reference.md`** (NEW):
- Artifact creation cheat sheet
- Coding standards quick ref
- Common commands table
- Validation commands

## 6. Implementation Plan

### Phase 1: Immediate Restructuring (Week 1)
1. Prune `.cursor/rules/prompts-artifacts-guidelines.mdc` to 50 lines
2. Enhance `AGENT_ENTRY.md` with critical links
3. Split `AgentQMS/knowledge/system.md` (extract operations to references)
4. Create `AgentQMS/knowledge/quick-reference.md`

### Phase 2: Add Missing References (Week 1-2)
1. Create `AgentQMS/knowledge/references/data-contracts.md` (links + quick ref)
2. Create `AgentQMS/knowledge/references/api-contracts.md` (links + quick ref)
3. Add coding standards quick ref to entry points
4. Update all entry points with prominent links

### Phase 3: Feedback Mechanism (Week 2)
1. Create `AgentQMS/knowledge/protocols/status-update.md`
2. Add status update reminders to entry points
3. Create status update template

### Phase 4: Validation & Automation (Week 3)
1. Add pre-commit validation hooks
2. Create validation checklist in entry points
3. Test and refine

## 7. Success Metrics

**Short-term (1 month)**:
- 50% reduction in filename/location violations
- 30% reduction in frontmatter violations
- Increased references to data contracts (track via link clicks/access)

**Medium-term (3 months)**:
- 80% reduction in all compliance violations
- Regular status updates from agents
- Self-correcting violations (agents fix before commit)

**Long-term (6 months)**:
- Near-zero compliance violations
- Automated compliance monitoring
- Documentation health dashboard operational

## 8. Artifact Generation Script Issues

### 8.1 Identified Problems

During testing and code review, several issues were identified that cause errors when AI agents attempt to use the artifact generation tools:

#### 8.1.1 Subprocess Call Failures

**Problem**: Multiple subprocess calls that can fail silently or with unclear errors:
- **Bug ID generation** (lines 149, 228 in `artifact_workflow.py`): Uses `subprocess.check_output(["uv", "run", "python", "scripts/bug_tools/next_bug_id.py"])` without timeout or clear error handling
- **Index updates** (line 305): Uses `subprocess.run()` with minimal error handling
- **Git branch detection** (line 26 in `agent_qms/toolbelt/core.py`): Uses `subprocess.run(["git", "rev-parse", ...])` which can fail if git is unavailable

**Impact**:
- Scripts fail when `uv` is not in PATH
- No timeout handling - can hang indefinitely
- Unclear error messages when subprocess fails
- AI agents see cryptic subprocess errors instead of actionable guidance

**Example Error Scenarios**:
```python
# Fails silently if uv not available:
subprocess.check_output(["uv", "run", ...])  # FileNotFoundError: [Errno 2] No such file or directory: 'uv'

# Fails if script doesn't exist:
subprocess.check_output([...])  # FileNotFoundError

# Fails if git not available (handled, but returns "unknown"):
subprocess.run(["git", ...])  # FileNotFoundError → returns "unknown"
```

#### 8.1.2 Path Resolution Issues

**Problem**: Bootstrap script (`scripts/_bootstrap.py`) attempts to find project root but can fail:
- Searches for markers (`pyproject.toml`, `.git`, `agent_qms`) but may not find them in all contexts
- Falls back to parent directory assumption which may be incorrect
- No clear error if path resolution fails

**Impact**: Scripts may import from wrong locations or fail to find dependencies

#### 8.1.3 Error Handling Gaps

**Problem**: Several error paths provide unclear feedback:
- **Line 195-198 in `artifact_workflow.py`**: Catches all exceptions but error message may not indicate root cause
- **Line 260-265**: Validation errors may not clearly indicate what failed
- **Line 327**: Index update failures are logged but don't prevent artifact creation (may be intentional)

**Impact**: AI agents see generic errors instead of actionable guidance

#### 8.1.4 Missing Dependencies

**Problem**: Scripts require external dependencies that may not be available:
- `jsonschema` for frontmatter validation
- `yaml` for YAML parsing
- `jinja2` for template rendering
- `zoneinfo` (Python 3.9+) for timezone handling

**Impact**: Import errors if dependencies not installed, unclear error messages

### 8.2 Root Causes

1. **Assumption of Environment**: Scripts assume `uv`, `git`, and all dependencies are available
2. **Lack of Defensive Programming**: Minimal validation of external dependencies
3. **Poor Error Messages**: Generic exceptions instead of actionable errors
4. **No Timeout Handling**: Subprocess calls can hang indefinitely
5. **Silent Failures**: Some failures (index updates, bundle updates) fail silently

### 8.3 Recommended Fixes

#### Immediate Fixes (High Priority)

1. **Add Timeout Handling**:
```python
# Before (line 149):
bug_id = subprocess.check_output(
    ["uv", "run", "python", "scripts/bug_tools/next_bug_id.py"],
    text=True
).strip()

# After:
try:
    bug_id = subprocess.check_output(
        ["uv", "run", "python", "scripts/bug_tools/next_bug_id.py"],
        text=True,
        timeout=30,  # 30 second timeout
        stderr=subprocess.PIPE
    ).strip()
except subprocess.TimeoutExpired:
    raise ValueError("Bug ID generation timed out. Check if uv is working.")
except FileNotFoundError:
    raise ValueError("'uv' not found in PATH. Install uv or use --bug-id flag.")
except subprocess.CalledProcessError as e:
    raise ValueError(f"Bug ID generation failed: {e.stderr.decode() if e.stderr else str(e)}")
```

2. **Better Error Messages**:
```python
# Add helper function for subprocess calls
def safe_subprocess_check_output(cmd, timeout=30, error_message=None):
    """Run subprocess with timeout and clear error messages."""
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            timeout=timeout,
            stderr=subprocess.PIPE
        ).strip()
    except FileNotFoundError:
        cmd_name = cmd[0] if cmd else "command"
        raise ValueError(
            f"'{cmd_name}' not found in PATH. "
            f"{error_message or 'Please install required tools.'}"
        )
    except subprocess.TimeoutExpired:
        raise ValueError(
            f"Command timed out: {' '.join(cmd)}. "
            f"{error_message or 'Check if process is hanging.'}"
        )
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode() if e.stderr else str(e)
        raise ValueError(
            f"Command failed: {' '.join(cmd)}\n"
            f"Error: {stderr_msg}\n"
            f"{error_message or 'Check command output above.'}"
        )
```

3. **Validate Dependencies at Startup**:
```python
def check_dependencies():
    """Check if required external dependencies are available."""
    missing = []

    # Check uv
    try:
        subprocess.run(["uv", "--version"], capture_output=True, timeout=5, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        missing.append("uv")

    # Check git (optional but recommended)
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=5, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass  # Git is optional, only warns

    if missing:
        print(f"⚠️  Warning: Missing dependencies: {', '.join(missing)}")
        print("   Some features may not work correctly.")
        print("   Install missing tools or use alternative methods.")
        return False
    return True
```

4. **Improve Path Resolution Error Handling**:
```python
def _load_bootstrap():
    # ... existing code ...
    if not found:
        raise RuntimeError(
            "Could not locate scripts/_bootstrap.py or project root.\n"
            "Make sure you're running from the project directory.\n"
            "Expected markers: pyproject.toml, .git, agent_qms/"
        )
```

#### Medium-Term Improvements

1. **Add Fallback Methods**: If `uv` is unavailable, provide alternative bug ID generation
2. **Cache Git Branch**: Cache git branch name to avoid repeated subprocess calls
3. **Better Validation Errors**: Include field names and expected values in validation errors
4. **Dependency Checking**: Check Python version and required packages at startup

#### Long-Term Improvements

1. **Configuration File**: Allow configuration of tool paths (`uv`, `git`) and timeouts
2. **Mock Mode**: Allow testing without external dependencies
3. **Retry Logic**: Add retry logic for transient failures
4. **Health Checks**: Add health check command to verify all dependencies

### 8.4 Testing Recommendations

1. **Test with Missing Dependencies**: Test behavior when `uv`, `git`, or Python packages are missing
2. **Test with Timeouts**: Simulate slow or hanging subprocess calls
3. **Test Path Resolution**: Test from different working directories
4. **Test Error Messages**: Verify error messages are actionable

## 9. Conclusion

The current AI documentation entry points suffer from verbosity, duplication, and poor organization, leading to recurring compliance violations. Additionally, the artifact generation scripts have several reliability issues that cause failures when AI agents attempt to use them.

**Key Changes**:
1. Ultra-concise entry points (50-100 lines)
2. Prominent critical documentation links
3. Single source of truth for artifact creation
4. Periodic feedback protocol
5. Pre-commit validation hooks
6. **Improved artifact script reliability** (timeout handling, better errors, dependency checks)

**Expected Impact**:
- Faster agent onboarding
- Reduced compliance violations
- Better adherence to coding standards
- Improved documentation usage
- Self-regulating system
- **Fewer artifact creation failures**
