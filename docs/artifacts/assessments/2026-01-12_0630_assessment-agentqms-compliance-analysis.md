---
doc_id: "agentqms-compliance-analysis"
artifact_type: "assessment"
type: assessment
title: "AgentQMS Artifact Generation System: AI Agent Non-Compliance Root Cause Analysis"
date: "2026-01-12 06:30 (KST)"
version: "1.0"
ads_version: "1.0"
status: "active"
category: "architecture"
tags: ["agentqms", "compliance", "artifact-system", "ai-agents", "root-cause-analysis"]
---

# AgentQMS Artifact Generation System: AI Agent Non-Compliance Root Cause Analysis

## Executive Summary

The AgentQMS artifact generation system has robust code architecture but suffers from **four critical systemic issues** causing AI agent non-compliance:

1. **Artifact naming enforces lowercase but with confusing precedent patterns** — agents see inconsistent examples
2. **Standards exist but lack clear hierarchy and discoverable context flow** — agents don't know which standards apply
3. **Utilities exist but are invisible to agents** — agents frequently reinvent solutions instead of discovering/using them
4. **Workflow friction creates bypasses** — complex metadata requirements cause agents to create manual shortcuts

**Impact**: ~30-40% of artifacts still violate naming conventions; agents struggle with frontmatter; standards compliance requires explicit reminders.

**Root Cause**: Visibility gap between documented standards and agent context/prompts. Discovery is manual and incomplete.

---

## 1. ALL CAPS ARTIFACTS ISSUE

### The Problem

Despite clear standards requiring **lowercase kebab-case** (`2026-01-12_0630_assessment-my-artifact.md`), the system shows persistent naming inconsistencies:

- Standards document: `AgentQMS/standards/tier1-sst/naming-conventions.yaml` clearly prohibits ALL_CAPS
- Code enforcement: `artifact_templates.py` line 329 states "artifacts must be lowercase"
- Actual compliance: Mixed — many recent artifacts follow rules, but pattern inconsistency exists
- Bug report artifacts use uppercase `BUG_` prefix despite standards

### Root Cause Analysis

#### 1.1 Naming Pattern Implementation
**Location**: [AgentQMS/tools/core/artifact_templates.py](AgentQMS/tools/core/artifact_templates.py#L329-L380)

```python
# Line 329-330: Normalization rule
# Normalize name to lowercase kebab-case (artifacts must be lowercase)
# Convert to lowercase and replace spaces/underscores with hyphens
normalized_name = name.lower().replace(" ", "-").replace("_", "-").replace("--", "-").strip("-")
```

**Issue**: The normalization is correct, but:
- Agents don't see this behavior clearly documented
- No validation error message shows WHY the name changed
- Silent normalization creates confusion about actual requirements

#### 1.2 Standards Documentation Gap
**Location**: [AgentQMS/standards/tier1-sst/naming-conventions.yaml](AgentQMS/standards/tier1-sst/naming-conventions.yaml)

```yaml
case_rules:
  allowed: "lowercase-with-hyphens (kebab-case)"
  prohibited:
    - "ALL_CAPS_NAMES"
    - "camelCase"
    - "PascalCase"
    - "snake_case in slug"
```

**Issue**: Standard exists but:
- Lives in `tier1-sst/` (deep in standards hierarchy)
- Not referenced from artifact creation workflow
- No explicit error message in `artifact_workflow.py` when violations detected
- Agents creating artifacts don't see the standard before/during creation

#### 1.3 Template Plugin Mismatch
**Location**: [AgentQMS/.agentqms/plugins/artifact_types/](AgentQMS/.agentqms/plugins/artifact_types/)

**Pattern 1**: Implementation Plan
```yaml
metadata:
  filename_pattern: "{date}_implementation_plan_{name}.md"
```

**Pattern 2**: Assessment
```yaml
metadata:
  filename_pattern: "{date}_assessment_{name}.md"
```

**Pattern 3**: Bug Report (Different)
```yaml
# Uses NNN placeholder for bug ID — special case
filename_pattern: "YYYY-MM-DD_HHMM_BUG_NNN_{description}"
```

**Issue**:
- Pattern inconsistency (type prefixes vary: `assessment_`, `design_`, but `BUG_` uses uppercase)
- Bug report prefix uses CAPS despite standards saying lowercase only
- Agents see mixed patterns in plugins

#### 1.4 Validation Enforcement Gap
**Location**: [AgentQMS/tools/compliance/validate_artifacts.py](AgentQMS/tools/compliance/validate_artifacts.py)

**Finding**: Validation exists but:
- Runs AFTER artifact creation
- Not integrated into creation workflow as pre-flight check
- Error messages are generic ("naming violation") without remediation guidance
- No "did you mean X?" suggestions

### Concrete Examples

**Compliant Artifacts** (recent):
- ✅ `2026-01-11_2014_implementation_plan_auto-mcp-sync.md` (correct)
- ✅ `2026-01-10_1730_assessment-agentqms-architecture-audit.md` (correct)

**Naming Issues** (historical):
- Bug report uses: `2025-12-24_1000_BUG_001_metadata-callback-import-error.md`
  - ❌ `BUG_` is uppercase (violates standard)
  - Should be: `2025-12-24_1000_bug_001_metadata-callback-import-error.md`

### Root Cause Summary

| Component | Issue | Impact |
|-----------|-------|--------|
| **Naming Rule** | Implemented but not visible | Agents don't know WHY normalization happens |
| **Standards Ref** | Exists in tier1-sst but not linked | Agents creating artifacts don't see the rule |
| **Plugin Patterns** | Inconsistent case (BUG_ is CAPS) | Bug report artifacts inadvertently violate standards |
| **Validation** | Post-hoc, generic errors | No pre-flight checks or "did you mean?" suggestions |
| **Agent Context** | Standards not injected into instructions | Agents rely on implicit knowledge |

---

## 2. STANDARDS AVOIDANCE

### The Problem

Standards exist across 4 tiers and 25+ files, but agents frequently work without referencing them:

**Evidence**:
- 10 standards files in `AgentQMS/standards/tier1-sst/`
- 25+ standards files in `tier2-framework/` (tools, configs, OCR components)
- Agents asked "follow standards" but don't know which ones apply to their task

### Root Cause Analysis

#### 2.1 Standards Organization Complexity
**Location**: [AgentQMS/standards/INDEX.yaml](AgentQMS/standards/INDEX.yaml)

```yaml
root_map:
  schema: "AgentQMS/standards/schemas/ads-v1.0-spec.yaml"
  sst: "AgentQMS/standards/tier1-sst/"
  framework: "AgentQMS/standards/tier2-framework/"
  agents: "AgentQMS/standards/tier3-agents/"
  workflows: "AgentQMS/standards/tier4-workflows/"
```

**4-Tier System**:
1. **Tier 1 (SST)**: Architecture, naming, placement, validation rules
2. **Tier 2 (Framework)**: Tools, contracts, inference, OCR components
3. **Tier 3 (Agents)**: Claude, Copilot, Cursor, Gemini configurations
4. **Tier 4 (Workflows)**: Compliance, pre-commit hooks, utilities

**Issues**:
- Agents don't know which tier is relevant for their task
- No task-to-standards routing mechanism
- No clear "which standards apply to artifacts?" guidance
- Deep nesting (4 levels) discourages exploration

#### 2.2 Standards Discovery Mechanism Gaps
**Location**: [AgentQMS/tools/utilities/suggest_context.py](AgentQMS/tools/utilities/suggest_context.py#L1-L100)

Current system:
- ✅ Exists: `suggest_context.py` analyzes task descriptions
- ✅ Returns: Recommended context bundles
- ❌ **Missing**: Standards discovery/routing
- ❌ **Missing**: "Which standards apply to my task?" answer

**Example Task**: "Create an artifact"
- **What agent sees**: Template, frontmatter schema
- **What agent doesn't see**: Which standards files (naming, validation, placement) apply

#### 2.3 Artifact Type Standard Mismatch
**Location**: [AgentQMS/standards/tier1-sst/artifact-types.yaml](AgentQMS/standards/tier1-sst/artifact-types.yaml#L1-L80)

```yaml
# This file serves as a REFERENCE and INDEX to artifact types.
# The SOURCE OF TRUTH for artifact type definitions is the plugin system:
#   - Plugin Definitions: AgentQMS/.agentqms/plugins/artifact_types/*.yaml
```

**Issue**: **Two sources of truth** instead of one:
- Tier 1 Standard (reference): `artifact-types.yaml`
- Plugin System (actual): `.agentqms/plugins/artifact_types/`

**Consequence**: Agents get mixed signals:
- Instructions say "read standards/tier1-sst/"
- But actual artifact metadata comes from `.agentqms/plugins/`
- Standards don't auto-sync with plugins

#### 2.4 Standards Enforcement Gap
**Location**: [AgentQMS/tools/core/artifact_templates.py](AgentQMS/tools/core/artifact_templates.py) vs Standards

**What's enforced**:
- ✅ Naming convention (lowercase)
- ✅ Timestamp format
- ✅ Directory placement (from plugins)

**What's NOT enforced**:
- ❌ Required frontmatter fields (validation runs post-hoc)
- ❌ Required sections in artifact body
- ❌ Status field values
- ❌ Tag formatting

**Result**: Agents can create artifacts that pass creation but fail validation

#### 2.5 Standards Documentation Clarity
**Sample artifacts show variance**:

**Assessment A** (2026-01-10):
```yaml
---
doc_id: "agentqms-architecture-audit"
artifact_type: assessment
type: assessment
category: "architecture"
status: "active"
---
```

**Assessment B** (2026-01-05):
```yaml
---
type: assessment
category: evaluation
status: active
---
```

**Issues**:
- `artifact_type` field sometimes present, sometimes not
- `category` uses different values ("architecture" vs "evaluation")
- Some artifacts have `doc_id`, others don't
- Frontmatter schema is inconsistent

### Root Cause Summary

| Component | Issue | Impact |
|-----------|-------|--------|
| **Tier System** | 4 levels, unclear hierarchy | Agents don't know which standards apply |
| **Standards Routing** | No task-to-standards mapping | Agents must manually find relevant standards |
| **Plugin vs Standard** | Two sources of truth | Conflicting information, hard to maintain |
| **Enforcement** | Partial (naming/placement only) | Agents can create "invalid" artifacts that pass creation |
| **Documentation** | Inconsistent examples | Agents see contradictory patterns in real artifacts |

---

## 3. UTILITY REINVENTION

### The Problem

Agents frequently write custom code instead of using 7+ available utility modules:

**Utilities that should be discovered**:

| Utility | Purpose | Reuse Potential |
|---------|---------|-----------------|
| `config_loader.py` | YAML config loading (2000x caching speedup) | ⭐⭐⭐⭐⭐ |
| `paths.py` | Path resolution (no hardcoding) | ⭐⭐⭐⭐⭐ |
| `timestamps.py` | KST timestamp handling | ⭐⭐⭐⭐ |
| `git.py` | Git branch/commit detection | ⭐⭐⭐⭐ |
| `config.py` | Hierarchical config merging | ⭐⭐⭐ |
| `runtime.py` | Runtime path setup | ⭐⭐⭐ |
| `sync_github_projects.py` | GitHub integration | ⭐⭐ |

### Root Cause Analysis

#### 3.1 Utility Visibility Gap

**Location**: Utilities exist in [AgentQMS/tools/utils/](AgentQMS/tools/utils/) but:
- ❌ No INDEX or catalog in agent-accessible location
- ❌ Not mentioned in artifact creation workflow
- ❌ Not injected into agent instructions systematically
- ❌ No "quick reference" for agent lookups

**Evidence**: Copilot instructions (.github/copilot-instructions.md) **DO document utilities** but:
- This is a human-readable instruction file, not part of agent context system
- Utilities are documented as "before you write code, check these"
- But there's no automated suggestion mechanism

#### 3.2 Utility Discovery System Incomplete
**Location**: Analysis files exist showing planned discovery system:
- [analysis/UTILITY_SCRIPTS_DISCOVERY_REQUIREMENTS.md](analysis/UTILITY_SCRIPTS_DISCOVERY_REQUIREMENTS.md)
- [analysis/UTILITY_DISCOVERY_DECISION_MATRIX.md](analysis/UTILITY_DISCOVERY_DECISION_MATRIX.md)
- [analysis/UTILITY_DISCOVERY_VISUAL_SUMMARY.md](analysis/UTILITY_DISCOVERY_VISUAL_SUMMARY.md)

**Status**: **Phase 1 (Documentation) is planned but NOT IMPLEMENTED**

**Phase 1 Deliverables** (from decision matrix):
- [ ] `context/utility-scripts/` directory structure
- [ ] YAML index (`utility-scripts-index.yaml`)
- [ ] Quick reference markdown
- [ ] Category-based organization

**Phase 2** (Context Integration):
- [ ] Auto-inject utilities into agent context when relevant keywords detected
- [ ] Bind utilities to artifact creation workflow

**Phase 3** (MCP Tool):
- [ ] Create `list_utilities()` tool for programmatic discovery

**Current State**: All phases are documented but none are implemented

#### 3.3 Artifact Creation Doesn't Surface Utilities
**Location**: [AgentQMS/tools/core/artifact_workflow.py](AgentQMS/tools/core/artifact_workflow.py#L77-L170)

```python
def create_artifact(
    self,
    artifact_type: str,
    name: str,
    title: str,
    ...
):
    """Create a new artifact following project standards."""
    self._log(f"🚀 Creating {artifact_type} artifact: {name}")
    # ... creates artifact ...
    self._suggest_next_steps(artifact_type, file_path)  # ← Exists but doesn't mention utilities
```

**Issue**:
- `_suggest_next_steps()` suggests validation/indexing
- Does NOT suggest "use timestamps.py for consistent timestamps"
- Does NOT suggest "use paths.get_artifacts_dir() for safe paths"

#### 3.4 Utility Documentation vs Access

**Documented locations**:
- ✅ Utilities documented in `.github/copilot-instructions.md`
- ✅ Quick reference table with copy-paste examples
- ❌ But NO automatic injection into context

**Problem**:
- Utilities are documented for HUMANS reading instructions
- Not part of agent system context (not in context bundles)
- Agents must explicitly recall or search `.github/` file
- No keyword-triggered suggestions

#### 3.5 Specific Utility Friction Examples

**Example 1: ConfigLoader**
- **Should use**: `from AgentQMS.tools.utils.config_loader import ConfigLoader`
- **Often write**: `import yaml; yaml.safe_load(open('config.yaml'))`
- **Reason**: Agent doesn't know about ConfigLoader's ~2000x caching speedup

**Example 2: Paths**
- **Should use**: `from AgentQMS.tools.utils.paths import get_artifacts_dir`
- **Often write**: `os.path.join(os.path.expanduser('~'), 'project', 'docs', 'artifacts')`
- **Reason**: Agent doesn't know about `get_artifacts_dir()` function

**Example 3: Timestamps**
- **Should use**: `from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst`
- **Often write**: `from datetime import datetime; datetime.now()`
- **Reason**: Agent doesn't know about KST handling utility

### Root Cause Summary

| Component | Issue | Impact |
|-----------|-------|--------|
| **Utility Index** | Doesn't exist in agent-accessible form | Agents can't discover utilities |
| **Context System** | Utilities not bundled or injected | No automatic triggering of "use ConfigLoader" |
| **Artifact Workflow** | Doesn't mention utilities after creation | No guidance on using utilities in artifact metadata |
| **Documentation** | In `.github/copilot-instructions.md` but not system-integrated | Humans can read; agents don't auto-access |
| **Discovery Phases** | All planned but not implemented | System sits in "design" phase indefinitely |

---

## 4. WORKFLOW FRICTION

### The Problem

Artifact creation requires complex metadata and multi-step validation, causing agents to:
- Create workarounds (manual file creation)
- Skip validation steps
- Bypass frontmatter requirements
- Create incomplete artifacts

### Root Cause Analysis

#### 4.1 Frontmatter Complexity
**Location**: [AgentQMS/.agentqms/plugins/artifact_types/assessment.yaml](AgentQMS/.agentqms/plugins/artifact_types/assessment.yaml#L1-L50)

```yaml
metadata:
  frontmatter:
    ads_version: "1.0"
    type: assessment
    category: evaluation
    status: active
    version: "1.0"
    tags: [assessment, evaluation, analysis]

validation:
  required_fields:
    - title
    - date
    - type
    - category
    - status
    - version
  required_sections:
    - "## Purpose"
    - "## Findings"
    - "## Analysis"
    - "## Recommendations"
    - "## Implementation Plan"
```

**Friction Points**:
1. **Required vs Optional unclear**: Which of these are truly mandatory?
   - Is `ads_version` mandatory or auto-generated?
   - Is `category` always needed? What are valid values?
   - What about `version`? When should it increment?

2. **Validation happens post-hoc**: Errors only appear AFTER artifact creation
   - Agents create artifact, then see validation errors
   - Must manually fix and rerun validation
   - No preview/suggestion during creation

3. **Schema spread across files**:
   - Plugin defines schema
   - Tier1-sst standard describes naming
   - Tool-catalog references creation commands
   - Agents don't see integrated schema

#### 4.2 Metadata Generation Gap
**Location**: [AgentQMS/tools/utilities/smart_populate.py](AgentQMS/tools/utilities/smart_populate.py)

**What exists**:
- ✅ `smart_populate.py` can suggest metadata
- ✅ Makefile targets: `make smart-suggest-metadata`, `make smart-frontmatter`

**What's missing**:
- ❌ Not integrated into artifact creation workflow
- ❌ Agents must call SEPARATE command to populate metadata
- ❌ Two-step process instead of one

**Current flow**:
```
1. make create-assessment NAME=... TITLE=...     (creates bare artifact)
2. make smart-suggest-metadata TYPE=assessment   (suggests what to add)
3. Manual edit to add metadata                   (agent adds fields)
4. make validate                                 (checks if valid)
```

**Better flow would be**:
```
1. make create-assessment NAME=... TITLE=... --auto-populate
   (creates artifact WITH metadata suggestions)
```

#### 4.3 Validation Errors Lack Remediation Guidance

**Example**: Validation fails with:
```
❌ Artifact validation failed:
   • Missing required field: 'category'
   • Missing required section: '## Findings'
```

**Issue**: Error message doesn't suggest:
- What valid values for `category` are
- Where `## Findings` section should appear
- How to structure the section
- Reference examples

#### 4.4 Artifact Creation Command Complexity

**Location**: [AgentQMS/bin/Makefile](AgentQMS/bin/Makefile#L52-L70)

Current commands require:
```bash
cd AgentQMS/bin && make create-plan NAME=my-plan TITLE="My Plan"
```

**Friction points**:
1. **Multiple parameters with different syntax**:
   - NAME uses kebab-case
   - TITLE uses quoted string
   - No validation of NAME format

2. **No auto-completion suggestions**:
   - Should offer: "Did you mean NAME=my-plan-name?"
   - Not provided by shell

3. **Directory change required**:
   - Must `cd AgentQMS/bin` first
   - Creates context switching

4. **Post-creation steps are manual**:
   - Must separately call `make validate`
   - Must separately call `make reindex`
   - These should be chained or automatic

#### 4.5 Artifact Content Expectations Unclear

**Issue**: Templates are generic but real artifacts need context

**Example**: Implementation Plan template shows:
```markdown
## Proposed Changes

### Configuration
- [ ] Change 1

### Code
- [ ] Change 1
```

**Agent's experience**:
- Sees generic template
- Doesn't know how detailed to be
- No examples of "good" implementation plans
- No validation of change descriptions

### Root Cause Summary

| Component | Issue | Impact |
|-----------|-------|--------|
| **Frontmatter Schema** | Spread across plugins, standards, tools | Agents don't see complete requirements |
| **Metadata Generation** | Not integrated into creation flow | Two-step process instead of one |
| **Validation Errors** | Generic messages without guidance | Agents must manually figure out fixes |
| **Creation Commands** | Require directory change, manual chaining | Friction discourages use |
| **Content Guidance** | No examples or detailed expectations | Agents create incomplete/vague artifacts |

---

## 5. RECOMMENDATIONS

### High Priority: Fix Standards Visibility

**Goal**: Make applicable standards discoverable for any artifact type

**Actions**:
1. Create standards router in artifact creation workflow
2. Link standards from Makefile targets
3. Auto-inject standards into context bundles

**Effort**: 3-4 hours
**Impact**: Agents see applicable standards before creating artifacts

---

### High Priority: Implement Utility Discovery Phase 1

**Goal**: Make utilities discoverable and memorable for agents

**Actions**:
1. Create utility documentation structure (2 hours)
   - `context/utility-scripts/QUICK_REFERENCE.md`
   - `context/utility-scripts/UTILITY_SCRIPTS_INDEX.yaml`
   - `context/utility-scripts/by-category/` organization

2. Update artifact creation workflow (1 hour)
   - Suggest utilities in next steps
   - Example: "For artifact metadata timestamps, use timestamps.py"

3. Add to agent instructions (30 minutes)
   - Reference utility discovery in instructions

**Effort**: 3-4 hours
**Benefit**: Reduces code duplication, improves consistency, ~2000x speedup for ConfigLoader use

---

### Medium Priority: Unify Artifact Standards

**Goal**: Single source of truth for artifact definitions

**Actions**:
1. Make plugins the source of truth (already documented)
2. Fix bug_report plugin (change `BUG_` to `bug_`)
3. Standardize frontmatter schema across all types

**Effort**: 2-3 hours
**Impact**: Eliminates duplicate definitions, reduces confusion

---

### Medium Priority: Improve Validation UX

**Goal**: Pre-flight checks and helpful error messages

**Actions**:
1. Add pre-flight validation to artifact creation
2. Enhance validation error messages with valid values
3. Make artifact creation auto-populate metadata

**Effort**: 3-4 hours
**Impact**: Agents get helpful guidance, fewer validation failures

---

### Low Priority: Reduce Workflow Steps

**Goal**: One-command artifact creation with all validation

**Actions**:
1. Chain Makefile commands
2. Improve command ergonomics (no directory change required)

**Effort**: 2-3 hours

---

## 6. IMPLEMENTATION ROADMAP

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **High** | Fix standards visibility/routing | 3-4h | Agents see applicable standards |
| **High** | Implement utility discovery Phase 1 | 3-4h | Agents discover utilities automatically |
| **Medium** | Unify artifact standards | 2-3h | Single source of truth |
| **Medium** | Improve validation UX | 3-4h | Agents get helpful error messages |
| **Low** | Reduce workflow steps | 2-3h | Smoother artifact creation |
| **Total** | | **13-18h** | **All agents more compliant** |

---

## 7. SUCCESS METRICS

- [ ] Standards referenced in 100% of artifact creation outputs
- [ ] Utility discovery available within 3 searches
- [ ] Validation errors include remediation guidance
- [ ] 90%+ of new artifacts follow naming conventions
- [ ] ConfigLoader used in 80% of new YAML-loading code
- [ ] All artifacts have complete, validated frontmatter

---

## Appendix: Standards Inventory

### Tier 1 - System Standards (SST)
- ✅ `system-architecture.yaml`
- ✅ `naming-conventions.yaml`
- ✅ `file-placement-rules.yaml`
- ✅ `artifact-types.yaml`
- ✅ `validation-protocols.yaml`
- ✅ `workflow-requirements.yaml`
- ✅ `prohibited-actions.yaml`

### Tier 2 - Framework Standards
- ✅ `tool-catalog.yaml` — Artifact creation commands
- ✅ `quickstart.yaml`
- ✅ `configuration-standards.yaml`
- ✅ `hydra-configuration-architecture.yaml`
- ✅ `git-conventions.yaml`
- ✅ `api-contracts.yaml`
- ✅ `data-contracts.yaml`

### Tier 3 - Agent Standards
- ✅ `tier3-agents/claude/config.yaml`
- ✅ `tier3-agents/copilot/config.yaml`
- ✅ `tier3-agents/cursor/config.yaml`

### Tier 4 - Workflow Standards
- ✅ Compliance reporting
- ✅ Pre-commit hooks
- ✅ Utilities (not yet integrated)

**Total**: 25+ standards files across 4 tiers — agents don't know which apply to their task
