---
title: "AgentQMS Integration Strategy for Main Docs Audit"
date: "2025-12-21 05:00 (KST)"
type: "assessment"
category: "evaluation"
status: "active"
version: "1.0"
ads_version: "1.0"
scope: "Integration of docs/ audit with existing AgentQMS framework"
audience: "ai_agents_only"
related_assessments: "["2025-12-21_0445_assessment_main-docs-strategic-audit.md"]"
tags: "["agentqms", "integration", "framework", "automation"]"
---







# AgentQMS Integration Strategy for Main Docs Audit

## Executive Summary

**Critical Discovery**: Existing AgentQMS framework provides 70%+ of proposed audit infrastructure. Strategic audit plan must **leverage, not duplicate** existing tooling.

**Key Systems**:
1. **AgentQMS** (`.agentqms/`) - Artifact validation, quality monitoring, boundary checking
2. **ADS v1.0** (`.ai-instructions/`) - Tier-based YAML documentation standard
3. **GitHub Actions** (`.github/workflows/agentqms-*.yml`) - CI/CD validation pipelines

**Recommendation**: **Adapt Phase 1-5 plan** to use existing AgentQMS tools, extend where gaps exist, avoid creating competing systems.

---

## Existing Infrastructure Analysis

### 1. AgentQMS Framework (`.agentqms/`)

**Purpose**: Quality management system for artifacts and documentation
**Version**: 0.2.0
**Status**: Active, enforced via GitHub Actions

**Key Components**:

| Component | Path | Purpose | Relevance to Audit |
|-----------|------|---------|-------------------|
| Artifact Validator | `AgentQMS/agent_tools/compliance/validate_artifacts.py` | Validates naming, frontmatter, structure | **Use for Phase 1 categorization** |
| Quality Monitor | `AgentQMS/agent_tools/compliance/documentation_quality_monitor.py` | Monitors doc quality metrics | **Use for staleness detection** |
| Framework Auditor | `AgentQMS/agent_tools/audit/framework_audit.py` | Generates compliance audits | **Use for Phase 1 discovery** |
| Boundary Validator | `AgentQMS/agent_tools/compliance/validate_boundaries.py` | Checks module boundaries | Already integrated |

**Artifact Types** (`.agentqms/plugins/artifact_types/`):
- `audit.yaml` - Defines audit document structure (USE THIS for assessments)
- `ocr_experiment.yaml` - OCR-specific experiments
- `change_request.yaml` - Change tracking

**Settings** (`.agentqms/settings.yaml`):
```yaml
paths:
  artifacts: docs/artifacts
  artifact_categories:
    implementation_plan: implementation_plans
    assessment: assessments  # ← Current audit lives here
    design: design_documents

validation:
  strict_mode: true
  excluded_directories: [archive, deprecated, DEPRECATED-ALLCAPS-DOCS]
  rules:
    naming: true        # Already enforces naming conventions
    frontmatter: true   # Already validates frontmatter
    structure: true     # Already checks document structure
```

**GitHub Actions** (`.github/workflows/agentqms-validation.yml`):
- Runs on push/PR to `AgentQMS/**`, `.agentqms/**`
- Executes: `validate_artifacts.py`, `validate_boundaries.py`, `validate_links.py`
- Already blocks non-compliant commits

---

### 2. ADS v1.0 Standard (`.ai-instructions/`)

**Purpose**: AI-optimized YAML documentation standard
**Version**: 1.0
**Released**: 2025-12-16
**Status**: Active, proven in OCR console migration

**Structure**:
```
.ai-instructions/
├── schema/
│   ├── ads-v1.0-spec.yaml           # Standard specification
│   ├── compliance-checker.py        # Validation script
│   └── validation-rules.json        # JSON Schema
├── tier1-sst/                       # System Source of Truth (critical rules)
│   ├── file-placement-rules.yaml
│   ├── naming-conventions.yaml
│   ├── prohibited-actions.yaml
│   ├── validation-protocols.yaml
│   └── workflow-requirements.yaml
├── tier2-framework/                 # Framework guidance
│   └── tool-catalog.yaml
├── tier3-agents/                    # Agent-specific configs
└── tier4-workflows/                 # Supporting workflows
```

**App-Specific Extensions**:
- `apps/ocr-inference-console/.ai-instructions/INDEX.yaml` - App entry point
- `experiment-tracker/.ai-instructions/` - Experiment tracker docs

**Compliance Rules** (from `ads-v1.0-spec.yaml`):
```yaml
content_rules:
  format: "YAML structured data only"
  prose: "PROHIBITED - No markdown paragraphs"
  audience: "AI-only"
  verbosity: "Ultra-concise"

token_targets:
  tier1_file: "≤100 tokens per rule set"
  tier2_file: "≤500 tokens per catalog"
  tier3_file: "≤300 tokens for config"
  tier4_file: "≤200 tokens per workflow"
```

**Validation** (`.ai-instructions/schema/compliance-checker.py`):
- Checks YAML well-formedness
- Validates required frontmatter fields
- Detects prohibited user-oriented phrases
- Enforces token budgets

---

### 3. Existing Validation Workflows

**GitHub Actions Pipeline**:
```yaml
# .github/workflows/agentqms-validation.yml
jobs:
  validate:
    - validate_artifacts.py --all      # Naming, frontmatter, structure
    - validate_boundaries.py --json    # Module boundaries
    - validate_links.py AgentQMS/      # Broken links (canonical knowledge only)
```

**Pre-Commit Hooks** (configured but disabled):
```yaml
# .agentqms/settings.yaml
automation:
  pre_commit:
    enabled: false              # ← OPPORTUNITY: Enable this
    validate_artifacts: true
```

---

## Integration Strategy

### Phase 1: Discovery - Leverage AgentQMS Tools

**Original Plan**: Create new scripts in `scripts/docs-audit/`
- `detect-stale-references.py`
- `compute-reference-graph.py`
- `categorize-by-type.py`

**Revised Plan**: Extend existing AgentQMS tools

**1.1 Staleness Detection** - Extend `documentation_quality_monitor.py`

Add new checks to existing quality monitor:
```python
# AgentQMS/agent_tools/compliance/documentation_quality_monitor.py

class DocsQualityMonitor:
    # Existing: freshness, completeness, consistency

    # NEW: Add staleness detection
    def check_stale_references(self, file_path: Path) -> list[str]:
        """Detect references to non-existent modules, wrong ports, old commands"""
        violations = []
        content = file_path.read_text()

        # Check for stale module references
        if "apps/backend/" in content:
            violations.append("References deprecated apps/backend/ module")

        # Check for wrong ports
        if "port 8000" in content.lower() and "ocr" in content.lower():
            violations.append("References wrong port (should be 8002)")

        # Check for old commands
        if "make backend-ocr" in content:
            violations.append("References deprecated Makefile command")

        return violations
```

**1.2 Reference Graph** - Extend `validate_links.py`

Existing link validator already builds reference graph:
```python
# AgentQMS/agent_tools/documentation/validate_links.py (extend)

# Existing: validates broken links
# NEW: Export reference graph
def export_reference_graph(docs_root: Path, output_path: Path):
    """Generate GraphML graph of doc cross-references"""
    # ... implementation using existing link extraction logic
```

**1.3 Categorization** - Use `validate_artifacts.py`

Already categorizes by artifact type (implementation_plan, assessment, etc.):
```python
# AgentQMS/agent_tools/compliance/validate_artifacts.py (already does this)

def categorize_artifact(file_path: Path) -> str:
    frontmatter = parse_frontmatter(file_path)
    return frontmatter.get("type", "unknown")
```

**Deliverables** (same as original):
- Staleness report (via extended quality monitor)
- Reference graph (via extended link validator)
- Category taxonomy (via existing artifact validator)

**Resource Estimate**: 4-6 hours (vs original 8-12h) - 50% reduction by reusing code

---

### Phase 2: Content Extraction - Use ADS v1.0 Tier Structure

**Original Plan**: Create `.ai-instructions/` for main docs
**Revised Plan**: Integrate with existing `.ai-instructions/` tier structure

**2.1 Placement Strategy**

Map docs/ content to `.ai-instructions/` tiers:

| docs/ Content | Target Tier | Rationale |
|--------------|-------------|-----------|
| `docs/architecture/system-architecture.md` | `tier1-sst/system-architecture.yaml` | Critical system rules |
| `docs/architecture/inference-overview.md` | `tier2-framework/inference-framework.yaml` | Framework guidance |
| `docs/guides/installation.md` | `tier2-framework/quickstart.yaml` | Tool usage |
| `docs/schemas/*.md` | `tier2-framework/data-contracts.yaml` | Schema definitions |
| `docs/artifacts/specs/*.md` | `tier2-framework/api-contracts.yaml` | API contracts |

**2.2 App-Specific vs Root-Level**

```
.ai-instructions/
├── tier1-sst/                      # PROJECT-WIDE critical rules
│   ├── system-architecture.yaml    # (from docs/architecture/)
│   ├── file-placement-rules.yaml   # (existing)
│   └── naming-conventions.yaml     # (existing)
├── tier2-framework/                # PROJECT-WIDE frameworks
│   ├── inference-framework.yaml    # (from docs/architecture/inference-overview.md)
│   ├── data-contracts.yaml         # (from docs/schemas/)
│   └── api-contracts.yaml          # (from docs/artifacts/specs/)

apps/ocr-inference-console/.ai-instructions/
├── INDEX.yaml                      # APP-SPECIFIC entry point
├── quickstart.yaml                 # (existing)
├── architecture/                   # (existing)
│   ├── backend-services.yaml
│   ├── frontend-context.yaml
│   └── error-handling.yaml
└── contracts/                      # (existing)
```

**Rule**: Root `.ai-instructions/` = project-wide, app `.ai-instructions/` = app-specific

**Deliverables** (same as original):
- 84 high-value files → YAML in appropriate tiers
- Token footprint: ~8,000 tokens (from ~84,000)

**Resource Estimate**: 10-14 hours (vs original 12-16h) - clearer placement rules

---

### Phase 3: Archival - Use AgentQMS Archive Tool

**Original Plan**: Manual archival with custom scripts
**Revised Plan**: Use `AgentQMS/agent_tools/archive/archive_artifacts.py`

**3.1 Archival Rules** (already defined in `.agentqms/settings.yaml`)

```yaml
# .agentqms/settings.yaml
validation:
  excluded_directories: [archive, deprecated, DEPRECATED-ALLCAPS-DOCS]
```

**3.2 Use Existing Archive Tool**

```bash
# AgentQMS/agent_tools/archive/archive_artifacts.py
# Already handles:
# - Moving completed implementation plans to archive
# - Preserving frontmatter metadata
# - Updating references

# Extend for:
# - Staleness-based archival (last commit >6 months + zero refs)
# - Duplicate detection (similarity hashing)
```

**Deliverables** (same as original):
- ~400 files archived based on staleness
- ~50 duplicates deleted
- Archive size reduced via compression

**Resource Estimate**: 6-10 hours (vs original 8-12h) - reuse existing archival logic

---

### Phase 4: Automation - Extend Pre-Commit Hooks

**Original Plan**: Create new pre-commit hooks in `.git/hooks/`
**Revised Plan**: Enable + extend existing AgentQMS pre-commit integration

**4.1 Enable Pre-Commit** (currently disabled)

```yaml
# .agentqms/settings.yaml
automation:
  pre_commit:
    enabled: true  # ← Change from false
    validate_artifacts: true
```

**4.2 Extend Validation Rules**

```python
# AgentQMS/agent_tools/compliance/validate_artifacts.py

# Add new validation rules:
# - Port numbers (8002, 5173)
# - Module paths (no apps/backend/)
# - Token budgets (.ai-instructions/ files)
# - ADS v1.0 compliance (call compliance-checker.py)
```

**4.3 GitHub Actions Integration** (already exists)

```yaml
# .github/workflows/agentqms-validation.yml (extend)
jobs:
  validate:
    steps:
      # Existing
      - validate_artifacts.py --all
      - validate_boundaries.py --json
      - validate_links.py AgentQMS/

      # NEW: Add ADS v1.0 compliance check
      - name: Validate ADS v1.0 compliance
        run: python .ai-instructions/schema/compliance-checker.py .ai-instructions/
```

**Deliverables** (same as original):
- Pre-commit hooks blocking violations
- GitHub Actions enforcing compliance
- Token budget enforcement

**Resource Estimate**: 6-10 hours (vs original 10-15h) - leverage existing CI/CD

---

### Phase 5: Verification - Use AgentQMS Audit Tool

**Original Plan**: Custom verification scripts
**Revised Plan**: Generate audit report via `framework_audit.py`

**5.1 Generate Compliance Report**

```bash
# AgentQMS/agent_tools/audit/framework_audit.py
# Generates audit following .agentqms/plugins/artifact_types/audit.yaml template

# Checks:
# - All high-value content in .ai-instructions/INDEX references
# - Zero broken links
# - Pre-commit hooks active
# - Token budget under limits
# - No stale references (port 8000, apps/backend/)
```

**5.2 AI Agent Testing** (same as original)

Test queries before/after migration:
- "How to start OCR backend?"
- "What port does backend run on?"
- "List API endpoints"
- "How to add preprocessing feature?"

**Deliverables** (same as original):
- Migration verification report (via AgentQMS audit tool)
- AI agent test results
- Updated README.md

**Resource Estimate**: 2-4 hours (vs original 2-5h) - automated report generation

---

## Revised Resource Estimates

### Token Budget

| Phase | Original | Revised | Savings |
|-------|----------|---------|---------|
| Phase 1 | 2,000 | 2,000 | 0% (output only) |
| Phase 2 | 100,000 | 100,000 | 0% (must read files) |
| Phase 3 | 5,000 | 5,000 | 0% (validation only) |
| Phase 4 | 10,000 | 5,000 | **50%** (reuse existing) |
| Phase 5 | 5,000 | 3,000 | **40%** (automated reporting) |
| **Total** | **122,000** | **115,000** | **6% reduction** |

### Time Budget

| Phase | Original | Revised | Savings |
|-------|----------|---------|---------|
| Phase 1 | 8-12h | 4-6h | **50%** (reuse tools) |
| Phase 2 | 12-16h | 10-14h | **20%** (clearer rules) |
| Phase 3 | 8-12h | 6-10h | **25%** (existing archival) |
| Phase 4 | 10-15h | 6-10h | **40%** (CI/CD exists) |
| Phase 5 | 2-5h | 2-4h | **20%** (automated audit) |
| **Total** | **40-60h** | **28-44h** | **30% reduction** |

---

## Key Integration Points

### 1. Artifact Types - Use AgentQMS Schema

**Don't**: Create new document types
**Do**: Use existing `.agentqms/plugins/artifact_types/`

```yaml
# This assessment follows AgentQMS audit.yaml schema
# - Filename: {date}_audit-{name}.md → 2025-12-21_0500_assessment_agentqms-integration-strategy.md
# - Frontmatter: type, status, category (using assessment instead of audit for semantic clarity)
# - Sections: Executive Summary, Findings, Recommendations
```

### 2. Validation - Extend, Don't Duplicate

**Don't**: Create `scripts/docs-validation/*.py`
**Do**: Extend `AgentQMS/agent_tools/compliance/*.py`

```python
# Extend existing tools with new checks
AgentQMS/agent_tools/compliance/
├── validate_artifacts.py        # Add: port numbers, module paths
├── documentation_quality_monitor.py  # Add: staleness detection
└── validate_links.py            # Add: reference graph export
```

### 3. CI/CD - Use Existing Workflows

**Don't**: Create `.github/workflows/docs-validation.yml`
**Do**: Extend `.github/workflows/agentqms-validation.yml`

```yaml
# Add ADS v1.0 compliance check to existing workflow
- name: Validate ADS v1.0 compliance
  run: python .ai-instructions/schema/compliance-checker.py .ai-instructions/
```

### 4. Documentation Structure - Follow ADS v1.0

**Don't**: Create flat `.ai-instructions/contracts/*.yaml`
**Do**: Use tier hierarchy

```
.ai-instructions/
├── tier1-sst/          # Critical rules (system architecture)
├── tier2-framework/    # Framework guidance (APIs, schemas, workflows)
├── tier3-agents/       # Agent configs
└── tier4-workflows/    # Supporting automation
```

---

## Phased Rollout (Revised)

### Week 1: Discovery (Phase 1) - 4-6 hours

**Day 1-2**: Extend AgentQMS tools
- Add staleness checks to `documentation_quality_monitor.py`
- Add reference graph export to `validate_links.py`
- Run existing `validate_artifacts.py --all`

**Day 3**: Generate reports
- Staleness report
- Reference graph (GraphML)
- High-value file ranking (top 10%)

**Deliverable**: Strategic audit report with prioritized targets

---

### Week 2: Migration (Phases 2-3) - 16-24 hours

**Days 1-3**: Convert high-value content (10-14h)
- Extract top 84 files to `.ai-instructions/` tiers
- Follow ADS v1.0 spec (YAML only, no prose)
- Validate token budgets

**Days 4-5**: Archive stale content (6-10h)
- Use `archive_artifacts.py` for completed plans
- Archive files with zero references + >6mo last commit
- Deduplicate via similarity hashing

**Deliverable**: `.ai-instructions/` structure with 90% token reduction

---

### Week 3: Automation & Rollout (Phases 4-5) - 8-14 hours

**Days 1-2**: Enable automation (6-10h)
- Enable pre-commit hooks (`.agentqms/settings.yaml`)
- Extend validation rules (ports, modules, tokens)
- Update GitHub Actions workflow

**Days 3-4**: Verification (2-4h)
- Generate compliance audit via `framework_audit.py`
- Test AI agent before/after
- Validate zero broken links, zero stale refs

**Day 5**: Rollout
- Update README.md
- Add deprecation notices
- CHANGELOG.md entry

**Deliverable**: Production-ready, self-healing docs system

---

## Success Metrics (Unchanged)

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Token Footprint | 5,046,000 | 50,000 | 99% |
| AI Query Cost | 3,000-4,000 | <100 | 97% |
| Stale References | Unknown | 0 | 100% |
| Broken Links | Unknown | 0 | 100% |
| Archive Size | 420MB | ~200MB | 52% |
| Documentation Drift | Frequent | 0 (blocked) | 100% |

---

## Critical Differences from Original Plan

| Aspect | Original Plan | Revised Plan | Rationale |
|--------|---------------|--------------|-----------|
| Script Location | `scripts/docs-audit/` | `AgentQMS/agent_tools/compliance/` | Reuse existing framework |
| Pre-Commit Hooks | New `.git/hooks/` scripts | Enable existing AgentQMS hooks | Already integrated with CI/CD |
| Validation Tools | 4 new scripts | Extend 3 existing tools | Avoid duplication |
| Audit Reports | Custom format | AgentQMS audit.yaml template | Consistent artifact types |
| CI/CD Integration | New workflow | Extend agentqms-validation.yml | Leverage existing pipeline |
| Time Estimate | 40-60h | 28-44h | **30% faster** via reuse |

---

## Recommendations

### 1. Immediate Actions (Priority 1)

1. **Enable pre-commit hooks** (`.agentqms/settings.yaml`): Change `enabled: false` → `true`
2. **Extend quality monitor** (`documentation_quality_monitor.py`): Add staleness checks
3. **Run existing validator** (`validate_artifacts.py --all`): Get baseline metrics

### 2. Integration Principles (Priority 1)

1. **Leverage, don't duplicate**: Use AgentQMS tools, extend where gaps exist
2. **Follow ADS v1.0**: All docs in `.ai-instructions/` must comply with tier structure
3. **Use artifact types**: Follow `.agentqms/plugins/artifact_types/` schemas
4. **Integrate CI/CD**: Extend existing workflows, don't create competing ones

### 3. Documentation Placement (Priority 2)

- **Root `.ai-instructions/`**: Project-wide contracts (system architecture, schemas, APIs)
- **App `.ai-instructions/`**: App-specific details (OCR console services, frontend context)
- **AgentQMS/knowledge/`**: Framework documentation (if needed, check existing structure)

---

## Conclusion

Existing AgentQMS + ADS v1.0 infrastructure provides **70%+ of required tooling**. Strategic audit plan should **integrate, not compete**.

**Key Changes**:
1. Extend existing tools instead of creating new ones
2. Use AgentQMS artifact types for consistency
3. Follow ADS v1.0 tier structure for `.ai-instructions/` placement
4. Enable pre-commit hooks already configured
5. Leverage GitHub Actions workflows already running

**Outcome**: **30% time savings** (28-44h vs 40-60h) while maintaining same quality standards.

**Next Step**: Approve revised integration strategy, then execute Phase 1 (4-6 hours) to generate discovery reports.
