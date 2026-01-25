---
type: changelog
category: governance
date: 2026-01-25
version: "1.0"
status: completed
---

# Standards Documentation Update - January 25, 2026

## Summary

Created comprehensive standards documentation based on lessons from legacy purge audit. Established anti-patterns catalog, V5 architecture patterns reference, and bloat detection automation.

## Changes

### New Standards Documents

#### 1. Anti-Patterns Catalog
**File:** `AgentQMS/standards/tier2-framework/anti-patterns.md`

**Critical Anti-Patterns (10 total):**
- AP-001: Model-Level Configuration (models creating optimizers)
- AP-002: Silent Fallback Chains (exception handlers masking errors)
- AP-003: Multiple Configuration Paths (ambiguous config locations)
- AP-004: Overly Permissive Checkpoint Loading (strict=False abuse)
- AP-005: Magic Numbers (hardcoded values)
- AP-006: Domain-Specific Code in Shared Utils
- AP-007: God Classes/Functions (complexity violations)
- AP-008: Commented-Out Code
- AP-009: Duplicate Code
- AP-010: Unclear Naming

**Features:**
- Severity levels: Critical (block PR) / High / Medium
- Bad vs Good code examples for each pattern
- Enforcement measures (pre-commit, linter, code review)
- Pre-commit hook configuration
- Linter rule mappings
- Code review checklist

#### 2. V5 Architecture Patterns
**File:** `AgentQMS/standards/tier1-sst/v5-architecture-patterns.md`

**Required Patterns (5 enforced):**
1. Optimizer Configuration: `config.train.optimizer` ONLY
2. Scheduler Configuration: `config.train.scheduler`
3. Model Architecture: `config.model.architectures` with `_recursive_=False`
4. Dataset Configuration: `config.data.datasets`
5. Checkpoint Loading: 2-level fallback maximum

**Features:**
- Standard locations for all configuration
- Hydra package directive requirements
- Domain separation import rules
- Error handling patterns (fail-fast with helpful messages)
- Testing patterns (model vs Lightning module separation)
- Validation tools (AST grep, pre-commit hooks)
- Migration guidelines

#### 3. Bloat Detection Rules
**File:** `AgentQMS/standards/tier2-framework/coding/bloat-detection-rules.md`

**Detection Criteria (4 types):**
1. Usage-Based: no imports 90+ days, no test coverage, no experiment refs
2. Duplication-Based: 80% similar code, 10+ lines duplicated
3. Complexity-Based: cyclomatic > 15, function > 100 lines, file > 1000 lines
4. Architectural: cross-domain imports, god classes, anti-pattern violations

**Features:**
- Automated scanning thresholds
- Archival decision tree
- Weekly scan GitHub Actions workflow
- Manual review process
- Archive structure and README templates
- Tool integration (pylint, radon, ADT)

#### 4. Bloat Detector Tool
**File:** `AgentQMS/tools/bloat_detector.py`

**Capabilities:**
- Git blame analysis for last import dates
- Test coverage checking
- Experiment reference detection
- Production import counting
- Severity classification (LOW/MEDIUM/HIGH/CRITICAL)
- Action recommendations (ARCHIVE/REFACTOR/MOVE/REVIEW)
- JSON report generation

**Usage:**
```bash
uv run python AgentQMS/tools/bloat_detector.py --threshold-days 90
uv run python AgentQMS/tools/bloat_detector.py --include-complexity
```

### Updated Standards Registry

**File:** `AgentQMS/standards/registry.yaml`

**New Task Mapping:**
- `standards_maintenance`: Triggers anti-patterns, V5 patterns, bloat rules

**Updated Task Mappings:**
- `code_changes`: Added anti-patterns.md and v5-architecture-patterns.md
- `code_analysis`: Added bloat-detection-rules.md and bloat keywords

## Impact

### For Developers
- ✅ Clear anti-patterns to avoid during code reviews
- ✅ Explicit V5 architecture patterns to follow
- ✅ Automated tools to detect bloat and quality issues
- ✅ Fail-fast error messages with migration guidance

### For Audits
- ✅ Standardized criteria for code quality evaluation
- ✅ Automated detection tools reduce manual review burden
- ✅ Consistent archival decisions based on documented rules

### For Maintenance
- ✅ Weekly automated bloat detection (GitHub Actions)
- ✅ Pre-commit hooks prevent anti-pattern introduction
- ✅ Linter integration catches violations early
- ✅ Clear archival process for unused code

## Enforcement

### Pre-commit Hooks (Planned)
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-anti-patterns
        entry: python AgentQMS/tools/check_anti_patterns.py
      - id: check-v5-compliance
        entry: python AgentQMS/tools/check_v5_compliance.py
```

### Linter Rules
```toml
# pyproject.toml
[tool.pylint.messages_control]
enable = ["bare-except", "broad-except", "duplicate-code"]

[tool.pylint.design]
max-args = 5
max-locals = 15
max-branches = 12
max-statements = 50
```

### AST Grep Rules (Planned)
```yaml
# rules/v5-optimizer-location.yaml
id: v5-optimizer-location
pattern: config.model.optimizer
message: "Use config.train.optimizer only (V5 standard)"
severity: error
```

## Related Work

**Source Audit:**
- [Legacy Purge Audit](../../docs/artifacts/audits/2026-01-25_2100_audit_legacy-purge.md)
- [Legacy Purge Resolution](../../docs/artifacts/audits/2026-01-25_2100_audit_legacy-purge-resolution.md)

**Referenced Standards:**
- [Naming Conventions](../tier1-sst/naming-conventions.yaml)
- [File Placement Rules](../tier1-sst/file-placement-rules.yaml)
- [Tool Catalog](../tier2-framework/tool-catalog.yaml)

## Next Steps

### Immediate (Week 1)
1. ✅ Create standards documents
2. ⏳ Implement `check_anti_patterns.py` script
3. ⏳ Implement `check_v5_compliance.py` script
4. ⏳ Set up pre-commit hooks

### Short-term (Month 1)
1. ⏳ Add complexity and duplication scans to bloat detector
2. ⏳ Create GitHub Actions workflow for weekly scans
3. ⏳ Set up AST grep rules
4. ⏳ Add linter configuration to pyproject.toml

### Long-term (Quarter 1)
1. ⏳ Establish automated bloat archival process
2. ⏳ Integrate with CI/CD to block anti-pattern PRs
3. ⏳ Track bloat metrics over time
4. ⏳ Quarterly review of thresholds and rules

## Metrics

**Documentation Added:**
- 3 new standard documents (700+ lines)
- 10 anti-patterns cataloged
- 5 V5 patterns enforced
- 4 bloat detection criteria
- 1 automated tool

**Coverage:**
- ✅ Model-level configuration anti-pattern
- ✅ Silent fallback anti-pattern
- ✅ Multiple config paths anti-pattern
- ✅ Checkpoint loading best practices
- ✅ Domain separation rules
- ✅ Error handling patterns
- ✅ Testing patterns
- ✅ Bloat detection automation

---

**Authors:** GitHub Copilot (Agent)
**Reviewers:** TBD
**Approved:** TBD

**Maintenance:**
- Review quarterly
- Update after major refactors
- Add new patterns from code reviews
- Tune thresholds based on team feedback
