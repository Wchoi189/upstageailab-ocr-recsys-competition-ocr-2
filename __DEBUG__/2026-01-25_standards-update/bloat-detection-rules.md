---
type: standard
category: governance
tier: 3
version: "1.0"
ads_version: "1.0"
status: active
created: 2026-01-25 21:00 (KST)
updated: 2026-01-25 21:00 (KST)
---

# Bloat Detection Rules

## Purpose

Automated criteria for identifying unused, low-value, or archival-candidate code. Used by bloat detection tools to flag code for review.

---

## Detection Criteria

### 1. Usage-Based Bloat

**Threshold: Code not actively used**

```yaml
unused_code_criteria:
  # Time-based thresholds
  no_imports_days: 90           # No imports in git blame for 90+ days
  no_commits_days: 180          # No commits touching file for 180+ days
  marked_experimental_days: 180 # "WIP" or "experimental" tag for 180+ days
  
  # Coverage thresholds
  test_coverage_min: 0.0        # No test coverage at all
  
  # Reference thresholds
  no_experiment_refs: true      # Not referenced in any experiment config
  no_production_refs: true      # Not imported by production code
  
  # Documentation
  missing_docstring: true       # No module/class docstring
  marked_deprecated: true       # Explicitly marked @deprecated
```

**Detection Script:**
```python
# AgentQMS/tools/bloat_detector.py
def detect_unused_code(threshold_days: int = 90) -> list[Path]:
    """Find code not used in specified time period."""
    
    candidates = []
    
    for py_file in Path("ocr/").rglob("*.py"):
        # Check git blame for last import
        last_import = get_last_import_date(py_file)
        if last_import and (datetime.now() - last_import).days > threshold_days:
            
            # Additional checks
            if not has_test_coverage(py_file):
                if not referenced_in_experiments(py_file):
                    if not imported_by_production(py_file):
                        candidates.append(py_file)
    
    return candidates
```

**Usage:**
```bash
uv run python AgentQMS/tools/bloat_detector.py \
  --threshold-days 90 \
  --output analysis/bloat-candidates.json
```

---

### 2. Duplication-Based Bloat

**Threshold: Similar code in multiple locations**

```yaml
duplication_criteria:
  # Similarity thresholds
  similarity_threshold: 0.8     # 80% similar code
  min_lines: 10                 # At least 10 lines duplicated
  min_tokens: 50                # At least 50 tokens duplicated
  
  # Scope
  check_functions: true         # Check function-level duplication
  check_classes: true           # Check class-level duplication
  check_modules: false          # Don't check whole modules (too slow)
  
  # Exclusions
  exclude_patterns:
    - "*/tests/*"               # Test code may have duplication
    - "*/migrations/*"          # Migration code is intentionally similar
    - "*/__init__.py"           # Init files often similar
```

**Detection Tools:**
```bash
# Option 1: Pylint (fast, basic)
uv run pylint ocr/ --disable=all --enable=duplicate-code

# Option 2: ADT sg-search (advanced, AST-based)
uv run adt analyze sg-search \
  --pattern "def $FUNC($PARAMS): $$$BODY" \
  --similarity 0.8 \
  --min-lines 10

# Option 3: CPD (Copy-Paste Detector)
uv run pmd cpd --minimum-tokens 50 --files ocr/
```

---

### 3. Complexity-Based Bloat

**Threshold: Code that's too complex**

```yaml
complexity_criteria:
  # Function-level
  cyclomatic_complexity_max: 15      # McCabe complexity
  cognitive_complexity_max: 20       # Human understandability
  function_lines_max: 100            # Lines of code
  function_params_max: 5             # Parameter count
  
  # Class-level
  class_lines_max: 500               # Lines per class
  class_methods_max: 20              # Methods per class
  inheritance_depth_max: 3           # Inheritance depth
  
  # File-level
  file_lines_max: 1000               # Lines per file
  file_classes_max: 5                # Classes per file
  import_depth_max: 5                # Nested import levels
  
  # Module-level
  module_dependencies_max: 20        # Number of imports
  circular_dependencies: false       # No circular imports allowed
```

**Detection Tools:**
```bash
# Radon (complexity metrics)
uv run radon cc ocr/ -a -nb --total-average

# Output format:
# F 28:0 MyClass.complex_method - C (16)
#   ↑ File  ↑ Location ↑ Name   ↑ Grade ↑ Complexity

# Wily (complexity tracking over time)
uv run wily build ocr/
uv run wily report ocr/core/lightning/base.py

# ADT complexity analysis
uv run adt analyze complexity \
  --target ocr/ \
  --threshold 10 \
  --output analysis/complexity-report.json
```

---

### 4. Architectural Bloat

**Threshold: Code violating architecture**

```yaml
architectural_bloat_criteria:
  # Domain separation
  cross_domain_imports: true         # Detection importing from recognition
  core_importing_domain: true        # Core importing from domain/*
  domain_specific_in_core: true      # Domain logic in core/
  
  # Abstraction violations
  god_classes: true                  # Classes > 500 lines
  god_functions: true                # Functions > 100 lines
  deep_nesting: 5                    # Nesting depth > 5
  
  # Anti-patterns
  model_creates_optimizer: true      # AP-001 violation
  silent_fallbacks: true             # AP-002 violation
  multiple_config_paths: true        # AP-003 violation
  
  # Deprecated patterns
  uses_deprecated_api: true          # Calls to @deprecated functions
  legacy_imports: true               # Imports from archived code
```

**Detection:**
```bash
# Check anti-patterns
uv run python AgentQMS/tools/check_anti_patterns.py

# Check domain separation
uv run adt analyze dependency-graph \
  --target ocr/domains/ \
  --check-cross-domain

# Check architectural violations
uv run python AgentQMS/tools/check_v5_compliance.py
```

---

## Archival Decision Tree

```
┌─────────────────────────────────────────┐
│ Is code used in production?            │
│ (imported by runners/, used in config) │
└───────────┬─────────────────────────────┘
            │
      NO ───┴─────> Archive immediately
            │
      YES ──┴──────────────────────────────┐
                                           │
            ┌──────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│ Has test coverage > 0%?                 │
└───────────┬─────────────────────────────┘
            │
      NO ───┴─────> Mark for testing or archive
            │        (30-day grace period)
      YES ──┴──────────────────────────────┐
                                           │
            ┌──────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│ Is it in correct location?              │
│ (domain-specific in domain/,            │
│  shared in core/)                       │
└───────────┬─────────────────────────────┘
            │
      NO ───┴─────> Move to correct location
            │
      YES ──┴──────────────────────────────┐
                                           │
            ┌──────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│ Complexity acceptable?                  │
│ (< thresholds above)                    │
└───────────┬─────────────────────────────┘
            │
      NO ───┴─────> Flag for refactoring
            │
      YES ──┴─────> Keep
```

---

## Automated Scanning

### Weekly Scan Job

```yaml
# .github/workflows/bloat-detection.yml
name: Weekly Bloat Detection

on:
  schedule:
    - cron: '0 0 * * 0'  # Sunday midnight
  workflow_dispatch:      # Manual trigger

jobs:
  detect-bloat:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for git blame
      
      - name: Run bloat detection
        run: |
          uv run python AgentQMS/tools/bloat_detector.py \
            --threshold-days 90 \
            --output bloat-report.json
      
      - name: Check for violations
        run: |
          uv run python AgentQMS/tools/check_anti_patterns.py
      
      - name: Complexity analysis
        run: |
          uv run radon cc ocr/ -a -nb --total-average > complexity.txt
      
      - name: Create issue if bloat found
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'Weekly Bloat Detection: Issues Found',
              body: 'Automated bloat detection found issues. See workflow run for details.',
              labels: ['bloat', 'automated', 'maintenance']
            })
```

---

## Manual Review Process

### 1. Generate Bloat Report

```bash
# Full scan
uv run python AgentQMS/tools/bloat_detector.py \
  --threshold-days 90 \
  --include-complexity \
  --include-duplication \
  --output analysis/bloat-report-$(date +%Y-%m-%d).json
```

### 2. Review Candidates

```bash
# View report
uv run python AgentQMS/tools/bloat_report_viewer.py \
  --input analysis/bloat-report-2026-01-25.json \
  --sort-by severity \
  --filter "severity >= HIGH"
```

### 3. Create Archive Plan

```bash
# Generate archive recommendations
uv run python AgentQMS/tools/generate_archive_plan.py \
  --bloat-report analysis/bloat-report-2026-01-25.json \
  --output docs/artifacts/audits/bloat-archive-plan-2026-01-25.md
```

### 4. Execute Archive

```bash
# Archive identified bloat
uv run python AgentQMS/tools/execute_archive.py \
  --plan docs/artifacts/audits/bloat-archive-plan-2026-01-25.md \
  --dry-run  # Preview changes first

# If OK, execute
uv run python AgentQMS/tools/execute_archive.py \
  --plan docs/artifacts/audits/bloat-archive-plan-2026-01-25.md \
  --execute
```

---

## Archival Guidelines

### What to Archive

✅ **Archive These:**
- Code unused for > 90 days
- Experimental features abandoned > 180 days
- Duplicate implementations (keep best one)
- Code violating V5 architecture
- Deprecated APIs with no usage
- Test code for archived features

❌ **Don't Archive These:**
- Core infrastructure code
- Active experiment dependencies
- Shared utilities with > 3 references
- Code with recent commits (< 90 days)
- Code with > 70% test coverage and active use

### Archive Structure

```
archive/
├── YYYY-MM-DD_<component-name>/
│   ├── ARCHIVE_README.md       # What, why, restoration guide
│   ├── <archived-code>/        # Original code structure
│   └── metadata.json           # Detection metadata
```

### Archive README Template

```markdown
# <Component> Archive - YYYY-MM-DD

## Archive Reason

[Detection category: unused/duplicate/complex/architectural]

## Decision Rationale

**Detection Metrics:**
- Last import: YYYY-MM-DD (XXX days ago)
- Test coverage: X%
- Experiment references: X
- Complexity score: X

**Why Archived:**
- [Specific reasons based on detection criteria]

## Archived Components

[List of files/modules]

## Restoration Instructions

[If needed in future, how to restore]

## Related Audits

[Links to relevant audits]
```

---

## Tooling

### Bloat Detector Implementation

```python
# AgentQMS/tools/bloat_detector.py
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import json

@dataclass
class BloatCandidate:
    file_path: Path
    reasons: list[str]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    metrics: dict
    recommended_action: str  # ARCHIVE, REFACTOR, MOVE, REVIEW

class BloatDetector:
    def __init__(self, config: dict):
        self.config = config
        self.candidates = []
    
    def scan_unused_code(self) -> list[BloatCandidate]:
        """Scan for unused code based on import history."""
        threshold = timedelta(days=self.config["no_imports_days"])
        # ... implementation
    
    def scan_duplication(self) -> list[BloatCandidate]:
        """Scan for duplicate code."""
        # ... implementation using pylint or AST comparison
    
    def scan_complexity(self) -> list[BloatCandidate]:
        """Scan for overly complex code."""
        # ... implementation using radon
    
    def scan_architectural(self) -> list[BloatCandidate]:
        """Scan for architectural violations."""
        # ... implementation checking anti-patterns
    
    def generate_report(self, output: Path):
        """Generate JSON report of all candidates."""
        report = {
            "scan_date": datetime.now().isoformat(),
            "config": self.config,
            "summary": {
                "total_candidates": len(self.candidates),
                "by_severity": self._count_by_severity(),
                "by_action": self._count_by_action()
            },
            "candidates": [
                {
                    "file": str(c.file_path),
                    "reasons": c.reasons,
                    "severity": c.severity,
                    "metrics": c.metrics,
                    "action": c.recommended_action
                }
                for c in self.candidates
            ]
        }
        
        output.write_text(json.dumps(report, indent=2))
```

---

## Related Standards

- [Anti-Patterns Catalog](../tier2-framework/anti-patterns.md)
- [V5 Architecture Patterns](../tier1-sst/v5-architecture-patterns.md)
- [File Placement Rules](../tier1-sst/file-placement-rules.yaml)

---

**Maintenance:**
- Review thresholds quarterly
- Update based on project evolution
- Adjust for team size and velocity
- Tune false positive rate

**Audit History:**
- 2026-01-25: Initial rules from Legacy Purge Audit experience
