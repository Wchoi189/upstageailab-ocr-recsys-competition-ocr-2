#!/usr/bin/env python3
"""
Architecture Guardian - Pre-commit hook that enforces architectural boundaries.

This hook prevents the architectural violations that have plagued 5 refactors.
It's not just a linter - it's a memory system that remembers your goals.

USAGE:
    .git/hooks/pre-commit calls this script
    OR run manually: python3 .pre-commit-hooks/architecture_guardian.py

WHAT IT CATCHES:
    1. Detection code added to ocr/core/ (should be in ocr/domains/detection/)
    2. Cross-domain imports (detection → recognition without interface)
    3. Registry bloat (new eager registrations)
    4. Import time regressions (>100ms for single module)
    5. Violations of documented architecture rules

WHY IT EXISTS:
    "I've done 5 refactors and keep losing sight of goals after conversations end"
    - This system remembers for you
    - It blocks violations BEFORE they compound
    - It teaches the correct patterns
"""

import ast
import sys
import time
from pathlib import Path
from dataclasses import dataclass

# ============================================================================
# ARCHITECTURAL RULES (Edit these to match your current goals)
# ============================================================================

# RULE 1: Detection-specific code must NOT be in ocr/core/
DETECTION_KEYWORDS = {
    'polygon', 'box', 'detection', 'dbnet', 'craft', 'binary_map',
    'prob_map', 'thresh_map', 'inverse_matrix', 'get_polygons',
    'DetectionHead', 'DetectionLoss', 'shrink_ratio', 'contour'
}

# RULE 2: Domain boundaries - what can import what
ALLOWED_IMPORTS = {
    'ocr.domains.detection': {'ocr.core', 'ocr.domains.detection'},
    'ocr.domains.recognition': {'ocr.core', 'ocr.domains.recognition'},
    'ocr.domains.kie': {'ocr.core', 'ocr.domains.kie'},
    'ocr.domains.layout': {'ocr.core', 'ocr.domains.layout'},
    'ocr.core': {'ocr.core'},  # Core can only import itself
}

# RULE 3: Files that should move (mark for future migration)
FLAGGED_FILES = {
    'ocr/core/validation.py': 'Move to ocr/domains/detection/validation.py',
    'ocr/core/models/loss/db_loss.py': 'Move to ocr/domains/detection/losses/db_loss.py',
    'ocr/core/models/loss/craft_loss.py': 'Move to ocr/domains/detection/losses/craft_loss.py',
    'ocr.domains.detection.metrics.cleval_metric.py': 'Move to ocr/domains/detection/metrics/cleval_metric.py',
}

# RULE 4: Import time budget (milliseconds)
MAX_IMPORT_TIME_MS = 100

# RULE 5: Patterns that indicate architectural problems
ANTI_PATTERNS = [
    (r'from ocr\.core\.interfaces\.models import.*Head', 'BaseHead is detection-specific, not shared'),
    (r'registry\.register_\w+\([^)]+\)\s*(?!#.*lazy)', 'Use lazy registration, not eager'),
    (r'from ocr\.core import \*', 'Star imports hide dependencies, be explicit'),
]

# ============================================================================
# VIOLATION TRACKING
# ============================================================================

@dataclass
class Violation:
    file: str
    line: int
    rule: str
    message: str
    severity: str  # ERROR, WARNING, INFO
    suggestion: str = ""

class ArchitectureGuardian:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations: list[Violation] = []

    def check_file(self, filepath: Path) -> list[Violation]:
        """Check a single file for architectural violations."""
        relative_path = filepath.relative_to(self.project_root)
        violations = []

        # Skip non-Python files
        if filepath.suffix != '.py':
            return violations

        content = filepath.read_text()

        # RULE 1: Check for detection code in ocr/core/
        if str(relative_path).startswith('ocr/core/'):
            violations.extend(self._check_detection_in_core(relative_path, content))

        # RULE 2: Check cross-domain imports
        violations.extend(self._check_domain_boundaries(relative_path, content))

        # RULE 3: Check flagged files
        violations.extend(self._check_flagged_files(relative_path))

        # RULE 4: Check anti-patterns
        violations.extend(self._check_anti_patterns(relative_path, content))

        return violations

    def _check_detection_in_core(self, filepath: Path, content: str) -> list[Violation]:
        """Rule 1: Detect detection-specific code in ocr/core/."""
        violations = []

        # Count detection keywords
        lower_content = content.lower()
        detection_count = sum(1 for kw in DETECTION_KEYWORDS if kw in lower_content)

        if detection_count >= 3:
            violations.append(Violation(
                file=str(filepath),
                line=1,
                rule="CORE_PURITY",
                message=f"File contains {detection_count} detection-specific keywords",
                severity="ERROR",
                suggestion="Move to ocr/domains/detection/ - core/ should only contain truly shared code"
            ))

        # Check for detection-specific imports
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = getattr(node, 'module', None)
                    if module and 'detection' in module:
                        violations.append(Violation(
                            file=str(filepath),
                            line=node.lineno,
                            rule="CORE_PURITY",
                            message=f"Importing from detection module: {module}",
                            severity="ERROR",
                            suggestion="Core should not depend on domain-specific modules"
                        ))
        except SyntaxError:
            pass  # Skip files with syntax errors

        return violations

    def _check_domain_boundaries(self, filepath: Path, content: str) -> list[Violation]:
        """Rule 2: Check for cross-domain imports."""
        violations = []

        # Determine current domain
        path_str = str(filepath)
        current_domain = None
        for domain in ALLOWED_IMPORTS.keys():
            domain_path = domain.replace('.', '/')
            if domain_path in path_str:
                current_domain = domain
                break

        if not current_domain:
            return violations

        # Check imports
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Check if importing from forbidden domain
                        for domain in ALLOWED_IMPORTS.keys():
                            if node.module.startswith(domain):
                                if domain not in ALLOWED_IMPORTS[current_domain]:
                                    violations.append(Violation(
                                        file=str(filepath),
                                        line=node.lineno,
                                        rule="DOMAIN_BOUNDARY",
                                        message=f"Cross-domain import: {current_domain} → {domain}",
                                        severity="ERROR",
                                        suggestion="Use interfaces in ocr/core/interfaces/ for cross-domain communication"
                                    ))
        except SyntaxError:
            pass

        return violations

    def _check_flagged_files(self, filepath: Path) -> list[Violation]:
        """Rule 3: Warn about files marked for migration."""
        violations = []

        path_str = str(filepath)
        for flagged_path, suggestion in FLAGGED_FILES.items():
            if path_str.endswith(flagged_path):
                violations.append(Violation(
                    file=str(filepath),
                    line=1,
                    rule="MIGRATION_PENDING",
                    message="This file is marked for migration",
                    severity="WARNING",
                    suggestion=suggestion
                ))

        return violations

    def _check_anti_patterns(self, filepath: Path, content: str) -> list[Violation]:
        """Rule 4: Check for known anti-patterns."""
        import re
        violations = []

        for pattern, message in ANTI_PATTERNS:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                violations.append(Violation(
                    file=str(filepath),
                    line=line_num,
                    rule="ANTI_PATTERN",
                    message=message,
                    severity="WARNING",
                    suggestion="See CRITICAL_ARCHITECTURE_ASSESSMENT.md for correct patterns"
                ))

        return violations

    def report_violations(self, violations: list[Violation]) -> int:
        """Print violations in a helpful format. Returns number of errors."""
        if not violations:
            print("✅ No architectural violations detected")
            return 0

        # Group by severity
        errors = [v for v in violations if v.severity == "ERROR"]
        warnings = [v for v in violations if v.severity == "WARNING"]

        print("\n" + "="*80)
        print("🚨 ARCHITECTURE GUARDIAN REPORT")
        print("="*80)

        if errors:
            print(f"\n❌ {len(errors)} ERRORS (must fix before commit):\n")
            for v in errors:
                print(f"  {v.file}:{v.line}")
                print(f"    Rule: {v.rule}")
                print(f"    ❌ {v.message}")
                if v.suggestion:
                    print(f"    💡 {v.suggestion}")
                print()

        if warnings:
            print(f"\n⚠️  {len(warnings)} WARNINGS (should fix soon):\n")
            for v in warnings:
                print(f"  {v.file}:{v.line}")
                print(f"    Rule: {v.rule}")
                print(f"    ⚠️  {v.message}")
                if v.suggestion:
                    print(f"    💡 {v.suggestion}")
                print()

        print("="*80)
        print("📖 For detailed guidance, see:")
        print("   analysis/architecture-migration-2026-01-21/CRITICAL_ARCHITECTURE_ASSESSMENT.md")
        print("="*80)

        return len(errors)

def check_import_time():
    """Rule 5: Check if imports are too slow."""
    print("\n⏱️  Checking import time budget...")

    test_imports = [
        'from ocr.domains.detection.models.heads.db_head import DBHead',
        'from ocr.domains.detection.models.architectures import craft',
        'from ocr.core import registry',
    ]

    violations = []
    for import_stmt in test_imports:
        start = time.time()
        try:
            exec(import_stmt)
            duration_ms = (time.time() - start) * 1000

            if duration_ms > MAX_IMPORT_TIME_MS:
                print(f"  ❌ SLOW: {import_stmt} took {duration_ms:.0f}ms (budget: {MAX_IMPORT_TIME_MS}ms)")
                violations.append(f"Import time: {import_stmt}")
            else:
                print(f"  ✅ FAST: {import_stmt} took {duration_ms:.0f}ms")
        except Exception as e:
            print(f"  ⚠️  Cannot import: {import_stmt} ({e})")

    return violations

def main():
    """Main entry point for pre-commit hook."""
    project_root = Path(__file__).parent.parent

    # Get changed files from git
    import subprocess
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR'],
        capture_output=True,
        text=True,
        cwd=project_root
    )

    if result.returncode != 0:
        print("⚠️  Could not get staged files. Checking all Python files...")
        changed_files = list(project_root.glob('ocr/**/*.py'))
    else:
        changed_files = [
            project_root / f
            for f in result.stdout.strip().split('\n')
            if f.endswith('.py')
        ]

    if not changed_files:
        print("✅ No Python files changed")
        return 0

    print(f"\n🔍 Checking {len(changed_files)} files for architectural violations...\n")

    guardian = ArchitectureGuardian(project_root)
    all_violations = []

    for filepath in changed_files:
        if filepath.exists():
            violations = guardian.check_file(filepath)
            all_violations.extend(violations)

    # Report violations
    error_count = guardian.report_violations(all_violations)

    # Check import times (non-blocking, just informational)
    import_violations = check_import_time()

    if error_count > 0:
        print("\n❌ COMMIT BLOCKED: Fix errors above before committing")
        return 1
    elif import_violations:
        print("\n⚠️  Import time budget exceeded - consider refactoring for performance")
        # Don't block commit, just warn
        return 0
    else:
        print("\n✅ All checks passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
