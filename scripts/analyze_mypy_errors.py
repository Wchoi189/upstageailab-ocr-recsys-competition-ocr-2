#!/usr/bin/env python3
"""
MyPy Error Analysis and Categorization Script

This script runs mypy and categorizes errors by type and severity to help
prioritize type annotation work in a large ML/research codebase.

Usage:
    python scripts/analyze_mypy_errors.py [--fix-imports]

Output:
    - mypy-errors-full.txt: Complete mypy output
    - mypy-errors-summary.txt: Categorized summary
    - mypy-errors-by-module.txt: Errors grouped by module
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class MypyErrorAnalyzer:
    """Analyzes and categorizes mypy errors."""

    def __init__(self):
        self.errors: List[str] = []
        self.error_categories: Dict[str, List[str]] = defaultdict(list)
        self.module_errors: Dict[str, List[str]] = defaultdict(list)
        self.error_codes: Dict[str, int] = defaultdict(int)

    def run_mypy(self) -> Tuple[bool, str]:
        """Run mypy and capture output."""
        print("🔍 Running mypy type checking...")
        try:
            result = subprocess.run(
                ["uv", "run", "mypy", ".", "--show-error-codes"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            print("⚠️ MyPy timed out after 5 minutes")
            return False, ""
        except FileNotFoundError:
            print("❌ Error: 'uv' not found. Please install uv first.")
            sys.exit(1)

    def parse_errors(self, output: str):
        """Parse mypy output into structured errors."""
        lines = output.split("\n")
        for line in lines:
            if not line.strip() or line.startswith("Found") or line.startswith("Success"):
                continue

            self.errors.append(line)

            # Extract module name
            match = re.match(r"^([^:]+\.py):", line)
            if match:
                module = match.group(1)
                self.module_errors[module].append(line)

            # Categorize by error type
            self._categorize_error(line)

            # Count error codes
            code_match = re.search(r"\[([a-z-]+)\]", line)
            if code_match:
                self.error_codes[code_match.group(1)] += 1

    def _categorize_error(self, error: str):
        """Categorize error by type."""
        error_lower = error.lower()

        if "import" in error_lower or "cannot find" in error_lower:
            self.error_categories["Import Errors"].append(error)
        elif "incompatible" in error_lower or "expected" in error_lower:
            self.error_categories["Type Incompatibility"].append(error)
        elif "has no attribute" in error_lower:
            self.error_categories["Attribute Errors"].append(error)
        elif "undefined" in error_lower or "not defined" in error_lower:
            self.error_categories["Undefined Names"].append(error)
        elif "any" in error_lower:
            self.error_categories["Any Type Issues"].append(error)
        elif "return" in error_lower:
            self.error_categories["Return Type Issues"].append(error)
        elif "argument" in error_lower:
            self.error_categories["Argument Type Issues"].append(error)
        else:
            self.error_categories["Other"].append(error)

    def generate_summary(self) -> str:
        """Generate a human-readable summary."""
        summary = []
        summary.append("=" * 80)
        summary.append("MYPY ERROR ANALYSIS SUMMARY")
        summary.append("=" * 80)
        summary.append(f"\nTotal Errors: {len(self.errors)}\n")

        # Error categories
        summary.append("\n📊 ERRORS BY CATEGORY:")
        summary.append("-" * 80)
        for category, errors in sorted(
            self.error_categories.items(), key=lambda x: len(x[1]), reverse=True
        ):
            summary.append(f"\n{category}: {len(errors)} errors")
            summary.append(f"  First 3 examples:")
            for error in errors[:3]:
                summary.append(f"    {error[:120]}...")

        # Error codes
        summary.append("\n\n📝 ERRORS BY CODE:")
        summary.append("-" * 80)
        for code, count in sorted(self.error_codes.items(), key=lambda x: x[1], reverse=True)[:10]:
            summary.append(f"  [{code}]: {count}")

        # Top modules with errors
        summary.append("\n\n📁 TOP 10 MODULES WITH ERRORS:")
        summary.append("-" * 80)
        sorted_modules = sorted(self.module_errors.items(), key=lambda x: len(x[1]), reverse=True)
        for module, errors in sorted_modules[:10]:
            summary.append(f"  {module}: {len(errors)} errors")

        # Recommendations
        summary.append("\n\n💡 RECOMMENDATIONS:")
        summary.append("-" * 80)
        summary.append(self._generate_recommendations())

        return "\n".join(summary)

    def _generate_recommendations(self) -> str:
        """Generate actionable recommendations."""
        recs = []

        # Import errors
        import_count = len(self.error_categories["Import Errors"])
        if import_count > 0:
            recs.append(
                f"\n1. Import Errors ({import_count}):\n"
                "   - Add missing library stubs to [[tool.mypy.overrides]]\n"
                "   - Run: pip install types-<library>"
            )

        # Type incompatibility
        incompat_count = len(self.error_categories["Type Incompatibility"])
        if incompat_count > 0:
            recs.append(
                f"\n2. Type Incompatibility ({incompat_count}):\n"
                "   - These are real type safety issues\n"
                "   - Priority: Review and fix in production code"
            )

        # Any type issues
        any_count = len(self.error_categories["Any Type Issues"])
        if any_count > 0:
            recs.append(
                f"\n3. Any Type Issues ({any_count}):\n"
                "   - Low priority for ML code\n"
                "   - Consider: warn_return_any = false"
            )

        # Suggested exclusions
        high_error_modules = [
            m
            for m, errs in self.module_errors.items()
            if len(errs) > 20 and not m.startswith("ocr/core/")
        ]
        if high_error_modules:
            recs.append(
                "\n4. Consider Excluding:\n"
                + "\n".join(f'   - "{m}"' for m in high_error_modules[:5])
            )

        return "\n".join(recs) if recs else "   All errors are manageable! 🎉"

    def save_reports(self, output: str):
        """Save analysis reports to files."""
        reports_dir = Path(".")

        # Full output
        with open(reports_dir / "mypy-errors-full.txt", "w") as f:
            f.write(output)
        print(f"✅ Saved: mypy-errors-full.txt")

        # Summary
        summary = self.generate_summary()
        with open(reports_dir / "mypy-errors-summary.txt", "w") as f:
            f.write(summary)
        print(f"✅ Saved: mypy-errors-summary.txt")
        print(f"\n{summary}")

        # By module
        with open(reports_dir / "mypy-errors-by-module.txt", "w") as f:
            f.write("ERRORS BY MODULE\n")
            f.write("=" * 80 + "\n\n")
            for module, errors in sorted(self.module_errors.items()):
                f.write(f"\n{module} ({len(errors)} errors)\n")
                f.write("-" * 80 + "\n")
                for error in errors:
                    f.write(f"{error}\n")
        print(f"✅ Saved: mypy-errors-by-module.txt")


def main():
    parser = argparse.ArgumentParser(description="Analyze mypy type checking errors")
    parser.add_argument(
        "--fix-imports",
        action="store_true",
        help="Suggest pip install commands for missing stubs",
    )
    args = parser.parse_args()

    analyzer = MypyErrorAnalyzer()

    # Run mypy
    success, output = analyzer.run_mypy()

    if success:
        print("✅ MyPy check passed! No errors found.")
        return 0

    # Analyze errors
    analyzer.parse_errors(output)

    # Save reports
    analyzer.save_reports(output)

    # Fix imports if requested
    if args.fix_imports:
        import_errors = analyzer.error_categories["Import Errors"]
        if import_errors:
            print("\n📦 Missing stub packages:")
            # Extract library names from import errors
            libraries = set()
            for error in import_errors:
                match = re.search(r'Cannot find implementation or library stub for module named "([^"]+)"', error)
                if match:
                    lib = match.group(1).split(".")[0]
                    libraries.add(lib)

            if libraries:
                print("\nRun these commands:")
                for lib in sorted(libraries):
                    print(f"  pip install types-{lib}")

    print(f"\n📈 Total errors: {len(analyzer.errors)}")
    print("\n💡 Next steps:")
    print("   1. Review mypy-errors-summary.txt")
    print("   2. Add problematic modules to pyproject.toml exclude list")
    print("   3. Fix critical errors in production code first")
    print("   4. Gradually improve type coverage")

    return 1


if __name__ == "__main__":
    sys.exit(main())
