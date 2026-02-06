#!/usr/bin/env python3
"""
Architecture Validation Hook

Enforces the domains-first architecture pattern for OCR project:
- ocr/domains/<domain>/*.py (domain-specific code)
- ocr/core/<domain>/*.py (shared infrastructure)

Prevents regression to flat structure.
"""

import sys
from pathlib import Path

# Allowed patterns
CORE_DOMAINS = {
    "models", "data", "utils", "transforms", "metrics", "losses",
    "inference", "communication", "lightning", "analysis", "evaluation",
    "interfaces", "registry", "infrastructure"
}
ALLOWED_CORE_PATHS = {f"ocr/core/{domain}" for domain in CORE_DOMAINS}

# Allowed top-level directories in ocr/
ALLOWED_TOP_LEVEL = {
    "core",            # Shared infrastructure
    "domains",         # Domain-specific logic (detection, recognition, etc.)
    "data",            # Data loading/processing
    "command_builder", # CLI tools
    "pipelines",       # Orchestration
    "synthetic_data",  # Data generation
    "validation",      # Validation scripts
    "services",        # Service integration (if any)
}

def validate_path(file_path: str) -> tuple[bool, str]:
    """
    Validate that a Python file follows domains-first architecture.

    Returns:
        (is_valid, error_message)
    """
    path = Path(file_path)

    # Only check files in ocr/ directory
    if not str(path).startswith("ocr/"):
        return True, ""

    # Ignore __init__.py and __pycache__
    if path.name == "__init__.py" or "__pycache__" in path.parts:
        return True, ""

    parts = path.parts
    # parts[0] is 'ocr'

    # Must have at least ocr/<top_level>/...
    if len(parts) < 3:
         # Files directly in ocr/ are typically not allowed, except maybe experiment_registry.py or similar
         if len(parts) == 2 and parts[1] in ("experiment_registry.py",):
             return True, ""

         return False, f"File {file_path} is too shallow. Put code in ocr/domains/ or ocr/core/."

    top_level = parts[1]

    if top_level not in ALLOWED_TOP_LEVEL:
        return False, (
            f"Directory 'ocr/{top_level}' is not an allowed top-level directory.\n"
            f"Allowed: {', '.join(sorted(ALLOWED_TOP_LEVEL))}"
        )

    # Specific checks for 'domains'
    if top_level == "domains":
        if len(parts) < 4:
            # ocr/domains/<domain>/file.py
             return False, f"File {file_path} in 'domains' must belong to a specific domain (e.g. ocr/domains/detection/)."

    # Specific checks for 'core'
    if top_level == "core":
        # Allow files directly in ocr/core/ (e.g. ocr/core/experiment.py)
        if len(parts) == 3:
            return True, ""

        if len(parts) >= 4:
            core_module = parts[2]
            if core_module not in CORE_DOMAINS:
                return False, (
                    f"Unknown core module 'ocr/core/{core_module}'.\n"
                    f"Allowed core modules: {', '.join(sorted(CORE_DOMAINS))}"
                )

    return True, ""


def main(filenames: list[str] = None, check_all: bool = False) -> int:
    """
    Main validation function.

    Args:
        filenames: List of files to check (from pre-commit)
        check_all: If True, check all Python files in ocr/

    Returns:
        0 if all files pass, 1 if any violations found
    """
    if check_all:
        # Check all files in ocr/ directory
        ocr_dir = Path("ocr")
        if not ocr_dir.exists():
            print("Error: ocr/ directory not found", file=sys.stderr)
            return 1

        filenames = [
            str(f) for f in ocr_dir.rglob("*.py")
            if "__pycache__" not in str(f)
        ]

    if not filenames:
        filenames = []
        # Read from stdin (pre-commit passes files this way)
        for line in sys.stdin:
            filenames.append(line.strip())

    violations = []

    for filename in filenames:
        if not filename or not filename.endswith(".py"):
            continue

        is_valid, error_msg = validate_path(filename)
        if not is_valid:
            violations.append(error_msg)

    if violations:
        print("❌ Architecture validation failed!\n", file=sys.stderr)
        for violation in violations:
            print(violation, file=sys.stderr)
            print("", file=sys.stderr)

        print(
            "💡 Tip: Organize code using the domains-first pattern:",
            file=sys.stderr
        )
        print("  - Domain-specific: ocr/domains/<domain>/", file=sys.stderr)
        print("  - Shared infrastructure: ocr/core/<domain>/", file=sys.stderr)
        print("", file=sys.stderr)

        return 1

    print(f"✅ All {len(filenames)} files pass architecture validation")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate domains-first architecture pattern"
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Files to check (from pre-commit hook)"
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Check all Python files in ocr/ directory"
    )

    args = parser.parse_args()

    sys.exit(main(filenames=args.filenames, check_all=args.check_all))
