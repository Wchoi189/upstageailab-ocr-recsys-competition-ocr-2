#!/usr/bin/env python3
"""
Architecture Validation Hook

Enforces the feature-first architecture pattern for OCR project:
- ocr/<feature>/<domain>/*.py (feature-specific code)
- ocr/core/<domain>/*.py (shared infrastructure)

Prevents regression to flat structure.
"""

import sys
from pathlib import Path

# Allowed patterns
CORE_DOMAINS = {"models", "data", "utils", "transforms", "metrics", "losses", "inference", "communication", "lightning", "analysis", "evaluation"}
ALLOWED_CORE_PATHS = {f"ocr/core/{domain}" for domain in CORE_DOMAINS}

# Known features
KNOWN_FEATURES = {
    "detection",
    "recognition",
    "perspective",
    "kie",
    "layout",
    "features",  # Features parent directory
    "core",  # Core infrastructure
}


def validate_path(file_path: str) -> tuple[bool, str]:
    """
    Validate that a Python file follows feature-first architecture.

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

    # Must have at least ocr/<feature>/<domain>/file.py structure
    if len(parts) < 4:
        # Special case: allow ocr/agents/*.py and ocr/communication/*.py
        # These are legacy top-level directories that we validate as exceptions
        if len(parts) == 3 and parts[1] in {"agents", "communication", "synthetic_data", "validation"}:
            return True, ""

        # Special case: allow ocr/core/<domain>/file.py (3 parts + filename = 4)
        if len(parts) == 4 and parts[1] == "core" and parts[2] in CORE_DOMAINS:
            return True, ""

        return False, (
            f"File {file_path} violates feature-first architecture.\n"
            f"  Expected: ocr/<feature>/<domain>/*.py or ocr/core/<domain>/*.py\n"
            f"  Got: {file_path}\n"
            f"  Please organize code by feature first, then by domain."
        )

    feature = parts[1]
    domain = parts[2]

    # Check if it's a core infrastructure file
    if feature == "core":
        if domain not in CORE_DOMAINS:
            return False, (
                f"File {file_path} uses unknown core domain '{domain}'.\n"
                f"  Allowed core domains: {', '.join(sorted(CORE_DOMAINS))}\n"
                f"  If this is feature-specific code, move to ocr/<feature>/{domain}/"
            )
        return True, ""

    # For feature-specific files, check structure
    if feature not in KNOWN_FEATURES:
        # Warn but don't fail for new features
        print(
            f"⚠️  Warning: '{feature}' is not in known features list.\n"
            f"   Known features: {', '.join(sorted(KNOWN_FEATURES))}\n"
            f"   If this is a new feature, please update scripts/hooks/validate_architecture.py",
            file=sys.stderr
        )

    # Domain can be anything for features, but warn for common mistakes
    if domain in {"__init__", "__pycache__"}:
        return False, f"Invalid domain name in {file_path}"

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
            "💡 Tip: Organize code using the feature-first pattern:",
            file=sys.stderr
        )
        print("  - Feature-specific: ocr/<feature>/<domain>/", file=sys.stderr)
        print("  - Shared infrastructure: ocr/core/<domain>/", file=sys.stderr)
        print("", file=sys.stderr)

        return 1

    print(f"✅ All {len(filenames)} files pass architecture validation")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate feature-first architecture pattern"
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
