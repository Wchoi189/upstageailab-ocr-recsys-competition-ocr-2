#!/usr/bin/env python3
"""
Path Usage Validator
Enforces the use of AgentQMS.tools.utils.paths instead of fragile sys.path hacks.
"""
import argparse
import re
import sys
from pathlib import Path

# Patterns that indicate fragile path handling
SUSPICIOUS_PATTERNS = [
    (
        r"sys\.path\.append",
        "Avoid manipulating sys.path directly. Use 'uv pip install -e .' or path utilities."
    ),
    (
        r"\.parent\.parent\.parent",
        "Excessive parent chaining detected (>2). Use AgentQMS.tools.utils.paths.get_project_root() instead."
    ),
    (
        r"os\.path\.abspath\(.*os\.path\.join\(.*os\.path\.dirname",
        "Fragile os.path construction. Use pathlib.Path and standard standard utilities."
    ),
]

# Allow exceptions with this comment
NOQA_MARKER = "noqa: path-hack"

def check_file(file_path: Path) -> list[str]:
    errors = []
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            if NOQA_MARKER in line:
                continue

            for pattern, message in SUSPICIOUS_PATTERNS:
                if re.search(pattern, line):
                    errors.append(f"{file_path}:{i}: {message}\n  > {line.strip()}")

    except Exception as e:
        errors.append(f"{file_path}: Failed to read/parse: {e}")

    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate Python path usage.")
    parser.add_argument("files", nargs="+", help="Files to check")
    args = parser.parse_args()

    all_errors = []
    for f in args.files:
        path = Path(f)
        if path.suffix == ".py" and path.exists():
            all_errors.extend(check_file(path))

    if all_errors:
        print("❌ found fragile path usage:")
        for error in all_errors:
            print(error)
        print(f"\n💡 To allow valid exceptions, append '# {NOQA_MARKER}' to the line.")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
