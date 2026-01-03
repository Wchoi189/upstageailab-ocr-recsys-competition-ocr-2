#!/usr/bin/env python3
import subprocess
from collections import Counter


def analyze_mypy():
    print("Running Mypy analysis on ocr/ directory...")
    try:
        result = subprocess.run(
            ["uv", "run", "mypy", "ocr/", "--no-error-summary"],
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        print("Error: 'uv' not found. Please install uv first.")
        return

    lines = result.stdout.splitlines()
    error_types = []
    files_with_errors = []

    for line in lines:
        if ":" in line:
            parts = line.split(":")
            if len(parts) >= 4:
                filename = parts[0]
                error_part = parts[-1].strip()
                if "[" in error_part and "]" in error_part:
                    error_type = error_part[error_part.find("[")+1 : error_part.find("]")]
                    error_types.append(error_type)
                files_with_errors.append(filename)

    print("\nTotal Errors:", len(lines))
    print("\nTop Error Types:")
    for error_type, count in Counter(error_types).most_common(10):
        print(f"  - {error_type}: {count}")

    print("\nFiles with most errors:")
    for filename, count in Counter(files_with_errors).most_common(10):
        print(f"  - {filename}: {count}")

if __name__ == "__main__":
    analyze_mypy()
