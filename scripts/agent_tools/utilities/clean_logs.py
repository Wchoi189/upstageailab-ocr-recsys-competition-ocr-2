#!/usr/bin/env python3
"""
Clean up log files by removing ANSI escape codes and making them human readable.
"""

import re
import sys
from pathlib import Path


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    # Pattern to match ANSI escape codes
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[mK]")
    return ansi_pattern.sub("", text)


def clean_log_file(file_path: Path) -> None:
    """Clean a single log file."""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"🧹 Cleaning log file: {file_path}")

    # Read the original content
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with file_path.open("r", encoding="latin-1") as f:
            content = f.read()

    # Strip ANSI codes
    cleaned_content = strip_ansi_codes(content)

    # Create backup
    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
    file_path.rename(backup_path)
    print(f"📦 Created backup: {backup_path}")

    # Write cleaned content
    with file_path.open("w", encoding="utf-8") as f:
        f.write(cleaned_content)

    print(f"✅ Cleaned log file: {file_path}")


def clean_all_logs(logs_dir: Path) -> None:
    """Clean all log files in a directory."""
    if not logs_dir.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return

    log_files = list(logs_dir.glob("*.out")) + list(logs_dir.glob("*.log"))

    if not log_files:
        print(f"📭 No log files found in: {logs_dir}")
        return

    print(f"🔍 Found {len(log_files)} log files to clean")

    for log_file in log_files:
        clean_log_file(log_file)
        print()


def main() -> None:
    """Main function."""
    if len(sys.argv) > 1:
        # Clean specific file
        file_path = Path(sys.argv[1])
        clean_log_file(file_path)
    else:
        # Clean all logs in the logs directory
        project_root = Path(__file__).resolve().parents[2]
        logs_dir = project_root / "logs"
        clean_all_logs(logs_dir)


if __name__ == "__main__":
    main()
