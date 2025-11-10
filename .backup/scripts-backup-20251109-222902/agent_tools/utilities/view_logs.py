#!/usr/bin/env python3
"""
View log files in a human-readable format.
Automatically strips ANSI escape codes and formats output nicely.
"""

import re
import sys
from pathlib import Path


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[mK]")
    return ansi_pattern.sub("", text)


def view_log_file(file_path: Path, lines: int | None = None) -> None:
    """View a log file with ANSI codes stripped."""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"📄 Viewing log file: {file_path}")
    print("=" * 80)

    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with file_path.open("r", encoding="latin-1") as f:
            content = f.read()

    # Strip ANSI codes
    cleaned_content = strip_ansi_codes(content)

    # Show last N lines if specified
    if lines:
        lines_list = cleaned_content.split("\n")
        cleaned_content = "\n".join(lines_list[-lines:])

    print(cleaned_content)
    print("=" * 80)


def list_log_files(logs_dir: Path) -> None:
    """List available log files."""
    if not logs_dir.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return

    log_files = sorted(logs_dir.rglob("*.out")) + sorted(logs_dir.rglob("*.log"))

    if not log_files:
        print(f"📭 No log files found in: {logs_dir}")
        return

    print(f"📋 Available log files in {logs_dir}:")
    for i, log_file in enumerate(log_files, 1):
        rel_path = log_file.relative_to(logs_dir)
        size = log_file.stat().st_size
        print(f"  {i:2d}. {rel_path} ({size:,} bytes)")


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/agent_tools/view_logs.py <log_file> [lines]")
        print("  python scripts/agent_tools/view_logs.py --list")
        print()
        print("Examples:")
        print("  python scripts/agent_tools/view_logs.py logs/streamlit/app.out")
        print("  python scripts/agent_tools/view_logs.py logs/streamlit/app.out 50")
        print("  python scripts/agent_tools/view_logs.py --list")
        return

    if sys.argv[1] == "--list":
        project_root = Path(__file__).resolve().parents[2]
        logs_dir = project_root / "logs"
        list_log_files(logs_dir)
        return

    file_path = Path(sys.argv[1])
    lines = int(sys.argv[2]) if len(sys.argv) > 2 else None

    view_log_file(file_path, lines)


if __name__ == "__main__":
    main()
