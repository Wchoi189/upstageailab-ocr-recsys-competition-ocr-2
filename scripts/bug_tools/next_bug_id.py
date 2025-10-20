#!/usr/bin/env python3
"""
Bug Report ID Manager

Provides a consistent, date-scoped incremental index for bug reports, generating IDs
like BUG-YYYYMMDD-###. Stores counters in .bug_index/ per date.

Usage:
    python scripts/bug_tools/next_bug_id.py           # prints next ID, e.g., BUG-20251021-001
    python scripts/bug_tools/next_bug_id.py --peek    # prints next without incrementing
    python scripts/bug_tools/next_bug_id.py --reset   # resets today's counter to 0

"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / ".bug_index"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def _state_file_for(date: dt.date) -> Path:
    return STATE_DIR / f"{date.strftime('%Y%m%d')}.counter"


def read_count(date: dt.date) -> int:
    f = _state_file_for(date)
    if not f.exists():
        return 0
    try:
        return int(f.read_text().strip())
    except Exception:
        return 0


def write_count(date: dt.date, value: int) -> None:
    f = _state_file_for(date)
    f.write_text(str(max(0, value)))


def next_bug_id(peek: bool = False) -> str:
    today = dt.date.today()
    count = read_count(today)
    bug_num = count + 1
    bug_id = f"BUG-{today.strftime('%Y%m%d')}-{bug_num:03d}"
    if not peek:
        write_count(today, bug_num)
    return bug_id


def reset_today() -> None:
    write_count(dt.date.today(), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--peek", action="store_true", help="Show next ID without incrementing")
    parser.add_argument("--reset", action="store_true", help="Reset today's counter to 0")
    args = parser.parse_args()

    if args.reset:
        reset_today()
        print("OK")
    else:
        print(next_bug_id(peek=args.peek))
