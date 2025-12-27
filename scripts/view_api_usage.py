#!/usr/bin/env python3
"""View Upstage API usage statistics.

Usage:
    python scripts/view_api_usage.py
    python scripts/view_api_usage.py --since 2025-12-01
"""

import argparse
from datetime import datetime

from ocr.utils.api_usage_tracker import get_tracker


def main():
    parser = argparse.ArgumentParser(description="View Upstage API usage statistics")
    parser.add_argument(
        "--since",
        type=str,
        help="Show usage since date (ISO format: YYYY-MM-DD)"
    )
    args = parser.parse_args()

    tracker = get_tracker()

    since_dt = None
    if args.since:
        try:
            since_dt = datetime.fromisoformat(args.since)
        except ValueError:
            print(f"Invalid date format: {args.since}. Use YYYY-MM-DD")
            return 1

    tracker.print_report(since=since_dt)
    return 0


if __name__ == "__main__":
    exit(main())
