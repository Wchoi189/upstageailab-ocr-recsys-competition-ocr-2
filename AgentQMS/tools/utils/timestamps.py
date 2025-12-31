#!/usr/bin/env python3
"""
Timestamp Utilities for AgentQMS

Provides timezone-aware timestamp utilities for artifact metadata.

Supports timezone configuration from environment, settings, or fallback to KST.

Usage:
    from AgentQMS.tools.utils.timestamps import get_kst_timestamp

    timestamp = get_kst_timestamp()  # Returns "2025-12-06 12:00 (KST)"
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from AgentQMS.tools.utils.config import load_config


def get_configured_timezone() -> str:
    """
    Get configured timezone from settings or environment.

    Priority:
    1. TZ environment variable
    2. .agentqms/settings.yaml framework.timezone
    3. Default to Asia/Seoul (KST)

    Returns:
        Timezone string (e.g., "Asia/Seoul", "UTC", "America/New_York")
    """
    # Check environment variable first
    if env_tz := os.getenv("TZ"):
        return env_tz

    # Check config file
    try:
        config = load_config()
        if config and "framework" in config:
            if "timezone" in config["framework"]:
                return config["framework"]["timezone"]
    except Exception:
        pass

    # Default to Asia/Seoul (KST)
    return "Asia/Seoul"


def get_timezone_abbr(timezone_str: str) -> str:
    """
    Get timezone abbreviation from IANA timezone name.
    """
    timezone_abbrs = {
        "Asia/Seoul": "KST",
        "Asia/Tokyo": "JST",
        "Asia/Shanghai": "CST",
        "Asia/Hong_Kong": "HKT",
        "Asia/Singapore": "SGT",
        "Asia/Bangkok": "ICT",
        "Asia/Dubai": "GST",
        "Asia/Kolkata": "IST",
        "UTC": "UTC",
        "Europe/London": "GMT",
        "Europe/Paris": "CET",
        "America/New_York": "EST",
        "America/Chicago": "CST",
        "America/Denver": "MST",
        "America/Los_Angeles": "PST",
        "Australia/Sydney": "AEDT",
        "Australia/Melbourne": "AEDT",
    }

    return timezone_abbrs.get(timezone_str, timezone_str)


def get_kst_timestamp(dt: datetime | None = None) -> str:
    """
    Get current or provided datetime formatted as KST timestamp.
    """
    tz_name = get_configured_timezone()
    tz_abbr = get_timezone_abbr(tz_name)

    if dt is None:
        if tz_name.lower() == "utc":
            dt = datetime.now(UTC)
        else:
            try:
                import zoneinfo

                tz = zoneinfo.ZoneInfo(tz_name)
                dt = datetime.now(tz)
            except (ImportError, Exception):
                kst_offset = timezone(timedelta(hours=9))
                dt = datetime.now(kst_offset)
    else:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

    return dt.strftime(f"%Y-%m-%d %H:%M ({tz_abbr})")


def parse_timestamp(timestamp_str: str) -> datetime | None:
    """Parse timestamp string back to datetime object."""
    if not timestamp_str:
        return None

    try:
        if "(" in timestamp_str and ")" in timestamp_str:
            dt_part = timestamp_str.split("(")[0].strip()
            timestamp_str = dt_part
    except Exception:
        pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    return None


def get_age_in_days(timestamp_str: str) -> int | None:
    """Get age of artifact in days based on timestamp."""
    dt = parse_timestamp(timestamp_str)
    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    age = now - dt
    return age.days


def format_timestamp_for_filename(dt: datetime | None = None) -> str:
    """Get timestamp formatted for artifact filename."""
    if dt is None:
        tz_name = get_configured_timezone()
        if tz_name.lower() == "utc":
            dt = datetime.now(UTC)
        else:
            try:
                import zoneinfo

                tz = zoneinfo.ZoneInfo(tz_name)
                dt = datetime.now(tz)
            except (ImportError, Exception):
                kst_offset = timezone(timedelta(hours=9))
                dt = datetime.now(kst_offset)

    return dt.strftime("%Y-%m-%d_%H%M")


def infer_artifact_filename_timestamp(file_path: str | Path) -> str:
    """
    Infer artifact timestamp for filename using intelligent fallback.
    """
    import subprocess
    from pathlib import Path

    file_path = Path(file_path)
    # Project root is 4 levels up from this file
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # Get configured timezone
    tz_name = get_configured_timezone()
    kst = timezone(timedelta(hours=9))

    def get_dt_filename(dt: datetime) -> str:
        # Ensure aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        # Convert to configured timezone
        if tz_name.lower() != "utc":
            try:
                import zoneinfo

                tz = zoneinfo.ZoneInfo(tz_name)
                dt = dt.astimezone(tz)
            except (ImportError, Exception):
                dt = dt.astimezone(kst)

        return dt.strftime("%Y-%m-%d_%H%M")

    try:
        # 1. Try frontmatter date
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            if content.startswith("---"):
                for line in content.splitlines():
                    if line.strip().startswith("date:"):
                        date_str = line.split(":", 1)[1].strip().strip("\"'")
                        fm_date = None
                        try:
                            # Try standard format
                            fm_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M (KST)")
                        except ValueError:
                            try:
                                if "(KST)" in date_str:
                                    fm_date = datetime.strptime(date_str.replace("(KST)", "").strip(), "%Y-%m-%d")
                                    fm_date = fm_date.replace(hour=12, minute=0)
                                else:
                                    fm_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                                    fm_date = fm_date.replace(hour=12, minute=0)
                            except ValueError:
                                pass

                        if fm_date:
                            return get_dt_filename(fm_date)
                        break
    except Exception:
        pass

    try:
        # 2. Try git creation date
        result = subprocess.run(
            ["git", "log", "--follow", "--format=%aI", "--diff-filter=A", "--", str(file_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            git_date_str = result.stdout.strip().split("\n")[-1]
            git_date = datetime.fromisoformat(git_date_str.replace("Z", "+00:00"))
            return get_dt_filename(git_date)
    except Exception:
        pass

    try:
        # 3. Try filesystem modification time
        mtime = file_path.stat().st_mtime
        fs_date = datetime.fromtimestamp(mtime, tz=UTC)
        return get_dt_filename(fs_date)
    except Exception:
        pass

    # 4. Fallback to current time
    return format_timestamp_for_filename()


def infer_artifact_date(file_path: str | Path) -> str:
    """Infer artifact date for frontmatter."""
    timestamp = infer_artifact_filename_timestamp(file_path)
    # Convert YYYY-MM-DD_HHMM to YYYY-MM-DD HH:MM (KST)
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d_%H%M")
        return dt.strftime("%Y-%m-%d %H:%M (KST)")
    except Exception:
        return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M (KST)")
