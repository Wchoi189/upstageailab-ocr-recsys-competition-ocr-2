#!/usr/bin/env python3
"""
Timestamp Utilities for AgentQMS

Provides timezone-aware timestamp utilities for artifact metadata.

Supports timezone configuration from environment, settings, or fallback to KST.

Usage:
    from AgentQMS.agent_tools.utils.timestamps import get_kst_timestamp

    timestamp = get_kst_timestamp()  # Returns "2025-12-06 12:00 (KST)"
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta, timezone

from AgentQMS.agent_tools.utils.config import load_config


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

    Common mappings:
    - Asia/Seoul → KST
    - Asia/Tokyo → JST
    - UTC → UTC
    - America/New_York → EST/EDT
    - Europe/London → GMT/BST

    Args:
        timezone_str: IANA timezone name

    Returns:
        Timezone abbreviation or original string if not found
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

    Format: "YYYY-MM-DD HH:MM (KST)"

    Args:
        dt: Optional datetime object (defaults to now)

    Returns:
        Formatted timestamp string
    """
    tz_name = get_configured_timezone()
    tz_abbr = get_timezone_abbr(tz_name)

    if dt is None:
        # Use current time in configured timezone
        if tz_name.lower() == "utc":
            dt = datetime.now(UTC)
        else:
            # Try to create timezone from IANA name
            try:
                import zoneinfo
                tz = zoneinfo.ZoneInfo(tz_name)
                dt = datetime.now(tz)
            except (ImportError, Exception):
                # Fallback: use KST offset (UTC+9)
                kst_offset = timezone(timedelta(hours=9))
                dt = datetime.now(kst_offset)
    else:
        # Ensure datetime has timezone info
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = dt.replace(tzinfo=UTC)

    return dt.strftime(f"%Y-%m-%d %H:%M ({tz_abbr})")


def parse_timestamp(timestamp_str: str) -> datetime | None:
    """
    Parse timestamp string back to datetime object.

    Supports formats:
    - "2025-12-06 12:00 (KST)"
    - "2025-12-06 12:00"
    - "2025-12-06T12:00:00"
    - "2025-12-06"

    Args:
        timestamp_str: Timestamp string to parse

    Returns:
        datetime object or None if parsing fails
    """
    if not timestamp_str:
        return None

    # Try formats with timezone abbr
    try:
        # Format: "2025-12-06 12:00 (KST)"
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
    """
    Get age of artifact in days based on timestamp.

    Args:
        timestamp_str: Timestamp string

    Returns:
        Age in days or None if parsing fails
    """
    dt = parse_timestamp(timestamp_str)
    if dt is None:
        return None

    # Make dt aware if it's naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    age = now - dt
    return age.days


def format_timestamp_for_filename(dt: datetime | None = None) -> str:
    """
    Get timestamp formatted for artifact filename.

    Format: "YYYY-MM-DD_HHMM" (e.g., "2025-12-06_1200")

    Args:
        dt: Optional datetime object (defaults to now)

    Returns:
        Formatted timestamp string for filename
    """
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
