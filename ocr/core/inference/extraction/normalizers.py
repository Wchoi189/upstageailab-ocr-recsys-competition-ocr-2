"""Normalizer functions for receipt field extraction.

This module provides utility functions to parse and normalize
common receipt field formats (currency, dates, phone numbers).
"""

from __future__ import annotations

import re
from datetime import date, time
from decimal import Decimal, InvalidOperation

# Currency patterns for Korean Won
CURRENCY_PATTERNS = [
    # Standard formats: 12,500원, ₩12,500, 12500원
    re.compile(r"₩?\s*([0-9,]+)\s*원?"),
    # With decimal: 12,500.00
    re.compile(r"₩?\s*([0-9,]+(?:\.[0-9]+)?)\s*원?"),
]

# Date patterns (Korean format variations)
DATE_PATTERNS = [
    # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    (re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"), "%Y-%m-%d"),
    # DD-MM-YYYY, DD/MM/YYYY
    (re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})"), "%d-%m-%Y"),
    # YY-MM-DD (2-digit year)
    (re.compile(r"(\d{2})[-/.](\d{1,2})[-/.](\d{1,2})"), "%y-%m-%d"),
    # YYYYMMDD (no separator)
    (re.compile(r"(\d{4})(\d{2})(\d{2})"), "%Y%m%d"),
]

# Time patterns
TIME_PATTERNS = [
    # HH:MM:SS, HH:MM
    re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?"),
    # HH시 MM분
    re.compile(r"(\d{1,2})시\s*(\d{2})분"),
]

# Phone patterns (Korean format)
PHONE_PATTERNS = [
    # 02-1234-5678, 010-1234-5678
    re.compile(r"(\d{2,3})[-)\s]?(\d{3,4})[-\s]?(\d{4})"),
    # 1588-1234 (special numbers)
    re.compile(r"(1\d{3})[-\s]?(\d{4})"),
]


def normalize_currency(text: str) -> Decimal | None:
    """Parse Korean currency to Decimal.

    Handles various currency formats:
    - "12,500원" → Decimal("12500")
    - "₩ 12,500" → Decimal("12500")
    - "12500" → Decimal("12500")
    - "12,500.00원" → Decimal("12500.00")

    Args:
        text: Raw text containing currency value

    Returns:
        Decimal value or None if parsing fails
    """
    if not text:
        return None

    text = text.strip()

    for pattern in CURRENCY_PATTERNS:
        match = pattern.search(text)
        if match:
            # Extract numeric part and remove commas
            value_str = match.group(1).replace(",", "")
            try:
                return Decimal(value_str)
            except InvalidOperation:
                continue

    return None


def normalize_date(text: str) -> date | None:
    """Parse various date formats to date object.

    Handles various date formats:
    - "2024-12-25" → date(2024, 12, 25)
    - "2024/12/25" → date(2024, 12, 25)
    - "2024.12.25" → date(2024, 12, 25)
    - "24-12-25" → date(2024, 12, 25)
    - "20241225" → date(2024, 12, 25)

    Args:
        text: Raw text containing date

    Returns:
        date object or None if parsing fails
    """
    if not text:
        return None

    text = text.strip()

    for pattern, _ in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()

            try:
                if len(groups) == 3:
                    # Determine year, month, day order
                    a, b, c = int(groups[0]), int(groups[1]), int(groups[2])

                    # If first number looks like a year (4 digits or >= 24)
                    if len(groups[0]) == 4 or a > 31:
                        year, month, day = a, b, c
                    # If last number looks like a year (4 digits)
                    elif len(groups[2]) == 4:
                        day, month, year = a, b, c
                    # 2-digit year
                    elif a < 100:
                        # Assume YY-MM-DD
                        year = 2000 + a if a < 50 else 1900 + a
                        month, day = b, c
                    else:
                        continue

                    # Validate ranges
                    if not (1 <= month <= 12 and 1 <= day <= 31):
                        continue

                    return date(year, month, day)

            except (ValueError, TypeError):
                continue

    return None


def normalize_time(text: str) -> time | None:
    """Parse time string to time object.

    Handles various time formats:
    - "14:30" → time(14, 30)
    - "14:30:00" → time(14, 30, 0)
    - "14시 30분" → time(14, 30)
    - "2:30 PM" → time(14, 30) (basic AM/PM)

    Args:
        text: Raw text containing time

    Returns:
        time object or None if parsing fails
    """
    if not text:
        return None

    text = text.strip()

    # Check for AM/PM
    is_pm = "pm" in text.lower() or "오후" in text
    is_am = "am" in text.lower() or "오전" in text

    for pattern in TIME_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()

            try:
                hour = int(groups[0])
                minute = int(groups[1])
                second = int(groups[2]) if len(groups) > 2 and groups[2] else 0

                # Handle 12-hour format
                if is_pm and hour < 12:
                    hour += 12
                elif is_am and hour == 12:
                    hour = 0

                # Validate ranges
                if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                    continue

                return time(hour, minute, second)

            except (ValueError, TypeError):
                continue

    return None


def normalize_phone(text: str) -> str | None:
    """Normalize Korean phone number format.

    Standardizes phone numbers to dashed format:
    - "010 1234 5678" → "010-1234-5678"
    - "02)1234-5678" → "02-1234-5678"
    - "1588 1234" → "1588-1234"

    Args:
        text: Raw text containing phone number

    Returns:
        Normalized phone number or None if parsing fails
    """
    if not text:
        return None

    text = text.strip()

    for pattern in PHONE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()

            # Format as dashed number
            if len(groups) == 3:
                return f"{groups[0]}-{groups[1]}-{groups[2]}"
            elif len(groups) == 2:
                return f"{groups[0]}-{groups[1]}"

    return None


def normalize_business_number(text: str) -> str | None:
    """Normalize Korean business registration number (사업자번호).

    Standard format: XXX-XX-XXXXX

    Args:
        text: Raw text containing business number

    Returns:
        Normalized business number or None if parsing fails
    """
    if not text:
        return None

    # Remove all non-digit characters
    digits = re.sub(r"\D", "", text)

    # Korean business number is 10 digits
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"

    return None


__all__ = [
    "normalize_currency",
    "normalize_date",
    "normalize_time",
    "normalize_phone",
    "normalize_business_number",
]
