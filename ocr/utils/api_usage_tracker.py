"""Upstage API usage tracking for free tier management.

Tracks API calls to Document Parse and Solar APIs to monitor usage against
free tier limits (valid until March 2026).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_USAGE_FILE = Path("data/ops/upstage_api_usage.json")


@dataclass
class APIUsageRecord:
    """Single API call record."""

    timestamp: str
    api_type: str  # "document_parse" or "solar"
    status: str  # "success", "error", "rate_limited"
    response_time_ms: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class APIUsageStats:
    """Aggregated API usage statistics."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rate_limited_calls: int = 0
    total_response_time_ms: float = 0.0
    first_call: str | None = None
    last_call: str | None = None
    calls_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def avg_response_time_ms(self) -> float:
        """Calculate average response time."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_response_time_ms / self.successful_calls


class UpstageAPITracker:
    """Track and persist Upstage API usage."""

    def __init__(self, usage_file: Path | None = None):
        """Initialize tracker.

        Args:
            usage_file: Path to JSON file for storing usage records
        """
        self.usage_file = usage_file or DEFAULT_USAGE_FILE
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[APIUsageRecord] = []
        self._load_records()

    def _load_records(self):
        """Load existing usage records from file."""
        if not self.usage_file.exists():
            logger.info(f"No existing usage file found at {self.usage_file}")
            return

        try:
            with open(self.usage_file) as f:
                data = json.load(f)
                self.records = [APIUsageRecord(**record) for record in data.get("records", [])]
            logger.info(f"Loaded {len(self.records)} usage records from {self.usage_file}")
        except Exception as e:
            logger.error(f"Failed to load usage records: {e}")
            self.records = []

    def _save_records(self):
        """Persist usage records to file."""
        try:
            data = {
                "records": [asdict(record) for record in self.records],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.usage_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.records)} usage records to {self.usage_file}")
        except Exception as e:
            logger.error(f"Failed to save usage records: {e}")

    def record_call(
        self,
        api_type: str,
        status: str,
        response_time_ms: float | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Record a single API call.

        Args:
            api_type: Type of API ("document_parse" or "solar")
            status: Call status ("success", "error", "rate_limited")
            response_time_ms: Response time in milliseconds
            error_message: Error message if status is "error"
            metadata: Additional metadata (e.g., image filename, word count)
        """
        record = APIUsageRecord(
            timestamp=datetime.now().isoformat(),
            api_type=api_type,
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata or {},
        )
        self.records.append(record)
        self._save_records()

    def get_stats(self, since: datetime | None = None) -> APIUsageStats:
        """Get aggregated usage statistics.

        Args:
            since: Optional datetime to filter records from

        Returns:
            APIUsageStats with aggregated metrics
        """
        records = self.records
        if since:
            records = [r for r in records if datetime.fromisoformat(r.timestamp) >= since]

        if not records:
            return APIUsageStats()

        stats = APIUsageStats(
            total_calls=len(records),
            successful_calls=sum(1 for r in records if r.status == "success"),
            failed_calls=sum(1 for r in records if r.status == "error"),
            rate_limited_calls=sum(1 for r in records if r.status == "rate_limited"),
            first_call=records[0].timestamp,
            last_call=records[-1].timestamp,
        )

        # Calculate total response time and calls by type
        for record in records:
            if record.response_time_ms:
                stats.total_response_time_ms += record.response_time_ms
            stats.calls_by_type[record.api_type] = stats.calls_by_type.get(record.api_type, 0) + 1

        return stats

    def print_report(self, since: datetime | None = None):
        """Print usage report to console.

        Args:
            since: Optional datetime to filter records from
        """
        stats = self.get_stats(since)

        print("\n=== Upstage API Usage Report ===")
        print(f"Total Calls: {stats.total_calls}")
        print(f"  ✓ Successful: {stats.successful_calls} ({stats.success_rate:.1f}%)")
        print(f"  ✗ Failed: {stats.failed_calls}")
        print(f"  ⏸ Rate Limited: {stats.rate_limited_calls}")
        print(f"\nAverage Response Time: {stats.avg_response_time_ms:.1f} ms")
        print(f"\nCalls by Type:")
        for api_type, count in stats.calls_by_type.items():
            print(f"  - {api_type}: {count}")
        if stats.first_call and stats.last_call:
            print(f"\nFirst Call: {stats.first_call}")
            print(f"Last Call: {stats.last_call}")
        print("=" * 35)


# Global tracker instance
_tracker: UpstageAPITracker | None = None


def get_tracker() -> UpstageAPITracker:
    """Get or create global API tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = UpstageAPITracker()
    return _tracker


__all__ = [
    "APIUsageRecord",
    "APIUsageStats",
    "UpstageAPITracker",
    "get_tracker",
]
