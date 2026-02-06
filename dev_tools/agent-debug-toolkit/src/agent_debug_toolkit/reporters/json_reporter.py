"""JSON reporter for analysis results."""

from __future__ import annotations

import json

from agent_debug_toolkit.analyzers.base import AnalysisReport


class JSONReporter:
    """
    Generate JSON output from analysis reports.

    Supports pretty-printing and compact output.
    """

    def __init__(self, indent: int = 2, compact: bool = False):
        """
        Initialize the reporter.

        Args:
            indent: Indentation level for pretty-printing
            compact: If True, output compact JSON without whitespace
        """
        self.indent = None if compact else indent

    def format(self, report: AnalysisReport) -> str:
        """Format a single report as JSON."""
        return json.dumps(report.to_dict(), indent=self.indent)

    def format_multiple(self, reports: list[AnalysisReport]) -> str:
        """Format multiple reports as a JSON array."""
        data = {
            "reports": [r.to_dict() for r in reports],
            "total_findings": sum(len(r.results) for r in reports),
        }
        return json.dumps(data, indent=self.indent)

    def save(self, report: AnalysisReport, path: str) -> None:
        """Save a report to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.format(report))
