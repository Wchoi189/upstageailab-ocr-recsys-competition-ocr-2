"""Markdown reporter for analysis results."""

from __future__ import annotations

from typing import Any

from agent_debug_toolkit.analyzers.base import AnalysisReport, AnalysisResult


class MarkdownReporter:
    """
    Generate Markdown output from analysis reports.

    Produces formatted, readable reports suitable for documentation
    or AI agent consumption.
    """

    def __init__(self, include_code_snippets: bool = True, max_snippet_lines: int = 5):
        """
        Initialize the reporter.

        Args:
            include_code_snippets: Whether to include source code snippets
            max_snippet_lines: Maximum lines per code snippet
        """
        self.include_code_snippets = include_code_snippets
        self.max_snippet_lines = max_snippet_lines

    def format(self, report: AnalysisReport) -> str:
        """Format a single report as Markdown."""
        lines = [
            f"# {report.analyzer_name}",
            "",
            f"**Target**: `{report.target_path}`  ",
            f"**Findings**: {len(report.results)}",
            "",
        ]

        # Summary section
        if report.summary:
            lines.append("## Summary")
            lines.append("")
            self._format_summary(report.summary, lines)
            lines.append("")

        # Findings section
        if report.results:
            lines.append("## Findings")
            lines.append("")

            # Group by category
            by_category: dict[str, list[AnalysisResult]] = {}
            for r in report.results:
                by_category.setdefault(r.category, []).append(r)

            for category, results in by_category.items():
                lines.append(f"### {self._format_category(category)} ({len(results)})")
                lines.append("")

                for result in results:
                    self._format_result(result, lines)

                lines.append("")
        else:
            lines.append("*No findings*")
            lines.append("")

        return "\n".join(lines)

    def format_multiple(self, reports: list[AnalysisReport], title: str = "Analysis Report") -> str:
        """Format multiple reports into a single Markdown document."""
        lines = [
            f"# {title}",
            "",
            f"**Reports**: {len(reports)}  ",
            f"**Total Findings**: {sum(len(r.results) for r in reports)}",
            "",
            "---",
            "",
        ]

        for report in reports:
            lines.append(self.format(report))
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _format_summary(self, summary: dict[str, Any], lines: list[str]) -> None:
        """Format the summary section."""
        for key, value in summary.items():
            if isinstance(value, dict):
                lines.append(f"**{self._format_key(key)}**:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"- **{self._format_key(key)}**: {value}")

    def _format_result(self, result: AnalysisResult, lines: list[str]) -> None:
        """Format a single analysis result."""
        # Main line with location and pattern
        lines.append(f"- **Line {result.line}** in `{result.file.split('/')[-1]}`")
        lines.append(f"  - Pattern: `{result.pattern}`")

        if result.context:
            lines.append(f"  - Context: {result.context}")

        # Code snippet
        if self.include_code_snippets and result.code_snippet:
            snippet_lines = result.code_snippet.split("\n")[:self.max_snippet_lines]
            if snippet_lines:
                lines.append("  ```python")
                for line in snippet_lines:
                    lines.append(f"  {line}")
                lines.append("  ```")

        lines.append("")

    def _format_category(self, category: str) -> str:
        """Format a category name for display."""
        return category.replace("_", " ").title()

    def _format_key(self, key: str) -> str:
        """Format a key name for display."""
        return key.replace("_", " ").title()

    def save(self, report: AnalysisReport, path: str) -> None:
        """Save a report to a Markdown file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.format(report))
