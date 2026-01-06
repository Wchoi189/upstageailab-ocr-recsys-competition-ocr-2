"""
Base analyzer infrastructure for AST-based code analysis.

Provides abstract base class and common data structures used by all analyzers.
"""

from __future__ import annotations

import ast
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class AnalysisResult:
    """
    A single analysis finding from an AST analyzer.

    Attributes:
        file: Absolute path to the analyzed file
        line: Line number (1-indexed) where the pattern was found
        column: Column number (0-indexed) where the pattern starts
        pattern: String representation of the detected pattern
        context: Additional context about the finding
        category: Category/type of the finding for filtering
        code_snippet: The actual source code around the finding
        metadata: Additional analyzer-specific data
    """

    file: str
    line: int
    column: int
    pattern: str
    context: str
    category: str = "general"
    code_snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.file}:{self.line}:{self.column} [{self.category}] {self.pattern}"


@dataclass
class AnalysisReport:
    """
    Collection of analysis results with metadata.

    Attributes:
        analyzer_name: Name of the analyzer that produced these results
        target_path: Path that was analyzed
        results: List of analysis findings
        summary: Summary statistics and insights
    """

    analyzer_name: str
    target_path: str
    results: list[AnalysisResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: AnalysisResult) -> None:
        """Add a result to the report."""
        self.results.append(result)

    def filter_by_category(self, category: str) -> list[AnalysisResult]:
        """Filter results by category."""
        return [r for r in self.results if r.category == category]

    def filter_by_component(self, component: str) -> list[AnalysisResult]:
        """Filter results by component name in pattern."""
        return [r for r in self.results if component.lower() in r.pattern.lower()]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "analyzer_name": self.analyzer_name,
            "target_path": self.target_path,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "total_findings": len(self.results),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# {self.analyzer_name} Report",
            "",
            f"**Target**: `{self.target_path}`",
            f"**Findings**: {len(self.results)}",
            "",
        ]

        if self.summary:
            lines.append("## Summary")
            for key, value in self.summary.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if self.results:
            lines.append("## Findings")
            lines.append("")

            # Group by category
            by_category: dict[str, list[AnalysisResult]] = {}
            for r in self.results:
                by_category.setdefault(r.category, []).append(r)

            for category, items in by_category.items():
                lines.append(f"### {category.title()} ({len(items)})")
                lines.append("")
                for item in items:
                    lines.append(f"- **Line {item.line}**: `{item.pattern}`")
                    if item.context:
                        lines.append(f"  - {item.context}")
                lines.append("")

        return "\n".join(lines)


class BaseAnalyzer(ABC):
    """
    Abstract base class for all AST analyzers.

    Subclasses should implement:
    - visit_* methods for AST node types they care about
    - analyze_file() or override the default implementation

    The base class provides:
    - File parsing with error handling
    - Result collection infrastructure
    - Report generation
    """

    name: str = "BaseAnalyzer"

    def __init__(self):
        self._current_file: str = ""
        self._source_lines: list[str] = []
        self._results: list[AnalysisResult] = []

    def analyze_file(self, file_path: str | Path) -> AnalysisReport:
        """
        Analyze a single Python file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            AnalysisReport with all findings
        """
        path = Path(file_path).resolve()
        self._current_file = str(path)
        self._results = []

        try:
            source = path.read_text(encoding="utf-8")
            self._source_lines = source.splitlines()
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            return AnalysisReport(
                analyzer_name=self.name,
                target_path=str(path),
                summary={"error": f"Syntax error: {e}"},
            )
        except FileNotFoundError:
            return AnalysisReport(
                analyzer_name=self.name,
                target_path=str(path),
                summary={"error": f"File not found: {path}"},
            )

        # Run the visitor
        self.visit(tree)

        report = AnalysisReport(
            analyzer_name=self.name,
            target_path=str(path),
            results=self._results.copy(),
            summary=self._generate_summary(),
        )

        return report

    def analyze_directory(
        self, directory: str | Path, pattern: str = "*.py", recursive: bool = True
    ) -> AnalysisReport:
        """
        Analyze all Python files in a directory.

        Args:
            directory: Directory path to analyze
            pattern: Glob pattern for file matching
            recursive: Whether to search recursively

        Returns:
            Combined AnalysisReport from all files
        """
        dir_path = Path(directory).resolve()
        glob_method = dir_path.rglob if recursive else dir_path.glob

        all_results: list[AnalysisResult] = []
        files_analyzed = 0
        errors = 0

        for py_file in sorted(glob_method(pattern)):
            if py_file.is_file():
                report = self.analyze_file(py_file)
                all_results.extend(report.results)
                files_analyzed += 1
                if "error" in report.summary:
                    errors += 1

        return AnalysisReport(
            analyzer_name=self.name,
            target_path=str(dir_path),
            results=all_results,
            summary={
                "files_analyzed": files_analyzed,
                "total_findings": len(all_results),
                "errors": errors,
            },
        )

    def analyze_source(self, source: str, filename: str = "<string>") -> AnalysisReport:
        """
        Analyze Python source code directly.

        Args:
            source: Python source code as string
            filename: Virtual filename for error messages

        Returns:
            AnalysisReport with findings
        """
        self._current_file = filename
        self._source_lines = source.splitlines()
        self._results = []

        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            return AnalysisReport(
                analyzer_name=self.name,
                target_path=filename,
                summary={"error": f"Syntax error: {e}"},
            )

        self.visit(tree)

        return AnalysisReport(
            analyzer_name=self.name,
            target_path=filename,
            results=self._results.copy(),
            summary=self._generate_summary(),
        )

    @abstractmethod
    def visit(self, node: ast.AST) -> None:
        """
        Visit an AST node. Subclasses must implement this.

        Typically calls self.generic_visit(node) and implements
        specific visit_* methods for nodes of interest.
        """
        pass

    def generic_visit(self, node: ast.AST) -> None:
        """Visit all child nodes."""
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def _add_result(
        self,
        node: ast.AST,
        pattern: str,
        context: str = "",
        category: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an analysis result for an AST node."""
        line = getattr(node, "lineno", 0)
        column = getattr(node, "col_offset", 0)

        # Get code snippet (3 lines of context)
        snippet = self._get_code_snippet(line, context_lines=1)

        result = AnalysisResult(
            file=self._current_file,
            line=line,
            column=column,
            pattern=pattern,
            context=context,
            category=category,
            code_snippet=snippet,
            metadata=metadata or {},
        )
        self._results.append(result)

    def _get_code_snippet(self, line: int, context_lines: int = 1) -> str:
        """Get source code snippet around a line."""
        if not self._source_lines or line < 1:
            return ""

        start = max(0, line - 1 - context_lines)
        end = min(len(self._source_lines), line + context_lines)

        return "\n".join(self._source_lines[start:end])

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics. Subclasses can override."""
        categories: dict[str, int] = {}
        for r in self._results:
            categories[r.category] = categories.get(r.category, 0) + 1

        return {
            "total_findings": len(self._results),
            "by_category": categories,
        }

    def _unparse_safe(self, node: ast.AST) -> str:
        """Safely unparse an AST node to source code."""
        try:
            return ast.unparse(node)
        except Exception:
            return f"<{node.__class__.__name__}>"
