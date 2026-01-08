"""
Intelligent symbol search for Python codebases.

Provides qualified path resolution, reverse lookup, fuzzy matching,
and cross-reference detection for classes and functions.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_debug_toolkit.analyzers.base import AnalysisReport, BaseAnalyzer
from agent_debug_toolkit.precomputes.symbol_table import SymbolTable, SymbolDefinition


@dataclass
class SearchResult:
    """
    Result from symbol search operation.

    Attributes:
        name: Simple name (e.g., "TimmBackbone")
        full_name: Fully qualified path (e.g., "ocr.core.models.encoder.TimmBackbone")
        file_path: Absolute path to definition file
        line_number: Line where symbol is defined
        kind: "class" or "function"
        import_paths: Alternative import paths to this symbol
        usage_sites: Files that import/use this symbol
        confidence: Match confidence for fuzzy searches (0.0-1.0)
    """
    name: str
    full_name: str
    file_path: str
    line_number: int
    kind: str
    import_paths: list[str] = field(default_factory=list)
    usage_sites: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "kind": self.kind,
            "import_paths": self.import_paths,
            "usage_sites": self.usage_sites,
            "confidence": self.confidence,
        }


class IntelligentSearcher(BaseAnalyzer):
    """
    Search for symbols using qualified paths, class names, and fuzzy matching.

    Leverages SymbolTable for indexing and provides multiple search modes:
    - Qualified path resolution (ocr.core.models.X → file location)
    - Reverse lookup (class name → all import paths)
    - Fuzzy matching (typo correction)
    - Cross-reference detection (usage sites)
    """

    name = "IntelligentSearcher"

    def __init__(self, root_path: str | Path, module_root: str | Path):
        """
        Initialize searcher with symbol table.

        Args:
            root_path: Directory to search within
            module_root: Python package root for module name resolution
        """
        super().__init__()
        self.root_path = Path(root_path).resolve()
        self.module_root = Path(module_root).resolve()

        # Build symbol table on initialization
        self.symbol_table = SymbolTable(str(self.root_path), str(self.module_root))
        self.symbol_table.build()

    def search(
        self,
        query: str,
        fuzzy: bool = True,
        threshold: float = 0.6
    ) -> list[SearchResult]:
        """
        Main search entry point.

        Tries multiple search strategies:
        1. Exact qualified path match
        2. Exact simple name match across all symbols
        3. Fuzzy matching if enabled

        Args:
            query: Symbol name or qualified path
            fuzzy: Enable fuzzy matching
            threshold: Minimum similarity for fuzzy matches (0.0-1.0)

        Returns:
            List of SearchResult objects, sorted by confidence
        """
        results: list[SearchResult] = []

        # Strategy 1: Exact qualified path match
        exact_match = self.resolve_qualified_path(query)
        if exact_match:
            results.append(exact_match)
            return results

        # Strategy 2: Reverse lookup by simple name
        reverse_results = self.reverse_lookup(query)
        if reverse_results:
            results.extend(reverse_results)

        # Strategy 3: Fuzzy matching if no exact matches and fuzzy enabled
        if not results and fuzzy:
            fuzzy_results = self.fuzzy_search(query, threshold)
            results.extend(fuzzy_results)

        # Sort by confidence (descending)
        results.sort(key=lambda r: r.confidence, reverse=True)

        return results

    def resolve_qualified_path(self, path: str) -> SearchResult | None:
        """
        Resolve a fully qualified path to its definition.

        Args:
            path: Qualified path (e.g., "ocr.core.models.encoder.TimmBackbone")

        Returns:
            SearchResult if found, None otherwise
        """
        symbol = self.symbol_table.lookup(path)
        if not symbol:
            return None

        return self._symbol_to_result(symbol, confidence=1.0)

    def reverse_lookup(self, name: str) -> list[SearchResult]:
        """
        Find all symbols matching a simple name.

        Args:
            name: Simple class/function name (e.g., "TimmBackbone")

        Returns:
            List of all symbols with matching simple names
        """
        results: list[SearchResult] = []

        # Iterate through all symbols in the table
        for full_name, symbol in self.symbol_table._symbols.items():
            if symbol.name == name:
                result = self._symbol_to_result(symbol, confidence=1.0)

                # Add alternative import paths
                result.import_paths = self._find_alternative_imports(symbol)

                results.append(result)

        return results

    def fuzzy_search(self, query: str, threshold: float = 0.6) -> list[SearchResult]:
        """
        Find symbols with similar names using fuzzy matching.

        Args:
            query: Search query (possibly with typos)
            threshold: Minimum similarity ratio (0.0-1.0)

        Returns:
            List of SearchResult with confidence scores
        """
        results: list[SearchResult] = []

        for full_name, symbol in self.symbol_table._symbols.items():
            # Compare with both simple name and full name
            simple_ratio = difflib.SequenceMatcher(None, query.lower(), symbol.name.lower()).ratio()
            full_ratio = difflib.SequenceMatcher(None, query.lower(), full_name.lower()).ratio()

            # Use the better match
            confidence = max(simple_ratio, full_ratio)

            if confidence >= threshold:
                result = self._symbol_to_result(symbol, confidence=confidence)
                results.append(result)

        return results

    def find_usage_sites(self, symbol_name: str) -> list[str]:
        """
        Find files that import or use a symbol (placeholder).

        Note: Full implementation would require grepping for imports.
        For now, returns empty list as usage detection is complex.

        Args:
            symbol_name: Symbol to search for

        Returns:
            List of file paths that use this symbol
        """
        # TODO: Implement using grep_search or import analysis
        # For now, return empty list
        return []

    def _symbol_to_result(self, symbol: SymbolDefinition, confidence: float = 1.0) -> SearchResult:
        """Convert SymbolDefinition to SearchResult."""
        return SearchResult(
            name=symbol.name,
            full_name=symbol.full_name,
            file_path=symbol.file_path,
            line_number=symbol.line_number,
            kind=symbol.kind,
            confidence=confidence,
        )

    def _find_alternative_imports(self, symbol: SymbolDefinition) -> list[str]:
        """
        Find alternative import paths for a symbol.

        For example, if symbol is at ocr.core.models.encoder.backbone.TimmBackbone,
        it might also be importable as ocr.core.models.encoder.TimmBackbone if
        __init__.py exports it.

        Args:
            symbol: Symbol to find alternatives for

        Returns:
            List of alternative qualified paths
        """
        alternatives = [symbol.full_name]

        # Check if symbol is exported in parent __init__.py
        # This is a simplified heuristic - could be enhanced
        parts = symbol.full_name.split(".")

        # Try removing intermediate packages (e.g., backbone module)
        if len(parts) > 2:
            # Try parent package export
            parent_path = ".".join(parts[:-2] + [parts[-1]])
            if parent_path != symbol.full_name:
                alternatives.append(parent_path)

        return alternatives

    # BaseAnalyzer abstract method implementations
    def visit(self, node: Any) -> None:
        """Not used for this analyzer (uses SymbolTable instead)."""
        pass

    def analyze_file(self, file_path: str | Path) -> AnalysisReport:
        """
        Not the primary interface for this analyzer.
        Use search() method instead.
        """
        return AnalysisReport(
            analyzer_name=self.name,
            target_path=str(file_path),
            summary={"note": "Use IntelligentSearcher.search() method instead"},
        )


def format_search_results_markdown(results: list[SearchResult], query: str) -> str:
    """
    Format search results as markdown.

    Args:
        results: List of search results
        query: Original search query

    Returns:
        Formatted markdown string
    """
    if not results:
        return f"# Search Results: '{query}'\n\nNo results found."

    lines = [
        f"# Search Results: '{query}'",
        "",
        f"**Found**: {len(results)} result(s)",
        "",
    ]

    for i, result in enumerate(results, 1):
        lines.append(f"## {i}. {result.name}")
        lines.append("")
        lines.append(f"- **Full Name**: `{result.full_name}`")
        lines.append(f"- **Type**: {result.kind}")
        lines.append(f"- **Location**: [{result.file_path}:{result.line_number}](file://{result.file_path}#L{result.line_number})")

        if result.confidence < 1.0:
            lines.append(f"- **Confidence**: {result.confidence:.1%} (fuzzy match)")

        if result.import_paths and len(result.import_paths) > 1:
            lines.append("- **Import Paths**:")
            for path in result.import_paths:
                lines.append(f"  - `{path}`")

        if result.usage_sites:
            lines.append("- **Used In**:")
            for site in result.usage_sites[:5]:  # Limit to 5
                lines.append(f"  - `{site}`")

        lines.append("")

    return "\n".join(lines)
