"""
ComplexityMetricsAnalyzer - Calculate code complexity metrics.

Metrics:
- Cyclomatic complexity (McCabe)
- Nesting depth
- Lines of code (logical)
- Parameter count
- Return statement count

Outputs:
- Per-function metrics
- File-level aggregates
- Threshold violations
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseAnalyzer, AnalysisReport


@dataclass
class FunctionMetrics:
    """Complexity metrics for a single function."""

    name: str
    qualified_name: str
    file: str
    line: int
    cyclomatic_complexity: int = 1
    max_nesting_depth: int = 0
    lines_of_code: int = 0
    param_count: int = 0
    return_count: int = 0
    is_async: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file": self.file,
            "line": self.line,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "max_nesting_depth": self.max_nesting_depth,
            "lines_of_code": self.lines_of_code,
            "param_count": self.param_count,
            "return_count": self.return_count,
            "is_async": self.is_async,
        }


class ComplexityMetricsAnalyzer(BaseAnalyzer):
    """
    Analyze Python code for complexity metrics.

    Calculates:
    - Cyclomatic complexity (decision points + 1)
    - Maximum nesting depth
    - Logical lines of code
    - Parameter count
    - Return statement count

    Use cases:
    - Identify overly complex functions
    - Find refactoring candidates
    - Code review prioritization
    """

    name = "ComplexityMetricsAnalyzer"

    # Decision nodes that increase cyclomatic complexity
    DECISION_NODES = (
        ast.If,
        ast.For,
        ast.While,
        ast.ExceptHandler,
        ast.With,
        ast.Assert,
        ast.comprehension,
    )

    # Boolean operators that add complexity
    BOOL_OPS = (ast.And, ast.Or)

    def __init__(
        self,
        complexity_threshold: int = 10,
        nesting_threshold: int = 4,
        param_threshold: int = 5,
    ):
        """
        Initialize the analyzer with thresholds.

        Args:
            complexity_threshold: Flag functions with complexity >= this
            nesting_threshold: Flag functions with nesting >= this
            param_threshold: Flag functions with params >= this
        """
        super().__init__()
        self.complexity_threshold = complexity_threshold
        self.nesting_threshold = nesting_threshold
        self.param_threshold = param_threshold

        self._function_metrics: list[FunctionMetrics] = []
        self._current_class: str = ""
        self._scope_stack: list[str] = []

    def analyze_file(self, file_path: str | Path) -> AnalysisReport:
        """Analyze a single file for complexity metrics."""
        self._function_metrics = []
        self._current_class = ""
        self._scope_stack = []
        return super().analyze_file(file_path)

    def visit(self, node: ast.AST) -> None:
        """Visit AST nodes to calculate metrics."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        visitor(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class context for qualified names."""
        old_class = self._current_class
        self._current_class = node.name
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()
        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Analyze function complexity."""
        self._analyze_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Analyze async function complexity."""
        self._analyze_function(node, is_async=True)

    def _analyze_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> None:
        """Calculate metrics for a function."""
        # Build qualified name
        if self._scope_stack:
            qualified_name = ".".join(self._scope_stack) + "." + node.name
        else:
            qualified_name = node.name

        # Calculate cyclomatic complexity
        complexity = self._calculate_cyclomatic_complexity(node)

        # Calculate max nesting depth
        max_depth = self._calculate_max_nesting(node)

        # Calculate lines of code (end - start + 1)
        loc = node.end_lineno - node.lineno + 1 if node.end_lineno else 1

        # Count parameters
        args = node.args
        param_count = (
            len(args.args)
            + len(args.posonlyargs)
            + len(args.kwonlyargs)
            + (1 if args.vararg else 0)
            + (1 if args.kwarg else 0)
        )
        # Don't count 'self' or 'cls'
        if args.args and args.args[0].arg in ("self", "cls"):
            param_count -= 1

        # Count return statements
        return_count = self._count_returns(node)

        metrics = FunctionMetrics(
            name=node.name,
            qualified_name=qualified_name,
            file=self._current_file,
            line=node.lineno,
            cyclomatic_complexity=complexity,
            max_nesting_depth=max_depth,
            lines_of_code=loc,
            param_count=param_count,
            return_count=return_count,
            is_async=is_async,
        )
        self._function_metrics.append(metrics)

        # Add results for threshold violations
        violations = []
        if complexity >= self.complexity_threshold:
            violations.append(f"complexity={complexity}")
        if max_depth >= self.nesting_threshold:
            violations.append(f"nesting={max_depth}")
        if param_count >= self.param_threshold:
            violations.append(f"params={param_count}")

        if violations:
            self._add_result(
                node,
                f"{qualified_name}: {', '.join(violations)}",
                context=f"LOC={loc}, returns={return_count}",
                category="high_complexity",
                metadata=metrics.to_dict(),
            )
        else:
            self._add_result(
                node,
                f"{qualified_name}: complexity={complexity}, nesting={max_depth}",
                context=f"LOC={loc}, params={param_count}",
                category="normal",
                metadata=metrics.to_dict(),
            )

        # Continue visiting nested functions
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Calculate cyclomatic complexity (McCabe).

        Complexity = 1 + number of decision points
        Decision points: if, for, while, except, with, assert, and, or
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision nodes
            if isinstance(child, self.DECISION_NODES):
                complexity += 1
            # Boolean operators (each adds a decision point)
            elif isinstance(child, ast.BoolOp):
                # and/or with N values has N-1 operators
                complexity += len(child.values) - 1
            # Ternary expressions (x if cond else y)
            elif isinstance(child, ast.IfExp):
                complexity += 1

        return complexity

    def _calculate_max_nesting(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        return self._get_nesting_depth(node, 0)

    def _get_nesting_depth(self, node: ast.AST, current_depth: int) -> int:
        """Recursively calculate nesting depth."""
        max_depth = current_depth

        # Nodes that increase nesting
        nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try)

        for child in ast.iter_child_nodes(node):
            if isinstance(child, nesting_nodes):
                child_depth = self._get_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Don't count nested function/class definitions
                continue
            else:
                child_depth = self._get_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def _count_returns(self, node: ast.AST) -> int:
        """Count return statements in a function."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                count += 1
            # Don't count returns in nested functions
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child != node:
                continue
        return count

    def get_metrics(self) -> list[FunctionMetrics]:
        """Get all function metrics."""
        return self._function_metrics

    def get_high_complexity_functions(self) -> list[FunctionMetrics]:
        """Get functions exceeding complexity threshold."""
        return [
            m for m in self._function_metrics
            if m.cyclomatic_complexity >= self.complexity_threshold
        ]

    def get_deeply_nested_functions(self) -> list[FunctionMetrics]:
        """Get functions exceeding nesting threshold."""
        return [
            m for m in self._function_metrics
            if m.max_nesting_depth >= self.nesting_threshold
        ]

    def get_sorted_by_complexity(self, limit: int = 10) -> list[FunctionMetrics]:
        """Get top N functions by complexity."""
        return sorted(
            self._function_metrics,
            key=lambda m: m.cyclomatic_complexity,
            reverse=True,
        )[:limit]

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary with aggregate metrics."""
        if not self._function_metrics:
            return {
                "total_functions": 0,
                "average_complexity": 0,
                "max_complexity": 0,
                "high_complexity_count": 0,
                "deep_nesting_count": 0,
            }

        complexities = [m.cyclomatic_complexity for m in self._function_metrics]
        high_complexity = self.get_high_complexity_functions()
        deep_nesting = self.get_deeply_nested_functions()

        return {
            "total_functions": len(self._function_metrics),
            "average_complexity": sum(complexities) / len(complexities),
            "max_complexity": max(complexities),
            "min_complexity": min(complexities),
            "high_complexity_count": len(high_complexity),
            "deep_nesting_count": len(deep_nesting),
            "thresholds": {
                "complexity": self.complexity_threshold,
                "nesting": self.nesting_threshold,
                "params": self.param_threshold,
            },
            "top_complex_functions": [
                {"name": m.qualified_name, "complexity": m.cyclomatic_complexity, "line": m.line}
                for m in self.get_sorted_by_complexity(5)
            ],
        }
