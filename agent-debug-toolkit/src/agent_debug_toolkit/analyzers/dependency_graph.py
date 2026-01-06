"""
DependencyGraphAnalyzer - Build module and class dependency graphs.

Traces:
- Import dependencies (import X, from X import Y)
- Class instantiation dependencies
- Function call dependencies

Outputs:
- Dependency graph with nodes and edges
- Circular dependency detection
- Dependents lookup (what depends on X)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseAnalyzer, AnalysisReport


@dataclass
class DependencyNode:
    """A node in the dependency graph."""

    name: str
    node_type: str  # 'module', 'class', 'function'
    file: str = ""
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.node_type,
            "file": self.file,
            "line": self.line,
        }


@dataclass
class DependencyEdge:
    """An edge in the dependency graph."""

    source: str
    target: str
    edge_type: str  # 'import', 'from_import', 'call', 'instantiate'
    file: str = ""
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type,
            "file": self.file,
            "line": self.line,
        }


class DependencyGraphAnalyzer(BaseAnalyzer):
    """
    Analyze Python code to build dependency graphs.

    Tracks:
    - Module imports (import X, from X import Y)
    - Class/function definitions
    - Call relationships

    Use cases:
    - Find circular dependencies
    - Understand module coupling
    - Navigate codebase structure
    """

    name = "DependencyGraphAnalyzer"

    def __init__(self, include_stdlib: bool = False):
        """
        Initialize the analyzer.

        Args:
            include_stdlib: Whether to include standard library imports
        """
        super().__init__()
        self.include_stdlib = include_stdlib
        self._nodes: dict[str, DependencyNode] = {}
        self._edges: list[DependencyEdge] = []
        self._current_scope: list[str] = []

        # Standard library module prefixes (common ones)
        self._stdlib_prefixes = frozenset({
            "abc", "argparse", "ast", "asyncio", "base64", "collections",
            "concurrent", "contextlib", "copy", "csv", "dataclasses",
            "datetime", "decimal", "enum", "functools", "glob", "hashlib",
            "html", "http", "importlib", "inspect", "io", "itertools",
            "json", "logging", "math", "multiprocessing", "operator", "os",
            "pathlib", "pickle", "platform", "pprint", "queue", "random",
            "re", "shutil", "socket", "sqlite3", "ssl", "statistics",
            "string", "subprocess", "sys", "tempfile", "threading", "time",
            "traceback", "types", "typing", "unittest", "urllib", "uuid",
            "warnings", "weakref", "xml", "zipfile",
        })

    def analyze_file(self, file_path: str | Path) -> AnalysisReport:
        """Analyze a single file and build dependency graph."""
        self._nodes = {}
        self._edges = []
        self._current_scope = []
        return super().analyze_file(file_path)

    def analyze_directory(
        self, directory: str | Path, pattern: str = "*.py", recursive: bool = True
    ) -> AnalysisReport:
        """Analyze directory and build combined dependency graph."""
        self._nodes = {}
        self._edges = []
        self._current_scope = []
        return super().analyze_directory(directory, pattern, recursive)

    def visit(self, node: ast.AST) -> None:
        """Visit AST nodes to build dependency graph."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        visitor(node)

    def visit_Module(self, node: ast.Module) -> None:
        """Track module as root node."""
        module_name = Path(self._current_file).stem if self._current_file else "<module>"
        self._add_node(module_name, "module", 1)
        self._current_scope.append(module_name)
        self.generic_visit(node)
        if self._current_scope:
            self._current_scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class definitions."""
        self._add_node(node.name, "class", node.lineno)

        # Track base class dependencies
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                self._add_edge(node.name, base_name, "inherits", node.lineno)

        self._current_scope.append(node.name)
        self.generic_visit(node)
        self._current_scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function definitions."""
        scope_prefix = ".".join(self._current_scope) + "." if self._current_scope else ""
        func_name = f"{scope_prefix}{node.name}"
        self._add_node(func_name, "function", node.lineno)
        self._current_scope.append(node.name)
        self.generic_visit(node)
        self._current_scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        """Track import statements."""
        for alias in node.names:
            module_name = alias.name
            if self._should_include_import(module_name):
                self._add_node(module_name, "module", node.lineno)
                source = self._current_scope[-1] if self._current_scope else "<module>"
                self._add_edge(source, module_name, "import", node.lineno)
                self._add_result(
                    node,
                    f"import {module_name}",
                    context=f"Imports module: {module_name}",
                    category="import",
                    metadata={"module": module_name, "alias": alias.asname},
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from...import statements."""
        module_name = node.module or ""
        if not self._should_include_import(module_name):
            self.generic_visit(node)
            return

        for alias in node.names:
            imported_name = alias.name
            full_name = f"{module_name}.{imported_name}" if module_name else imported_name

            self._add_node(full_name, "import", node.lineno)
            source = self._current_scope[-1] if self._current_scope else "<module>"
            self._add_edge(source, full_name, "from_import", node.lineno)

            self._add_result(
                node,
                f"from {module_name} import {imported_name}",
                context=f"Imports {imported_name} from {module_name}",
                category="from_import",
                metadata={
                    "module": module_name,
                    "name": imported_name,
                    "alias": alias.asname,
                },
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track function/class instantiation calls."""
        call_name = self._get_name(node.func)
        if call_name:
            source = self._current_scope[-1] if self._current_scope else "<module>"
            self._add_edge(source, call_name, "call", node.lineno)
        self.generic_visit(node)

    def _should_include_import(self, module_name: str) -> bool:
        """Check if import should be included based on settings."""
        if self.include_stdlib:
            return True
        # Check if it's a stdlib module
        root_module = module_name.split(".")[0]
        return root_module not in self._stdlib_prefixes

    def _add_node(self, name: str, node_type: str, line: int) -> None:
        """Add a node to the graph."""
        if name not in self._nodes:
            self._nodes[name] = DependencyNode(
                name=name,
                node_type=node_type,
                file=self._current_file,
                line=line,
            )

    def _add_edge(self, source: str, target: str, edge_type: str, line: int) -> None:
        """Add an edge to the graph."""
        edge = DependencyEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            file=self._current_file,
            line=line,
        )
        self._edges.append(edge)

    def _get_name(self, node: ast.expr) -> str | None:
        """Extract name from an expression node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        return None

    def get_graph(self) -> dict[str, Any]:
        """Get the dependency graph as a dictionary."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def find_cycles(self) -> list[list[str]]:
        """Detect circular dependencies in the graph."""
        # Build adjacency list
        adj: dict[str, set[str]] = {}
        for edge in self._edges:
            adj.setdefault(edge.source, set()).add(edge.target)

        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in self._nodes:
            if node not in visited:
                dfs(node)

        return cycles

    def get_dependents(self, target: str) -> list[str]:
        """Find all nodes that depend on the target."""
        return [edge.source for edge in self._edges if edge.target == target]

    def get_dependencies(self, source: str) -> list[str]:
        """Find all nodes that the source depends on."""
        return [edge.target for edge in self._edges if edge.source == source]

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of the dependency graph."""
        lines = ["graph TD"]

        # Add nodes with styling based on type
        type_styles = {
            "module": ":::module",
            "class": ":::class",
            "function": ":::function",
            "import": ":::import",
        }

        for name, node in self._nodes.items():
            safe_name = name.replace(".", "_").replace("-", "_")
            style = type_styles.get(node.node_type, "")
            lines.append(f"    {safe_name}[{name}]{style}")

        # Add edges
        edge_arrows = {
            "import": "-->|import|",
            "from_import": "-->|from|",
            "call": "-.->|call|",
            "inherits": "==>|inherits|",
        }

        for edge in self._edges:
            source = edge.source.replace(".", "_").replace("-", "_")
            target = edge.target.replace(".", "_").replace("-", "_")
            arrow = edge_arrows.get(edge.edge_type, "-->")
            lines.append(f"    {source} {arrow} {target}")

        return "\n".join(lines)

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary with graph statistics."""
        cycles = self.find_cycles()

        # Count by type
        node_types: dict[str, int] = {}
        for node in self._nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

        edge_types: dict[str, int] = {}
        for edge in self._edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "cycles_detected": len(cycles),
            "cycles": cycles[:5] if cycles else [],  # Limit to first 5
        }
