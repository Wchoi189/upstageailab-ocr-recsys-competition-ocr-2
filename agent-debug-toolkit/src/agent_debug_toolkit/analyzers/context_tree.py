"""
ContextTreeAnalyzer - Generate annotated directory trees with semantic context.

Provides AI agents with rich context about directory structure by:
- Extracting module/package docstrings
- Listing key exports from __init__.py
- Identifying special directories (tests, configs, models, data)
- Annotating with class/function purposes
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_debug_toolkit.analyzers.base import AnalysisReport, AnalysisResult


@dataclass
class FileContext:
    """Metadata extracted from a Python file."""

    path: Path
    module_docstring: str | None = None
    exports: list[str] = field(default_factory=list)
    key_classes: list[tuple[str, str]] = field(default_factory=list)  # (name, docstring)
    key_functions: list[tuple[str, str]] = field(default_factory=list)  # (name, docstring)
    is_init: bool = False


@dataclass
class DirectoryNode:
    """Represents a directory tree node with context."""

    path: Path
    name: str
    is_dir: bool
    context: FileContext | None = None
    children: list[DirectoryNode] = field(default_factory=list)
    depth: int = 0
    category: str | None = None  # tests, configs, models, data, etc.


# Special directory categories
SPECIAL_DIRS = {
    "tests": "🧪 Tests",
    "test": "🧪 Tests",
    "configs": "⚙️ Config",
    "config": "⚙️ Config",
    "models": "🤖 Models",
    "model": "🤖 Models",
    "data": "📊 Data",
    "datasets": "📊 Data",
    "utils": "🔧 Utils",
    "core": "⚡ Core",
    "inference": "🔮 Inference",
    "training": "📈 Training",
    "trainer": "📈 Training",
}


class ContextTreeAnalyzer:
    """Analyze directory structure and generate annotated trees."""

    def __init__(self, max_depth: int = 3, filter_pattern: str | None = None):
        """
        Initialize the analyzer.

        Args:
            max_depth: Maximum directory depth to traverse
            filter_pattern: Optional glob pattern to filter files
        """
        self.max_depth = max_depth
        self.filter_pattern = filter_pattern

    def analyze_directory(self, path: str | Path) -> AnalysisReport:
        """
        Analyze a directory and generate an annotated tree.

        Args:
            path: Path to directory to analyze

        Returns:
            AnalysisReport with tree structure in findings
        """
        path = Path(path).resolve()

        if not path.exists():
            return AnalysisReport(
                target=str(path),
                analyzer="ContextTreeAnalyzer",
                findings=[],
                summary={"error": f"Path not found: {path}"},
            )

        if not path.is_dir():
            return AnalysisReport(
                target=str(path),
                analyzer="ContextTreeAnalyzer",
                findings=[],
                summary={"error": f"Not a directory: {path}"},
            )

        # Build the tree
        root_node = self._build_tree(path, depth=0)

        # Convert tree to findings (use _results list from base)
        self._results = []
        self._tree_to_findings(root_node)

        summary = {
            "total_directories": self._count_dirs(root_node),
            "total_files": self._count_files(root_node),
            "max_depth_reached": self._get_max_depth(root_node),
        }

        return AnalysisReport(
            analyzer_name="ContextTreeAnalyzer",
            target_path=str(path),
            results=self._results,
            summary=summary,
        )

    def _build_tree(self, path: Path, depth: int) -> DirectoryNode:
        """Recursively build directory tree with context."""
        if depth > self.max_depth:
            return None

        name = path.name if depth > 0 else path.as_posix()
        node = DirectoryNode(
            path=path,
            name=name,
            is_dir=path.is_dir(),
            depth=depth,
        )

        # Determine category
        node.category = SPECIAL_DIRS.get(path.name.lower())

        if path.is_file():
            # Extract file context if it's a Python file
            if path.suffix == ".py":
                node.context = self._extract_file_context(path)
            return node

        # Directory - traverse children
        try:
            children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))

            for child in children:
                # Skip hidden files and common ignores
                if child.name.startswith((".","__pycache__")):
                    continue

                child_node = self._build_tree(child, depth + 1)
                if child_node:
                    node.children.append(child_node)

            # If directory has __init__.py, extract its context
            init_file = path / "__init__.py"
            if init_file.exists():
                node.context = self._extract_file_context(init_file)

        except PermissionError:
            pass

        return node

    def _extract_file_context(self, file_path: Path) -> FileContext:
        """Extract context from a Python file using AST."""
        context = FileContext(path=file_path, is_init=file_path.name == "__init__.py")

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)

            # Extract module docstring
            if tree.body and isinstance(tree.body[0], ast.Expr):
                if isinstance(tree.body[0].value, ast.Constant):
                    if isinstance(tree.body[0].value.value, str):
                        context.module_docstring = tree.body[0].value.value.strip()

            # Extract __all__ exports
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                context.exports = [
                                    elt.value for elt in node.value.elts
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                                ]

            # Extract key classes (first 3)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node) or ""
                    # Take first line of docstring
                    doc_first_line = docstring.split("\n")[0] if docstring else ""
                    context.key_classes.append((node.name, doc_first_line))
                    if len(context.key_classes) >= 3:
                        break

            # Extract key functions (first 3, excluding private)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    docstring = ast.get_docstring(node) or ""
                    doc_first_line = docstring.split("\n")[0] if docstring else ""
                    context.key_functions.append((node.name, doc_first_line))
                    if len(context.key_functions) >= 3:
                        break

        except (SyntaxError, UnicodeDecodeError):
            pass

        return context

    def _tree_to_findings(self, node: DirectoryNode, parent_path: str = ""):
        """Convert tree structure to findings for reporting."""
        if node is None:
            return

        # Create path representation
        if parent_path:
            path_repr = f"{parent_path}/{node.name}"
        else:
            path_repr = node.name

        # Determine the finding category and details
        category = "directory" if node.is_dir else "file"

        details: dict[str, Any] = {
            "depth": node.depth,
            "is_dir": node.is_dir,
        }

        if node.category:
            details["category"] = node.category

        if node.context:
            if node.context.module_docstring:
                details["docstring"] = node.context.module_docstring
            if node.context.exports:
                details["exports"] = node.context.exports
            if node.context.key_classes:
                details["key_classes"] = [
                    {"name": name, "doc": doc} for name, doc in node.context.key_classes
                ]
            if node.context.key_functions:
                details["key_functions"] = [
                    {"name": name, "doc": doc} for name, doc in node.context.key_functions
                ]

        self._results.append(
            AnalysisResult(
                file=path_repr,
                line=0,
                column=0,
                pattern=node.name,
                context="",
                category=category,
                code_snippet="",
                metadata=details,
            )
        )

        # Process children
        for child in node.children:
            self._tree_to_findings(child, path_repr)

    def _count_dirs(self, node: DirectoryNode) -> int:
        """Count total directories in tree."""
        if node is None or not node.is_dir:
            return 0
        count = 1
        for child in node.children:
            count += self._count_dirs(child)
        return count

    def _count_files(self, node: DirectoryNode) -> int:
        """Count total files in tree."""
        if node is None:
            return 0
        if not node.is_dir:
            return 1
        count = 0
        for child in node.children:
            count += self._count_files(child)
        return count

    def _get_max_depth(self, node: DirectoryNode) -> int:
        """Get maximum depth reached in tree."""
        if node is None or not node.children:
            return node.depth if node else 0
        return max(self._get_max_depth(child) for child in node.children)


def format_tree_markdown(report: AnalysisReport) -> str:
    """Format context tree as markdown with emojis and annotations."""
    lines = [f"# Context Tree: {report.target_path}\n"]

    # Group findings by parent directory
    current_dir = None
    indent_level = {}

    for finding in report.results:
        depth = finding.metadata.get("depth", 0)
        is_dir = finding.metadata.get("is_dir", False)

        indent = "  " * depth

        if is_dir:
            # Directory entry
            icon = "📁"
            category_label = finding.metadata.get("category", "")
            if category_label:
                header_line = f"{indent}{icon} **{finding.pattern}/** - *{category_label}*"
            else:
                header_line = f"{indent}{icon} **{finding.pattern}/**"

            # Add docstring if available
            if "docstring" in finding.metadata:
                doc = finding.metadata["docstring"].split("\n")[0]
                header_line += f" - *{doc}*"

            lines.append(header_line)

            # Add exports if it's __init__.py
            if "exports" in finding.metadata and finding.metadata["exports"]:
                exports_str = ", ".join(finding.metadata["exports"][:5])
                if len(finding.metadata["exports"]) > 5:
                    exports_str += f", ... ({len(finding.metadata['exports'])} total)"
                lines.append(f"{indent}  - Exports: {exports_str}")

        else:
            # File entry
            icon = "📄"
            file_line = f"{indent}{icon} `{finding.pattern}`"

            # Add key classes/functions
            if "key_classes" in finding.metadata and finding.metadata["key_classes"]:
                classes = finding.metadata["key_classes"]
                class_names = [c["name"] for c in classes]
                file_line += f" - Classes: {', '.join(class_names)}"

            if "key_functions" in finding.metadata and finding.metadata["key_functions"]:
                funcs = finding.metadata["key_functions"]
                func_names = [f["name"] for f in funcs]
                file_line += f" - Functions: {', '.join(func_names)}"

            lines.append(file_line)

    # Add summary
    lines.append(f"\n**Summary**: {report.summary['total_directories']} directories, {report.summary['total_files']} files")

    return "\n".join(lines)
