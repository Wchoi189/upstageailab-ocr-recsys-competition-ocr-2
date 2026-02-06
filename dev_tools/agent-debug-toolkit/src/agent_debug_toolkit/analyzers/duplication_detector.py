"""
DuplicationDetector - Find duplicate/similar code blocks via AST hashing.

Algorithm:
1. Normalize AST (remove variable names, comments)
2. Hash normalized subtrees (functions, code blocks)
3. Compare hashes to find duplicates
4. Report similar code with line ranges and similarity score

Output:
- duplicate_groups: [[file:line, file:line], ...]
- similarity_score: Float 0-1
- suggested_action: "Extract to function" | "Create shared module"
"""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import BaseAnalyzer, AnalysisReport, AnalysisResult


@dataclass
class CodeBlock:
    """Represents a code block for duplication analysis."""

    file: str
    name: str
    start_line: int
    end_line: int
    ast_hash: str
    normalized_ast: str
    line_count: int

    def __hash__(self) -> int:
        return hash((self.file, self.start_line, self.ast_hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodeBlock):
            return False
        return self.file == other.file and self.start_line == other.start_line


@dataclass
class DuplicateGroup:
    """A group of duplicate code blocks."""

    blocks: list[CodeBlock] = field(default_factory=list)
    similarity_score: float = 1.0

    @property
    def suggested_action(self) -> str:
        """Suggest refactoring action based on duplicate location."""
        if not self.blocks:
            return "No action needed"

        files = set(b.file for b in self.blocks)
        if len(files) == 1:
            return "Extract to function within module"
        else:
            return "Extract to shared utility module"


class ASTNormalizer(ast.NodeTransformer):
    """
    Normalize AST for comparison by replacing variable names with placeholders.

    This makes code blocks with different variable names but same structure
    compare as equal.
    """

    def __init__(self):
        self.var_counter = 0
        self.var_map: dict[str, str] = {}

    def _get_normalized_name(self, name: str) -> str:
        """Get normalized placeholder name for a variable."""
        if name not in self.var_map:
            self.var_map[name] = f"_var_{self.var_counter}"
            self.var_counter += 1
        return self.var_map[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Normalize variable names, but keep built-in names."""
        builtins = {
            "True",
            "False",
            "None",
            "print",
            "len",
            "range",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "open",
            "super",
            "self",
            "cls",
        }
        if node.id not in builtins:
            node.id = self._get_normalized_name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Normalize function argument names."""
        if node.arg not in ("self", "cls"):
            node.arg = self._get_normalized_name(node.arg)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Normalize function definition including its name."""
        # Normalize the function name for comparison
        node.name = "_func_"
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Normalize async function definition including its name."""
        node.name = "_func_"
        self.generic_visit(node)
        return node


class DuplicationDetector(BaseAnalyzer):
    """
    Detect duplicate/similar code blocks in Python code.

    Uses AST hashing to find code with similar structure regardless of
    variable naming. Configurable minimum line count and similarity threshold.
    """

    name = "DuplicationDetector"

    def __init__(
        self,
        min_lines: int = 5,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize the duplication detector.

        Args:
            min_lines: Minimum lines for a code block to be considered
            similarity_threshold: Similarity threshold (0.0-1.0), 1.0 = exact match only
        """
        super().__init__()
        self.min_lines = min_lines
        self.similarity_threshold = similarity_threshold
        self._code_blocks: list[CodeBlock] = []
        self._duplicate_groups: list[DuplicateGroup] = []

    def analyze_file(self, file_path: str | Path) -> AnalysisReport:
        """Analyze a single file for code duplication."""
        path = Path(file_path).resolve()
        self._current_file = str(path)
        self._results = []
        self._code_blocks = []  # Reset for single file analysis
        self._duplicate_groups = []

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

        # Extract and hash code blocks
        self._extract_code_blocks(tree, str(path))

        # Find duplicates (only within this single file)
        self._find_duplicates_in_blocks(self._code_blocks)

        # Generate results
        self._generate_results()

        return AnalysisReport(
            analyzer_name=self.name,
            target_path=str(path),
            results=self._results.copy(),
            summary=self._generate_summary(),
        )

    def analyze_directory(
        self, directory: str | Path, pattern: str = "*.py", recursive: bool = True
    ) -> AnalysisReport:
        """
        Analyze all Python files in a directory for duplication.

        This enables cross-file duplicate detection.
        """
        dir_path = Path(directory).resolve()
        glob_method = dir_path.rglob if recursive else dir_path.glob

        # Reset state for directory analysis
        self._code_blocks = []
        self._duplicate_groups = []
        self._results = []

        files_analyzed = 0
        errors = 0

        # First pass: extract all code blocks from all files
        for py_file in sorted(glob_method(pattern)):
            if py_file.is_file():
                try:
                    source = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=str(py_file))
                    self._extract_code_blocks(tree, str(py_file))
                    files_analyzed += 1
                except (SyntaxError, FileNotFoundError):
                    errors += 1

        # Find duplicates across all files
        self._find_duplicates_in_blocks(self._code_blocks)

        # Generate results
        self._generate_results()

        return AnalysisReport(
            analyzer_name=self.name,
            target_path=str(dir_path),
            results=self._results,
            summary={
                "files_analyzed": files_analyzed,
                "errors": errors,
                "total_code_blocks": len(self._code_blocks),
                "duplicate_groups": len(self._duplicate_groups),
                "total_duplicates": sum(len(g.blocks) for g in self._duplicate_groups),
            },
        )

    def visit(self, node: ast.AST) -> None:
        """Visit AST nodes - not used directly, uses _extract_code_blocks instead."""
        pass

    def analyze_source(self, source: str, filename: str = "<string>") -> AnalysisReport:
        """Analyze Python source code directly."""
        self._current_file = filename
        self._source_lines = source.splitlines()
        self._results = []
        self._code_blocks = []
        self._duplicate_groups = []

        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            return AnalysisReport(
                analyzer_name=self.name,
                target_path=filename,
                summary={"error": f"Syntax error: {e}"},
            )

        # Extract and hash code blocks
        self._extract_code_blocks(tree, filename)

        # Find duplicates
        self._find_duplicates_in_blocks(self._code_blocks)

        # Generate results
        self._generate_results()

        return AnalysisReport(
            analyzer_name=self.name,
            target_path=filename,
            results=self._results.copy(),
            summary=self._generate_summary(),
        )

    def _extract_code_blocks(self, tree: ast.AST, file_path: str) -> None:
        """Extract function and class method definitions as code blocks."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Calculate line count
                start_line = node.lineno
                end_line = getattr(node, "end_lineno", start_line)
                line_count = end_line - start_line + 1

                if line_count >= self.min_lines:
                    # Normalize and hash the AST
                    normalizer = ASTNormalizer()
                    # Create a copy to avoid modifying original
                    normalized = normalizer.visit(ast.parse(ast.unparse(node)))
                    normalized_str = ast.dump(normalized, annotate_fields=False)
                    ast_hash = hashlib.md5(normalized_str.encode()).hexdigest()

                    block = CodeBlock(
                        file=file_path,
                        name=node.name,
                        start_line=start_line,
                        end_line=end_line,
                        ast_hash=ast_hash,
                        normalized_ast=normalized_str,
                        line_count=line_count,
                    )
                    self._code_blocks.append(block)

    def _find_duplicates_in_blocks(self, blocks: list[CodeBlock]) -> None:
        """Find duplicate code blocks by comparing hashes."""
        # Group blocks by hash for exact matches
        hash_groups: dict[str, list[CodeBlock]] = defaultdict(list)
        for block in blocks:
            hash_groups[block.ast_hash].append(block)

        # Create duplicate groups for exact matches
        self._duplicate_groups = []
        for ast_hash, group_blocks in hash_groups.items():
            if len(group_blocks) > 1:
                group = DuplicateGroup(blocks=group_blocks, similarity_score=1.0)
                self._duplicate_groups.append(group)

    def _generate_results(self) -> None:
        """Generate analysis results from duplicate groups."""
        for group in self._duplicate_groups:
            if len(group.blocks) < 2:
                continue

            # Use first block as reference
            first_block = group.blocks[0]

            locations = [f"{b.file}:{b.start_line}-{b.end_line} ({b.name})" for b in group.blocks]

            self._results.append(
                AnalysisResult(
                    file=first_block.file,
                    line=first_block.start_line,
                    column=0,
                    pattern=f"Duplicate code: {first_block.name}",
                    context=f"Found {len(group.blocks)} identical code blocks",
                    category="duplicate",
                    metadata={
                        "locations": locations,
                        "similarity_score": group.similarity_score,
                        "suggested_action": group.suggested_action,
                        "line_count": first_block.line_count,
                    },
                )
            )

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_code_blocks": len(self._code_blocks),
            "duplicate_groups": len(self._duplicate_groups),
            "total_duplicates": sum(len(g.blocks) for g in self._duplicate_groups),
            "min_lines_threshold": self.min_lines,
            "similarity_threshold": self.similarity_threshold,
        }

    def get_duplicate_groups(self) -> list[DuplicateGroup]:
        """Get all detected duplicate groups."""
        return self._duplicate_groups

    def get_duplicates_by_file(self) -> dict[str, list[DuplicateGroup]]:
        """Get duplicates grouped by file."""
        by_file: dict[str, list[DuplicateGroup]] = defaultdict(list)
        for group in self._duplicate_groups:
            for block in group.blocks:
                if group not in by_file[block.file]:
                    by_file[block.file].append(group)
        return dict(by_file)
