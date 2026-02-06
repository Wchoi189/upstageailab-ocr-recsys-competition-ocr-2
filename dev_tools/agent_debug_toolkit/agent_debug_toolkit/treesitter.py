"""
Agent Debug Toolkit - Tree-Sitter Integration Module

Provides multi-language AST parsing via tree-sitter with pre-compiled grammars.
Uses tree-sitter-languages for 30+ language support without compilation.

Tools:
    - parse_code: Parse code into AST
    - run_query: Execute tree-sitter queries
    - get_ast_text: Get prettified AST representation
    - list_languages: List available languages
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Check for tree-sitter availability
try:
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    get_language = None
    get_parser = None


# Supported languages (subset of tree-sitter-languages)
SUPPORTED_LANGUAGES = [
    "python", "javascript", "typescript", "tsx", "java", "go", "rust",
    "c", "cpp", "c_sharp", "kotlin", "swift", "ruby", "php",
    "html", "css", "json", "yaml", "bash", "lua", "sql",
    "markdown", "toml", "dockerfile", "make",
]


@dataclass
class ASTNode:
    """Simplified representation of an AST node."""
    type: str
    text: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    children: list["ASTNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "children_count": len(self.children),
        }


@dataclass
class QueryMatch:
    """Result from a tree-sitter query."""
    capture_name: str
    node_type: str
    text: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture": self.capture_name,
            "type": self.node_type,
            "text": self.text,
            "line": self.start_line,
        }


@dataclass
class ParseReport:
    """Report from a parse operation."""
    success: bool
    message: str
    language: str = ""
    root_type: str = ""
    node_count: int = 0
    ast_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "language": self.language,
            "root_type": self.root_type,
            "node_count": self.node_count,
            "ast_text": self.ast_text[:2000] if len(self.ast_text) > 2000 else self.ast_text,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class QueryReport:
    """Report from a query operation."""
    success: bool
    message: str
    query: str = ""
    language: str = ""
    matches: list[QueryMatch] = field(default_factory=list)
    match_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "query": self.query[:200] if len(self.query) > 200 else self.query,
            "language": self.language,
            "match_count": self.match_count,
            "matches": [m.to_dict() for m in self.matches[:50]],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# --- Helper Functions ---

def _count_nodes(node) -> int:
    """Recursively count nodes in AST."""
    count = 1
    for child in node.children:
        count += _count_nodes(child)
    return count


def _node_to_text(node, indent: int = 0, max_depth: int = 5) -> str:
    """Convert AST node to indented text representation."""
    if indent > max_depth * 2:
        return "  " * indent + "...\n"

    text = node.text.decode("utf-8") if isinstance(node.text, bytes) else str(node.text)
    text_preview = text[:30].replace("\n", "\\n") if len(text) > 30 else text.replace("\n", "\\n")

    lines = ["  " * indent + f"{node.type}"]
    if node.child_count == 0 and text_preview:
        lines[0] += f': "{text_preview}"'
    lines[0] += f" [{node.start_point[0]+1}:{node.start_point[1]}]"

    if indent < max_depth * 2:
        for child in node.children:
            lines.append(_node_to_text(child, indent + 1, max_depth))

    return "\n".join(lines)


# --- Main Functions ---

def parse_code(
    code: str,
    lang: str,
    max_depth: int = 5,
) -> ParseReport:
    """
    Parse code into an AST.

    Args:
        code: Source code to parse
        lang: Language identifier (e.g., "python", "javascript")
        max_depth: Maximum depth for AST text output

    Returns:
        ParseReport with AST information
    """
    if not TREE_SITTER_AVAILABLE:
        return ParseReport(
            success=False,
            message="tree-sitter-languages not installed. Install with: pip install tree-sitter-languages",
        )

    lang = lang.lower()
    if lang not in SUPPORTED_LANGUAGES:
        return ParseReport(
            success=False,
            message=f"Unsupported language: {lang}. Supported: {', '.join(SUPPORTED_LANGUAGES[:10])}...",
        )

    try:
        parser = get_parser(lang)
        code_bytes = code.encode("utf-8") if isinstance(code, str) else code
        tree = parser.parse(code_bytes)

        root = tree.root_node
        node_count = _count_nodes(root)
        ast_text = _node_to_text(root, max_depth=max_depth)

        return ParseReport(
            success=True,
            message=f"Parsed {node_count} nodes",
            language=lang,
            root_type=root.type,
            node_count=node_count,
            ast_text=ast_text,
        )

    except Exception as e:
        return ParseReport(
            success=False,
            message=f"Parse error: {e}",
            language=lang,
        )


def run_query(
    code: str,
    query: str,
    lang: str,
    max_results: int = 50,
) -> QueryReport:
    """
    Run a tree-sitter query against code.

    Args:
        code: Source code to query
        query: Tree-sitter query in S-expression format
        lang: Language identifier
        max_results: Maximum number of matches to return

    Returns:
        QueryReport with matches
    """
    if not TREE_SITTER_AVAILABLE:
        return QueryReport(
            success=False,
            message="tree-sitter-languages not installed",
        )

    lang = lang.lower()
    if lang not in SUPPORTED_LANGUAGES:
        return QueryReport(
            success=False,
            message=f"Unsupported language: {lang}",
        )

    try:
        language = get_language(lang)
        parser = get_parser(lang)
        code_bytes = code.encode("utf-8") if isinstance(code, str) else code
        tree = parser.parse(code_bytes)

        # Compile and run query
        ts_query = language.query(query)
        captures = ts_query.captures(tree.root_node)

        matches = []
        for node, capture_name in captures[:max_results]:
            text = code_bytes[node.start_byte:node.end_byte].decode("utf-8")
            matches.append(QueryMatch(
                capture_name=capture_name,
                node_type=node.type,
                text=text,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
            ))

        return QueryReport(
            success=True,
            message=f"Found {len(matches)} match(es)",
            query=query,
            language=lang,
            matches=matches,
            match_count=len(matches),
        )

    except Exception as e:
        return QueryReport(
            success=False,
            message=f"Query error: {e}",
            query=query,
            language=lang,
        )


def list_languages() -> list[str]:
    """Get list of supported languages."""
    return SUPPORTED_LANGUAGES.copy()


def is_available() -> bool:
    """Check if tree-sitter is available."""
    return TREE_SITTER_AVAILABLE


def get_version() -> str | None:
    """Get tree-sitter version if available."""
    if not TREE_SITTER_AVAILABLE:
        return None
    try:
        import tree_sitter
        return getattr(tree_sitter, "__version__", "unknown")
    except Exception:
        return None
