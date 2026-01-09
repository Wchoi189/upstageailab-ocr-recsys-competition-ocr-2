"""
Tests for the tree-sitter integration module.

Tests cover:
- parse_code: Multi-language AST parsing
- run_query: Tree-sitter query execution
- list_languages: Available language support
"""

import pytest
from agent_debug_toolkit.treesitter import (
    parse_code,
    run_query,
    list_languages,
    is_available,
    get_version,
)


# Skip all tests if tree-sitter is not available
pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="tree-sitter-languages not installed"
)


# --- Test parse_code ---

class TestParseCode:
    """Tests for AST parsing."""

    def test_parse_python(self):
        """Should parse Python code."""
        code = "def hello():\n    print('Hello')"
        report = parse_code(code, "python")

        assert report.success
        assert report.language == "python"
        assert report.root_type == "module"
        assert report.node_count > 0

    def test_parse_javascript(self):
        """Should parse JavaScript code."""
        code = "function hello() { console.log('Hello'); }"
        report = parse_code(code, "javascript")

        assert report.success
        assert report.language == "javascript"
        assert report.root_type == "program"

    def test_parse_typescript(self):
        """Should parse TypeScript code."""
        code = "const greeting: string = 'Hello';"
        report = parse_code(code, "typescript")

        assert report.success
        assert report.language == "typescript"

    def test_parse_go(self):
        """Should parse Go code."""
        code = "package main\n\nfunc main() {}"
        report = parse_code(code, "go")

        assert report.success
        assert report.language == "go"

    def test_parse_unsupported_language(self):
        """Should handle unsupported language."""
        report = parse_code("code", "unsupported_lang")

        assert not report.success
        assert "unsupported" in report.message.lower()

    def test_ast_text_output(self):
        """Should produce AST text output."""
        code = "x = 1"
        report = parse_code(code, "python")

        assert report.success
        assert len(report.ast_text) > 0
        assert "module" in report.ast_text.lower() or "assignment" in report.ast_text.lower()

    def test_report_to_json(self):
        """Should produce valid JSON."""
        import json
        code = "def foo(): pass"
        report = parse_code(code, "python")

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert "node_count" in parsed


# --- Test run_query ---

class TestRunQuery:
    """Tests for tree-sitter queries."""

    def test_query_function_definitions(self):
        """Should find function definitions with query."""
        code = "def foo():\n    pass\n\ndef bar():\n    pass"
        query = "(function_definition name: (identifier) @func_name)"
        report = run_query(code, query, "python")

        assert report.success
        assert report.match_count >= 2

    def test_query_with_captures(self):
        """Should capture named nodes."""
        code = "x = 1\ny = 2"
        query = "(assignment left: (identifier) @var)"
        report = run_query(code, query, "python")

        assert report.success
        # Should find x and y
        names = [m.text for m in report.matches]
        assert "x" in names
        assert "y" in names

    def test_query_javascript(self):
        """Should query JavaScript code."""
        code = "const x = 1; let y = 2;"
        query = "(variable_declarator name: (identifier) @name)"
        report = run_query(code, query, "javascript")

        assert report.success
        assert report.match_count >= 2

    def test_query_invalid_syntax(self):
        """Should handle invalid query syntax."""
        code = "x = 1"
        query = "((invalid query syntax"
        report = run_query(code, query, "python")

        assert not report.success
        assert "error" in report.message.lower()

    def test_query_max_results(self):
        """Should limit results."""
        code = "a=1\nb=2\nc=3\nd=4\ne=5"
        query = "(assignment left: (identifier) @var)"
        report = run_query(code, query, "python", max_results=2)

        assert report.success
        assert len(report.matches) <= 2


# --- Test utility functions ---

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_list_languages(self):
        """Should return list of supported languages."""
        languages = list_languages()

        assert isinstance(languages, list)
        assert "python" in languages
        assert "javascript" in languages
        assert len(languages) >= 10

    def test_is_available(self):
        """Should detect availability."""
        result = is_available()
        assert result is True  # We're in this test because it's available

    def test_get_version(self):
        """Should return version string."""
        version = get_version()
        assert version is not None or version == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
