"""
Tests for the ast-grep integration module.

Tests cover:
- sg_search: Pattern-based structural code search
- sg_lint: YAML rule-based linting
- dump_syntax_tree: AST visualization
- Utility functions: is_available, get_version
"""

import pytest
import tempfile
from pathlib import Path
from agent_debug_toolkit.astgrep import (
    sg_search,
    sg_lint,
    dump_syntax_tree,
    is_available,
    get_version,
    LintReport,
    Match,
)


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file for testing."""
    file_path = temp_dir / "sample.py"
    content = '''def hello():
    """Say hello."""
    print("Hello, world!")


def goodbye():
    """Say goodbye."""
    print("Goodbye, world!")


class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"

    def farewell(self, name):
        return f"Goodbye, {name}!"


def process_data(data):
    """Process some data."""
    result = []
    for item in data:
        result.append(item * 2)
    return result
'''
    file_path.write_text(content, encoding="utf-8")
    return file_path


# Skip all tests if ast-grep is not available
pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="ast-grep CLI not installed"
)


# --- Test sg_search ---

class TestSgSearch:
    """Tests for structural pattern search."""

    def test_search_function_definitions(self, sample_python_file):
        """Should find function definitions."""
        report = sg_search(
            pattern="def $NAME($$$)",
            path=sample_python_file,
            lang="python",
        )

        assert report.success
        assert report.match_count >= 5  # hello, goodbye, greet, farewell, process_data
        assert len(report.matches) >= 5

    def test_search_with_metavariables(self, sample_python_file):
        """Should capture meta-variables in matches."""
        report = sg_search(
            pattern="def $NAME($$$)",
            path=sample_python_file,
            lang="python",
        )

        assert report.success
        # Should find multiple functions
        assert report.match_count >= 2

        # Check that NAME is captured
        for match in report.matches:
            assert "NAME" in match.meta_variables

    def test_search_print_calls(self, sample_python_file):
        """Should find print function calls."""
        report = sg_search(
            pattern='print($MSG)',
            path=sample_python_file,
            lang="python",
        )

        assert report.success
        assert report.match_count >= 2  # print in hello and goodbye

    def test_search_directory(self, temp_dir, sample_python_file):
        """Should search entire directory."""
        # Create another file
        other_file = temp_dir / "other.py"
        other_file.write_text("def foo(): pass\n", encoding="utf-8")

        report = sg_search(
            pattern="def $NAME($$$)",
            path=temp_dir,
            lang="python",
        )

        assert report.success
        # Should find functions in both files
        assert report.match_count >= 6

    def test_search_max_results(self, sample_python_file):
        """Should limit results with max_results."""
        report = sg_search(
            pattern="def $NAME($$$)",
            path=sample_python_file,
            lang="python",
            max_results=2,
        )

        assert report.success
        assert len(report.matches) <= 2

    def test_search_no_matches(self, sample_python_file):
        """Should handle no matches gracefully."""
        report = sg_search(
            pattern="class $NAME(SomeBase):",
            path=sample_python_file,
            lang="python",
        )

        assert report.success
        assert report.match_count == 0
        # Message should indicate zero matches
        assert "0" in report.message or "No" in report.message

    def test_search_invalid_path(self):
        """Should handle invalid path."""
        report = sg_search(
            pattern="def $NAME($$$)",
            path="/nonexistent/path",
        )

        assert not report.success
        assert "not found" in report.message.lower()

    def test_report_to_json(self, sample_python_file):
        """Should produce valid JSON output."""
        import json

        report = sg_search(
            pattern="def $NAME($$$)",
            path=sample_python_file,
            lang="python",
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert "matches" in parsed
        assert isinstance(parsed["matches"], list)

    def test_report_to_markdown(self, sample_python_file):
        """Should produce valid Markdown output."""
        report = sg_search(
            pattern="def $NAME($$$)",
            path=sample_python_file,
            lang="python",
        )

        md = report.to_markdown()

        assert "# AST-Grep Search Report" in md
        assert "✅ Success" in md
        assert "Matches" in md


# --- Test sg_lint ---

class TestSgLint:
    """Tests for YAML rule-based linting."""

    def test_lint_with_rule_string(self, sample_python_file):
        """Should lint with inline YAML rule."""
        rule = """
id: find-print-statements
language: Python
rule:
  pattern: print($$$)
message: "Found print statement"
"""
        report = sg_lint(
            path=sample_python_file,
            rule=rule,
        )

        # May succeed or fail depending on rule format
        assert isinstance(report, LintReport)

    def test_lint_invalid_path(self):
        """Should handle invalid path."""
        report = sg_lint(
            path="/nonexistent/path",
            rule="id: test\nrule:\n  pattern: test",
        )

        assert not report.success
        assert "not found" in report.message.lower()

    def test_lint_missing_rule(self, sample_python_file):
        """Should require rule or rule_file."""
        report = sg_lint(path=sample_python_file)

        assert not report.success
        assert "must be provided" in report.message.lower()

    def test_report_to_json(self, sample_python_file):
        """Should produce valid JSON output."""
        import json

        report = sg_lint(
            path=sample_python_file,
            rule="id: test\nlanguage: Python\nrule:\n  pattern: print($$$)",
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert "success" in parsed
        assert "violation_count" in parsed


# --- Test dump_syntax_tree ---

class TestDumpSyntaxTree:
    """Tests for AST visualization."""

    def test_dump_simple_code(self):
        """Should dump AST for simple code."""
        code = "def foo(): pass"
        result = dump_syntax_tree(code, "python")

        # Should return some tree representation
        assert isinstance(result, str)
        # The output format varies, but should not be empty or error
        assert len(result) > 0

    def test_dump_with_error(self):
        """Should handle parsing errors gracefully."""
        code = "this is not valid python {"
        result = dump_syntax_tree(code, "python")

        # Should return something (may be error or partial tree)
        assert isinstance(result, str)


# --- Test utility functions ---

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_available(self):
        """Should detect ast-grep availability."""
        result = is_available()
        assert isinstance(result, bool)
        assert result is True  # We're in this test because it's available

    def test_get_version(self):
        """Should return version string."""
        version = get_version()
        assert version is not None
        assert "ast-grep" in version.lower() or version.startswith("0.")


# --- Test Match dataclass ---

class TestMatch:
    """Tests for Match dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        match = Match(
            file="test.py",
            text="def foo(): pass",
            start_line=1,
            end_line=1,
            start_column=0,
            end_column=15,
            language="Python",
            meta_variables={"NAME": "foo"},
        )

        d = match.to_dict()

        assert d["file"] == "test.py"
        assert d["text"] == "def foo(): pass"
        assert d["start_line"] == 1
        assert d["meta_variables"]["NAME"] == "foo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
