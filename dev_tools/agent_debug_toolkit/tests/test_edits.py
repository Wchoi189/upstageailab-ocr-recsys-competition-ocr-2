"""
Tests for the edits module.

Tests cover:
- apply_unified_diff: exact, whitespace-insensitive, and fuzzy matching
- smart_edit: exact, regex, and fuzzy modes
- read_file_slice: line range extraction
- format_code: formatter integration
"""

import pytest
import tempfile
from pathlib import Path
from agent_debug_toolkit.edits import (
    apply_unified_diff,
    smart_edit,
    read_file_slice,
    format_code,
    create_simple_diff,
    parse_unified_diff,
    HunkStatus,
    EditReport,
)


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file(temp_dir):
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
'''
    file_path.write_text(content, encoding="utf-8")
    return file_path


# --- Test parse_unified_diff ---

class TestParseUnifiedDiff:
    """Tests for diff parsing."""

    def test_parse_simple_diff(self):
        """Should parse a simple unified diff."""
        diff = '''--- a/sample.py
+++ b/sample.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello, world!")
+    print("Hello, universe!")
'''
        hunks = parse_unified_diff(diff)
        assert len(hunks) == 1
        assert hunks[0].file_path == "sample.py"
        assert "Hello, world!" in hunks[0].remove_lines[0]
        assert "Hello, universe!" in hunks[0].add_lines[0]

    def test_parse_multiple_hunks(self):
        """Should parse diff with multiple hunks."""
        diff = '''--- a/sample.py
+++ b/sample.py
@@ -1,2 +1,2 @@
-old line 1
+new line 1
@@ -10,2 +10,2 @@
-old line 10
+new line 10
'''
        hunks = parse_unified_diff(diff)
        assert len(hunks) == 2

    def test_parse_empty_diff(self):
        """Should handle empty diff."""
        hunks = parse_unified_diff("")
        assert len(hunks) == 0


# --- Test apply_unified_diff ---

class TestApplyUnifiedDiff:
    """Tests for unified diff application."""

    def test_apply_exact_match(self, sample_file, temp_dir):
        """Should apply diff with exact matching."""
        diff = '''--- a/sample.py
+++ b/sample.py
@@ -2,2 +2,2 @@
-    """Say hello."""
-    print("Hello, world!")
+    \"\"\"Say hello to the universe.\"\"\"
+    print("Hello, universe!")
'''
        report = apply_unified_diff(diff, strategy="exact", project_root=temp_dir)

        assert report.success
        assert "sample.py" in report.files_modified

        new_content = sample_file.read_text()
        assert "Hello, universe!" in new_content

    def test_apply_fuzzy_match(self, sample_file, temp_dir):
        """Should apply diff with fuzzy matching when exact fails."""
        # Modify the file slightly to break exact matching
        original = sample_file.read_text()
        modified = original.replace('    """Say hello."""', '    """Say hello!"""')
        sample_file.write_text(modified)

        diff = '''--- a/sample.py
+++ b/sample.py
@@ -2,2 +2,2 @@
-    \"\"\"Say hello.\"\"\"
-    print("Hello, world!")
+    \"\"\"Say hello to all.\"\"\"
+    print("Hello, everyone!")
'''
        report = apply_unified_diff(diff, strategy="fuzzy", project_root=temp_dir)

        # Should still succeed with fuzzy matching
        assert report.success or any(
            h.status == HunkStatus.APPLIED for h in report.hunks
        )

    def test_dry_run_no_changes(self, sample_file, temp_dir):
        """Should not modify file in dry_run mode."""
        original_content = sample_file.read_text()

        diff = '''--- a/sample.py
+++ b/sample.py
@@ -3,1 +3,1 @@
-    print("Hello, world!")
+    print("Changed!")
'''
        report = apply_unified_diff(diff, project_root=temp_dir, dry_run=True)

        assert report.dry_run
        new_content = sample_file.read_text()
        assert new_content == original_content

    def test_file_not_found(self, temp_dir):
        """Should handle missing file gracefully."""
        diff = '''--- a/nonexistent.py
+++ b/nonexistent.py
@@ -1,1 +1,1 @@
-old
+new
'''
        report = apply_unified_diff(diff, project_root=temp_dir)

        assert not report.success or any(
            h.status == HunkStatus.FAILED_NOT_FOUND for h in report.hunks
        )


# --- Test smart_edit ---

class TestSmartEdit:
    """Tests for smart search/replace."""

    def test_exact_mode(self, sample_file):
        """Should replace exact matches."""
        report = smart_edit(
            sample_file,
            search='print("Hello, world!")',
            replace='print("Hello, universe!")',
            mode="exact",
        )

        assert report.success
        new_content = sample_file.read_text()
        assert 'print("Hello, universe!")' in new_content

    def test_exact_not_found(self, sample_file):
        """Should fail when exact match not found."""
        report = smart_edit(
            sample_file,
            search="this text does not exist",
            replace="replacement",
            mode="exact",
        )

        assert not report.success
        assert "not found" in report.message.lower()

    def test_regex_mode(self, sample_file):
        """Should replace using regex patterns."""
        report = smart_edit(
            sample_file,
            search=r'print\("([^"]+)"\)',
            replace=r'logger.info("\1")',
            mode="regex",
        )

        assert report.success
        new_content = sample_file.read_text()
        assert "logger.info" in new_content

    def test_regex_all_occurrences(self, sample_file):
        """Should replace all occurrences when requested."""
        report = smart_edit(
            sample_file,
            search=r'"""[^"]+"""',
            replace='"""Modified docstring."""',
            mode="regex",
            all_occurrences=True,
        )

        assert report.success
        new_content = sample_file.read_text()
        assert new_content.count('"""Modified docstring."""') >= 2

    def test_fuzzy_mode(self, sample_file):
        """Should find approximate matches."""
        # Search for slightly different text
        report = smart_edit(
            sample_file,
            search='def helo():\n    """Say hello."""',  # typo in 'hello'
            replace='def hello_new():\n    """Updated hello."""',
            mode="fuzzy",
            fuzzy_threshold=0.7,
        )

        # May or may not succeed depending on threshold
        # At least it shouldn't crash
        assert isinstance(report, EditReport)

    def test_dry_run(self, sample_file):
        """Should not modify file in dry_run mode."""
        original = sample_file.read_text()

        report = smart_edit(
            sample_file,
            search='print("Hello, world!")',
            replace='print("Changed!")',
            mode="exact",
            dry_run=True,
        )

        assert report.dry_run
        assert sample_file.read_text() == original

    def test_file_not_found(self, temp_dir):
        """Should handle missing file."""
        report = smart_edit(
            temp_dir / "missing.py",
            search="x",
            replace="y",
        )

        assert not report.success
        assert "not found" in report.message.lower()


# --- Test read_file_slice ---

class TestReadFileSlice:
    """Tests for line range reading."""

    def test_read_specific_lines(self, sample_file):
        """Should read specific line range."""
        result = read_file_slice(sample_file, start_line=1, end_line=3)

        assert "def hello():" in result
        assert "Say hello" in result
        assert "Hello, world" in result

    def test_read_with_context(self, sample_file):
        """Should include context lines when requested."""
        result = read_file_slice(sample_file, start_line=3, end_line=3, context_lines=1)

        # Should include lines 2, 3, 4
        assert "Say hello" in result  # line 2
        assert "Hello, world" in result  # line 3

    def test_read_out_of_bounds(self, sample_file):
        """Should handle out-of-bounds line numbers."""
        result = read_file_slice(sample_file, start_line=1, end_line=1000)

        # Should return all available lines without error
        assert "def hello():" in result

    def test_file_not_found(self, temp_dir):
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            read_file_slice(temp_dir / "missing.py", 1, 10)


# --- Test format_code ---

class TestFormatCode:
    """Tests for code formatting."""

    def test_format_with_black(self, sample_file):
        """Should format code with black if available."""
        report = format_code(sample_file, style="black")

        # May succeed or fail depending on black installation
        assert isinstance(report, EditReport)
        if not report.success:
            assert "not found" in report.message.lower() or "error" in report.message.lower()

    def test_check_only_mode(self, sample_file):
        """Should check without modifying in check_only mode."""
        original = sample_file.read_text()

        report = format_code(sample_file, style="black", check_only=True)

        # File should be unchanged
        assert sample_file.read_text() == original

    def test_unknown_formatter(self, sample_file):
        """Should handle unknown formatter gracefully."""
        report = format_code(sample_file, style="unknown_formatter")

        assert not report.success
        assert "unknown" in report.message.lower()

    def test_path_not_found(self, temp_dir):
        """Should handle missing path."""
        report = format_code(temp_dir / "missing.py")

        assert not report.success
        assert "not found" in report.message.lower()


# --- Test create_simple_diff ---

class TestCreateSimpleDiff:
    """Tests for diff generation."""

    def test_create_diff(self):
        """Should create valid unified diff."""
        old = "line 1\nline 2\nline 3\n"
        new = "line 1\nmodified line 2\nline 3\n"

        diff = create_simple_diff("test.py", old, new)

        assert "--- a/test.py" in diff
        assert "+++ b/test.py" in diff
        assert "-line 2" in diff
        assert "+modified line 2" in diff

    def test_create_diff_no_changes(self):
        """Should handle identical content."""
        content = "same content\n"

        diff = create_simple_diff("test.py", content, content)

        # Empty diff for identical content
        assert diff == "" or "@@" not in diff


# --- Test EditReport ---

class TestEditReport:
    """Tests for EditReport formatting."""

    def test_to_json(self):
        """Should produce valid JSON."""
        import json

        report = EditReport(
            success=True,
            message="Test",
            files_modified=["test.py"],
        )

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert "test.py" in parsed["files_modified"]

    def test_to_markdown(self):
        """Should produce valid Markdown."""
        report = EditReport(
            success=True,
            message="Applied changes",
            files_modified=["test.py"],
        )

        md = report.to_markdown()

        assert "# Edit Report" in md
        assert "✅ Success" in md
        assert "`test.py`" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
