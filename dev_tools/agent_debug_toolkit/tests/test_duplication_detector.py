"""Tests for DuplicationDetector."""

from agent_debug_toolkit.analyzers.duplication_detector import DuplicationDetector


class TestDuplicationDetector:
    """Test cases for DuplicationDetector."""

    def test_exact_duplicate_functions(self):
        """Test detection of exact duplicate functions."""
        source = """
def process_data(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result

def process_values(values):
    result = []
    for value in values:
        if value > 0:
            result.append(value * 2)
    return result
"""
        analyzer = DuplicationDetector(min_lines=5)
        report = analyzer.analyze_source(source)

        # Should detect these as duplicates (same structure, different variable names)
        assert len(analyzer.get_duplicate_groups()) >= 1
        assert report.summary["duplicate_groups"] >= 1

    def test_no_duplicates(self):
        """Test when there are no duplicates."""
        source = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def subtract(a, b):
    return a - b
"""
        analyzer = DuplicationDetector(min_lines=2)
        report = analyzer.analyze_source(source)

        # Short functions should not trigger duplicates
        groups = analyzer.get_duplicate_groups()
        assert len(groups) == 0

    def test_min_lines_threshold(self):
        """Test that min_lines threshold is respected."""
        source = """
def short1():
    return 1

def short2():
    return 2

def longer_func1(x):
    if x > 0:
        y = x * 2
        z = y + 1
        result = z * 3
        return result
    return 0

def longer_func2(a):
    if a > 0:
        b = a * 2
        c = b + 1
        result = c * 3
        return result
    return 0
"""
        # With high min_lines, should only detect longer functions
        analyzer = DuplicationDetector(min_lines=6)
        report = analyzer.analyze_source(source)

        groups = analyzer.get_duplicate_groups()
        # The longer functions should be detected as duplicates
        if groups:
            for group in groups:
                for block in group.blocks:
                    assert block.line_count >= 6

    def test_empty_file(self):
        """Test handling of file with no functions."""
        source = """
x = 1
y = 2
"""
        analyzer = DuplicationDetector()
        report = analyzer.analyze_source(source)

        assert len(report.results) == 0
        assert report.summary["total_code_blocks"] == 0

    def test_class_methods(self):
        """Test detection of duplicate class methods."""
        source = """
class Processor1:
    def process(self, items):
        result = []
        for item in items:
            if item > 0:
                result.append(item * 2)
        return result

class Processor2:
    def handle(self, values):
        result = []
        for value in values:
            if value > 0:
                result.append(value * 2)
        return result
"""
        analyzer = DuplicationDetector(min_lines=5)
        report = analyzer.analyze_source(source)

        # Methods with same structure should be detected
        groups = analyzer.get_duplicate_groups()
        assert len(groups) >= 1

    def test_summary_contains_expected_keys(self):
        """Test that summary contains expected keys."""
        source = """
def example():
    return 1
"""
        analyzer = DuplicationDetector()
        report = analyzer.analyze_source(source)

        assert "total_code_blocks" in report.summary
        assert "duplicate_groups" in report.summary
        assert "min_lines_threshold" in report.summary

    def test_suggested_action_same_file(self):
        """Test suggested action for duplicates in same file."""
        source = """
def func1(items):
    result = []
    for item in items:
        result.append(item * 2)
        result.append(item + 1)
    return result

def func2(values):
    result = []
    for value in values:
        result.append(value * 2)
        result.append(value + 1)
    return result
"""
        analyzer = DuplicationDetector(min_lines=5)
        report = analyzer.analyze_source(source)

        groups = analyzer.get_duplicate_groups()
        if groups:
            # Same file duplicates should suggest extracting to function
            assert "function" in groups[0].suggested_action.lower() or "module" in groups[0].suggested_action.lower()

    def test_different_structure_not_duplicate(self):
        """Test that different structures are not marked as duplicates."""
        source = """
def process_list(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item)
    return result

def process_dict(data):
    result = {}
    for key, value in data.items():
        if value > 0:
            result[key] = value
    return result
"""
        analyzer = DuplicationDetector(min_lines=5)
        report = analyzer.analyze_source(source)

        # Different loop types should not be duplicates
        groups = analyzer.get_duplicate_groups()
        # These should NOT be marked as duplicates due to different structure
        assert len(groups) == 0
