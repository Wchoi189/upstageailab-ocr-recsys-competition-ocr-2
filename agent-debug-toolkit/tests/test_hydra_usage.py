"""
Tests for HydraUsageAnalyzer.
"""

import json
import pytest
from agent_debug_toolkit.analyzers.hydra_usage import HydraUsageAnalyzer
from tests.fixtures.sample_code import SAMPLE_HYDRA_USAGE


class TestHydraUsageAnalyzer:
    """Tests for HydraUsageAnalyzer."""

    def test_detects_hydra_imports(self):
        """Should detect hydra-related imports."""
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source(SAMPLE_HYDRA_USAGE, "test.py")

        # Should find hydra imports
        import_results = [r for r in report.results if r.category == "import"]
        assert len(import_results) >= 2  # hydra and hydra.utils

        patterns = [r.pattern for r in import_results]
        assert any("hydra" in p for p in patterns)
        assert any("instantiate" in p for p in patterns)

    def test_detects_hydra_main_decorator(self):
        """Should detect @hydra.main decorated functions."""
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source(SAMPLE_HYDRA_USAGE, "test.py")

        entrypoint_results = [r for r in report.results if r.category == "entry_point"]
        assert len(entrypoint_results) == 1
        assert "main" in entrypoint_results[0].pattern

    def test_detects_instantiate_calls(self):
        """Should detect hydra.utils.instantiate() calls."""
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source(SAMPLE_HYDRA_USAGE, "test.py")

        inst_results = [r for r in report.results if r.category == "instantiation"]
        assert len(inst_results) >= 2  # model and optimizer

        patterns = [r.pattern for r in inst_results]
        assert any("cfg.model" in p for p in patterns)
        assert any("cfg.optimizer" in p or "optimizer" in p.lower() for p in patterns)

    def test_detects_target_patterns(self):
        """Should detect _target_ patterns in dictionaries."""
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source(SAMPLE_HYDRA_USAGE, "test.py")

        target_results = [r for r in report.results if r.category == "config_pattern"]
        assert len(target_results) >= 1

        patterns = [r.pattern for r in target_results]
        assert any("_target_" in p for p in patterns)

    def test_generates_summary_with_categories(self):
        """Should generate summary with finding categories."""
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source(SAMPLE_HYDRA_USAGE, "test.py")

        assert "total_findings" in report.summary
        assert "by_category" in report.summary
        assert report.summary["total_findings"] > 0

    def test_empty_file(self):
        """Should handle empty files gracefully."""
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source("", "empty.py")

        assert len(report.results) == 0
        assert report.summary.get("total_findings", 0) == 0

    def test_no_hydra_usage(self):
        """Should handle files without Hydra usage."""
        code = '''
def regular_function():
    x = 1 + 2
    return x
'''
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source(code, "regular.py")

        assert len(report.results) == 0

    def test_syntax_error_handling(self):
        """Should handle files with syntax errors."""
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source("def broken(", "error.py")

        assert "error" in report.summary


class TestHydraUsageAnalyzerOutput:
    """Tests for output formatting."""

    def test_json_output(self):
        """Should produce valid JSON."""
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source(SAMPLE_HYDRA_USAGE, "test.py")

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert "analyzer_name" in parsed
        assert parsed["analyzer_name"] == "HydraUsageAnalyzer"
        assert "results" in parsed
        assert isinstance(parsed["results"], list)

    def test_markdown_output(self):
        """Should produce valid Markdown."""
        analyzer = HydraUsageAnalyzer()
        report = analyzer.analyze_source(SAMPLE_HYDRA_USAGE, "test.py")

        md = report.to_markdown()

        assert "# HydraUsageAnalyzer" in md
        assert "Findings" in md or "Finding" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
