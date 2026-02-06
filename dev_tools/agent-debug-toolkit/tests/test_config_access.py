"""
Tests for ConfigAccessAnalyzer.
"""

import pytest
from agent_debug_toolkit.analyzers.config_access import ConfigAccessAnalyzer
from tests.fixtures.sample_code import SAMPLE_CONFIG_ACCESS


class TestConfigAccessAnalyzer:
    """Tests for ConfigAccessAnalyzer."""

    def test_detects_attribute_access(self):
        """Should detect cfg.X patterns."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        # Should find multiple config accesses
        assert len(report.results) > 0

        # Should find cfg.encoder access
        patterns = [r.pattern for r in report.results]
        assert any("cfg.encoder" in p for p in patterns)
        assert any("cfg.decoder" in p for p in patterns)

    def test_detects_self_cfg_pattern(self):
        """Should detect self.cfg.X patterns."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        patterns = [r.pattern for r in report.results]
        assert any("self.cfg" in p or "self.architecture_config_obj" in p for p in patterns)

    def test_detects_subscript_access(self):
        """Should detect cfg['key'] patterns."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        patterns = [r.pattern for r in report.results]
        assert any("[" in p and "]" in p for p in patterns)

    def test_detects_getattr_hasattr(self):
        """Should detect getattr/hasattr patterns."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        categories = [r.category for r in report.results]
        assert "getattr_call" in categories or "hasattr_check" in categories

    def test_categorizes_component_access(self):
        """Should identify component-related config access."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        categories = [r.category for r in report.results]
        assert "component_access" in categories

    def test_tracks_class_function_context(self):
        """Should track class and function context."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        # Should have context information
        for result in report.results:
            if result.metadata.get("class"):
                assert result.metadata["class"] == "OCRModel"
                break

    def test_filter_by_component(self):
        """Should filter results by component name."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        decoder_results = report.filter_by_component("decoder")
        assert len(decoder_results) > 0
        assert all("decoder" in r.pattern.lower() for r in decoder_results)

    def test_generates_summary(self):
        """Should generate summary with statistics."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        assert "total_findings" in report.summary
        assert "by_category" in report.summary
        assert report.summary["total_findings"] > 0

    def test_empty_file(self):
        """Should handle empty files gracefully."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source("", "empty.py")

        assert len(report.results) == 0
        assert report.summary.get("total_findings", 0) == 0

    def test_syntax_error_handling(self):
        """Should handle files with syntax errors."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source("def broken(", "error.py")

        assert "error" in report.summary


class TestConfigAccessAnalyzerOutput:
    """Tests for output formatting."""

    def test_json_output(self):
        """Should produce valid JSON."""
        import json

        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert "analyzer_name" in parsed
        assert "results" in parsed
        assert isinstance(parsed["results"], list)

    def test_markdown_output(self):
        """Should produce valid Markdown."""
        analyzer = ConfigAccessAnalyzer()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        md = report.to_markdown()

        assert "# ConfigAccessAnalyzer" in md
        assert "Findings" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
