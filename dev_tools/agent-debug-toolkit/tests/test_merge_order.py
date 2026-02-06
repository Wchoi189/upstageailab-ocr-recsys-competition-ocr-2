"""
Tests for MergeOrderTracker.
"""

import pytest
from agent_debug_toolkit.analyzers.merge_order import MergeOrderTracker
from tests.fixtures.sample_code import SAMPLE_CONFIG_ACCESS


class TestMergeOrderTracker:
    """Tests for MergeOrderTracker."""

    def test_detects_omega_conf_merge(self):
        """Should detect OmegaConf.merge() calls."""
        analyzer = MergeOrderTracker()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        merge_results = [r for r in report.results if r.category == "omegaconf_merge"]
        assert len(merge_results) >= 3  # The sample has 3 merge operations

    def test_detects_omega_conf_create(self):
        """Should detect OmegaConf.create() calls."""
        analyzer = MergeOrderTracker()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        create_results = [r for r in report.results if r.category == "omegaconf_create"]
        assert len(create_results) >= 1

    def test_tracks_merge_priority(self):
        """Should assign increasing priority to later merges."""
        analyzer = MergeOrderTracker()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        merge_results = [r for r in report.results if r.category == "omegaconf_merge"]
        priorities = [r.metadata.get("priority", 0) for r in merge_results]

        # Priorities should be sequential within a function
        assert len(priorities) > 0
        # Later merges have higher priority (they win)
        assert max(priorities) > 1

    def test_identifies_winner_on_conflict(self):
        """Should identify which config wins on conflicts."""
        analyzer = MergeOrderTracker()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        # Check that sources are tracked
        for result in report.results:
            if result.category == "omegaconf_merge":
                sources = result.metadata.get("sources", [])
                assert len(sources) >= 2
                # Last source wins
                break

    def test_explain_precedence(self):
        """Should generate precedence explanation."""
        analyzer = MergeOrderTracker()
        analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        explanation = analyzer.explain_precedence()

        assert "Precedence" in explanation
        assert "Priority" in explanation

    def test_get_merge_sequence(self):
        """Should return ordered merge sequence."""
        analyzer = MergeOrderTracker()
        analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        sequence = analyzer.get_merge_sequence()

        assert len(sequence) >= 3
        # Should be ordered by line number
        lines = [op.line for op in sequence]
        assert lines == sorted(lines)

    def test_summary_includes_merge_stats(self):
        """Should include merge statistics in summary."""
        analyzer = MergeOrderTracker()
        report = analyzer.analyze_source(SAMPLE_CONFIG_ACCESS, "test.py")

        assert "merge_operations" in report.summary
        assert "create_operations" in report.summary
        assert "total_omegaconf_ops" in report.summary


class TestMergeOrderTrackerRealWorld:
    """Tests using patterns from actual OCRModel."""

    def test_three_priority_merge_pattern(self):
        """Should correctly identify the 3-priority merge pattern."""
        code = """
def _prepare_component_configs(self, cfg):
    merged_config = OmegaConf.create({})

    # Priority 1: Architecture config
    if self.architecture_config_obj:
        merged_config = OmegaConf.merge(merged_config, self.architecture_config_obj)

    # Priority 2: Top-level overrides
    top_level = {"decoder": cfg.decoder}
    merged_config = OmegaConf.merge(merged_config, top_level)

    # Priority 3: Explicit overrides (highest priority)
    if cfg.component_overrides:
        merged_config = OmegaConf.merge(merged_config, cfg.component_overrides)

    return merged_config
"""
        analyzer = MergeOrderTracker()
        report = analyzer.analyze_source(code, "test.py")

        merge_results = [r for r in report.results if r.category == "omegaconf_merge"]

        # Should have 3 merges
        assert len(merge_results) == 3

        # Last merge should have highest priority
        priorities = [r.metadata["priority"] for r in merge_results]
        assert priorities[-1] == max(priorities)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
