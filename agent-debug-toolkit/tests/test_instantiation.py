"""
Tests for ComponentInstantiationTracker.
"""

import json
import pytest
from agent_debug_toolkit.analyzers.instantiation import ComponentInstantiationTracker
from tests.fixtures.sample_code import SAMPLE_COMPONENT_INSTANTIATION


class TestComponentInstantiationTracker:
    """Tests for ComponentInstantiationTracker."""

    def test_detects_factory_calls(self):
        """Should detect get_*_by_cfg patterns."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        factory_results = [r for r in report.results if r.category == "factory_call"]
        assert len(factory_results) >= 3  # encoder, decoder, head

        call_names = [r.metadata.get("call_name", "") for r in factory_results]
        assert any("get_encoder_by_cfg" in c for c in call_names)
        assert any("get_decoder_by_cfg" in c for c in call_names)
        assert any("get_head_by_cfg" in c for c in call_names)

    def test_detects_registry_calls(self):
        """Should detect registry.create_* patterns."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        # Registry calls are categorized as factory_call
        registry_results = [r for r in report.results if "create_architecture_components" in r.pattern]
        assert len(registry_results) >= 1

        patterns = [r.pattern for r in registry_results]
        assert any("create_architecture_components" in p for p in patterns)

    def test_identifies_component_type(self):
        """Should identify the component type from call names."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        component_types = [r.metadata.get("component_type") for r in report.results]
        assert "encoder" in component_types
        assert "decoder" in component_types
        assert "head" in component_types

    def test_tracks_config_source(self):
        """Should track what config is passed to the factory."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        # Find decoder factory call
        decoder_results = [
            r for r in report.results
            if r.metadata.get("component_type") == "decoder"
        ]
        assert len(decoder_results) >= 1

        # Should track that cfg.decoder is the config source
        config_sources = [r.metadata.get("config_source", "") for r in decoder_results]
        assert any("cfg.decoder" in cs or "decoder" in cs.lower() for cs in config_sources)

    def test_tracks_target_variable(self):
        """Should track what variable receives the component."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        # Find assignments
        assignments = [r for r in report.results if r.metadata.get("target_variable")]
        assert len(assignments) >= 1

        targets = [r.metadata.get("target_variable", "") for r in assignments]
        assert any("encoder" in t.lower() for t in targets)

    def test_tracks_class_context(self):
        """Should track class context of instantiations."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        for result in report.results:
            if result.metadata.get("class"):
                assert result.metadata["class"] == "ModelFactory"
                break

    def test_find_component_source(self):
        """Should find where a component is instantiated."""
        tracker = ComponentInstantiationTracker()
        tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        decoder_sources = tracker.find_component_source("decoder")
        assert len(decoder_sources) >= 1

        # find_component_source returns list of dicts
        patterns = [r.get("pattern", "") for r in decoder_sources]
        assert any("decoder" in p.lower() for p in patterns)

    def test_get_instantiation_flow(self):
        """Should return ordered flow of instantiations."""
        tracker = ComponentInstantiationTracker()
        tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        flow = tracker.get_instantiation_flow()
        assert len(flow) >= 3

        # Should be in line order (flow returns list of dicts)
        lines = [f["line"] for f in flow]
        assert lines == sorted(lines)

    def test_generates_summary_with_breakdown(self):
        """Should generate summary with component breakdown."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        assert "total_findings" in report.summary
        assert "by_category" in report.summary
        assert "by_component_type" in report.summary
        assert report.summary["total_findings"] > 0

    def test_empty_file(self):
        """Should handle empty files gracefully."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source("", "empty.py")

        assert len(report.results) == 0
        assert report.summary.get("total_findings", 0) == 0

    def test_no_instantiations(self):
        """Should handle files without instantiation patterns."""
        code = '''
def regular_function():
    x = 1 + 2
    return x
'''
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(code, "regular.py")

        assert len(report.results) == 0

    def test_syntax_error_handling(self):
        """Should handle files with syntax errors."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source("def broken(", "error.py")

        assert "error" in report.summary

    def test_additional_factories(self):
        """Should support custom factory function names."""
        code = '''
class CustomModel:
    def setup(self, cfg):
        self.backbone = create_custom_backbone(cfg.backbone)
'''
        tracker = ComponentInstantiationTracker(
            additional_factories=["create_custom_backbone"]
        )
        report = tracker.analyze_source(code, "test.py")

        # Should find the custom factory call
        factory_calls = [r for r in report.results if "create_custom_backbone" in r.pattern]
        assert len(factory_calls) >= 1


class TestComponentInstantiationTrackerOutput:
    """Tests for output formatting."""

    def test_json_output(self):
        """Should produce valid JSON."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert "analyzer_name" in parsed
        assert parsed["analyzer_name"] == "ComponentInstantiationTracker"
        assert "results" in parsed
        assert isinstance(parsed["results"], list)

    def test_markdown_output(self):
        """Should produce valid Markdown."""
        tracker = ComponentInstantiationTracker()
        report = tracker.analyze_source(SAMPLE_COMPONENT_INSTANTIATION, "test.py")

        md = report.to_markdown()

        assert "# ComponentInstantiationTracker" in md
        assert "Findings" in md or "Finding" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
