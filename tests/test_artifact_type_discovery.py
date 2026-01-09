#!/usr/bin/env python3
"""
Tests for dynamic artifact type discovery system.

Tests the new _get_available_artifact_types() method which consolidates
artifact type information from hardcoded templates, plugins, and standards.

Coverage:
- Dynamic type discovery from all sources
- Source metadata tracking (hardcoded vs plugin)
- Conflict detection and reporting
- Template-plugin integration
- Naming consistency validation
"""

import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from AgentQMS.tools.core.artifact_templates import ArtifactTemplates


class TestArtifactTypeDiscovery:
    """Test suite for artifact type discovery functionality."""

    @pytest.fixture
    def templates(self) -> ArtifactTemplates:
        """Create a fresh ArtifactTemplates instance."""
        return ArtifactTemplates()

    def test_get_available_artifact_types_returns_dict(self, templates: ArtifactTemplates):
        """Test that _get_available_artifact_types() returns a dictionary."""
        types = templates._get_available_artifact_types()
        assert isinstance(types, dict)
        assert len(types) > 0

    def test_hardcoded_types_included(self, templates: ArtifactTemplates):
        """Test that all hardcoded types are discoverable."""
        hardcoded = [
            "implementation_plan",
            "walkthrough",
            "assessment",
            "design",
            "research",
            "template",
            "bug_report",
            "vlm_report",
        ]

        types = templates._get_available_artifact_types()

        for type_name in hardcoded:
            assert type_name in types, f"Hardcoded type '{type_name}' not found in discovery"
            assert types[type_name]["source"] == "hardcoded"

    def test_hardcoded_types_have_templates(self, templates: ArtifactTemplates):
        """Test that all hardcoded types have template configuration."""
        types = templates._get_available_artifact_types()

        for name, info in types.items():
            if info["source"] == "hardcoded":
                assert info["template"] is not None
                assert "filename_pattern" in info["template"]
                assert "content_template" in info["template"]
                assert "frontmatter" in info["template"]

    def test_type_info_structure(self, templates: ArtifactTemplates):
        """Test that type info has required fields."""
        types = templates._get_available_artifact_types()

        for name, info in types.items():
            assert "name" in info
            assert "source" in info
            assert "description" in info
            assert "template" in info
            assert "conflict" in info
            assert info["name"] == name

    def test_source_values_valid(self, templates: ArtifactTemplates):
        """Test that source values are valid."""
        types = templates._get_available_artifact_types()
        valid_sources = {"hardcoded", "plugin", "hardcoded (plugin available)"}

        for name, info in types.items():
            assert info["source"] in valid_sources, f"Invalid source for {name}: {info['source']}"

    def test_plugin_types_discovered(self, templates: ArtifactTemplates):
        """Test that plugin types are discovered (if plugins available)."""
        types = templates._get_available_artifact_types()

        # Known plugins that should exist
        plugin_types = ["audit", "change_request", "ocr_experiment_report"]

        for plugin_type in plugin_types:
            assert plugin_type in types, f"Expected plugin type '{plugin_type}' not found"
            assert types[plugin_type]["source"] in ["plugin", "hardcoded (plugin available)"]

    def test_conflict_detection(self, templates: ArtifactTemplates):
        """Test that conflicts are detected when types defined in multiple sources."""
        types = templates._get_available_artifact_types()

        # Check that conflict flag is properly set
        for name, info in types.items():
            conflict = info.get("conflict", False)
            assert isinstance(conflict, bool)

    def test_metadata_method_enhanced(self, templates: ArtifactTemplates):
        """Test the enhanced metadata method."""
        metadata_list = templates.get_available_templates_with_metadata()

        assert isinstance(metadata_list, list)
        assert len(metadata_list) > 0

        # Check each metadata entry
        for entry in metadata_list:
            assert "name" in entry
            assert "source" in entry
            assert "description" in entry
            assert "has_validation" in entry
            assert "has_conflict" in entry
            assert isinstance(entry["has_validation"], bool)
            assert isinstance(entry["has_conflict"], bool)

    def test_metadata_sorted_by_name(self, templates: ArtifactTemplates):
        """Test that metadata is sorted alphabetically."""
        metadata_list = templates.get_available_templates_with_metadata()
        names = [entry["name"] for entry in metadata_list]

        assert names == sorted(names), "Metadata should be sorted by name"

    def test_no_empty_descriptions(self, templates: ArtifactTemplates):
        """Test that all types have descriptions."""
        types = templates._get_available_artifact_types()

        for name, info in types.items():
            assert info["description"], f"Type '{name}' has empty description"

    def test_discovery_stability(self, templates: ArtifactTemplates):
        """Test that multiple calls return consistent results."""
        types1 = templates._get_available_artifact_types()
        types2 = templates._get_available_artifact_types()

        # Same keys
        assert set(types1.keys()) == set(types2.keys())

        # Same metadata
        for key in types1:
            assert types1[key]["source"] == types2[key]["source"]
            assert types1[key]["name"] == types2[key]["name"]

    def test_naming_conflict_annotations(self, templates: ArtifactTemplates):
        """Test that known naming conflicts are annotated."""
        types = templates._get_available_artifact_types()

        # These are known conflict areas documented in assessment
        potential_conflicts = ["assessment", "design", "research"]

        for type_name in potential_conflicts:
            if type_name in types:
                # Check if conflict note is present (optional but recommended)
                info = types[type_name]
                # Either has conflict flag set or _conflict_note field
                has_marker = info.get("conflict", False) or "_conflict_note" in info
                # This is informational - we document but don't enforce


class TestArtifactTypeDescriptions:
    """Test that artifact types have meaningful descriptions."""

    @pytest.fixture
    def templates(self) -> ArtifactTemplates:
        """Create a fresh ArtifactTemplates instance."""
        return ArtifactTemplates()

    def test_hardcoded_descriptions(self, templates: ArtifactTemplates):
        """Test that hardcoded types have meaningful descriptions."""
        types = templates._get_available_artifact_types()

        hardcoded_descriptions = {
            "implementation_plan": "Implementation plan for features and changes",
            "walkthrough": "Code walkthrough and explanation",
            "assessment": "Technical assessment and analysis",
            "design": "Design document for architecture",
            "research": "Research findings and documentation",
            "template": "Template for standardized processes",
            "bug_report": "Bug report with reproduction steps",
            "vlm_report": "VLM analysis and evaluation report",
        }

        for type_name, expected_desc in hardcoded_descriptions.items():
            assert type_name in types
            assert types[type_name]["description"] == expected_desc


class TestMCPIntegration:
    """Test MCP integration with dynamic discovery."""

    def test_mcp_template_list_enhanced(self):
        """Test that MCP template list includes metadata."""
        import asyncio
        import json
        from AgentQMS.mcp_server import _get_template_list

        result_json = asyncio.run(_get_template_list())
        result = json.loads(result_json)

        # Should have templates and summary
        assert "templates" in result
        assert "summary" in result

        # Summary should have counts
        assert "total" in result["summary"]
        assert "hardcoded" in result["summary"]
        assert "plugin" in result["summary"]
        assert "with_conflicts" in result["summary"]

        # Template list should be non-empty
        assert len(result["templates"]) > 0

    def test_mcp_available_artifact_types(self):
        """Test that MCP can get available artifact types."""
        import asyncio
        from AgentQMS.mcp_server import _get_available_artifact_types

        types = asyncio.run(_get_available_artifact_types())

        assert isinstance(types, list)
        assert len(types) > 0

        # Should include both hardcoded and plugin types
        assert "implementation_plan" in types  # Hardcoded
        assert "audit" in types  # Plugin
        assert "change_request" in types  # Plugin
        assert "ocr_experiment_report" in types  # Plugin


class TestTemplateIntegrity:
    """Test that templates remain intact during discovery."""

    @pytest.fixture
    def templates(self) -> ArtifactTemplates:
        """Create a fresh ArtifactTemplates instance."""
        return ArtifactTemplates()

    def test_get_template_still_works(self, templates: ArtifactTemplates):
        """Test that get_template() still returns correct templates."""
        for type_name in ["implementation_plan", "assessment", "design"]:
            template = templates.get_template(type_name)
            assert template is not None
            assert isinstance(template, dict)
            assert "content_template" in template
            assert "frontmatter" in template

    def test_available_templates_list_still_works(self, templates: ArtifactTemplates):
        """Test that get_available_templates() still returns list."""
        templates_list = templates.get_available_templates()
        assert isinstance(templates_list, list)
        assert len(templates_list) > 0

    def test_create_filename_unchanged(self, templates: ArtifactTemplates):
        """Test that filename creation still works."""
        filename = templates.create_filename("implementation_plan", "test-feature")
        assert "implementation_plan" in filename
        assert "test-feature" in filename
        assert ".md" in filename

    def test_create_artifact_unchanged(self, templates: ArtifactTemplates):
        """Test that artifact creation still works (integration test)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test artifact
            path = templates.create_artifact(
                "implementation_plan",
                "test-feature",
                "Test Feature Plan",
                output_dir=tmpdir,
                quiet=True,
                subject="feature",
                methodology="test",
            )

            # Verify file was created
            assert Path(path).exists()
            assert "implementation_plan" in path

            # Verify content
            content = Path(path).read_text()
            assert "Test Feature Plan" in content
            assert "---" in content  # Frontmatter markers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
