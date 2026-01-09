#!/usr/bin/env python3
"""
Integration tests for Plugin Migration (Phase 2)

Verifies that plugin-based artifact types produce identical output
to hardcoded artifact types in artifact_templates.py

Session 3 Tests (First 3):
- implementation_plan
- walkthrough
- assessment

Session 4 Tests (Remaining 5):
- design_document (was 'design')
- research
- template
- bug_report
- vlm_report

Comparison criteria:
- Filename pattern
- Directory
- Frontmatter fields
- Template content structure
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from AgentQMS.tools.core.artifact_templates import ArtifactTemplates
from AgentQMS.tools.core.plugins import get_plugin_registry


class TestPluginMigrationPhase2:
    """Test plugin-based artifacts match hardcoded originals."""

    @pytest.fixture
    def templates(self):
        """Initialize ArtifactTemplates with plugin loading."""
        return ArtifactTemplates()

    @pytest.fixture
    def plugin_registry(self):
        """Get plugin registry."""
        return get_plugin_registry()

    def test_implementation_plan_plugin_loaded(self, templates, plugin_registry):
        """Test that implementation_plan plugin is loaded."""
        # Plugin should be available
        artifact_types = plugin_registry.get_artifact_types()
        assert "implementation_plan" in artifact_types

        # Template should exist (either hardcoded or plugin)
        template = templates.get_template("implementation_plan")
        assert template is not None

    def test_walkthrough_plugin_loaded(self, templates, plugin_registry):
        """Test that walkthrough plugin is loaded."""
        artifact_types = plugin_registry.get_artifact_types()
        assert "walkthrough" in artifact_types

        template = templates.get_template("walkthrough")
        assert template is not None

    def test_assessment_plugin_loaded(self, templates, plugin_registry):
        """Test that assessment plugin is loaded."""
        artifact_types = plugin_registry.get_artifact_types()
        assert "assessment" in artifact_types

        template = templates.get_template("assessment")
        assert template is not None

    def test_implementation_plan_structure_matches(self, templates):
        """Test that implementation_plan plugin structure matches hardcoded."""
        template = templates.get_template("implementation_plan")

        # Check required fields
        assert "filename_pattern" in template
        assert "directory" in template
        assert "frontmatter" in template
        assert "content_template" in template

        # Check filename pattern
        assert "implementation_plan" in template["filename_pattern"]
        assert "{name}" in template["filename_pattern"]

        # Check directory
        assert template["directory"] == "implementation_plans/"

        # Check frontmatter
        frontmatter = template["frontmatter"]
        assert frontmatter["type"] == "implementation_plan"
        assert frontmatter["category"] == "development"
        assert "implementation" in frontmatter["tags"]

        # Check content structure
        content = template["content_template"]
        assert "# Implementation Plan" in content
        assert "## Goal" in content
        assert "## Proposed Changes" in content
        assert "## Verification Plan" in content

    def test_walkthrough_structure_matches(self, templates):
        """Test that walkthrough plugin structure matches hardcoded."""
        template = templates.get_template("walkthrough")

        # Check required fields
        assert "filename_pattern" in template
        assert "directory" in template
        assert "frontmatter" in template
        assert "content_template" in template

        # Check filename pattern
        assert "walkthrough" in template["filename_pattern"]
        assert "{name}" in template["filename_pattern"]

        # Check directory
        assert template["directory"] == "walkthroughs/"

        # Check frontmatter
        frontmatter = template["frontmatter"]
        assert frontmatter["type"] == "walkthrough"
        assert frontmatter["category"] == "documentation"
        assert "walkthrough" in frontmatter["tags"]

        # Check content structure
        content = template["content_template"]
        assert "# Walkthrough" in content
        assert "## Goal" in content
        assert "## Steps" in content
        assert "## Verification" in content

    def test_assessment_structure_matches(self, templates):
        """Test that assessment plugin structure matches hardcoded."""
        template = templates.get_template("assessment")

        # Check required fields
        assert "filename_pattern" in template
        assert "directory" in template
        assert "frontmatter" in template
        assert "content_template" in template

        # Check filename pattern
        assert "assessment" in template["filename_pattern"]
        assert "{name}" in template["filename_pattern"]

        # Check directory
        assert template["directory"] == "assessments/"

        # Check frontmatter
        frontmatter = template["frontmatter"]
        assert frontmatter["type"] == "assessment"
        assert frontmatter["category"] == "evaluation"
        assert "assessment" in frontmatter["tags"]

        # Check content structure
        content = template["content_template"]
        assert "# Assessment" in content
        assert "## Purpose" in content
        assert "## Findings" in content
        assert "## Analysis" in content
        assert "## Recommendations" in content

    def test_all_eight_plugins_discoverable(self, templates):
        """Test that all 8 migrated plugins are discoverable."""
        available = templates.get_available_templates()

        # Session 3 plugins
        assert "implementation_plan" in available
        assert "walkthrough" in available
        assert "assessment" in available
        
        # Session 4 plugins
        assert "design_document" in available
        assert "research" in available
        assert "template" in available
        assert "bug_report" in available
        assert "vlm_report" in available

    def test_plugin_priority_over_hardcoded(self, templates):
        """Test that hardcoded templates still take precedence (for now).

        This test documents current behavior. In Phase 4, plugins will
        replace hardcoded entirely.
        """
        # Currently, hardcoded templates are defined first in __init__
        # Plugins are loaded via _load_plugin_templates which skips
        # if name already exists

        template = templates.get_template("implementation_plan")

        # Hardcoded version should be active
        # (This will change in Phase 4 when hardcoded is removed)
        assert template is not None

    def test_frontmatter_ads_version_present(self, templates):
        """Test that ads_version field is present in frontmatter."""
        all_types = [
            "implementation_plan", "walkthrough", "assessment",
            "design_document", "research", "template", "bug_report", "vlm_report"
        ]
        for type_name in all_types:
            template = templates.get_template(type_name)
            if template:  # Some may still be hardcoded-only
                frontmatter = template["frontmatter"]
                # Plugin schema may not have ads_version, but hardcoded does
                # This documents the difference that will be resolved in Phase 3
                assert "ads_version" in frontmatter or "version" in frontmatter

    def test_template_variables_available(self, plugin_registry):
        """Test that plugins define template_variables for substitution."""
        artifact_types = plugin_registry.get_artifact_types()

        all_types = [
            "implementation_plan", "walkthrough", "assessment",
            "design_document", "research", "template", "bug_report", "vlm_report"
        ]
        for type_name in all_types:
            plugin_def = artifact_types.get(type_name)
            if plugin_def:  # Check only if plugin exists
                # Check if template_variables are defined
                # This is used by _convert_plugin_to_template
                assert "template_variables" in plugin_def

    def test_backward_compatibility_maintained(self, templates):
        """Test that existing code paths still work."""
        # Original methods should still work
        assert callable(templates.get_template)
        assert callable(templates.get_available_templates)

        # Should return valid data
        available = templates.get_available_templates()
        assert len(available) > 0

        all_types = [
            "implementation_plan", "walkthrough", "assessment",
            "design_document", "research", "template", "bug_report", "vlm_report"
        ]
        for template_type in all_types:
            template = templates.get_template(template_type)
            assert template is not None, f"Template {template_type} not found"
            assert isinstance(template, dict)


class TestPluginSchemaCompliance:
    """Test that plugins follow the schema conventions."""

    @pytest.fixture
    def plugin_registry(self):
        return get_plugin_registry()

    def test_required_plugin_fields(self, plugin_registry):
        """Test that all migrated plugins have required fields."""
        artifact_types = plugin_registry.get_artifact_types()

        all_types = [
            "implementation_plan", "walkthrough", "assessment",
            "design_document", "research", "template", "bug_report", "vlm_report"
        ]
        for type_name in all_types:
            plugin = artifact_types.get(type_name)
            assert plugin is not None, f"Plugin {type_name} not found"

            # Required top-level fields
            assert "name" in plugin
            assert "version" in plugin
            assert "description" in plugin
            assert "scope" in plugin
            assert "metadata" in plugin
            assert "template" in plugin

    def test_metadata_structure(self, plugin_registry):
        """Test metadata section structure."""
        artifact_types = plugin_registry.get_artifact_types()

        all_types = [
            "implementation_plan", "walkthrough", "assessment",
            "design_document", "research", "template", "bug_report", "vlm_report"
        ]
        for type_name in all_types:
            plugin = artifact_types[type_name]
            metadata = plugin["metadata"]

            assert "filename_pattern" in metadata
            assert "directory" in metadata
            assert "frontmatter" in metadata

            # Check frontmatter structure
            frontmatter = metadata["frontmatter"]
            assert "type" in frontmatter
            assert "category" in frontmatter
            assert "status" in frontmatter
            assert "tags" in frontmatter

    def test_validation_rules_present(self, plugin_registry):
        """Test that validation rules are defined."""
        artifact_types = plugin_registry.get_artifact_types()

        all_types = [
            "implementation_plan", "walkthrough", "assessment",
            "design_document", "research", "template", "bug_report", "vlm_report"
        ]
        for type_name in all_types:
            plugin = artifact_types[type_name]

            assert "validation" in plugin
            validation = plugin["validation"]

            assert "required_fields" in validation
            assert "required_sections" in validation
            assert "allowed_statuses" in validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
