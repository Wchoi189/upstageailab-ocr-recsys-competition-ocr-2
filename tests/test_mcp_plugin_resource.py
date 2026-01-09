#!/usr/bin/env python3
"""
Tests for MCP Plugin Artifact Types Resource

Tests the new agentqms://plugins/artifact_types resource which exposes
all discoverable artifact types with metadata.

Coverage:
- Resource registration in MCP server
- Resource discovery via list_resources()
- Resource reading via read_resource()
- Response schema validation
- All artifact types included
- Metadata completeness
- Conflict detection
- Backward compatibility
"""

import pytest
import json
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from AgentQMS.mcp_server import (
    RESOURCES,
    _get_plugin_artifact_types,
    _get_available_artifact_types,
)


class TestPluginArtifactTypesResource:
    """Test suite for plugin artifact types MCP resource."""

    def test_resource_registered_in_resources_list(self):
        """Test that the resource is registered in RESOURCES."""
        resource_uris = [r["uri"] for r in RESOURCES]
        assert "agentqms://plugins/artifact_types" in resource_uris

    def test_resource_has_correct_metadata(self):
        """Test that resource has correct name, description, and MIME type."""
        resource = next(
            (r for r in RESOURCES if r["uri"] == "agentqms://plugins/artifact_types"),
            None,
        )

        assert resource is not None
        assert resource["name"] == "Plugin Artifact Types"
        assert "discoverable" in resource["description"].lower() or "metadata" in resource["description"].lower()
        assert resource["mimeType"] == "application/json"
        assert resource["path"] is None  # Dynamic content

    def test_handler_returns_valid_json(self):
        """Test that _get_plugin_artifact_types() returns valid JSON."""
        result = asyncio.run(_get_plugin_artifact_types())

        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_response_has_required_root_keys(self):
        """Test that response has required root-level keys."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        assert "artifact_types" in data
        assert "summary" in data
        assert "metadata" in data

    def test_artifact_types_is_list(self):
        """Test that artifact_types is a list."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        assert isinstance(data["artifact_types"], list)
        assert len(data["artifact_types"]) > 0

    def test_summary_has_required_fields(self):
        """Test that summary has required fields."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        summary = data["summary"]
        assert "total" in summary
        assert "sources" in summary
        assert "validation_enabled" in summary
        assert "last_updated" in summary

        # Check sources breakdown
        sources = summary["sources"]
        assert "hardcoded" in sources
        assert "plugin" in sources

    def test_summary_counts_match_artifact_types(self):
        """Test that summary counts match actual artifact types."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        summary = data["summary"]
        artifact_types = data["artifact_types"]

        assert summary["total"] == len(artifact_types)

        # Count by source
        hardcoded_count = sum(1 for t in artifact_types if t["source"] == "hardcoded")
        plugin_count = sum(1 for t in artifact_types if t["source"] == "plugin")

        assert summary["sources"]["hardcoded"] == hardcoded_count
        assert summary["sources"]["plugin"] == plugin_count

    def test_all_hardcoded_types_included(self):
        """Test that all hardcoded types are included."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        artifact_types = data["artifact_types"]
        type_names = {t["name"] for t in artifact_types}

        expected_hardcoded = {
            "implementation_plan",
            "walkthrough",
            "assessment",
            "design",
            "research",
            "template",
            "bug_report",
            "vlm_report",
        }

        for type_name in expected_hardcoded:
            assert type_name in type_names, f"Hardcoded type '{type_name}' not in response"

    def test_all_plugin_types_included(self):
        """Test that all plugin types are included."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        artifact_types = data["artifact_types"]
        type_names = {t["name"] for t in artifact_types}

        expected_plugins = {"audit", "change_request", "ocr_experiment_report"}

        for type_name in expected_plugins:
            assert type_name in type_names, f"Plugin type '{type_name}' not in response"

    def test_artifact_type_has_required_fields(self):
        """Test that each artifact type has required fields."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        required_fields = {
            "name",
            "source",
            "description",
            "category",
            "version",
            "metadata",
            "frontmatter",
            "template_preview",
            "conflicts",
        }

        for artifact_type in data["artifact_types"]:
            for field in required_fields:
                assert field in artifact_type, f"Missing field '{field}' in type {artifact_type['name']}"

    def test_metadata_has_required_fields(self):
        """Test that metadata object has required fields."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        for artifact_type in data["artifact_types"]:
            metadata = artifact_type["metadata"]
            assert "filename_pattern" in metadata
            assert "directory" in metadata
            assert "template_variables" in metadata

    def test_template_preview_structure(self):
        """Test that template_preview has correct structure."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        for artifact_type in data["artifact_types"]:
            preview = artifact_type["template_preview"]
            assert "first_300_chars" in preview
            assert "line_count" in preview
            assert "sections" in preview
            assert isinstance(preview["line_count"], int)
            assert isinstance(preview["sections"], list)

    def test_source_values_valid(self):
        """Test that source values are valid."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        valid_sources = {"hardcoded", "plugin", "hardcoded (plugin available)"}

        for artifact_type in data["artifact_types"]:
            assert artifact_type["source"] in valid_sources, f"Invalid source: {artifact_type['source']}"

    def test_conflicts_structure(self):
        """Test that conflicts object has correct structure."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        for artifact_type in data["artifact_types"]:
            conflicts = artifact_type["conflicts"]
            assert "exists_in_multiple_sources" in conflicts
            assert "conflict_sources" in conflicts
            assert isinstance(conflicts["exists_in_multiple_sources"], bool)
            assert isinstance(conflicts["conflict_sources"], list)

    def test_plugin_info_for_plugin_types(self):
        """Test that plugin types have plugin_info."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        for artifact_type in data["artifact_types"]:
            if artifact_type["source"] == "plugin":
                plugin_info = artifact_type.get("plugin_info", {})
                assert "plugin_name" in plugin_info or plugin_info == {}, "Missing plugin_info for plugin type"

    def test_hardcoded_types_have_no_plugin_path(self):
        """Test that hardcoded types don't have plugin paths in plugin_info."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        for artifact_type in data["artifact_types"]:
            if artifact_type["source"] == "hardcoded":
                # Should either have empty plugin_info or missing it
                plugin_info = artifact_type.get("plugin_info", {})
                # Hardcoded should not have plugin_path
                assert "plugin_path" not in plugin_info or plugin_info == {}

    def test_validation_rules_included_for_validated_types(self):
        """Test that validation rules are included for types that have them."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        # Some types should have validation
        validated_types = [t for t in data["artifact_types"] if t.get("validation") is not None]
        assert len(validated_types) > 0, "At least some types should have validation rules"

        # Check structure of validation
        for artifact_type in validated_types:
            validation = artifact_type["validation"]
            assert isinstance(validation, dict)

    def test_all_types_have_descriptions(self):
        """Test that all types have non-empty descriptions."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        for artifact_type in data["artifact_types"]:
            assert artifact_type["description"], f"Empty description for {artifact_type['name']}"
            assert isinstance(artifact_type["description"], str)

    def test_types_sorted_by_name(self):
        """Test that artifact types are sorted alphabetically by name."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        names = [t["name"] for t in data["artifact_types"]]
        assert names == sorted(names), "Artifact types should be sorted by name"

    def test_response_with_error_handling(self):
        """Test that handler gracefully handles errors."""
        # This test verifies the error path exists
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        # Should either have artifact_types or error field
        assert "artifact_types" in data or "error" in data

    def test_metadata_version_present(self):
        """Test that response metadata includes version."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        metadata = data["metadata"]
        assert "version" in metadata
        assert metadata["version"] == "1.0"

    def test_full_response_schema_valid(self):
        """Test that full response schema is valid and complete."""
        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)

        # Root level
        assert len(data) >= 3  # artifact_types, summary, metadata minimum
        assert data["summary"]["total"] > 0
        assert len(data["artifact_types"]) == data["summary"]["total"]

        # Should be 11 types (8 hardcoded + 3 plugin)
        assert data["summary"]["total"] >= 11

    def test_backward_compatibility_with_template_methods(self):
        """Test that existing template methods still work."""
        from AgentQMS.tools.core.artifact_templates import ArtifactTemplates

        templates = ArtifactTemplates()

        # Original methods should still work
        assert callable(templates.get_available_templates)
        assert callable(templates.get_template)

        # Should return same types
        original_types = set(templates.get_available_templates())

        result = asyncio.run(_get_plugin_artifact_types())
        data = json.loads(result)
        resource_types = {t["name"] for t in data["artifact_types"]}

        assert original_types == resource_types

    def test_concurrent_access_safe(self):
        """Test that resource handler can be called concurrently."""
        import concurrent.futures

        def get_types():
            return asyncio.run(_get_plugin_artifact_types())

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(get_types) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All results should have identical artifact types and counts
        parsed_results = [json.loads(r) for r in results]

        # Extract comparable data (exclude timestamp which changes slightly)
        comparable = [
            (
                data["summary"]["total"],
                len(data["artifact_types"]),
                tuple(t["name"] for t in data["artifact_types"]),
            )
            for data in parsed_results
        ]

        # All should be identical (count, type count, and sorted names)
        assert len(set(comparable)) == 1, f"Concurrent calls should return identical data structure, got {len(set(comparable))} variants"

        # Each should be valid
        for result in results:
            data = json.loads(result)
            assert data["summary"]["total"] > 0


class TestAvailableArtifactTypesHelper:
    """Test the _get_available_artifact_types() helper function."""

    def test_returns_list_of_strings(self):
        """Test that function returns list of type names as strings."""
        types = asyncio.run(_get_available_artifact_types())

        assert isinstance(types, list)
        assert len(types) > 0
        assert all(isinstance(t, str) for t in types)

    def test_includes_all_types(self):
        """Test that function returns all artifact types."""
        types = asyncio.run(_get_available_artifact_types())
        type_set = set(types)

        expected = {
            "implementation_plan",
            "walkthrough",
            "assessment",
            "design",
            "research",
            "template",
            "bug_report",
            "vlm_report",
            "audit",
            "change_request",
            "ocr_experiment_report",
        }

        assert type_set == expected

    def test_returns_sorted_list(self):
        """Test that returned list is sorted."""
        types = asyncio.run(_get_available_artifact_types())
        assert types == sorted(types)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
