#!/usr/bin/env python3
"""
Integration tests for Unified MCP Server: Plugin Artifact Types Resource

Validates that unified_server exposes `agentqms://plugins/artifact_types` and
returns proper JSON via read_resource().
"""

import asyncio
import json
import pytest

# Import unified server module
from scripts.mcp import unified_server


class TestUnifiedServerPluginArtifactTypes:
    def test_resource_list_contains_plugin_artifact_types(self):
        resources = asyncio.run(unified_server.list_resources())
        uris = [str(r.uri) for r in resources]
        assert "agentqms://plugins/artifact_types" in uris

    def test_resource_metadata_correct(self):
        resources = asyncio.run(unified_server.list_resources())
        res = next((r for r in resources if str(r.uri) == "agentqms://plugins/artifact_types"), None)
        assert res is not None
        assert res.name == "Plugin Artifact Types"
        assert res.mimeType == "application/json"
        # Description optional but should mention metadata or discoverable
        assert ("metadata" in (res.description or "").lower()) or ("discover" in (res.description or "").lower())

    def test_read_resource_returns_valid_json(self):
        contents = asyncio.run(unified_server.read_resource("agentqms://plugins/artifact_types"))
        assert isinstance(contents, list) and len(contents) >= 1
        payload = contents[0].content
        assert isinstance(payload, str)
        data = json.loads(payload)
        assert isinstance(data, dict)
        assert "artifact_types" in data and isinstance(data["artifact_types"], list)
        assert "summary" in data and isinstance(data["summary"], dict)

    def test_includes_all_11_types(self):
        contents = asyncio.run(unified_server.read_resource("agentqms://plugins/artifact_types"))
        data = json.loads(contents[0].content)
        names = {t["name"] for t in data["artifact_types"]}
        expected = {
            "implementation_plan",
            "walkthrough",
            "assessment",
            "design",
            "design_document",
            "research",
            "template",
            "bug_report",
            "vlm_report",
            "audit",
            "change_request",
            "ocr_experiment_report",
        }
        assert expected.issubset(names)
        assert len(names) >= 11

    def test_response_schema_minimum_fields(self):
        contents = asyncio.run(unified_server.read_resource("agentqms://plugins/artifact_types"))
        data = json.loads(contents[0].content)
        # Root keys
        assert set(data.keys()) >= {"artifact_types", "summary", "metadata"}
        # Each type minimal fields
        for t in data["artifact_types"]:
            assert "name" in t
            assert "source" in t
            assert "description" in t


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
