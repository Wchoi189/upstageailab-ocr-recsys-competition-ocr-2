"""
Unit tests for sync_registry.py v2.0 compiler.
Tests schema validation, cycle detection, registry generation, and DOT graphs.
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from AgentQMS.tools.sync_registry import (
    validate_header,
    detect_cycles,
    generate_dot_graph,
    compute_pulse_delta,
    generate_registry,
    extract_ads_header,
    SchemaValidationError,
)


@pytest.fixture
def valid_schema() -> Dict[str, Any]:
    """Fixture providing a valid ADS v2.0 schema."""
    return {
        "required": [
            "ads_version", "id", "type", "agent", "tier", "priority",
            "validates_with", "compliance_status", "memory_footprint", "dependencies"
        ],
        "properties": {
            "ads_version": {"const": "2.0"},
            "id": {"pattern": r"^[A-Z]{2,3}-[0-9]{3}$"},
            "type": {
                "enum": ["rule_set", "agent_configuration", "tool_catalog",
                        "workflow_definition", "component_interface"]
            },
            "agent": {"enum": ["claude", "copilot", "cursor", "gemini", "qwen", "all"]},
            "tier": {"minimum": 1, "maximum": 4},
            "priority": {"enum": ["critical", "high", "medium", "low"]},
            "compliance_status": {"enum": ["pass", "fail", "unknown", "pending"]},
        }
    }


@pytest.fixture
def valid_header() -> Dict[str, Any]:
    """Fixture providing a valid ADS v2.0 header."""
    return {
        "ads_version": "2.0",
        "id": "SC-001",
        "type": "rule_set",
        "agent": "all",
        "tier": 1,
        "priority": "critical",
        "validates_with": "AgentQMS/standards/schemas/compliance-checker.py",
        "compliance_status": "pass",
        "memory_footprint": 80,
        "dependencies": [],
        "keywords": ["naming", "conventions"],
        "description": "Test standard",
    }


class TestSchemaValidation:
    """Test ADS header schema validation."""

    def test_valid_header(self, valid_header, valid_schema):
        """Valid header should pass validation."""
        errors = validate_header(valid_header, valid_schema)
        assert errors == []

    def test_missing_required_field(self, valid_header, valid_schema):
        """Missing required field should fail validation."""
        del valid_header["id"]
        errors = validate_header(valid_header, valid_schema)
        assert any("Missing required field: id" in e for e in errors)

    def test_invalid_ads_version(self, valid_header, valid_schema):
        """Wrong ADS version should fail validation."""
        valid_header["ads_version"] = "1.0"
        errors = validate_header(valid_header, valid_schema)
        assert any("Invalid ads_version" in e for e in errors)

    def test_invalid_id_format(self, valid_header, valid_schema):
        """Invalid ID format should fail validation."""
        valid_header["id"] = "invalid-id"
        errors = validate_header(valid_header, valid_schema)
        assert any("Invalid id format" in e for e in errors)

    def test_invalid_tier(self, valid_header, valid_schema):
        """Tier outside 1-4 range should fail validation."""
        valid_header["tier"] = 5
        errors = validate_header(valid_header, valid_schema)
        assert any("Invalid tier" in e for e in errors)

    def test_invalid_enum_value(self, valid_header, valid_schema):
        """Invalid enum value should fail validation."""
        valid_header["priority"] = "invalid"
        errors = validate_header(valid_header, valid_schema)
        assert any("Invalid priority" in e for e in errors)

    def test_dependencies_not_list(self, valid_header, valid_schema):
        """Dependencies not being a list should fail validation."""
        valid_header["dependencies"] = "not-a-list"
        errors = validate_header(valid_header, valid_schema)
        assert any("dependencies must be an array" in e for e in errors)


class TestCycleDetection:
    """Test circular dependency detection."""

    def test_no_cycles(self):
        """No circular dependencies should pass."""
        standards = {
            "SC-001": {"dependencies": []},
            "SC-002": {"dependencies": ["SC-001"]},
            "SC-003": {"dependencies": ["SC-002"]},
        }
        cycle = detect_cycles(standards)
        assert cycle is None

    def test_simple_cycle(self):
        """Simple A -> B -> A cycle should be detected."""
        standards = {
            "SC-001": {"dependencies": ["SC-002"]},
            "SC-002": {"dependencies": ["SC-001"]},
        }
        cycle = detect_cycles(standards)
        assert cycle is not None
        assert "SC-001" in cycle and "SC-002" in cycle

    def test_complex_cycle(self):
        """Complex A -> B -> C -> A cycle should be detected."""
        standards = {
            "SC-001": {"dependencies": ["SC-002"]},
            "SC-002": {"dependencies": ["SC-003"]},
            "SC-003": {"dependencies": ["SC-001"]},
        }
        cycle = detect_cycles(standards)
        assert cycle is not None
        assert len(cycle) >= 3

    def test_self_reference_cycle(self):
        """Self-referencing standard should be detected."""
        standards = {
            "SC-001": {"dependencies": ["SC-001"]},
        }
        cycle = detect_cycles(standards)
        assert cycle is not None
        assert cycle == ["SC-001", "SC-001"]

    def test_missing_dependency_no_crash(self):
        """Missing dependency reference should not crash."""
        standards = {
            "SC-001": {"dependencies": ["SC-999"]},
        }
        cycle = detect_cycles(standards)
        # Should not crash, missing dep ignored
        assert cycle is None


class TestDOTGraphGeneration:
    """Test DOT graph generation."""

    def test_empty_graph(self):
        """Empty standards should generate minimal graph."""
        dot = generate_dot_graph({})
        assert "digraph AgentQMS_Architecture" in dot
        assert "}" in dot

    def test_single_standard(self):
        """Single standard should appear in graph."""
        standards = {
            "SC-001": {
                "tier": 1,
                "priority": "critical",
                "description": "Test standard",
                "dependencies": [],
            }
        }
        dot = generate_dot_graph(standards)
        assert "SC-001" in dot
        assert "Constitution (Tier 1)" in dot
        assert "penwidth=2" in dot  # Critical priority styling

    def test_dependency_edge(self):
        """Dependencies should create edges in graph."""
        standards = {
            "SC-001": {"tier": 1, "priority": "critical", "dependencies": []},
            "SC-002": {"tier": 1, "priority": "high", "dependencies": ["SC-001"]},
        }
        dot = generate_dot_graph(standards)
        assert '"SC-001" -> "SC-002"' in dot

    def test_tier_subgraphs(self):
        """Standards should be grouped by tier."""
        standards = {
            "SC-001": {"tier": 1, "priority": "critical", "dependencies": []},
            "FW-001": {"tier": 2, "priority": "high", "dependencies": []},
        }
        dot = generate_dot_graph(standards)
        assert "cluster_tier1" in dot
        assert "cluster_tier2" in dot
        assert "Constitution (Tier 1)" in dot
        assert "Framework (Tier 2)" in dot


class TestPulseDelta:
    """Test semantic diff (Pulse Delta) computation."""

    def test_first_compilation(self):
        """First compilation should mark all as added."""
        new_standards = {
            "SC-001": {},
            "SC-002": {},
        }
        delta = compute_pulse_delta(None, new_standards)
        assert len(delta["added"]) == 2
        assert "SC-001" in delta["added"]
        assert "SC-002" in delta["added"]
        assert delta["removed"] == []

    def test_no_changes(self):
        """No changes should show all unchanged."""
        old_registry = {
            "tier1_sst": {"naming": "path/to/naming.yaml"},
        }
        new_standards = {
            "naming": {"path": "path/to/naming.yaml"},
        }
        delta = compute_pulse_delta(old_registry, new_standards)
        assert "naming" in delta["unchanged"]

    def test_additions_and_removals(self):
        """Additions and removals should be tracked."""
        old_registry = {
            "tier1_sst": {
                "naming": "path/to/naming.yaml",
                "old_standard": "path/to/old.yaml",
            },
        }
        new_standards = {
            "naming": {"path": "path/to/naming.yaml"},
            "new_standard": {"path": "path/to/new.yaml"},
        }
        delta = compute_pulse_delta(old_registry, new_standards)
        assert "new_standard" in delta["added"]
        assert "old_standard" in delta["removed"]


class TestRegistryGeneration:
    """Test registry.yaml generation."""

    def test_basic_registry(self):
        """Basic registry should have correct structure."""
        standards = {
            "SC-001": {
                "ads_version": "2.0",
                "tier": 1,
                "keywords": ["test"],
                "dependencies": [],
            }
        }
        pulse_delta = {"added": ["SC-001"], "removed": [], "modified": [], "unchanged": []}

        registry = generate_registry(standards, pulse_delta)

        assert registry["ads_version"] == "2.0"
        assert registry["type"] == "unified_registry"
        assert "metadata" in registry
        assert registry["metadata"]["total_standards"] == 1
        assert "SC-001" in registry["standards"]

    def test_tier_indexing(self):
        """Registry should index standards by tier."""
        standards = {
            "SC-001": {"tier": 1, "keywords": [], "dependencies": []},
            "FW-001": {"tier": 2, "keywords": [], "dependencies": []},
        }
        pulse_delta = {"added": [], "removed": [], "modified": [], "unchanged": []}

        registry = generate_registry(standards, pulse_delta)

        assert "tier1" in registry["tier_index"]
        assert "tier2" in registry["tier_index"]
        assert "SC-001" in registry["tier_index"]["tier1"]
        assert "FW-001" in registry["tier_index"]["tier2"]

    def test_keyword_indexing(self):
        """Registry should index standards by keywords."""
        standards = {
            "SC-001": {
                "tier": 1,
                "keywords": ["naming", "conventions"],
                "dependencies": [],
            }
        }
        pulse_delta = {"added": [], "removed": [], "modified": [], "unchanged": []}

        registry = generate_registry(standards, pulse_delta)

        assert "naming" in registry["keyword_index"]
        assert "conventions" in registry["keyword_index"]
        assert "SC-001" in registry["keyword_index"]["naming"]

    def test_dependency_summary(self):
        """Registry should include dependency statistics."""
        standards = {
            "SC-001": {"tier": 1, "keywords": [], "dependencies": []},
            "SC-002": {"tier": 1, "keywords": [], "dependencies": ["SC-001"]},
            "FW-001": {"tier": 2, "keywords": [], "dependencies": []},
        }
        pulse_delta = {"added": [], "removed": [], "modified": [], "unchanged": []}

        registry = generate_registry(standards, pulse_delta)

        assert "dependency_summary" in registry
        assert registry["dependency_summary"]["total_dependencies"] == 1
        assert registry["dependency_summary"]["standards_with_deps"] == 1
        assert "FW-001" in registry["dependency_summary"]["orphan_standards"]


class TestHeaderExtraction:
    """Test ADS header extraction from YAML files."""

    def test_extract_valid_header(self, tmp_path):
        """Extract header from valid YAML file."""
        yaml_content = """
ads_version: '2.0'
id: SC-001
type: rule_set
agent: all
tier: 1
priority: critical
validates_with: test
compliance_status: pass
memory_footprint: 80
dependencies: []
keywords:
  - test
description: Test standard

# Content below
rules:
  - rule1
  - rule2
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        header = extract_ads_header(yaml_file)

        assert header["ads_version"] == "2.0"
        assert header["id"] == "SC-001"
        assert header["tier"] == 1
        assert "keywords" in header

    def test_extract_invalid_yaml(self, tmp_path):
        """Invalid YAML should raise SchemaValidationError."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: [")

        with pytest.raises(SchemaValidationError):
            extract_ads_header(yaml_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
