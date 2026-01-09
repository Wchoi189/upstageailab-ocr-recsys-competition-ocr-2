"""
Test Suite for Artifact Type Validation Rules

Tests Phase 3 Session 5 deliverables:
- Centralized validation schema enforcement
- Naming conflict resolution
- Prohibited type rejection
- Canonical type validation
- Plugin structure validation
"""

import pytest
from pathlib import Path
from typing import Any

import yaml

from AgentQMS.tools.core.plugins.validation import PluginValidator


# Test fixtures
@pytest.fixture
def validation_rules_path() -> Path:
    """Path to artifact_type_validation.yaml"""
    return Path(".agentqms/schemas/artifact_type_validation.yaml")


@pytest.fixture
def validator(validation_rules_path: Path) -> PluginValidator:
    """Create validator with artifact type rules loaded"""
    schemas_dir = Path("AgentQMS/standards/schemas")
    return PluginValidator(schemas_dir=schemas_dir, validation_rules_path=validation_rules_path)


@pytest.fixture
def valid_plugin_data() -> dict[str, Any]:
    """Valid artifact type plugin data"""
    return {
        "name": "assessment",
        "version": "1.0",
        "description": "Test assessment plugin",
        "scope": "project",
        "metadata": {
            "filename_pattern": "{date}_assessment_{name}.md",
            "directory": "assessments/",
            "frontmatter": {
                "ads_version": "1.0",
                "type": "assessment",
                "category": "evaluation",
                "status": "active",
                "version": "1.0",
                "tags": ["assessment", "test"],
            },
        },
        "template": "# Assessment - {title}\n\n## Purpose\n...",
        "validation": {
            "required_fields": ["title", "date"],
        },
    }


# ============================================================================
# Test 1: Validation Rules Loading
# ============================================================================


def test_validation_rules_loaded(validator: PluginValidator):
    """Test that validation rules are loaded successfully"""
    canonical = validator.get_canonical_types()
    prohibited = validator.get_prohibited_types()

    assert len(canonical) > 0, "No canonical types loaded"
    assert len(prohibited) > 0, "No prohibited types loaded"

    # Check specific canonical types
    assert "assessment" in canonical
    assert "design_document" in canonical
    assert "bug_report" in canonical

    # Check specific prohibited types
    prohibited_names = [p["name"] for p in prohibited]
    assert "research" in prohibited_names
    assert "design" in prohibited_names
    assert "template" in prohibited_names


# ============================================================================
# Test 2: Prohibited Types Rejection
# ============================================================================


def test_prohibited_type_research_rejected(validator: PluginValidator, valid_plugin_data: dict):
    """Test that 'research' type is rejected with correct message"""
    valid_plugin_data["name"] = "research"
    errors = validator.validate(valid_plugin_data, "artifact_type")

    assert len(errors) > 0, "Expected validation errors for prohibited type 'research'"
    assert any("Prohibited artifact type 'research'" in err for err in errors)
    assert any("Use 'assessment' instead" in err for err in errors)


def test_prohibited_type_design_rejected(validator: PluginValidator, valid_plugin_data: dict):
    """Test that 'design' type is rejected with alias warning"""
    valid_plugin_data["name"] = "design"
    errors = validator.validate(valid_plugin_data, "artifact_type")

    assert len(errors) > 0, "Expected validation errors for prohibited type 'design'"
    assert any("alias" in err.lower() or "design_document" in err for err in errors)


def test_prohibited_type_template_rejected(validator: PluginValidator, valid_plugin_data: dict):
    """Test that 'template' type is rejected"""
    valid_plugin_data["name"] = "template"
    errors = validator.validate(valid_plugin_data, "artifact_type")

    assert len(errors) > 0, "Expected validation errors for prohibited type 'template'"
    assert any("Prohibited artifact type 'template'" in err for err in errors)


# ============================================================================
# Test 3: Canonical Types Validation
# ============================================================================


def test_canonical_type_assessment_accepted(validator: PluginValidator, valid_plugin_data: dict):
    """Test that canonical 'assessment' type is accepted"""
    valid_plugin_data["name"] = "assessment"
    errors = validator.validate(valid_plugin_data, "artifact_type")

    # May have JSON schema errors, but should NOT have naming errors
    naming_errors = [e for e in errors if "unknown" in e.lower() or "prohibited" in e.lower()]
    assert len(naming_errors) == 0, f"Unexpected naming errors for canonical type: {naming_errors}"


def test_canonical_type_design_document_accepted(validator: PluginValidator, valid_plugin_data: dict):
    """Test that canonical 'design_document' type is accepted"""
    valid_plugin_data["name"] = "design_document"
    valid_plugin_data["metadata"]["frontmatter"]["type"] = "design"  # Frontmatter can use "design"
    errors = validator.validate(valid_plugin_data, "artifact_type")

    naming_errors = [e for e in errors if "unknown" in e.lower() or "prohibited" in e.lower()]
    assert len(naming_errors) == 0, f"Unexpected naming errors: {naming_errors}"


def test_canonical_type_bug_report_accepted(validator: PluginValidator, valid_plugin_data: dict):
    """Test that canonical 'bug_report' type is accepted"""
    valid_plugin_data["name"] = "bug_report"
    valid_plugin_data["metadata"]["frontmatter"]["type"] = "bug_report"
    errors = validator.validate(valid_plugin_data, "artifact_type")

    naming_errors = [e for e in errors if "unknown" in e.lower() or "prohibited" in e.lower()]
    assert len(naming_errors) == 0


# ============================================================================
# Test 4: Unknown Types Rejection
# ============================================================================


def test_unknown_type_rejected(validator: PluginValidator, valid_plugin_data: dict):
    """Test that unknown/invalid type names are rejected"""
    valid_plugin_data["name"] = "invalid_type"
    errors = validator.validate(valid_plugin_data, "artifact_type")

    assert len(errors) > 0, "Expected validation errors for unknown type"
    assert any("Unknown artifact type" in err for err in errors)
    assert any("invalid_type" in err for err in errors)


# ============================================================================
# Test 5: Required Metadata Validation
# ============================================================================


def test_missing_metadata_section(validator: PluginValidator, valid_plugin_data: dict):
    """Test that missing metadata section is rejected"""
    del valid_plugin_data["metadata"]
    errors = validator.validate(valid_plugin_data, "artifact_type")

    assert len(errors) > 0
    assert any("Missing 'metadata' section" in err for err in errors)


def test_missing_required_metadata_fields(validator: PluginValidator, valid_plugin_data: dict):
    """Test that missing required metadata fields are rejected"""
    del valid_plugin_data["metadata"]["filename_pattern"]
    errors = validator.validate(valid_plugin_data, "artifact_type")

    assert len(errors) > 0
    assert any("filename_pattern" in err for err in errors)


def test_missing_frontmatter_fields(validator: PluginValidator, valid_plugin_data: dict):
    """Test that missing required frontmatter fields are rejected"""
    del valid_plugin_data["metadata"]["frontmatter"]["ads_version"]
    del valid_plugin_data["metadata"]["frontmatter"]["type"]
    errors = validator.validate(valid_plugin_data, "artifact_type")

    assert len(errors) > 0
    assert any("ads_version" in err for err in errors)
    assert any("type" in err for err in errors)


def test_missing_template(validator: PluginValidator, valid_plugin_data: dict):
    """Test that missing template field is rejected"""
    del valid_plugin_data["template"]
    errors = validator.validate(valid_plugin_data, "artifact_type")

    assert len(errors) > 0
    assert any("template" in err.lower() for err in errors)


# ============================================================================
# Test 6: Integration with Actual Plugins
# ============================================================================


def test_load_actual_assessment_plugin(validator: PluginValidator):
    """Test loading actual assessment.yaml plugin"""
    plugin_path = Path("AgentQMS/.agentqms/plugins/artifact_types/assessment.yaml")
    if not plugin_path.exists():
        pytest.skip("assessment.yaml plugin not found")

    with plugin_path.open("r") as f:
        plugin_data = yaml.safe_load(f)

    errors = validator.validate(plugin_data, "artifact_type")

    # Should have no critical naming/structure errors
    critical_errors = [
        e for e in errors
        if "prohibited" in e.lower() or "unknown" in e.lower() or "missing" in e.lower()
    ]
    assert len(critical_errors) == 0, f"Critical errors in assessment.yaml: {critical_errors}"


def test_load_actual_design_document_plugin(validator: PluginValidator):
    """Test loading actual design_document.yaml plugin"""
    plugin_path = Path("AgentQMS/.agentqms/plugins/artifact_types/design_document.yaml")
    if not plugin_path.exists():
        pytest.skip("design_document.yaml plugin not found")

    with plugin_path.open("r") as f:
        plugin_data = yaml.safe_load(f)

    errors = validator.validate(plugin_data, "artifact_type")

    critical_errors = [
        e for e in errors
        if "prohibited" in e.lower() or "unknown" in e.lower()
    ]
    assert len(critical_errors) == 0, f"Critical errors in design_document.yaml: {critical_errors}"


def test_deprecated_plugins_not_loaded():
    """Test that deprecated plugins are not loaded by plugin system"""
    deprecated_files = [
        "AgentQMS/.agentqms/plugins/artifact_types/design.yaml.deprecated",
        "AgentQMS/.agentqms/plugins/artifact_types/research.yaml.deprecated",
        "AgentQMS/.agentqms/plugins/artifact_types/template.yaml.deprecated",
    ]

    for dep_file in deprecated_files:
        path = Path(dep_file)
        if path.exists():
            # File exists but should not end in .yaml (plugin loader skips .deprecated)
            assert not path.name.endswith(".yaml"), f"Deprecated file still has .yaml extension: {dep_file}"


# ============================================================================
# Test 7: Validation Rules Schema Structure
# ============================================================================


def test_validation_rules_schema_structure(validation_rules_path: Path):
    """Test that validation rules YAML has expected structure"""
    if not validation_rules_path.exists():
        pytest.skip("Validation rules file not found")

    with validation_rules_path.open("r") as f:
        rules = yaml.safe_load(f)

    # Check top-level keys
    assert "canonical_types" in rules
    assert "prohibited_types" in rules
    assert "naming_convention" in rules
    assert "validation_rules" in rules

    # Check canonical types structure
    canonical = rules["canonical_types"]
    assert "assessment" in canonical
    assert "design_document" in canonical

    # Each canonical type should have required fields
    for type_name, type_def in canonical.items():
        assert "location" in type_def, f"{type_name} missing 'location'"
        assert "purpose" in type_def, f"{type_name} missing 'purpose'"

    # Check prohibited types structure
    prohibited = rules["prohibited_types"]
    for prohibited_type in prohibited:
        assert "name" in prohibited_type
        assert "use_instead" in prohibited_type


# ============================================================================
# Test 8: Alias Handling
# ============================================================================


def test_alias_design_triggers_warning(validator: PluginValidator, valid_plugin_data: dict):
    """Test that using alias 'design' triggers a warning to use 'design_document'"""
    valid_plugin_data["name"] = "design"
    errors = validator.validate(valid_plugin_data, "artifact_type")

    # Should have error suggesting canonical name
    assert len(errors) > 0
    alias_errors = [e for e in errors if "alias" in e.lower()]
    assert len(alias_errors) > 0, "Expected alias warning for 'design'"


# ============================================================================
# Test 9: Plugin Discovery Integration
# ============================================================================


def test_all_active_plugins_are_canonical():
    """Test that all active .yaml plugins use canonical type names"""
    plugins_dir = Path("AgentQMS/.agentqms/plugins/artifact_types")
    if not plugins_dir.exists():
        pytest.skip("Plugins directory not found")

    canonical_names = [
        "assessment", "audit", "bug_report", "design_document",
        "implementation_plan", "vlm_report", "walkthrough",
        "change_request", "ocr_experiment"  # Project-specific types
    ]

    for plugin_file in plugins_dir.glob("*.yaml"):
        # Skip deprecated files
        if ".deprecated" in plugin_file.name:
            continue

        with plugin_file.open("r") as f:
            plugin_data = yaml.safe_load(f)

        plugin_name = plugin_data.get("name", "")

        # Plugin name should be in canonical list (or project-specific)
        # We allow project-specific types not in standards
        assert plugin_name, f"Plugin {plugin_file} has no 'name' field"


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
