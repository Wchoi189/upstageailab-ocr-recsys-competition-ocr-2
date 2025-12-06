#!/usr/bin/env python3
"""Quick test to verify branch metadata is captured in artifacts."""

import sys
from pathlib import Path

# Add AgentQMS to path
sys.path.insert(0, str(Path(__file__).parent))

from AgentQMS.toolkit.core.artifact_templates import ArtifactTemplates


def test_branch_metadata():
    """Test that artifact creation includes branch metadata."""

    # Test 1: Create artifact without explicit branch (should auto-detect or default to main)
    print("✓ Test 1: Create artifact without explicit branch...")
    templates = ArtifactTemplates()
    artifact_frontmatter = templates.create_frontmatter(
        template_type="implementation_plan",
        title="Test Metadata Capture"
    )

    # Parse YAML frontmatter from the generated string
    lines = artifact_frontmatter.split("\n")
    assert lines[0] == "---", "Frontmatter should start with ---"
    # Extract YAML lines (between --- markers)
    yaml_lines = []
    for i in range(1, len(lines)):
        if lines[i] == "---":
            break
        yaml_lines.append(lines[i])

    # Check for branch field
    branch_found = False
    date_found = False
    branch_value = None
    date_value = None

    for line in yaml_lines:
        if line.startswith("branch:"):
            branch_found = True
            branch_value = line.split(":", 1)[1].strip().strip('"\'')
            print(f"  ✓ Auto-detected branch: {branch_value}")
        if line.startswith("date:"):
            date_found = True
            date_value = line.split(":", 1)[1].strip().strip('"\'')
            print(f"  ✓ Generated date: {date_value}")

    assert branch_found, "Missing 'branch' field in artifact metadata"
    assert date_found, "Missing 'date' field in artifact metadata"
    assert branch_value in ["main", "feature/outputs-reorg"], \
        f"Unexpected branch value: {branch_value}"

    # Test 2: Create artifact with explicit branch override
    print("\n✓ Test 2: Create artifact with explicit branch override...")
    artifact_override = templates.create_frontmatter(
        template_type="implementation_plan",
        title="Test Branch Override",
        branch="custom/feature-branch"
    )

    assert "branch: \"custom/feature-branch\"" in artifact_override, \
        f"Branch override failed, expected 'branch: \"custom/feature-branch\"' in:\n{artifact_override}"
    print("  ✓ Branch override worked: custom/feature-branch")

    # Test 3: Verify date format
    print("\n✓ Test 3: Verify date format...")
    # Should match pattern YYYY-MM-DD HH:MM (TZ)
    assert len(date_value) > 10, f"Date too short: {date_value}"
    assert "(" in date_value and ")" in date_value, f"Missing timezone info: {date_value}"
    print(f"  ✓ Date format valid: {date_value}")

    print("\n✅ All tests passed! Branch metadata is working correctly.")
    return True


if __name__ == "__main__":
    try:
        success = test_branch_metadata()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
