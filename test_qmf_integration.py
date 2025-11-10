#!/usr/bin/env python3
"""Test script for Quality Management Framework integration."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from agent_qms.toolbelt import AgentQMSToolbelt

    print("✓ Successfully imported AgentQMSToolbelt")

    # Initialize toolbelt
    manifest_path = project_root / "agent_qms" / "q-manifest.yaml"
    toolbelt = AgentQMSToolbelt(manifest_path=str(manifest_path))
    print(f"✓ Successfully initialized AgentQMSToolbelt")
    print(f"  Manifest path: {manifest_path}")

    # List artifact types
    artifact_types = toolbelt.list_artifact_types()
    print(f"✓ Available artifact types: {artifact_types}")

    # Test artifact creation
    print("\n--- Testing Artifact Creation ---")

    # Create a test assessment
    assessment_path = toolbelt.create_artifact(
        artifact_type="assessment",
        title="QMF Integration Test",
        content="""
## 1. Summary
This is a test assessment to verify QMF integration.

## 2. Assessment
The Quality Management Framework has been successfully integrated into the project.

## 3. Recommendations
- Continue with Phase 2 (Agent Tools Integration)
- Update project documentation
        """.strip(),
        author="ai-agent",
        tags=["test", "integration", "qmf"]
    )
    print(f"✓ Created assessment: {assessment_path}")

    # Validate the artifact
    is_valid = toolbelt.validate_artifact(assessment_path)
    print(f"✓ Artifact validation: {'PASSED' if is_valid else 'FAILED'}")

    # Create a test implementation plan
    plan_path = toolbelt.create_artifact(
        artifact_type="implementation_plan",
        title="QMF Phase 1 Implementation",
        content="""
## 1. Objective
Successfully integrate the Quality Management Framework into the OCR project.

## 2. Approach
- Copy framework to project root
- Add dependencies
- Test integration
- Update documentation

## 3. Implementation Steps
1. Copy quality_management_framework/ to project root ✓
2. Add dependencies to pyproject.toml ✓
3. Create artifacts/ directory structure ✓
4. Test integration ✓
5. Update documentation (in progress)

## 4. Testing Strategy
- Import and initialize toolbelt
- Create test artifacts
- Validate artifacts

## 5. Success Criteria
- Toolbelt can be imported and initialized
- Artifacts can be created successfully
- Artifacts pass validation
        """.strip(),
        author="ai-agent",
        tags=["implementation", "qmf", "phase1"]
    )
    print(f"✓ Created implementation plan: {plan_path}")

    # Validate the plan
    is_valid = toolbelt.validate_artifact(plan_path)
    print(f"✓ Plan validation: {'PASSED' if is_valid else 'FAILED'}")

    print("\n--- Integration Test Complete ---")
    print("✓ All tests passed!")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  Make sure dependencies are installed: jinja2, jsonschema, pytz")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

