#!/usr/bin/env python3
"""
AgentQMS Artifact Creation Guide Tool

Provides interactive guidance for creating artifacts with proper format, location, and usage.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

try:
    from agent_qms.toolbelt import AgentQMSToolbelt
    AGENTQMS_AVAILABLE = True
except ImportError:
    AGENTQMS_AVAILABLE = False


def show_format_guide():
    """Show format guidance for artifacts."""
    print("=" * 70)
    print("ARTIFACT FORMAT GUIDE")
    print("=" * 70)
    print()
    print("Required Frontmatter:")
    print("-" * 70)
    print("""---
title: "Artifact Title"
date: "YYYY-MM-DD"
type: "assessment"  # or "implementation_plan"
category: "category_name"
status: "draft"  # or "active", "deprecated"
version: "1.0"
tags: ["tag1", "tag2"]
---""")
    print()
    print("Structure:")
    print("-" * 70)
    print("""
## 1. Summary
Brief overview of the artifact.

## 2. [Main Content Sections]
Detailed content based on artifact type.

## 3. Recommendations / Next Steps
Action items or follow-up.
""")
    print()


def show_location_guide():
    """Show location guidance for artifacts."""
    print("=" * 70)
    print("ARTIFACT LOCATION GUIDE")
    print("=" * 70)
    print()
    print("Artifact Storage:")
    print("-" * 70)
    print("""
All artifacts are stored in: artifacts/

Structure:
artifacts/
├── assessments/          # Assessment artifacts
│   └── assessment-name.md
└── implementation_plans/ # Implementation plan artifacts
    └── plan-name.md
""")
    print()
    print("Naming Convention:")
    print("-" * 70)
    print("""
- Use semantic names (not timestamp-based)
- Use kebab-case: assessment-name.md
- Be descriptive: ocr-pipeline-optimization.md
- Avoid: 2025-01-09_assessment.md (timestamp-based)
""")
    print()


def show_usage_guide():
    """Show usage guidance for AgentQMS toolbelt."""
    print("=" * 70)
    print("ARTIFACT USAGE GUIDE")
    print("=" * 70)
    print()

    if not AGENTQMS_AVAILABLE:
        print("⚠️  AgentQMS not available. Install dependencies:")
        print("   uv sync")
        print()
        return

    print("Preferred Method: AgentQMS Toolbelt")
    print("-" * 70)
    print("""
from agent_qms.toolbelt import AgentQMSToolbelt

toolbelt = AgentQMSToolbelt()

# Create an assessment
artifact_path = toolbelt.create_artifact(
    artifact_type="assessment",
    title="My Assessment",
    content="## Summary\\n...",
    author="ai-agent",
    tags=["tag1", "tag2"]
)

# Create an implementation plan
artifact_path = toolbelt.create_artifact(
    artifact_type="implementation_plan",
    title="My Implementation Plan",
    content="## Objective\\n...",
    author="ai-agent",
    tags=["implementation", "plan"]
)
""")
    print()

    print("Available Artifact Types:")
    print("-" * 70)
    try:
        toolbelt = AgentQMSToolbelt()
        artifact_types = toolbelt.list_artifact_types()
        for atype in artifact_types:
            print(f"  - {atype}")
    except Exception as e:
        print(f"  ⚠️  Error listing types: {e}")
    print()

    print("Artifact Locations:")
    print("-" * 70)
    print("  - assessments → artifacts/assessments/")
    print("  - implementation_plans → artifacts/implementation_plans/")
    print()


def show_examples():
    """Show example usage."""
    print("=" * 70)
    print("ARTIFACT CREATION EXAMPLES")
    print("=" * 70)
    print()

    print("Example 1: Create Assessment")
    print("-" * 70)
    print("""
from agent_qms.toolbelt import AgentQMSToolbelt

toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="assessment",
    title="OCR Pipeline Performance Assessment",
    content='''## Summary
Current OCR pipeline performance analysis.

## Assessment
- Training time: 20-30s per epoch
- Memory usage: 3-4GB
- Accuracy: 0.85 hmean

## Recommendations
- Optimize data loading
- Enable mixed precision training
''',
    author="ai-agent",
    tags=["performance", "pipeline", "ocr"]
)
print(f"Created: {artifact_path}")
""")
    print()

    print("Example 2: Create Implementation Plan")
    print("-" * 70)
    print("""
from agent_qms.toolbelt import AgentQMSToolbelt

toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="implementation_plan",
    title="Streamlit UI Refactoring Plan",
    content='''## Objective
Refactor monolithic Streamlit app to multi-page architecture.

## Approach
- Extract pages to separate files
- Create shared utilities module
- Update navigation

## Implementation Steps
1. Create pages/ directory
2. Extract preprocessing page
3. Extract inference page
4. Update main app

## Testing Strategy
- Test each page independently
- Verify navigation works
- Check session state persistence

## Success Criteria
- All pages functional
- Navigation works
- No regressions
''',
    author="ai-agent",
    tags=["refactoring", "streamlit", "ui"]
)
print(f"Created: {artifact_path}")
""")
    print()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AgentQMS Artifact Creation Guide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python artifact_guide.py                    # Show all guides
  python artifact_guide.py --format           # Show format guide only
  python artifact_guide.py --location         # Show location guide only
  python artifact_guide.py --usage            # Show usage guide only
  python artifact_guide.py --examples         # Show examples only
        """
    )

    parser.add_argument(
        "--format",
        action="store_true",
        help="Show format guidance"
    )
    parser.add_argument(
        "--location",
        action="store_true",
        help="Show location guidance"
    )
    parser.add_argument(
        "--usage",
        action="store_true",
        help="Show usage guidance"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show usage examples"
    )

    args = parser.parse_args()

    # If no specific option, show all
    if not any([args.format, args.location, args.usage, args.examples]):
        show_format_guide()
        show_location_guide()
        show_usage_guide()
        show_examples()
    else:
        if args.format:
            show_format_guide()
        if args.location:
            show_location_guide()
        if args.usage:
            show_usage_guide()
        if args.examples:
            show_examples()


if __name__ == "__main__":
    main()

