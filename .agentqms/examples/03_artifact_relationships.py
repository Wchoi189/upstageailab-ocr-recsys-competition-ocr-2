"""
Example 3: Artifact Relationships and Dependencies

This example demonstrates:
- Adding dependencies between artifacts
- Querying dependency trees
- Status propagation
"""

from agent_qms.toolbelt import AgentQMSToolbelt, StateManager, state_api


def main():
    print("=== Example 3: Artifact Relationships ===\n")

    # Initialize
    toolbelt = AgentQMSToolbelt()
    state_mgr = StateManager()

    # Create a series of related artifacts
    print("Creating related artifacts...")

    # 1. Assessment
    assessment_path = toolbelt.create_artifact(
        artifact_type="assessment",
        title="Feature X Assessment",
        content="# Assessment\n\nAnalysis of feature X requirements...",
        tags=["assessment", "feature-x"]
    )
    print(f"Created assessment: {assessment_path}")

    # 2. Implementation plan (depends on assessment)
    plan_path = toolbelt.create_artifact(
        artifact_type="implementation_plan",
        title="Feature X Implementation Plan",
        content="# Implementation Plan\n\nDetailed plan for feature X...",
        tags=["plan", "feature-x"]
    )
    print(f"Created plan: {plan_path}")

    # 3. Another plan (also depends on assessment)
    plan2_path = toolbelt.create_artifact(
        artifact_type="implementation_plan",
        title="Feature X Testing Plan",
        content="# Testing Plan\n\nTest strategy for feature X...",
        tags=["plan", "testing", "feature-x"]
    )
    print(f"Created testing plan: {plan2_path}\n")

    # Add dependencies
    print("Adding dependencies...")
    state_mgr.add_artifact_dependency(plan_path, assessment_path)
    state_mgr.add_artifact_dependency(plan2_path, assessment_path)
    print(f"  {plan_path} depends on {assessment_path}")
    print(f"  {plan2_path} depends on {assessment_path}\n")

    # Query dependencies
    print("Querying dependencies...")
    deps = state_mgr.get_artifact_dependencies(plan_path)
    print(f"Dependencies of {plan_path}:")
    for dep in deps:
        print(f"  - {dep}")
    print()

    # Reverse lookup
    print("Reverse lookup (what depends on assessment):")
    depending = state_mgr.get_artifacts_depending_on(assessment_path)
    for art in depending:
        print(f"  - {art}")
    print()

    # Get dependency tree
    print("Dependency tree:")
    tree = state_mgr.get_dependency_tree(plan_path)

    def print_tree(node, indent=0):
        print("  " * indent + f"- {node['path']}")
        for dep in node['dependencies']:
            print_tree(dep, indent + 1)

    print_tree(tree)
    print()

    # Status propagation
    print("Testing status propagation...")
    print(f"Marking assessment as deprecated...")

    # This will automatically deprecate all dependent artifacts
    updated = state_mgr.propagate_status_update(assessment_path, "deprecated")

    print(f"Deprecated artifacts:")
    for art_path in updated:
        artifact = state_mgr.get_artifact(art_path)
        print(f"  - {art_path} (status: {artifact['status']})")
    print()

    # Verify all are deprecated
    print("Verification:")
    for path in [assessment_path, plan_path, plan2_path]:
        artifact = state_mgr.get_artifact(path)
        print(f"  {path}: {artifact['status']}")
    print()

    print("Example completed successfully!")


if __name__ == "__main__":
    main()
