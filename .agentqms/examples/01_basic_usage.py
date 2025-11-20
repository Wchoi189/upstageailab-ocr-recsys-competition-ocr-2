"""
Example 1: Basic State Tracking Usage

This example demonstrates the basic usage of the state tracking system:
- Creating artifacts with automatic state tracking
- Querying artifacts
- Viewing statistics
"""

from agent_qms.toolbelt import AgentQMSToolbelt, state_api


def main():
    print("=== Example 1: Basic State Tracking Usage ===\n")

    # Initialize toolbelt (state tracking enabled by default)
    toolbelt = AgentQMSToolbelt()

    # Create an implementation plan
    print("Creating implementation plan...")
    plan_path = toolbelt.create_artifact(
        artifact_type="implementation_plan",
        title="Feature X Implementation",
        content="# Implementation Plan\n\nDetailed plan for feature X...",
        tags=["feature", "backend"],
        author="ai-agent"
    )
    print(f"Created: {plan_path}")
    print()

    # Query the artifact
    print("Querying artifact...")
    artifact = state_api.get_artifact(plan_path)
    if artifact:
        print(f"Path: {artifact['path']}")
        print(f"Type: {artifact['type']}")
        print(f"Status: {artifact['status']}")
        print(f"Created: {artifact['created_at']}")
        print()

    # Get recent artifacts
    print("Recent artifacts:")
    recent = state_api.get_recent_artifacts(limit=5)
    for i, art in enumerate(recent, 1):
        print(f"{i}. {art['path']} ({art['status']})")
    print()

    # Get statistics
    print("Framework statistics:")
    stats = state_api.get_statistics()
    print(f"Total artifacts created: {stats['total_artifacts_created']}")
    print(f"Total artifacts validated: {stats['total_artifacts_validated']}")
    print(f"Total sessions: {stats['total_sessions']}")
    print()

    # Get state health
    print("State health:")
    health = state_api.get_state_health()
    print(f"State valid: {health['is_valid']}")
    print(f"Total artifacts: {health['total_artifacts']}")
    print(f"State file size: {health['state_file_size_bytes']} bytes")
    print()

    print("Example completed successfully!")


if __name__ == "__main__":
    main()
