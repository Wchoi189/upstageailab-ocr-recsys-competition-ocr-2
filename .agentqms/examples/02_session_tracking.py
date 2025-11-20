"""
Example 2: Session Tracking

This example demonstrates session lifecycle management:
- Starting and ending sessions
- Tracking goals, outcomes, and challenges
- Restoring context from previous sessions
"""

from agent_qms.toolbelt import AgentQMSToolbelt, SessionManager, StateManager, state_api


def main():
    print("=== Example 2: Session Tracking ===\n")

    # Initialize managers
    state_mgr = StateManager()
    session_mgr = SessionManager(state_mgr)
    toolbelt = AgentQMSToolbelt()

    # Start a new session
    print("Starting new session...")
    session_id = session_mgr.start_session(
        branch="feature/api-v2",
        phase="implementation",
        goals=[
            "Design API endpoints",
            "Implement authentication",
            "Write tests",
            "Update documentation"
        ]
    )
    print(f"Session started: {session_id}\n")

    # Simulate work: Create an artifact
    print("Creating design document...")
    doc_path = toolbelt.create_artifact(
        artifact_type="assessment",
        title="API V2 Design Assessment",
        content="# API V2 Design\n\nDesign for new API version...",
        tags=["api", "design"]
    )
    print(f"Created: {doc_path}\n")

    # Track progress
    print("Tracking progress...")
    session_mgr.add_outcome("API endpoints designed")
    session_mgr.add_outcome("Authentication implemented")
    session_mgr.add_challenge("Database migration needed")
    session_mgr.add_decision(
        decision="Use JWT for authentication",
        rationale="Industry standard, good security"
    )
    print("Progress tracked\n")

    # Check active session
    print("Current session status:")
    active = state_api.get_active_session()
    if active:
        print(f"Session ID: {active['session_id']}")
        print(f"Branch: {active['branch']}")
        print(f"Goals: {len(active['context']['goals'])}")
        print(f"Outcomes: {len(active['context']['outcomes'])}")
        print(f"Challenges: {len(active['context']['challenges'])}")
        print(f"Decisions: {len(active['context']['decisions'])}")
        print()

    # End the session
    print("Ending session...")
    session_mgr.end_session(
        summary="API V2 design completed, implementation in progress",
        outcomes=["Design approved", "Authentication working"],
        challenges=["Need to handle database migration"]
    )
    print("Session ended\n")

    # Later: Restore context from this session
    print("Restoring context from session...")
    context = session_mgr.restore_session_context(session_id)
    print(f"Session: {context['session_id']}")
    print(f"Branch: {context['branch']}")
    print(f"Phase: {context['phase']}")
    print(f"Duration: {context['duration_minutes']:.1f} minutes")
    print(f"\nGoals:")
    for goal in context['goals']:
        print(f"  - {goal}")
    print(f"\nOutcomes:")
    for outcome in context['outcomes']:
        print(f"  - {outcome}")
    print(f"\nChallenges:")
    for challenge in context['challenges']:
        print(f"  - {challenge}")
    print()

    print("Example completed successfully!")


if __name__ == "__main__":
    main()
