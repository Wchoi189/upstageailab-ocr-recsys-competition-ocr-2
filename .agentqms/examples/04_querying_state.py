"""
Example 4: Querying State and Sessions

This example demonstrates various ways to query state:
- Finding artifacts by type, status, tags
- Searching sessions by criteria
- Analyzing statistics
"""

from agent_qms.toolbelt import state_api


def main():
    print("=== Example 4: Querying State ===\n")

    # Get framework info
    print("Framework Information:")
    info = state_api.get_framework_info()
    print(f"Name: {info['name']}")
    print(f"Version: {info['version']}")
    print()

    # Get current context
    print("Current Context:")
    context = state_api.get_current_context()
    print(f"Active session: {context['active_session_id']}")
    print(f"Current branch: {context['current_branch']}")
    print(f"Current phase: {context['current_phase']}")
    print(f"Active artifacts: {len(context['active_artifacts'])}")
    print()

    # Query artifacts by type
    print("Implementation Plans:")
    plans = state_api.get_artifacts_by_type("implementation_plan")
    for i, plan in enumerate(plans[:5], 1):
        print(f"{i}. {plan['path']}")
        print(f"   Status: {plan['status']}, Created: {plan['created_at']}")
    print(f"Total: {len(plans)} plans\n")

    # Query artifacts by status
    print("Draft Artifacts:")
    drafts = state_api.get_artifacts_by_status("draft")
    for i, draft in enumerate(drafts[:5], 1):
        print(f"{i}. {draft['path']} ({draft['type']})")
    print(f"Total: {len(drafts)} drafts\n")

    # Get recent artifacts
    print("Most Recent Artifacts:")
    recent = state_api.get_recent_artifacts(limit=10)
    for i, art in enumerate(recent, 1):
        print(f"{i}. {art['path']}")
        print(f"   Type: {art['type']}, Status: {art['status']}")
        print(f"   Created: {art['created_at']}")
    print()

    # Get recent sessions
    print("Recent Sessions:")
    sessions = state_api.get_recent_sessions(limit=5)
    for i, session in enumerate(sessions, 1):
        duration = session['statistics'].get('duration_minutes')
        print(f"{i}. {session['session_id']}")
        print(f"   Branch: {session['branch']}, Phase: {session['phase']}")
        if duration:
            print(f"   Duration: {duration:.1f} minutes")
        print(f"   Artifacts created: {len(session['artifacts_created'])}")
    print()

    # Search sessions by branch
    print("Searching for sessions on 'main' branch:")
    main_sessions = state_api.search_sessions(branch="main")
    print(f"Found {len(main_sessions)} sessions on main branch")
    for session in main_sessions[:3]:
        print(f"  - {session['session_id']} (Phase: {session['phase']})")
    print()

    # Statistics
    print("Framework Statistics:")
    stats = state_api.get_statistics()
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Total artifacts created: {stats['total_artifacts_created']}")
    print(f"Total artifacts validated: {stats['total_artifacts_validated']}")
    print(f"Total artifacts deployed: {stats['total_artifacts_deployed']}")
    if stats['last_session_timestamp']:
        print(f"Last session: {stats['last_session_timestamp']}")
    print()

    # Artifact count
    print("Artifact Count:")
    count = state_api.get_artifact_count()
    print(f"Total artifacts in index: {count}\n")

    # Session count
    print("Session Count:")
    count = state_api.get_session_count()
    print(f"Total sessions: {count}\n")

    print("Example completed successfully!")


if __name__ == "__main__":
    main()
