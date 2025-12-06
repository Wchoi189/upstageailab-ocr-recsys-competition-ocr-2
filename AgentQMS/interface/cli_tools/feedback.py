#!/usr/bin/env python3
"""
Agent Feedback System (Agent-Only Version)
Allows AI agents to report issues and suggest improvements
"""

import sys

from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path


def agent_feedback():
    """Agent feedback interface."""
    print("🤖 Agent Feedback System (AGENT-ONLY)")
    print("======================================")
    print()
    print("⚠️  WARNING: This tool is for AI agents only!")
    print("   Humans should use the main project tools.")
    print()

    ensure_project_root_on_sys_path()

    try:
        from AgentQMS.agent_tools.utilities.agent_feedback import main

        main()
    except ImportError as e:
        print(f"❌ Error importing feedback tool: {e}")
        print("   Make sure you're running from the agent directory")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(agent_feedback())
