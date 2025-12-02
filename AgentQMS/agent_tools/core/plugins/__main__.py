"""
Entry point for running the plugins module as a CLI.

Usage:
    python -m AgentQMS.agent_tools.core.plugins --list
    python -m AgentQMS.agent_tools.core.plugins --validate
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

