"""
Entry point for running the plugins module as a CLI.

Usage:
    python -m AgentQMS.tools.core.plugins --list
    python -m AgentQMS.tools.core.plugins --validate
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
