#!/usr/bin/env python3
"""
Agent Tools Discovery Helper - Legacy Shim

This is a compatibility shim. The canonical implementation is in:
    AgentQMS.agent_tools.core.discover

See AgentQMS/toolkit/README.md for migration guidance.
"""

# Re-export from canonical location
from AgentQMS.agent_tools.core.discover import main, show_tools

__all__ = ["show_tools", "main"]

if __name__ == "__main__":
    main()
