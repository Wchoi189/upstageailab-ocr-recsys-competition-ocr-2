#!/usr/bin/env python3
"""
Artifact Monitoring and Compliance System - Legacy Shim

This is a compatibility shim. The canonical implementation is in:
    AgentQMS.agent_tools.compliance.monitor_artifacts

See AgentQMS/toolkit/README.md for migration guidance.
"""

# Re-export from canonical location
from AgentQMS.agent_tools.compliance.monitor_artifacts import (
    ArtifactMonitor,
    main,
)

__all__ = ["ArtifactMonitor", "main"]

if __name__ == "__main__":
    import sys
    sys.exit(main())
