#!/usr/bin/env python3
"""
Wrapper script for artifact creation.
Delegates to AgentQMS/tools/core/artifact_workflow.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from AgentQMS.tools.core.artifact_workflow import main

if __name__ == "__main__":
    # The main() in artifact_workflow.py expects 'create' as the first subcommand
    # But the MCP server passes arguments like: --type ... --name ...
    # We need to adapt the arguments if necessary, or ensure artifact_workflow handles them.

    # artifact_workflow.py usage: python artifact_workflow.py create --type ...
    # MCP server calls: python create-artifact.py --type ...

    # We need to prepend 'create' to argv if it's not there,
    # BUT artifact_workflow.py uses argparse with subcommands.

    if len(sys.argv) > 1 and sys.argv[1] != "create":
        sys.argv.insert(1, "create")

    sys.exit(main())
