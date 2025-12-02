#!/usr/bin/env python3
"""Quick documentation update workflow."""

from AgentQMS.toolkit.utils.runtime import ensure_project_root_on_sys_path
from AgentQMS.toolkit.maintenance.regenerate_docs import main as regenerate_main

ensure_project_root_on_sys_path()


def main() -> None:
    """Quick documentation update."""
    print("🚀 Quick Documentation Update")
    print("This will regenerate the AI handbook index and validate the structure.")
    print()

    # Run the regeneration workflow
    regenerate_main()


if __name__ == "__main__":
    main()
