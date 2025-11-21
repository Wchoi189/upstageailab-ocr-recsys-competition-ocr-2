#!/usr/bin/env python3
"""
Complete state tracking workflow script.

This script runs the complete state tracking workflow:
1. Initialize/update state with current project data
2. Display current state
3. Optionally sync README with state

Usage:
    python state_workflow.py [--sync-readme] [--force]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable] + cmd,
            cwd=project_root,
            check=True,
            capture_output=False
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run complete state tracking workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Initialize state and show status
  %(prog)s --sync-readme      # Also sync README with state
  %(prog)s --force            # Force re-initialization
        """
    )
    parser.add_argument(
        "--sync-readme",
        action="store_true",
        help="Also sync README.md with current state"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-initialization of state"
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip initialization step"
    )

    args = parser.parse_args()

    success = True

    # Step 1: Initialize state
    if not args.skip_init:
        init_cmd = ["scripts/agent_tools/utilities/initialize_state.py"]
        if args.force:
            init_cmd.append("--force")

        if not run_command(init_cmd, "Step 1: Initializing State"):
            success = False

    # Step 2: Show state
    show_cmd = ["scripts/agent_tools/utilities/show_state.py"]
    if not run_command(show_cmd, "Step 2: Displaying Current State"):
        success = False

    # Step 3: Sync README (optional)
    if args.sync_readme:
        sync_cmd = ["scripts/agent_tools/utilities/sync_readme_state.py"]
        if not run_command(sync_cmd, "Step 3: Syncing README with State"):
            success = False

    if success:
        print(f"\n{'='*60}")
        print("  ✅ Workflow Complete!")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("  ⚠️  Workflow completed with errors")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

