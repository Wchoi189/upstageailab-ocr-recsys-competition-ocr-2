#!/usr/bin/env python3
"""
Project Compass CLI - Dedicated tool for project management tasks.
Separated from Experiment Tracker Kit (ETK) to ensure separation of concerns.

Commands:
    check-env     : Validate environment against lock state
    session-init  : Initialize or update current session context
"""

import argparse
import sys

# NOTE: Run this CLI as a module: uv run python -m project_compass.cli
try:
    from project_compass.src.core import EnvironmentChecker, SessionManager
    from project_compass.src.wizard import SprintContextWizard
    from project_compass.scripts import session_manager as legacy_session_manager
except ImportError:
    # Fallback for development/local execution context
    try:
        from src.core import EnvironmentChecker, SessionManager
        from src.wizard import SprintContextWizard
        from scripts import session_manager as legacy_session_manager
    except ImportError as e:
        print(f"CRITICAL ERROR: Failed to import Project Compass core components.\nDetail: {e}")
        print("Hint: Run as module: uv run python -m project_compass.cli")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Project Compass CLI - AI Project Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # check-env command (Environment Guard)
    check_env_parser = subparsers.add_parser("check-env", help="Validate environment against Project Compass lock state")
    check_env_parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")

    # session-init command (Session Management)
    session_init_parser = subparsers.add_parser("session-init", help="Initialize or update current session context")
    session_init_parser.add_argument("--objective", "-o", required=True, help="Primary goal for this session")
    session_init_parser.add_argument("--name", "-n", help="Descriptive session name (format: domain-action-target)")
    session_init_parser.add_argument(
        "--pipeline",
        "-p",
        default="kie",
        choices=["text_detection", "text_recognition", "layout_analysis", "kie", "roadmap"],
        help="Active pipeline (default: kie)",
    )

    # wizard command (Interactive Session Setup)
    subparsers.add_parser("wizard", help="Interactive Sprint Context setup")

    # session-export command (Wrapper for legacy script)
    session_export_parser = subparsers.add_parser("session-export", help="Archive current session to history")
    session_export_parser.add_argument("--note", "-n", help="Note for the manifest")
    session_export_parser.add_argument("--force", "-f", action="store_true", help="Bypass stale session check")

    # update-status command (Manual Compass Update)
    update_status_parser = subparsers.add_parser("update-status", help="Update compass.json project status")
    update_status_parser.add_argument("--phase", help="Current project phase")
    update_status_parser.add_argument("--health", choices=["healthy", "degraded", "blocked"], help="Overall health status")
    update_status_parser.add_argument("--note", help="Status note")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "check-env":
            print("🔒 Environment Guard: Checking against Project Compass lock state...\n")
            checker = EnvironmentChecker()
            passed, errors, warnings = checker.check_all()

            if warnings:
                for warning in warnings:
                    print(f"⚠️  {warning}")
                print()

            if errors:
                print("❌ ENVIRONMENT BREACH DETECTED\n")
                for error in errors:
                    print(f"  ✗ {error}\n")
                print("\n🔧 Path Restoration Instructions:")
                print("   1. Ensure you are using the correct UV binary")
                print("   2. Run: uv sync")
                print('   3. Verify with: uv run python -c "import torch; print(torch.__version__)"')
                sys.exit(1)
            else:
                print("✅ Environment validated against Compass lock state")
                if args.strict and warnings:
                    print("\n❌ Strict mode: warnings treated as errors")
                    sys.exit(1)
                sys.exit(0)

        elif args.command == "session-init":
            print("📋 Session Management: Initializing session...\n")

            # NEW: Run environment check first
            print("🔒 Environment Guard: Verifying environment...")
            checker = EnvironmentChecker()
            passed, errors, warnings = checker.check_all()

            if warnings:
                for warning in warnings:
                    print(f"⚠️  {warning}")

            if errors:
                print("\n❌ Environment validation failed:")
                for error in errors:
                    print(f"  ✗ {error}")
                print("\n🔧 Fix environment issues before initializing session.")
                print("   Run: uv run python -m project_compass.cli check-env")
                sys.exit(1)

            print("✅ Environment validated\n")

            # Initialize session
            manager = SessionManager()
            success, message = manager.init_session(
                objective=args.objective,
                session_name=args.name,
                active_pipeline=args.pipeline
            )

            if success:
                print(f"✅ {message}")
                print(f"📂 Updated: {manager.paths.current_session}")
                sys.exit(0)
            else:
                print(f"❌ {message}")
                sys.exit(1)

        elif args.command == "wizard":
            wizard = SprintContextWizard()
            wizard.run()
            sys.exit(0)

        elif args.command == "session-export":
            print("📦 Session Management: Exporting session...\n")
            # Invoke legacy logic
            legacy_session_manager.export_session(note=args.note, force=args.force)
            sys.exit(0)

        elif args.command == "session-export":
            print("📦 Session Management: Exporting session...\\n")
            # Invoke legacy logic
            legacy_session_manager.export_session(note=args.note, force=args.force)
            sys.exit(0)

        elif args.command == "update-status":
            print("📊 Updating compass.json status...\\n")
            manager = SessionManager()
            success = manager._update_compass_status(
                phase=args.phase,
                health=args.health,
                note=args.note
            )
            if success:
                print("✅ compass.json updated successfully")
                sys.exit(0)
            else:
                print("❌ Failed to update compass.json")
                sys.exit(1)

    except Exception as e:
        print(f"❌ ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
