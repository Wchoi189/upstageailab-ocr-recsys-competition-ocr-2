#!/usr/bin/env python3
"""
Project Compass V2 CLI - Vessel Pulse Management

Commands:
    check-env      : Validate environment (lightweight)
    pulse-init     : Initialize a new pulse (work cycle)
    pulse-status   : Show current pulse status
    pulse-sync     : Register staging artifacts
    pulse-export   : Archive pulse to history
    pulse-checkpoint: Evaluate pulse maturity
    spec-constitution: Establish project principles
    spec-specify     : Create baseline specification
    spec-plan        : Create implementation plan
    spec-tasks       : Generate actionable tasks"""

import argparse
import sys
from pathlib import Path

try:
    from project_compass.src.core import (
        PulseManager,
        EnvironmentChecker,
        VesselPaths,
    )
    from project_compass.src.state_schema import PipelinePhase, ArtifactType
    from project_compass.src.pulse_exporter import export_pulse, register_artifact
except ImportError:
    # Fallback for development
    try:
        from src.core import PulseManager, EnvironmentChecker, VesselPaths
        from src.state_schema import PipelinePhase, ArtifactType
        from src.pulse_exporter import export_pulse, register_artifact
    except ImportError as e:
        print(f"CRITICAL ERROR: Failed to import Vessel V2 modules.\nDetail: {e}")
        print("Hint: Run as module: uv run python -m project_compass.cli")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Project Compass V2 CLI - Vessel Pulse Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start a new pulse
    %(prog)s pulse-init --id rec-optimize-vocab --obj "Optimize vocabulary handling" --milestone rec-opt

    # Register an artifact
    %(prog)s pulse-sync --path design-doc.md --type design

    # Export pulse to history
    %(prog)s pulse-export
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # check-env command
    check_env_parser = subparsers.add_parser(
        "check-env",
        help="Validate environment (UV, Python)"
    )
    check_env_parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")

    # pulse-init command
    pulse_init_parser = subparsers.add_parser(
        "pulse-init",
        help="Initialize a new pulse (work cycle)"
    )
    pulse_init_parser.add_argument(
        "--id", "-i",
        required=True,
        help="Pulse ID: domain-action-target (e.g., recognition-optimize-vocab)"
    )
    pulse_init_parser.add_argument(
        "--obj", "-o",
        required=True,
        help="Objective: 20-500 char description"
    )
    pulse_init_parser.add_argument(
        "--milestone", "-m",
        required=True,
        help="Milestone ID from star-chart (e.g., rec-opt)"
    )
    pulse_init_parser.add_argument(
        "--phase", "-p",
        default="kie",
        choices=[p.value for p in PipelinePhase],
        help="Active pipeline phase (default: kie)"
    )

    # pulse-status command
    subparsers.add_parser("pulse-status", help="Show current pulse status")

    # pulse-sync command
    pulse_sync_parser = subparsers.add_parser(
        "pulse-sync",
        help="Register staging artifact in manifest"
    )
    pulse_sync_parser.add_argument(
        "--path", "-p",
        required=True,
        help="Artifact path (relative to pulse_staging/artifacts/)"
    )
    pulse_sync_parser.add_argument(
        "--type", "-t",
        required=True,
        choices=[t.value for t in ArtifactType],
        help="Artifact type"
    )
    pulse_sync_parser.add_argument(
        "--milestone", "-m",
        help="Override milestone ID (defaults to pulse milestone)"
    )

    # pulse-export command
    pulse_export_parser = subparsers.add_parser(
        "pulse-export",
        help="Archive pulse to history (with staging audit)"
    )
    pulse_export_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip staging audit (NOT RECOMMENDED)"
    )

    # pulse-checkpoint command
    checkpoint_parser = subparsers.add_parser(
        "pulse-checkpoint",
        help="Evaluate pulse maturity and token burden"
    )
    checkpoint_parser.add_argument(
        "--burden",
        choices=["low", "medium", "high"],
        help="Update token burden level"
    )

    # Spec Kit Integration Commands
    spec_constitution_parser = subparsers.add_parser(
        "spec-constitution",
        help="Establish project principles for spec-driven development"
    )
    spec_constitution_parser.add_argument(
        "principles",
        help="Project principles and development guidelines"
    )

    spec_specify_parser = subparsers.add_parser(
        "spec-specify",
        help="Create baseline specification document"
    )
    spec_specify_parser.add_argument(
        "--scope", "-s",
        help="Scope and context for the specification"
    )
    spec_specify_parser.add_argument(
        "--requirements", "-r",
        help="Key requirements to capture"
    )

    spec_plan_parser = subparsers.add_parser(
        "spec-plan",
        help="Create implementation plan from specification"
    )
    spec_plan_parser.add_argument(
        "--approach", "-a",
        help="Implementation approach and strategy"
    )

    spec_tasks_parser = subparsers.add_parser(
        "spec-tasks",
        help="Generate actionable tasks from implementation plan"
    )
    spec_tasks_parser.add_argument(
        "--focus", "-f",
        help="Specific area to focus task generation on"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "check-env":
            print("🔒 Environment Guard: Checking...\n")
            checker = EnvironmentChecker()
            passed, errors, warnings = checker.check_all()

            for warning in warnings:
                print(f"⚠️  {warning}")

            if errors:
                print("\n❌ ENVIRONMENT CHECK FAILED\n")
                for error in errors:
                    print(f"  ✗ {error}")
                sys.exit(1)
            else:
                print("✅ Environment validated")
                if args.strict and warnings:
                    print("\n❌ Strict mode: warnings treated as errors")
                    sys.exit(1)
                sys.exit(0)

        elif args.command == "pulse-init":
            print("🚀 Pulse Init: Starting new work cycle...\n")

            # Environment check first
            checker = EnvironmentChecker()
            passed, errors, _ = checker.check_all()
            if not passed:
                print("❌ Environment check failed. Fix errors before starting pulse.")
                for e in errors:
                    print(f"  ✗ {e}")
                sys.exit(1)

            manager = PulseManager()
            success, message = manager.init_pulse(
                pulse_id=args.id,
                objective=args.obj,
                milestone_id=args.milestone,
                phase=args.phase,
            )

            if success:
                print(f"✅ {message}")
                print("📁 Staging: project_compass/pulse_staging/artifacts/")
                print("📋 Rules injected from vault/")
                sys.exit(0)
            else:
                print(f"❌ {message}")
                sys.exit(1)

        elif args.command == "pulse-status":
            manager = PulseManager()
            status = manager.get_pulse_status()

            if not status["active"]:
                print("💤 " + status["message"])
            else:
                print("🔥 Active Pulse")
                print(f"   ID: {status['pulse_id']}")
                print(f"   Objective: {status['objective'][:80]}...")
                print(f"   Milestone: {status['milestone_id']}")
                print(f"   Artifacts: {status['artifact_count']}")
                print(f"   Rules: {status['instructions_count']}")
                print(f"   Token Burden: {status['token_burden']}")
            sys.exit(0)

        elif args.command == "pulse-sync":
            paths = VesselPaths()

            # Verify file exists
            artifact_file = paths.staging_dir / "artifacts" / args.path
            if not artifact_file.exists():
                print(f"❌ File not found: {artifact_file}")
                print("   Create file in: project_compass/pulse_staging/artifacts/")
                sys.exit(1)

            success, message = register_artifact(
                state_path=paths.vessel_state,
                artifact_path=args.path,
                artifact_type=args.type,
                milestone_id=args.milestone,
            )

            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
                sys.exit(1)
            sys.exit(0)

        elif args.command == "pulse-export":
            print("📦 Pulse Export: Archiving to history...\n")

            paths = VesselPaths()

            if args.force:
                print("⚠️  Force mode: skipping staging audit (NOT RECOMMENDED)")

            result = export_pulse(
                state_path=paths.vessel_state,
                staging_path=paths.staging_dir,
                history_path=paths.history_dir,
            )

            if result["status"] == "SUCCESS":
                print(f"✅ {result['message']}")
                print(f"📁 Exported to: {result['export_path']}")
            else:
                print(f"❌ EXPORT BLOCKED: {result['message']}")
                print(f"   🔧 Action: {result['action_required']}")
                sys.exit(1)
            sys.exit(0)

        elif args.command == "pulse-checkpoint":
            manager = PulseManager()
            state = manager.load_state()

            if not state.active_pulse:
                print("💤 No active pulse to checkpoint")
                sys.exit(1)

            if args.burden:
                state.active_pulse.token_burden = args.burden
                manager.save_state(state)
                print(f"✅ Token burden updated to: {args.burden}")

            # Show maturity assessment
            artifact_count = len(state.active_pulse.artifacts)
            burden = state.active_pulse.token_burden

            print("\n📊 Pulse Maturity Assessment")
            print(f"   Artifacts: {artifact_count}")
            print(f"   Token Burden: {burden}")

            if burden == "high" or artifact_count >= 5:
                print("\n⚠️  RECOMMENDATION: Export pulse and start fresh")
                print("   Run: uv run python -m project_compass.cli pulse-export")
            else:
                print("\n✅ Pulse can continue")
            sys.exit(0)

        # Spec Kit Integration Commands
        elif args.command == "spec-constitution":
            print("📋 Spec Constitution: Establishing project principles...\n")
            from datetime import datetime

            try:
                # Create constitution file in staging
                staging_dir = Path("project_compass/pulse_staging/artifacts")
                constitution_file = staging_dir / "constitution.md"

                constitution_content = f"""# Project Constitution

## Principles
{args.principles}

## Established
Date: {datetime.now().isoformat()}
Tool: Project Compass v2 CLI
"""

                staging_dir.mkdir(parents=True, exist_ok=True)
                constitution_file.write_text(constitution_content)

                print("✅ Project constitution established")
                print(f"📄 File: {constitution_file}")
                print("💡 Register with: uv run compass pulse-sync --path constitution.md --type requirements")

            except Exception as e:
                print(f"❌ Error establishing constitution: {str(e)}")
                sys.exit(1)

        elif args.command == "spec-specify":
            print("📝 Spec Specify: Creating baseline specification...\n")
            from datetime import datetime

            try:
                # Create specification file in staging
                staging_dir = Path("project_compass/pulse_staging/artifacts")
                spec_file = staging_dir / "specification.md"

                scope_text = getattr(args, 'scope', "General project scope")
                requirements_text = getattr(args, 'requirements', "TBD")

                spec_content = f"""# Project Specification

## Scope
{scope_text}

## Requirements
{requirements_text}

## Status
- Created: {datetime.now().isoformat()}
- Tool: Project Compass v2 CLI
- Status: Draft
"""

                staging_dir.mkdir(parents=True, exist_ok=True)
                spec_file.write_text(spec_content)

                print("✅ Specification created")
                print(f"📄 File: {spec_file}")
                print("💡 Register with: uv run compass pulse-sync --path specification.md --type specification")

            except Exception as e:
                print(f"❌ Error creating specification: {str(e)}")
                sys.exit(1)

        elif args.command == "spec-plan":
            print("📋 Spec Plan: Creating implementation plan...\n")
            from datetime import datetime

            try:
                # Create implementation plan file in staging
                staging_dir = Path("project_compass/pulse_staging/artifacts")
                plan_file = staging_dir / "implementation_plan.md"

                approach_text = getattr(args, 'approach', "Standard implementation approach")

                plan_content = f"""# Implementation Plan

## Approach
{approach_text}

## High-Level Steps
1. **Analysis Phase**
   - Requirements review
   - Architecture design
   - Risk assessment

2. **Development Phase**
   - Core implementation
   - Testing strategy
   - Integration planning

3. **Validation Phase**
   - Quality assurance
   - Performance testing
   - Deployment preparation

## Success Criteria
- All requirements met
- Code quality standards maintained
- Performance benchmarks achieved

## Timeline
TBD - To be determined based on scope and resources

## Status
- Created: {datetime.now().isoformat()}
- Tool: Project Compass v2 CLI
- Status: Draft
"""

                staging_dir.mkdir(parents=True, exist_ok=True)
                plan_file.write_text(plan_content)

                print("✅ Implementation plan created")
                print(f"📄 File: {plan_file}")
                print("💡 Register with: uv run compass pulse-sync --path implementation_plan.md --type implementation_plan")

            except Exception as e:
                print(f"❌ Error creating implementation plan: {str(e)}")
                sys.exit(1)

        elif args.command == "spec-tasks":
            print("🎯 Spec Tasks: Generating actionable tasks...\n")
            from datetime import datetime

            try:
                # Create tasks file in staging
                staging_dir = Path("project_compass/pulse_staging/artifacts")
                tasks_file = staging_dir / "tasks.md"

                focus_text = getattr(args, 'focus', "General development tasks")

                tasks_content = f"""# Actionable Tasks

## Focus Area: {focus_text}

## Task Breakdown

### Phase 1: Foundation
- [ ] Set up development environment
- [ ] Initialize project structure
- [ ] Configure CI/CD pipeline
- [ ] Establish coding standards

### Phase 2: Core Development
- [ ] Implement core functionality
- [ ] Write unit tests
- [ ] Integration testing
- [ ] Documentation

### Phase 3: Validation & Deployment
- [ ] Performance testing
- [ ] Security review
- [ ] User acceptance testing
- [ ] Production deployment

## Priority Matrix
- **High Priority**: Environment setup, core functionality
- **Medium Priority**: Testing, documentation
- **Low Priority**: Optimization, advanced features

## Status
- Created: {datetime.now().isoformat()}
- Tool: Project Compass v2 CLI
- Status: Draft
"""

                staging_dir.mkdir(parents=True, exist_ok=True)
                tasks_file.write_text(tasks_content)

                print("✅ Tasks generated")
                print(f"📄 File: {tasks_file}")
                print("💡 Register with: uv run compass pulse-sync --path tasks.md --type implementation_plan")

            except Exception as e:
                print(f"❌ Error generating tasks: {str(e)}")
                sys.exit(1)

    except Exception as e:
        print(f"❌ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
