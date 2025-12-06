#!/usr/bin/env python3
"""
AI Agent Artifact Workflow Integration

This script provides a unified interface for AI agents to create, validate,
and manage artifacts following the project's standards.

Usage:
    python artifact_workflow.py create --type implementation_plan --name "my-feature" --title "My Feature Plan"
    python artifact_workflow.py validate --file path/to/artifact.md
    python artifact_workflow.py validate --all
    python artifact_workflow.py update-indexes
    python artifact_workflow.py check-compliance
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import cast

from AgentQMS.agent_tools.compliance.validate_artifacts import ArtifactValidator
from AgentQMS.agent_tools.compliance.validate_boundaries import BoundaryValidator
from AgentQMS.agent_tools.core.artifact_templates import (
    ArtifactTemplates,
    create_artifact,
)
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


def _assert_clean_boundaries() -> None:
    violations = BoundaryValidator().validate()
    errors = [v for v in violations if v.severity == "error"]
    if errors:
        formatted = "\n".join(f"- {v.path}: {v.message}" for v in errors)
        raise RuntimeError(
            "Boundary validation failed. Resolve the following issues before creating artifacts:\n"
            f"{formatted}"
        )


_assert_clean_boundaries()

# Try to import context bundle functions for hooks
try:
    from AgentQMS.agent_tools.core.context_bundle import (
        list_available_bundles,
        load_bundle_definition,
        validate_bundle_files,
    )

    CONTEXT_BUNDLES_AVAILABLE = True
except ImportError:
    CONTEXT_BUNDLES_AVAILABLE = False


class ArtifactWorkflow:
    """Unified workflow for AI agents to manage artifacts."""

    def __init__(self, artifacts_root: str | Path | None = None):
        # Default to the configured artifacts directory if none is provided
        if artifacts_root is None:
            # Import lazily to avoid circular imports at module load time
            from AgentQMS.agent_tools.utils.paths import get_artifacts_dir

            root = get_artifacts_dir()
        else:
            root = artifacts_root

        self.artifacts_root = Path(root)
        self.templates = ArtifactTemplates()
        self.validator = ArtifactValidator(self.artifacts_root)

    def create_artifact(
        self, artifact_type: str, name: str, title: str, auto_validate: bool = True, auto_update_indexes: bool = True, track: bool = True, **kwargs
    ) -> str:
        """Create a new artifact following project standards.

        Note: Implementation plans use Blueprint Protocol Template (PROTO-GOV-003).
        See AgentQMS/knowledge/protocols/governance/implementation_plan_protocol.md

        Args:
            artifact_type: Type of artifact to create
            name: Artifact name (kebab-case)
            title: Artifact title
            auto_validate: Automatically validate after creation (default: True)
            auto_update_indexes: Automatically update indexes after creation (default: True)
            track: Auto-register in tracking DB (default: True for trackable types)
            **kwargs: Additional arguments passed to create_artifact
        """
        print(f"🚀 Creating {artifact_type} artifact: {name}")

        try:
            # Create the artifact
            file_path: str = create_artifact(
                artifact_type, name, title, str(self.artifacts_root), **kwargs
            )

            print(f"✅ Created artifact: {file_path}")

            # Auto-execution: Validate the created artifact
            if auto_validate:
                print("🔍 Validating created artifact...")
                results = self.validator.validate_single_file(Path(file_path))

                if results["valid"]:
                    print("✅ Artifact validation passed")
                else:
                    print("❌ Artifact validation failed:")
                    for error in results["errors"]:
                        print(f"   • {error}")
                    # Continue even if validation fails - return path

            # Auto-execution: Register in tracking DB
            if track:
                self._register_in_tracking(artifact_type, file_path, title, kwargs.get("owner"))

            # Auto-execution: Update indexes
            if auto_update_indexes:
                print("📝 Updating indexes...")
                self.update_indexes()

            # Hook: Notify bundle system about artifact change
            self.update_bundles_on_artifact_change(file_path)

            # Auto-execution: Suggest next steps
            self._suggest_next_steps(artifact_type, file_path)

            return file_path

        except Exception as e:
            print(f"❌ Error creating artifact: {e}")
            raise

    def validate_artifact(self, file_path: str) -> bool:
        """Validate a specific artifact."""
        print(f"🔍 Validating artifact: {file_path}")

        result = self.validator.validate_single_file(Path(file_path))

        if result["valid"]:
            print("✅ Artifact validation passed")
            return True
        else:
            print("❌ Artifact validation failed:")
            for error in result["errors"]:
                print(f"   • {error}")
            return False

    def validate_all(self) -> bool:
        """Validate all artifacts."""
        print("🔍 Validating all artifacts...")

        results = self.validator.validate_all()
        report = self.validator.generate_report(results)
        print(report)

        # Check if any validation failed
        failed_count = sum(1 for r in results if not r["valid"])
        if failed_count > 0:
            print(f"\n❌ {failed_count} artifacts failed validation")
            return False
        else:
            print("\n✅ All artifacts passed validation")
            return True

    def update_indexes(self) -> bool:
        """Update all artifact indexes."""
        print("📝 Updating artifact indexes...")

        try:
            # Run the index updater
            result = subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).parent.parent.parent
                        / "toolkit"
                        / "documentation"
                        / "update_artifact_indexes.py"
                    ),
                    "--all",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ Indexes updated successfully")
                return True
            else:
                print(f"❌ Error updating indexes: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error running index updater: {e}")
            return False

    def check_compliance(self) -> dict:
        """Check overall compliance with artifact standards."""
        print("📊 Checking artifact compliance...")

        results = self.validator.validate_all()
        total_files = len(results)
        valid_files = sum(1 for r in results if r["valid"])
        invalid_files = total_files - valid_files

        compliance_rate = (valid_files / total_files * 100) if total_files > 0 else 0

        compliance_report = {
            "total_files": total_files,
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "compliance_rate": compliance_rate,
            "violations": [r for r in results if not r["valid"]],
        }

        print("📈 Compliance Report:")
        print(f"   Total files: {total_files}")
        print(f"   Valid files: {valid_files}")
        print(f"   Invalid files: {invalid_files}")
        print(f"   Compliance rate: {compliance_rate:.1f}%")

        if invalid_files > 0:
            print("\n❌ Violations found:")
            violations = compliance_report.get("violations", [])
            if isinstance(violations, list):
                for violation in violations:
                    print(f"   • {violation['file']}")
                    for error in violation["errors"]:
                        print(f"     - {error}")

        return compliance_report

    def _register_in_tracking(
        self, artifact_type: str, file_path: str, title: str, owner: str | None
    ) -> None:
        """Register artifact in tracking database.

        Args:
            artifact_type: Type of artifact
            file_path: Path to artifact file
            title: Artifact title
            owner: Artifact owner
        """
        try:
            from AgentQMS.agent_tools.utilities.tracking_integration import (
                register_artifact_in_tracking,
            )

            print("📊 Registering in tracking database...")
            result = register_artifact_in_tracking(
                artifact_type, file_path, title, owner, track_flag=True
            )

            if result.get("tracked"):
                print(
                    f"✅ Registered in tracking DB: {result.get('tracking_type')} "
                    f"(key: {result.get('tracking_key')})"
                )
            elif not result.get("should_track"):
                # Not an error - this artifact type just isn't tracked
                pass
            else:
                # Failed to track but should have been tracked
                reason = result.get("reason", "unknown")
                print(f"⚠️  Tracking registration skipped: {reason}")

        except ImportError:
            # Tracking integration not available - silently continue
            pass
        except Exception as e:
            # Don't fail artifact creation if tracking fails
            print(f"⚠️  Tracking registration failed: {e}")

    def _suggest_next_steps(self, artifact_type: str, file_path: str) -> None:
        """Suggest next steps after artifact creation."""
        print("\n💡 Suggested next steps:")

        # Suggest validation if not already done
        print("   1. Review the artifact:")
        print(f"      {file_path}")

        # Suggest compliance check
        print("   2. Run compliance check:")
        print("      cd AgentQMS/interface && make compliance")

        # Suggest context loading if applicable
        if artifact_type == "implementation_plan":
            print("   3. Load planning context:")
            print("      cd AgentQMS/interface && make context-plan")
        elif artifact_type == "bug_report":
            print("   3. Load debugging context:")
            print("      cd AgentQMS/interface && make context-debug")

        # Suggest related workflows
        if artifact_type in ["implementation_plan", "design"]:
            print("   4. Consider creating related artifacts:")
            print("      cd AgentQMS/interface && make create-assessment NAME=... TITLE=...")

        print()

    def update_bundles_on_artifact_change(self, artifact_path: str) -> None:
        """
        Hook called after artifact creation/modification.

        Checks if the artifact should be included in any bundles and logs
        a message. Bundle definitions are manually curated, so this just
        provides visibility for potential bundle updates.

        Args:
            artifact_path: Path to the artifact that was created/modified
        """
        if not CONTEXT_BUNDLES_AVAILABLE:
            return  # Silent skip if bundle system not available

        try:
            artifact_path_obj = Path(artifact_path)

            # Check if any bundles reference this artifact or its directory
            available_bundles = list_available_bundles()

            if not available_bundles:
                return

            # Log that artifact was created (for potential bundle inclusion)
            # Note: Bundle definitions are manually curated, so we just log
            # This hook can be extended later to suggest bundle updates

            # Check if artifact is already in any bundle
            for bundle_name in available_bundles:
                try:
                    bundle_def = load_bundle_definition(bundle_name)
                    bundle_files = validate_bundle_files(bundle_def)

                    # Check if artifact matches any bundle file pattern
                    artifact_relative = artifact_path_obj.relative_to(
                        self.artifacts_root.parent.parent
                    )
                    if str(artifact_relative) in bundle_files:
                        # Artifact is already in a bundle - all good
                        return
                except Exception:
                    # Skip bundles that can't be loaded
                    continue

            # Artifact not in any bundle - this is expected for most artifacts
            # Only log at debug level or when explicitly requested
            # For now, we'll keep this silent to avoid noise

        except Exception:
            # Silently fail - bundle updates are not critical
            pass

    def get_available_templates(self) -> list[str]:
        """Get list of available artifact templates."""
        return cast("list[str]", self.templates.get_available_templates())

    def show_template_info(self, template_type: str) -> None:
        """Show information about a specific template."""
        template = self.templates.get_template(template_type)
        if not template:
            print(f"❌ Unknown template type: {template_type}")
            return

        print(f"📋 Template: {template_type}")
        print(f"   Directory: {template['directory']}")
        print(f"   Filename pattern: {template['filename_pattern']}")
        print(f"   Frontmatter fields: {', '.join(template['frontmatter'].keys())}")

    def interactive_create(self) -> str:
        """Interactive artifact creation for AI agents."""
        print("🤖 AI Agent Artifact Creation")
        print("=" * 40)

        # Show available templates
        templates = self.get_available_templates()
        print("Available artifact types:")
        for i, template_type in enumerate(templates, 1):
            print(f"  {i}. {template_type}")

        # Get artifact type
        while True:
            try:
                choice = input(f"\nSelect artifact type (1-{len(templates)}): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(templates):
                    artifact_type = templates[int(choice) - 1]
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
            except KeyboardInterrupt:
                print("\n❌ Cancelled by user")
                return ""

        # Get artifact name
        name = input("Enter artifact name (kebab-case): ").strip()
        if not name:
            print("❌ Name is required")
            return ""

        # Get title
        title = input("Enter artifact title: ").strip()
        if not title:
            print("❌ Title is required")
            return ""

        # Get additional information
        print("\nAdditional information (optional):")
        description = input("Description: ").strip()
        tags = input("Tags (comma-separated): ").strip()

        # Prepare kwargs
        kwargs = {}
        if description:
            kwargs["description"] = description
        if tags:
            kwargs["tags"] = tags  # Keep as string to match expected type

        # Create artifact
        try:
            file_path = self.create_artifact(artifact_type, name, title, **kwargs)
            print(f"\n🎉 Successfully created artifact: {file_path}")
            return file_path
        except Exception as e:
            print(f"\n❌ Failed to create artifact: {e}")
            return ""


def main():
    """Main entry point for the artifact workflow."""
    parser = argparse.ArgumentParser(description="AI Agent Artifact Workflow")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new artifact")
    create_parser.add_argument("--type", required=True, help="Artifact type")
    create_parser.add_argument("--name", required=True, help="Artifact name")
    create_parser.add_argument("--title", required=True, help="Artifact title")
    create_parser.add_argument("--description", help="Artifact description")
    create_parser.add_argument("--tags", help="Comma-separated tags")
    create_parser.add_argument(
        "--branch", help="Git branch name (auto-detected if not provided, defaults to main)"
    )
    create_parser.add_argument(
        "--interactive", action="store_true", help="Interactive mode"
    )
    validate_parser = subparsers.add_parser("validate", help="Validate artifacts")
    validate_parser.add_argument("--file", help="Validate specific file")
    validate_parser.add_argument(
        "--all", action="store_true", help="Validate all artifacts"
    )
    # Update indexes command
    subparsers.add_parser("update-indexes", help="Update artifact indexes")

    # Check compliance command
    subparsers.add_parser("check-compliance", help="Check artifact compliance")

    # List templates command
    subparsers.add_parser("list-templates", help="List available templates")

    # Template info command
    template_info_parser = subparsers.add_parser(
        "template-info", help="Show template information"
    )
    template_info_parser.add_argument("--type", required=True, help="Template type")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    workflow = ArtifactWorkflow()

    try:
        if args.command == "create":
            if args.interactive:
                file_path = workflow.interactive_create()
                return 0 if file_path else 1
            else:
                kwargs = {}
                if args.description:
                    kwargs["description"] = args.description
                if args.tags:
                    kwargs["tags"] = [tag.strip() for tag in args.tags.split(",")]
                if args.branch:
                    kwargs["branch"] = args.branch

                file_path = workflow.create_artifact(
                    args.type, args.name, args.title, **kwargs
                )
                return 0 if file_path else 1

        elif args.command == "validate":
            if args.file:
                success = workflow.validate_artifact(args.file)
                return 0 if success else 1
            elif args.all:
                success = workflow.validate_all()
                return 0 if success else 1
            else:
                print("❌ Please specify --file or --all")
                return 1

        elif args.command == "update-indexes":
            success = workflow.update_indexes()
            return 0 if success else 1

        elif args.command == "check-compliance":
            compliance_report = workflow.check_compliance()
            return 0 if compliance_report["invalid_files"] == 0 else 1

        elif args.command == "list-templates":
            templates = workflow.get_available_templates()
            print("Available artifact templates:")
            for template in templates:
                print(f"  - {template}")
            return 0

        elif args.command == "template-info":
            workflow.show_template_info(args.type)
            return 0

        else:
            print(f"❌ Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
