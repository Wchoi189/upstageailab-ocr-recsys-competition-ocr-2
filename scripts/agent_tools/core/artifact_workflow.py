#!/usr/bin/env python3
"""
AI Agent Artifact Workflow Integration

This script provides a unified CLI interface for AI agents to create, validate,
and manage artifacts following the project's standards.

Uses AgentQMS toolbelt for validated artifact creation (validates BEFORE creation).

Usage:
    # Primary workflow: Create empty template, then fill in manually
    python artifact_workflow.py create --type bug_report --name "my-bug" --title "My Bug Report" --bug-id "BUG-YYYYMMDD-###" --severity "High"

    # Alternative workflow: Create with full content from file (for programmatic creation)
    python artifact_workflow.py create --type bug_report --name "my-bug" --title "My Bug Report" --content-file path/to/content.md --bug-id "BUG-YYYYMMDD-###" --severity "High"

    # Other commands
    python artifact_workflow.py validate --file path/to/artifact.md
    python artifact_workflow.py validate --all
    python artifact_workflow.py update-indexes
    python artifact_workflow.py check-compliance
    python artifact_workflow.py list-templates
    python artifact_workflow.py template-info --type implementation_plan

Workflow Notes:
    - PRIMARY: Create empty template with placeholders, then fill in manually
    - ALTERNATIVE: Use --content-file for programmatic creation with full content
    - Templates render with default placeholders (e.g., "Brief description of the bug.")
    - User content (if provided) is appended after the template
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_bootstrap():
    if "scripts._bootstrap" in sys.modules:
        return sys.modules["scripts._bootstrap"]

    current_dir = Path(__file__).resolve().parent
    for directory in (current_dir, *tuple(current_dir.parents)):
        candidate = directory / "_bootstrap.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "scripts._bootstrap", candidate
            )
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


_BOOTSTRAP = _load_bootstrap()
setup_project_paths = _BOOTSTRAP.setup_project_paths
get_path_resolver = _BOOTSTRAP.get_path_resolver

setup_project_paths()

from agent_qms.toolbelt import AgentQMSToolbelt, ValidationError
from scripts.agent_tools.compliance.validate_artifacts import ArtifactValidator

# Try to import context bundle functions for hooks
try:
    from scripts.agent_tools.core.context_bundle import (
        list_available_bundles,
        load_bundle_definition,
        validate_bundle_files,
    )

    CONTEXT_BUNDLES_AVAILABLE = True
except ImportError:
    CONTEXT_BUNDLES_AVAILABLE = False


class ArtifactWorkflow:
    """Unified workflow for AI agents to manage artifacts.

    Uses AgentQMS toolbelt for validated artifact creation.
    """

    def __init__(self, artifacts_root: str = "artifacts"):
        self.artifacts_root = Path(artifacts_root)
        self.toolbelt = AgentQMSToolbelt()
        self.validator = ArtifactValidator(str(artifacts_root))

    def create_artifact(
        self, artifact_type: str, name: str, title: str, **kwargs
    ) -> str:
        """Create a new artifact following project standards with validation BEFORE creation.

        Note: Implementation plans use Blueprint Protocol Template (PROTO-GOV-003).
        Uses AgentQMS toolbelt which validates before creation.

        **Workflow:**
        - PRIMARY: Create empty template with placeholders, then fill in manually
        - ALTERNATIVE: Use content_file parameter for programmatic creation with full content
        - Templates render with default placeholders (e.g., "Brief description of the bug.")
        - User content (if provided) is appended after the template

        Args:
            artifact_type: Type of artifact (e.g., 'implementation_plan', 'assessment', 'bug_report')
            name: Artifact name (used for reference, title is used for filename)
            title: Artifact title (used for filename generation)
            **kwargs: Additional parameters:
                - content: Markdown content (default: empty, creates template with placeholders)
                - content_file: Path to file containing full markdown content (alternative to content)
                - description: Short description (converted to "## Description\n\n{description}\n" if no content)
                - author: Author name (default: "ai-agent")
                - tags: List of tags (default: [])
                - bug_id: Bug ID for bug_report type (auto-generated if not provided)
                - severity: Severity level for bug_report type (default: "Medium")
        """
        # Show rules and template info BEFORE creation
        print("=" * 70)
        print("ARTIFACT CREATION RULES")
        print("=" * 70)
        self.show_template_info(artifact_type)
        print()
        print("⚠️  CRITICAL RULES:")
        print("  - Use artifact workflow script (you are using it now)")
        print("  - Never create artifacts manually")
        print("  - Follow naming conventions (kebab-case, semantic names)")
        print("  - Include required frontmatter (title, date, type, category, status, version)")
        print("  - Place in correct location (shown above)")
        print("=" * 70)
        print()

        # Pre-generation validation: Check filename and location BEFORE creation
        try:
            # Get artifact metadata to show expected location
            artifact_meta = None
            for atype in self.toolbelt.manifest['artifact_types']:
                if atype['name'] == artifact_type:
                    artifact_meta = atype
                    break

            if not artifact_meta:
                raise ValueError(f"Unknown artifact type: {artifact_type}. Available: {', '.join(self.toolbelt.list_artifact_types())}")

            # For bug reports, generate bug ID early if not provided (needed for filename)
            if artifact_type == "bug_report" and "bug_id" not in kwargs:
                import subprocess
                bug_id = subprocess.check_output(
                    ["uv", "run", "python", "scripts/bug_tools/next_bug_id.py"],
                    text=True
                ).strip()
                kwargs["bug_id"] = bug_id

            # Generate expected filename
            slug = title.lower().replace(' ', '-').replace('_', '-')
            if artifact_type == "bug_report":
                # Bug reports use bug ID prefix
                bug_id = kwargs.get("bug_id", "")
                if bug_id:
                    expected_filename = f"{bug_id}_{slug}.md"
                else:
                    expected_filename = f"{slug}.md"
            else:
                expected_filename = f"{slug}.md"
            expected_location = artifact_meta['location']
            # Resolve path relative to project root (same as toolbelt does)
            expected_path = self.toolbelt.root_path.parent / expected_location / expected_filename

            print(f"📋 Pre-Generation Validation:")
            print(f"   Artifact Type: {artifact_type}")
            print(f"   Expected Filename: {expected_filename}")
            print(f"   Expected Location: {expected_location}")
            print(f"   Expected Path: {expected_path}")
            print()

            # Validate filename BEFORE creation
            try:
                self.toolbelt._validate_filename(expected_filename)
            except Exception as e:
                print(f"❌ Filename validation failed: {e}")
                print(f"   Expected: {expected_filename}")
                raise

            # Check if file already exists
            if expected_path.exists():
                print(f"❌ Artifact already exists: {expected_path}")
                raise ValueError(f"Artifact already exists: {expected_path}")

            print(f"✅ Pre-generation validation passed")
            print(f"🚀 Creating {artifact_type} artifact: {name}")
            print(f"   Title: {title}")
            print()

        except (ValueError, Exception) as e:
            print(f"❌ Pre-generation validation failed: {e}")
            print("   Artifact will NOT be created. Fix issues above and try again.")
            raise

        try:
            # Extract parameters from kwargs
            content = kwargs.pop("content", "")
            author = kwargs.pop("author", "ai-agent")
            tags = kwargs.pop("tags", [])
            description = kwargs.pop("description", "")
            content_file = kwargs.pop("content_file", None)

            # Load content from file if provided
            if content_file:
                content_file_path = Path(content_file)
                if not content_file_path.exists():
                    raise ValueError(f"Content file not found: {content_file}")
                content = content_file_path.read_text(encoding="utf-8")

            # Convert tags from string to list if needed
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

            # If description provided but no content, use description as content
            if description and not content:
                content = f"## Description\n\n{description}\n"

            # For bug reports, generate bug ID and pass additional parameters
            if artifact_type == "bug_report":
                # Generate bug ID if not provided
                if "bug_id" not in kwargs:
                    import subprocess
                    bug_id = subprocess.check_output(
                        ["uv", "run", "python", "scripts/bug_tools/next_bug_id.py"],
                        text=True
                    ).strip()
                    kwargs["bug_id"] = bug_id

                # Set default severity if not provided
                if "severity" not in kwargs:
                    kwargs["severity"] = "Medium"

            # Use AgentQMS toolbelt which validates BEFORE creation
            file_path: str = self.toolbelt.create_artifact(
                artifact_type=artifact_type,
                title=title,
                content=content,
                author=author,
                tags=tags,
                **kwargs  # Pass through bug_id, severity, and other kwargs
            )

            print(f"✅ Created artifact: {file_path}")
            print("✅ Validation passed (validated before creation)")

            # Update indexes
            print("📝 Updating indexes...")
            self.update_indexes()

            # Hook: Notify bundle system about artifact change
            self.update_bundles_on_artifact_change(file_path)

            return file_path

        except ValidationError as e:
            print(f"❌ Validation failed: {e}")
            raise
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
                        Path(__file__).parent.parent
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

    def update_status(self, file_path: str, new_status: str = None, auto_detect: bool = False) -> str:
        """Update artifact status in frontmatter.

        Args:
            file_path: Path to artifact file
            new_status: New status value (if None and auto_detect=True, detects from Progress Tracker)
            auto_detect: If True, automatically detect status from Progress Tracker

        Returns:
            Path to updated artifact
        """
        print(f"🔄 Updating artifact status: {file_path}")

        # Auto-detect status from Progress Tracker if requested
        if auto_detect and new_status is None:
            detected_status = self.toolbelt.detect_completion_from_progress_tracker(file_path)
            if detected_status:
                new_status = detected_status
                print(f"📊 Detected status from Progress Tracker: {detected_status}")
            else:
                print("⚠️  Could not detect status from Progress Tracker. Please specify --status")
                return ""

        if new_status is None:
            print("❌ No status specified. Use --status or --auto-detect")
            return ""

        try:
            updated_path = self.toolbelt.update_artifact_status(file_path, new_status)
            print(f"✅ Updated artifact status to: {new_status}")
            print(f"   File: {updated_path}")

            # Check for any remaining mismatches
            mismatch = self.toolbelt.check_status_mismatch(updated_path)
            if mismatch:
                print(f"⚠️  Warning: Status mismatch detected!")
                print(f"   Frontmatter: {mismatch['frontmatter_status']}")
                print(f"   Progress Tracker: {mismatch['detected_status']}")
            else:
                print("✅ Status matches Progress Tracker")

            return updated_path
        except ValidationError as e:
            print(f"❌ Validation failed: {e}")
            raise
        except Exception as e:
            print(f"❌ Error updating status: {e}")
            raise

    def check_status_mismatches(self) -> list:
        """Check all implementation plans for status mismatches.

        Returns:
            List of mismatches found
        """
        print("🔍 Checking for status mismatches...")

        mismatches = []
        implementation_plans_dir = Path("artifacts/implementation_plans")

        if not implementation_plans_dir.exists():
            print("⚠️  No implementation plans directory found")
            return mismatches

        for plan_file in implementation_plans_dir.glob("*.md"):
            mismatch = self.toolbelt.check_status_mismatch(str(plan_file))
            if mismatch:
                mismatches.append(mismatch)
                print(f"⚠️  Mismatch found: {plan_file.name}")
                print(f"   Frontmatter: {mismatch['frontmatter_status']}")
                print(f"   Progress Tracker: {mismatch['detected_status']}")

        if not mismatches:
            print("✅ No status mismatches found")

        return mismatches

    def get_available_templates(self) -> list[str]:
        """Get list of available artifact types from AgentQMS."""
        return self.toolbelt.list_artifact_types()

    def show_template_info(self, template_type: str) -> None:
        """Show information about a specific artifact type."""
        artifact_types = self.toolbelt.list_artifact_types()
        if template_type not in artifact_types:
            print(f"❌ Unknown artifact type: {template_type}")
            print(f"   Available types: {', '.join(artifact_types)}")
            return

        # Get artifact metadata from manifest
        artifact_meta = None
        for atype in self.toolbelt.manifest['artifact_types']:
            if atype['name'] == template_type:
                artifact_meta = atype
                break

        if artifact_meta:
            print(f"📋 Artifact Type: {template_type}")
            print(f"   Description: {artifact_meta.get('description', 'N/A')}")
            print(f"   📍 Location: {artifact_meta.get('location', 'N/A')}")
            print(f"   📄 Template: {artifact_meta.get('template', 'N/A')}")
            print(f"   ✅ Schema: {artifact_meta.get('schema', 'N/A')}")

            # Show template file path if it exists
            template_path = self.toolbelt.root_path / artifact_meta.get('template', '')
            if template_path.exists():
                print(f"   📖 Template file: {template_path}")
                print(f"      (Review this file to see required structure)")
        else:
            print(f"❌ Could not find metadata for: {template_type}")

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
    create_parser.add_argument("--content-file", help="Path to file containing full markdown content (for bug reports with detailed content)")
    create_parser.add_argument("--tags", help="Comma-separated tags")
    create_parser.add_argument("--bug-id", help="Bug ID (for bug_report type, auto-generated if not provided)")
    create_parser.add_argument("--severity", help="Severity level (for bug_report type, default: Medium)")
    create_parser.add_argument(
        "--interactive", action="store_true", help="Interactive mode"
    )

    # Validate command
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

    # Update status command
    update_status_parser = subparsers.add_parser(
        "update-status", help="Update artifact status in frontmatter"
    )
    update_status_parser.add_argument("--file", required=True, help="Path to artifact file")
    update_status_parser.add_argument(
        "--status", help="New status value (draft, in-progress, completed)"
    )
    update_status_parser.add_argument(
        "--auto-detect", action="store_true",
        help="Auto-detect status from Progress Tracker"
    )

    # Check status mismatches command
    subparsers.add_parser(
        "check-status-mismatches", help="Check for status mismatches between frontmatter and Progress Tracker"
    )

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
                if args.content_file:
                    kwargs["content_file"] = args.content_file
                if args.tags:
                    kwargs["tags"] = [tag.strip() for tag in args.tags.split(",")]
                if args.bug_id:
                    kwargs["bug_id"] = args.bug_id
                if args.severity:
                    kwargs["severity"] = args.severity

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

        elif args.command == "update-status":
            if not args.status and not args.auto_detect:
                print("❌ Please specify --status or --auto-detect")
                return 1
            updated_path = workflow.update_status(
                args.file, args.status, args.auto_detect
            )
            return 0 if updated_path else 1

        elif args.command == "check-status-mismatches":
            mismatches = workflow.check_status_mismatches()
            return 0 if len(mismatches) == 0 else 1

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
