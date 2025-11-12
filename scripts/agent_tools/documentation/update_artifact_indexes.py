#!/usr/bin/env python3
"""
Artifact Index Updater

This script automatically updates INDEX.md files in artifact directories
to maintain accurate listings of all artifacts.

Usage:
    python update_artifact_indexes.py
    python update_artifact_indexes.py --directory docs/artifacts/assessments/
    python update_artifact_indexes.py --all
"""

import argparse
from datetime import datetime
from pathlib import Path


class ArtifactIndexUpdater:
    """Updates INDEX.md files in artifact directories."""

    def __init__(
        self, artifacts_root: str = "docs/artifacts", public_only: bool = True
    ):
        self.artifacts_root = Path(artifacts_root)
        self.public_only = public_only

        # Define directory structure and their purposes
        self.directories = {
            "assessments": {
                "title": "Assessments",
                "description": "Audits, evaluations, and assessments of system components and processes.",
                "icon": "📊",
            },
            "bug_reports": {
                "title": "Bug Reports",
                "description": "Documentation of bugs, issues, and their resolution.",
                "icon": "🐛",
            },
            "completed_plans": {
                "title": "Completed Plans",
                "description": "Archived implementation plans, migration plans, and completion summaries.",
                "icon": "✅",
            },
            "design_documents": {
                "title": "Design Documents",
                "description": "Architectural designs, specifications, and technical documentation.",
                "icon": "🏗️",
            },
            "experiment_logs": {
                "title": "Experiment Logs",
                "description": "Logs and results from experiments and testing.",
                "icon": "🧪",
            },
            "implementation_plans": {
                "title": "Implementation Plans",
                "description": "Active implementation plans and development roadmaps.",
                "icon": "🚀",
            },
            "research": {
                "title": "Research",
                "description": "Research findings, investigations, and analysis.",
                "icon": "🔬",
            },
            "templates": {
                "title": "Templates",
                "description": "Reusable templates and guidelines for creating artifacts.",
                "icon": "📋",
            },
        }

    def get_artifact_info(self, file_path: Path) -> dict:
        """Extract information from an artifact file."""
        info = {
            "filename": file_path.name,
            "title": file_path.stem,
            "date": None,
            "type": None,
            "status": None,
            "description": None,
            "tags": [],
        }

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Extract frontmatter
            if content.startswith("---"):
                frontmatter_end = content.find("---", 3)
                if frontmatter_end != -1:
                    frontmatter_content = content[3:frontmatter_end]

                    # Parse frontmatter
                    for line in frontmatter_content.split("\n"):
                        line = line.strip()
                        if ":" in line and not line.startswith("#"):
                            key, value = line.split(":", 1)
                            key = key.strip()
                            value = value.strip().strip("\"'")

                            if key == "title":
                                info["title"] = value
                            elif key == "date":
                                info["date"] = value
                            elif key == "type":
                                info["type"] = value
                            elif key == "status":
                                info["status"] = value
                            elif key == "tags":
                                if isinstance(value, str):
                                    info["tags"] = [t.strip() for t in value.split(",")]
                                elif isinstance(value, list):
                                    info["tags"] = value

            # Extract description from content (first paragraph after title)
            lines = content.split("\n")
            in_content = False
            description_lines = []

            for line in lines:
                line = line.strip()
                if line.startswith("# ") and not in_content:
                    in_content = True
                    continue
                elif in_content and line and not line.startswith("#"):
                    if line.startswith("##"):
                        break
                    description_lines.append(line)
                    if len(description_lines) >= 2:  # Take first 2 lines
                        break

            if description_lines:
                info["description"] = (
                    " ".join(description_lines)[:200] + "..."
                    if len(" ".join(description_lines)) > 200
                    else " ".join(description_lines)
                )

        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

        return info

    def is_internal(self, artifact: dict) -> bool:
        """Check if artifact has internal tag (should be suppressed in public indices)."""
        tags = artifact.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        return "internal" in tags or "Internal" in tags

    def filter_public(
        self, artifacts: list[dict], public_only: bool = True
    ) -> list[dict]:
        """Filter out internal docs if public_only is True."""
        if not public_only:
            return artifacts
        return [a for a in artifacts if not self.is_internal(a)]

    def generate_index_content(self, directory: Path, artifacts: list[dict]) -> str:
        """Generate INDEX.md content for a directory."""
        dir_name = directory.name
        dir_info = self.directories.get(
            dir_name,
            {
                "title": dir_name.title(),
                "description": f"Artifacts in the {dir_name} directory.",
                "icon": "📁",
            },
        )

        # Sort artifacts by date (newest first), handling None values
        artifacts.sort(key=lambda x: x.get("date") or "", reverse=True)

        lines = []
        lines.append(f"# {dir_info['icon']} {dir_info['title']}")
        lines.append("")
        lines.append(dir_info["description"])
        lines.append("")
        lines.append(
            f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"**Total Artifacts**: {len(artifacts)}")
        lines.append("")

        if not artifacts:
            lines.append("*No artifacts found in this directory.*")
            return "\n".join(lines)

        # Group by status if available
        status_groups = {}
        for artifact in artifacts:
            status = artifact.get("status", "unknown")
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(artifact)

        # Create sections by status
        status_order = [
            "active",
            "draft",
            "completed",
            "archived",
            "deprecated",
            "unknown",
        ]
        for status in status_order:
            if status in status_groups:
                status_artifacts = status_groups[status]
                status_title = status.title() if status != "unknown" else "Other"
                lines.append(f"## {status_title} ({len(status_artifacts)})")
                lines.append("")

                for artifact in status_artifacts:
                    # Create link
                    link = f"[{artifact['title']}]({artifact['filename']})"

                    # Add metadata
                    metadata = []
                    if artifact.get("date"):
                        try:
                            # Parse ISO date and format nicely
                            date_str = artifact["date"].split("T")[
                                0
                            ]  # Get just the date part
                            metadata.append(f"📅 {date_str}")
                        except Exception:
                            pass

                    if artifact.get("type"):
                        type_icon = {
                            "implementation_plan": "🚀",
                            "assessment": "📊",
                            "design": "🏗️",
                            "research": "🔬",
                            "template": "📋",
                            "bug_report": "🐛",
                            "session_note": "📝",
                        }.get(artifact["type"], "📄")
                        metadata.append(f"{type_icon} {artifact['type']}")

                    # Add description if available
                    description = ""
                    if artifact.get("description"):
                        description = f" - {artifact['description']}"

                    # Format the entry
                    if metadata:
                        lines.append(f"- {link} ({', '.join(metadata)}){description}")
                    else:
                        lines.append(f"- {link}{description}")

                lines.append("")

        # Add summary statistics
        lines.append("## Summary")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        for status in status_order:
            if status in status_groups:
                count = len(status_groups[status])
                status_display = status.title() if status != "unknown" else "Other"
                lines.append(f"| {status_display} | {count} |")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*This index is automatically generated. Do not edit manually.*")

        return "\n".join(lines)

    def update_directory_index(self, directory: Path) -> bool:
        """Update INDEX.md for a specific directory."""
        if not directory.exists():
            print(f"Directory does not exist: {directory}")
            return False

        # Find all markdown files (excluding INDEX.md)
        artifacts = []
        for file_path in directory.rglob("*.md"):
            if file_path.is_file() and file_path.name != "INDEX.md":
                artifact_info = self.get_artifact_info(file_path)
                artifacts.append(artifact_info)

        # Filter out internal docs if public_only
        artifacts = self.filter_public(artifacts, self.public_only)

        # Generate index content
        index_content = self.generate_index_content(directory, artifacts)

        # Write INDEX.md
        index_path = directory / "INDEX.md"
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(index_content)
            print(f"✅ Updated {index_path}")
            return True
        except Exception as e:
            print(f"❌ Error updating {index_path}: {e}")
            return False

    def update_all_indexes(self) -> dict[str, bool]:
        """Update all INDEX.md files in the artifacts directory."""
        results = {}

        # Update main directories
        for dir_name in self.directories:
            dir_path = self.artifacts_root / dir_name
            if dir_path.exists():
                results[str(dir_path)] = self.update_directory_index(dir_path)

        # Update subdirectories in completed_plans
        completed_plans_dir = self.artifacts_root / "completed_plans"
        if completed_plans_dir.exists():
            for subdir in completed_plans_dir.iterdir():
                if subdir.is_dir():
                    results[str(subdir)] = self.update_directory_index(subdir)

        return results

    def update_master_index(self) -> bool:
        """Update the master INDEX.md in the artifacts root."""
        master_index_path = self.artifacts_root / "MASTER_INDEX.md"

        # Collect all artifacts
        all_artifacts = []
        for dir_name in self.directories:
            dir_path = self.artifacts_root / dir_name
            if dir_path.exists():
                for file_path in dir_path.rglob("*.md"):
                    if file_path.is_file() and file_path.name != "INDEX.md":
                        artifact_info = self.get_artifact_info(file_path)
                        artifact_info["directory"] = dir_name
                        all_artifacts.append(artifact_info)

        # Filter out internal docs if public_only
        all_artifacts = self.filter_public(all_artifacts, self.public_only)

        # Sort by date (newest first), handling None values
        all_artifacts.sort(key=lambda x: x.get("date") or "", reverse=True)

        # Generate master index content
        lines = []
        lines.append("# Master Artifact Registry")
        lines.append("")
        lines.append(
            f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"**Total Artifacts**: {len(all_artifacts)}")
        lines.append("")

        # Group by directory
        by_directory = {}
        for artifact in all_artifacts:
            dir_name = artifact["directory"]
            if dir_name not in by_directory:
                by_directory[dir_name] = []
            by_directory[dir_name].append(artifact)

        lines.append("## By Category")
        for dir_name, artifacts in by_directory.items():
            dir_info = self.directories.get(
                dir_name, {"title": dir_name.title(), "icon": "📁"}
            )
            lines.append(
                f"- [{dir_info['icon']} {dir_info['title']}]({dir_name}/INDEX.md) ({len(artifacts)})"
            )

        lines.append("")

        # Group by status
        by_status = {}
        for artifact in all_artifacts:
            status = artifact.get("status", "unknown")
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(artifact)

        lines.append("## By Status")
        status_order = [
            "active",
            "draft",
            "completed",
            "archived",
            "deprecated",
            "unknown",
        ]
        for status in status_order:
            if status in by_status:
                count = len(by_status[status])
                status_display = status.title() if status != "unknown" else "Other"
                lines.append(f"- **{status_display}**: {count} artifacts")

        lines.append("")

        # Recent activity
        lines.append("## Recent Activity")
        recent_artifacts = all_artifacts[:10]  # Last 10 artifacts
        for artifact in recent_artifacts:
            date_str = artifact.get("date", "Unknown date")
            if "T" in date_str:
                date_str = date_str.split("T")[0]
            dir_info = self.directories.get(artifact["directory"], {"icon": "📁"})
            lines.append(
                f"- {date_str}: {dir_info['icon']} {artifact['title']} ({artifact['directory']})"
            )

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(
            "*This master index is automatically generated. Do not edit manually.*"
        )

        # Write master index
        try:
            with open(master_index_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"✅ Updated {master_index_path}")
            return True
        except Exception as e:
            print(f"❌ Error updating {master_index_path}: {e}")
            return False


def main():
    """Main entry point for the index updater."""
    parser = argparse.ArgumentParser(description="Update artifact INDEX.md files")
    parser.add_argument("--directory", help="Update index for specific directory")
    parser.add_argument("--all", action="store_true", help="Update all indexes")
    parser.add_argument(
        "--master", action="store_true", help="Update master index only"
    )
    parser.add_argument(
        "--artifacts-root",
        default="docs/artifacts",
        help="Root directory for artifacts",
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Include internal docs (default: exclude from public indices)",
    )

    args = parser.parse_args()

    updater = ArtifactIndexUpdater(
        args.artifacts_root, public_only=not args.include_internal
    )

    if args.directory:
        directory = Path(args.directory)
        success = updater.update_directory_index(directory)
        if success:
            print("✅ Directory index updated successfully")
        else:
            print("❌ Failed to update directory index")
            return 1

    elif args.master:
        success = updater.update_master_index()
        if success:
            print("✅ Master index updated successfully")
        else:
            print("❌ Failed to update master index")
            return 1

    elif args.all:
        print("Updating all artifact indexes...")
        results = updater.update_all_indexes()
        master_success = updater.update_master_index()

        successful = sum(1 for success in results.values() if success)
        total = len(results)

        print("\n📊 Results:")
        print(f"  - Directory indexes: {successful}/{total} updated")
        print(f"  - Master index: {'✅' if master_success else '❌'}")

        if successful == total and master_success:
            print("✅ All indexes updated successfully")
            return 0
        else:
            print("❌ Some indexes failed to update")
            return 1

    else:
        # Default: update all
        print("Updating all artifact indexes...")
        results = updater.update_all_indexes()
        master_success = updater.update_master_index()

        successful = sum(1 for success in results.values() if success)
        total = len(results)

        print("\n📊 Results:")
        print(f"  - Directory indexes: {successful}/{total} updated")
        print(f"  - Master index: {'✅' if master_success else '❌'}")

        if successful == total and master_success:
            print("✅ All indexes updated successfully")
            return 0
        else:
            print("❌ Some indexes failed to update")
            return 1


if __name__ == "__main__":
    exit(main())
