#!/usr/bin/env python3
"""
Frontmatter Generation Tool

Automatically adds frontmatter to files missing it based on file analysis and location.
Migrated from deprecated AgentQMS.toolkit.maintenance.add_frontmatter (v0.3.2+)
"""

import re
from datetime import datetime
from pathlib import Path


class FrontmatterGenerator:
    """Generates frontmatter for files missing it"""

    def __init__(self):
        self.valid_categories = {
            "development",
            "architecture",
            "evaluation",
            "compliance",
            "reference",
            "planning",
            "research",
            "troubleshooting",
        }
        self.valid_types = {
            "implementation_plan",
            "assessment",
            "design",
            "research",
            "template",
            "bug_report",
            "session_note",
            "completion_summary",
        }

        # Category mapping based on directory structure
        self.category_mapping = {
            "assessments": "evaluation",
            "design_documents": "architecture",
            "implementation_plans": "planning",
            "research": "research",
            "templates": "reference",
            "bug_reports": "troubleshooting",
            "completed_plans": "reference",
            "session_notes": "reference",
        }

        # Type mapping based on directory structure
        self.type_mapping = {
            "assessments": "assessment",
            "design_documents": "design",
            "implementation_plans": "implementation_plan",
            "research": "research",
            "templates": "template",
            "bug_reports": "bug_report",
            "session_notes": "session_note",
            "completion_summaries": "completion_summary",
        }

    def analyze_file(self, file_path: str) -> dict[str, str]:
        """Analyze file to determine appropriate frontmatter values"""
        path = Path(file_path)

        # Extract information from file path
        filename = path.stem
        directory = path.parent.name
        parent_directory = (
            path.parent.parent.name if path.parent.parent.name != "artifacts" else None
        )

        # Determine type from directory structure
        file_type = self._determine_type(directory, parent_directory, filename)

        # Determine category from directory structure
        category = self._determine_category(directory, parent_directory, filename)

        # Extract title from filename or content
        title = self._extract_title(filename, file_path)

        # Determine status
        status = self._determine_status(directory, filename)

        # Generate tags
        tags = self._generate_tags(file_type, category, directory)

        return {
            "title": title,
            "type": file_type,
            "category": category,
            "status": status,
            "tags": tags,
        }

    def _determine_type(
        self, directory: str, parent_directory: str | None, filename: str
    ) -> str:
        """Determine file type from directory structure"""
        # Check parent directory first
        if parent_directory and parent_directory in self.type_mapping:
            return self.type_mapping[parent_directory]

        # Check current directory
        if directory in self.type_mapping:
            return self.type_mapping[directory]

        # Check filename patterns
        if "assessment" in filename.lower():
            return "assessment"
        elif "design" in filename.lower():
            return "design"
        elif "implementation" in filename.lower() or "plan" in filename.lower():
            return "implementation_plan"
        elif "research" in filename.lower():
            return "research"
        elif "template" in filename.lower():
            return "template"
        elif "bug" in filename.lower():
            return "bug_report"
        elif "session" in filename.lower():
            return "session_note"
        elif "completion" in filename.lower() or "summary" in filename.lower():
            return "completion_summary"

        # Default fallback
        return "reference"

    def _determine_category(
        self, directory: str, parent_directory: str | None, filename: str
    ) -> str:
        """Determine category from directory structure"""
        # Check parent directory first
        if parent_directory and parent_directory in self.category_mapping:
            return self.category_mapping[parent_directory]

        # Check current directory
        if directory in self.category_mapping:
            return self.category_mapping[directory]

        # Check filename patterns
        if "assessment" in filename.lower():
            return "evaluation"
        elif "design" in filename.lower():
            return "architecture"
        elif "implementation" in filename.lower() or "plan" in filename.lower():
            return "planning"
        elif "research" in filename.lower():
            return "research"
        elif "template" in filename.lower():
            return "reference"
        elif "bug" in filename.lower():
            return "troubleshooting"

        # Default fallback
        return "reference"

    def _extract_title(self, filename: str, file_path: str) -> str:
        """Extract title from filename or file content"""
        # Remove timestamp and type prefix from filename
        title = filename

        # Remove timestamp pattern (YYYY-MM-DD_HHMM_)
        title = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{4}_", "", title)

        # Remove type prefixes
        type_prefixes = [
            "implementation-plan",
            "assessment-",
            "design-",
            "research-",
            "template-",
            "BUG_",
            "SESSION_",
        ]
        for prefix in type_prefixes:
            if title.startswith(prefix):
                title = title[len(prefix) :]
                break

        # Convert to title case and replace hyphens/underscores with spaces
        title = title.replace("-", " ").replace("_", " ")
        title = " ".join(word.capitalize() for word in title.split())

        # If title is empty or too short, try to read from file content
        if len(title) < 3:
            title = self._extract_title_from_content(file_path)

        return title

    def _extract_title_from_content(self, file_path: str) -> str:
        """Extract title from file content (first heading)"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read(1000)  # Read first 1000 chars

            # Look for markdown headings
            heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if heading_match:
                return heading_match.group(1).strip()

            # Look for any text that could be a title
            lines = content.split("\n")
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line and not line.startswith("#") and len(line) > 5:
                    return line[:50]  # Limit length

        except Exception:
            pass

        return "Untitled Document"

    def _determine_status(self, directory: str, filename: str) -> str:
        """Determine status based on directory and filename"""
        if "completed" in directory.lower():
            return "completed"
        elif "template" in directory.lower():
            return "template"
        elif "index" in filename.lower():
            return "active"
        else:
            return "active"

    def _generate_tags(
        self, file_type: str, category: str, directory: str
    ) -> list[str]:
        """Generate appropriate tags"""
        tags = [file_type, category]

        # Add directory-specific tags
        if directory in ["assessments", "design_documents", "implementation_plans"]:
            tags.append("documentation")
        elif directory in ["research"]:
            tags.append("analysis")
        elif directory in ["templates"]:
            tags.append("reference")

        return tags

    def generate_frontmatter(self, file_path: str) -> str:
        """Generate frontmatter for a file"""
        analysis = self.analyze_file(file_path)
        current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        frontmatter = "---\n"
        frontmatter += f'title: "{analysis["title"]}"\n'
        frontmatter += f'date: "{current_date}"\n'
        frontmatter += f'type: "{analysis["type"]}"\n'
        frontmatter += f'category: "{analysis["category"]}"\n'
        frontmatter += f'status: "{analysis["status"]}"\n'
        frontmatter += 'version: "1.0"\n'
        frontmatter += f"tags: {analysis['tags']}\n"
        frontmatter += "---\n\n"

        return frontmatter

    def add_frontmatter_to_file(self, file_path: str, dry_run: bool = False) -> bool:
        """Add frontmatter to a file"""
        try:
            # Read current content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Generate frontmatter
            frontmatter = self.generate_frontmatter(file_path)

            # Combine frontmatter with content
            new_content = frontmatter + content

            if not dry_run:
                # Write back to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

            return True

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

    def process_files(
        self, file_paths: list[str], dry_run: bool = False, limit: int | None = None
    ) -> dict[str, bool]:
        """Process multiple files"""
        results = {}
        count = 0

        for file_path in file_paths:
            if limit is not None and count >= limit:
                print(f"✋ Reached file limit ({limit}). Stopping.")
                break

            print(f"Processing: {file_path}")
            success = self.add_frontmatter_to_file(file_path, dry_run)
            results[file_path] = success
            count += 1

            if success:
                print(
                    f"✅ {'Would add' if dry_run else 'Added'} frontmatter to {file_path}"
                )
            else:
                print(f"❌ Failed to process {file_path}")

        return results


def main():
    """Main execution function"""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Add frontmatter to files missing it")
    parser.add_argument("--files", nargs="+", help="Specific files to process")
    parser.add_argument(
        "--all", action="store_true", help="Process all files missing frontmatter"
    )
    parser.add_argument(
        "--batch-process",
        action="store_true",
        help="Process all files in directory missing frontmatter",
    )
    parser.add_argument(
        "--directory",
        default="docs/artifacts",
        help="Directory to process (used with --batch-process)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of files to process (for testing)",
    )

    args = parser.parse_args()

    generator = FrontmatterGenerator()

    if args.batch_process or args.all:
        # Find all files missing frontmatter in the directory
        directory = Path(args.directory)
        file_paths = []

        for file_path in directory.rglob("*.md"):
            if file_path.is_file() and file_path.name not in ["INDEX.md", "README.md", "MASTER_INDEX.md"]:
                # Check if file is missing frontmatter
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read(100)  # Read first 100 chars
                        if not content.startswith("---"):
                            file_paths.append(str(file_path))
                except Exception:
                    pass
    elif args.files:
        file_paths = args.files
    else:
        print("Please specify --files, --all, or --batch-process")
        return

    if not file_paths:
        print("✅ No files found missing frontmatter")
        return

    print(f"Processing {len(file_paths)} files...")
    if args.limit:
        print(f"Maximum files to process: {args.limit}")
    results = generator.process_files(file_paths, dry_run=args.dry_run, limit=args.limit)

    # Summary
    successful = sum(1 for success in results.values() if success)
    print(f"\nSummary: {successful}/{len(results)} files processed successfully")


if __name__ == "__main__":
    main()
