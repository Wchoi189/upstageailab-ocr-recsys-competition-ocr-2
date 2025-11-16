import datetime
import json
import re
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from jinja2 import Template


class ValidationError(Exception):
    """Raised when artifact validation fails."""
    pass


class AgentQMSToolbelt:
    def __init__(self, manifest_path="agent_qms/q-manifest.yaml"):
        self.root_path = Path(manifest_path).parent
        with open(manifest_path, 'r') as f:
            self.manifest = yaml.safe_load(f)

    def _get_git_branch(self) -> str:
        """Get the current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.root_path.parent  # Run from project root
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If git is not available or not in a git repo, return "unknown"
            return "unknown"

    def list_artifact_types(self):
        """Returns a list of available artifact types."""
        return [atype['name'] for atype in self.manifest['artifact_types']]

    def _validate_filename(self, filename: str) -> None:
        """
        Validate filename follows project conventions.

        Rules:
        - No ALL CAPS filenames (except README.md, CHANGELOG.md)
        - Use lowercase with hyphens/underscores
        - No spaces
        """
        # Allow README.md and CHANGELOG.md
        if filename.upper() in ["README.MD", "CHANGELOG.MD"]:
            return

        # Check for ALL CAPS (excluding extension)
        name_without_ext = filename.rsplit('.', 1)[0]
        if name_without_ext.isupper() and name_without_ext:
            raise ValidationError(
                f"Filename '{filename}' violates convention: No ALL CAPS filenames "
                f"(except README.md, CHANGELOG.md). Use lowercase with hyphens/underscores."
            )

        # Check for spaces
        if ' ' in filename:
            raise ValidationError(
                f"Filename '{filename}' violates convention: No spaces in filenames. "
                f"Use hyphens or underscores instead."
            )

    def _validate_frontmatter(self, frontmatter: dict, schema: dict) -> None:
        """Validate frontmatter against schema before creation."""
        from jsonschema import validate, ValidationError as SchemaValidationError

        try:
            validate(instance=frontmatter, schema=schema)
        except SchemaValidationError as e:
            raise ValidationError(
                f"Frontmatter validation failed: {e.message}\n"
                f"Path: {'.'.join(str(p) for p in e.path)}"
            )

    def _validate_artifact_type(self, artifact_type: str) -> dict:
        """Validate artifact type exists and return metadata."""
        artifact_meta = None
        for atype in self.manifest['artifact_types']:
            if atype['name'] == artifact_type:
                artifact_meta = atype
                break

        if not artifact_meta:
            raise ValueError(f"Unknown artifact type: {artifact_type}. "
                           f"Available types: {', '.join(self.list_artifact_types())}")

        return artifact_meta

    def create_artifact(self, artifact_type: str, title: str, content: str, author: str = "ai-agent", tags: list = [], **kwargs):
        """
        Creates a new quality artifact with validation BEFORE creation.

        Args:
            artifact_type: The type of artifact to create (e.g., 'implementation_plan').
            title: The title of the artifact.
            content: The markdown content of the artifact.
            author: The author of the artifact.
            tags: A list of tags for the artifact.
            **kwargs: Additional parameters to pass to template (e.g., bug_id, severity for bug_report).

        Returns:
            The path to the newly created artifact.

        Raises:
            ValidationError: If validation fails before creation.
            ValueError: If artifact type is invalid.
        """
        # Validate artifact type
        artifact_meta = self._validate_artifact_type(artifact_type)

        # Generate filename with timestamp prefix for assessments, implementation_plans, and guides
        # Format: YYYY-MM-DD_HHMM_descriptive-name.md
        now_kst = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
        timestamp_prefix = now_kst.strftime("%Y-%m-%d_%H%M")

        # Check if artifact type requires timestamp prefix
        timestamped_types = ["assessment", "implementation_plan", "guide"]
        if artifact_type in timestamped_types:
            slug = title.lower().replace(' ', '-').replace('_', '-')
            filename = f"{timestamp_prefix}_{slug}.md"
        elif artifact_type == "bug_report":
            # Bug reports use bug ID prefix: BUG-YYYYMMDD-###_descriptive-name.md
            bug_id = kwargs.get("bug_id", "")
            slug = title.lower().replace(' ', '-').replace('_', '-')
            if bug_id:
                filename = f"{bug_id}_{slug}.md"
            else:
                filename = f"{slug}.md"
        else:
            slug = title.lower().replace(' ', '-').replace('_', '-')
            filename = f"{slug}.md"

        # Validate filename BEFORE creation
        self._validate_filename(filename)

        # Define output path
        output_dir = self.root_path.parent / artifact_meta['location']
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Check if file already exists
        if output_path.exists():
            raise ValueError(f"Artifact already exists: {output_path}")

        # Load schema for validation
        schema_path = self.root_path / artifact_meta['schema']
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Create frontmatter with timestamp in KST (includes hour and minute)
        now_kst = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
        timestamp_kst = now_kst.strftime("%Y-%m-%d %H:%M KST")
        branch_name = self._get_git_branch()

        # Create frontmatter with default values (standardized format)
        frontmatter = {
            "title": title,
            "author": author,
            "timestamp": timestamp_kst,  # Single timestamp field in KST format
            "branch": branch_name,  # Git branch name
            "status": "draft",
            "tags": tags
        }

        # Add artifact type-specific defaults
        if artifact_type == "bug_report":
            frontmatter["type"] = "bug_report"
            frontmatter["category"] = "troubleshooting"
            frontmatter["version"] = "1.0"
            frontmatter["status"] = kwargs.get("status", "open")
            frontmatter["bug_id"] = kwargs.get("bug_id", "")
            frontmatter["severity"] = kwargs.get("severity", "Medium")
        elif artifact_type == "implementation_plan":
            frontmatter["type"] = "implementation_plan"
            frontmatter["category"] = "development"
        elif artifact_type == "assessment":
            frontmatter["type"] = "assessment"
            frontmatter["category"] = "evaluation"
        elif artifact_type == "data_contract":
            # Ensure data-contract tag is present
            if "data-contract" not in frontmatter.get("tags", []):
                frontmatter["tags"] = frontmatter.get("tags", []) + ["data-contract"]

        # Merge any additional kwargs into frontmatter
        frontmatter.update(kwargs)

        # Validate frontmatter BEFORE creation
        self._validate_frontmatter(frontmatter, schema)

        # Load template
        template_path = self.root_path / artifact_meta['template']
        with open(template_path, 'r') as f:
            template = Template(f.read())

        # Render the full document with all frontmatter values
        rendered_document = template.render(**frontmatter)

        # Combine with user content
        final_content = rendered_document + "\n" + content

        # Create file only after all validations pass
        with open(output_path, 'w') as f:
            f.write(final_content)

        # Final validation: verify the created file
        try:
            self.validate_artifact(str(output_path))
        except Exception as e:
            # If validation fails, remove the file
            output_path.unlink()
            raise ValidationError(
                f"Created artifact failed validation: {e}\n"
                f"File has been removed."
            )

        return str(output_path)

    def validate_artifact(self, file_path: str):
        """
        Validates an artifact's frontmatter against its schema and conventions.

        Args:
            file_path: The path to the artifact file.

        Returns:
            True if valid, raises ValidationError otherwise.
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise ValidationError(f"Artifact file does not exist: {file_path}")

        # Validate filename
        self._validate_filename(file_path_obj.name)

        with open(file_path_obj, 'r') as f:
            content = f.read()

        # Extract frontmatter
        parts = content.split('---')
        if len(parts) < 3:
            raise ValidationError("Invalid frontmatter format. File must start with '---' and have closing '---'.")

        frontmatter_str = parts[1]
        try:
            frontmatter = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML in frontmatter: {e}")

        if not frontmatter:
            raise ValidationError("Frontmatter is empty or invalid.")

        # Validate timestamp is present in frontmatter (required for all artifacts)
        if "timestamp" not in frontmatter:
            raise ValidationError("Frontmatter must include 'timestamp' field with hour and minute in KST format (YYYY-MM-DD HH:MM KST).")

        # Validate branch is present in frontmatter (required for all artifacts)
        if "branch" not in frontmatter:
            raise ValidationError("Frontmatter must include 'branch' field with git branch name.")

        # Determine artifact type from path
        artifact_type_name = file_path_obj.parent.name

        artifact_meta = None
        for atype in self.manifest['artifact_types']:
            if atype['location'].strip('/').endswith(artifact_type_name):
                artifact_meta = atype
                break

        if not artifact_meta:
            raise ValidationError(
                f"Could not determine artifact type for {file_path}. "
                f"Expected directory to match one of: {', '.join(atype['location'] for atype in self.manifest['artifact_types'])}"
            )

        # Load schema
        schema_path = self.root_path / artifact_meta['schema']
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Validate against schema
        from jsonschema import validate, ValidationError as SchemaValidationError
        try:
            validate(instance=frontmatter, schema=schema)
        except SchemaValidationError as e:
            raise ValidationError(
                f"Schema validation failed: {e.message}\n"
                f"Path: {'.'.join(str(p) for p in e.path)}"
            )

        return True

    def update_artifact_status(self, file_path: str, new_status: str) -> str:
        """
        Update the status field in an artifact's frontmatter.

        Args:
            file_path: Path to the artifact file
            new_status: New status value (must be valid for artifact type)

        Returns:
            The path to the updated artifact

        Raises:
            ValidationError: If validation fails after update
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise ValidationError(f"Artifact file does not exist: {file_path}")

        # Read current content
        with open(file_path_obj, 'r') as f:
            content = f.read()

        # Extract frontmatter
        parts = content.split('---')
        if len(parts) < 3:
            raise ValidationError("Invalid frontmatter format. File must start with '---' and have closing '---'.")

        frontmatter_str = parts[1]
        try:
            frontmatter = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML in frontmatter: {e}")

        if not frontmatter:
            raise ValidationError("Frontmatter is empty or invalid.")

        # Update status
        frontmatter['status'] = new_status

        # Ensure timestamp is present (required for all artifacts)
        if 'timestamp' not in frontmatter:
            now_kst = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
            frontmatter['timestamp'] = now_kst.strftime("%Y-%m-%d %H:%M KST")

        # Ensure branch is present (required for all artifacts)
        if 'branch' not in frontmatter:
            frontmatter['branch'] = self._get_git_branch()

        # Determine artifact type from path
        artifact_type_name = file_path_obj.parent.name

        artifact_meta = None
        for atype in self.manifest['artifact_types']:
            if atype['location'].strip('/').endswith(artifact_type_name):
                artifact_meta = atype
                break

        if not artifact_meta:
            raise ValidationError(
                f"Could not determine artifact type for {file_path}. "
                f"Expected directory to match one of: {', '.join(atype['location'] for atype in self.manifest['artifact_types'])}"
            )

        # Load schema for validation
        schema_path = self.root_path / artifact_meta['schema']
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Validate updated frontmatter
        self._validate_frontmatter(frontmatter, schema)

        # Reconstruct frontmatter YAML
        updated_frontmatter_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False, allow_unicode=True)

        # Preserve original formatting as much as possible
        # Replace the frontmatter section
        updated_content = f"---{updated_frontmatter_str}---{parts[2]}"

        # Write updated content
        with open(file_path_obj, 'w') as f:
            f.write(updated_content)

        # Final validation
        try:
            self.validate_artifact(str(file_path_obj))
        except Exception as e:
            raise ValidationError(
                f"Updated artifact failed validation: {e}\n"
                f"File has been updated but may be invalid."
            )

        return str(file_path_obj)

    def detect_completion_from_progress_tracker(self, file_path: str) -> str | None:
        """
        Detect completion status from Progress Tracker in implementation plan.

        Args:
            file_path: Path to the artifact file

        Returns:
            Detected status ("completed", "in-progress", "draft") or None if not detectable
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            return None

        with open(file_path_obj, 'r') as f:
            content = f.read()

        # Look for Progress Tracker section
        # Pattern: **STATUS:** Completed / In Progress / Not Started
        import re
        status_patterns = [
            r'\*\*STATUS:\*\*\s*(Completed|Complete|COMPLETED|COMPLETE)',
            r'- \*\*STATUS:\*\*\s*(Completed|Complete|COMPLETED|COMPLETE)',
            r'STATUS.*?:\s*(Completed|Complete|COMPLETED|COMPLETE)',
        ]

        for pattern in status_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return "completed"

        # Check for "In Progress" or "In Progress"
        progress_patterns = [
            r'\*\*STATUS:\*\*\s*(In Progress|IN PROGRESS|In-Progress)',
            r'- \*\*STATUS:\*\*\s*(In Progress|IN PROGRESS|In-Progress)',
            r'STATUS.*?:\s*(In Progress|IN PROGRESS|In-Progress)',
        ]

        for pattern in progress_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return "in-progress"

        # Check for "Not Started"
        not_started_patterns = [
            r'\*\*STATUS:\*\*\s*(Not Started|NOT STARTED|Not-Started)',
            r'- \*\*STATUS:\*\*\s*(Not Started|NOT STARTED|Not-Started)',
        ]

        for pattern in not_started_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return "draft"

        return None

    def check_status_mismatch(self, file_path: str) -> dict | None:
        """
        Check if frontmatter status matches Progress Tracker status.

        Args:
            file_path: Path to the artifact file

        Returns:
            Dict with mismatch info if found, None if no mismatch
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            return None

        # Read frontmatter
        with open(file_path_obj, 'r') as f:
            content = f.read()

        parts = content.split('---')
        if len(parts) < 3:
            return None

        try:
            frontmatter = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            return None

        if not frontmatter or 'status' not in frontmatter:
            return None

        frontmatter_status = frontmatter['status']

        # Detect status from Progress Tracker
        detected_status = self.detect_completion_from_progress_tracker(str(file_path_obj))

        if detected_status and detected_status != frontmatter_status:
            return {
                'file': str(file_path_obj),
                'frontmatter_status': frontmatter_status,
                'detected_status': detected_status,
                'mismatch': True
            }

        return None
