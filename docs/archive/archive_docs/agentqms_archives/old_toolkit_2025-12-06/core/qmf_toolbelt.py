import datetime
import json
from pathlib import Path

import yaml
from jinja2 import Template


class QualityManagementToolbelt:
    """Helper for working with quality management artifacts defined in q-manifest.yaml."""

    def __init__(self, manifest_path: str | Path | None = None):
        if manifest_path is None:
            # Default to the framework's conventions manifest
            from AgentQMS.toolkit.utils.paths import get_project_conventions_dir

            manifest_path = get_project_conventions_dir() / "q-manifest.yaml"

        manifest_path = Path(manifest_path)
        self.root_path = manifest_path.parent
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)

    def list_artifact_types(self):
        """Returns a list of available artifact types."""
        return [atype['name'] for atype in self.manifest['artifact_types']]

    def create_artifact(self, artifact_type: str, title: str, content: str, author: str = "ai-agent", tags: list = []):
        """
        Creates a new quality artifact.

        Args:
            artifact_type: The type of artifact to create (e.g., 'implementation_plan').
            title: The title of the artifact.
            content: The markdown content of the artifact.
            author: The author of the artifact.
            tags: A list of tags for the artifact.

        Returns:
            The path to the newly created artifact.
        """
        artifact_meta = None
        for atype in self.manifest['artifact_types']:
            if atype['name'] == artifact_type:
                artifact_meta = atype
                break

        if not artifact_meta:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

        # Generate filename
        slug = title.lower().replace(' ', '-').replace('_', '-')
        filename = f"{slug}.md"

        # Define output path
        output_dir = self.root_path.parent / artifact_meta['location']
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Load template
        template_path = self.root_path / artifact_meta['template']
        with open(template_path, 'r') as f:
            template = Template(f.read())

        # Create frontmatter
        frontmatter = {
            "title": title,
            "author": author,
            "date": datetime.date.today().isoformat(),
            "status": "draft",
            "tags": tags
        }

        # Render the full document
        rendered_document = template.render(**frontmatter)

        # Combine with user content
        final_content = rendered_document + "\n" + content

        with open(output_path, 'w') as f:
            f.write(final_content)

        return str(output_path)

    def validate_artifact(self, file_path: str):
        """
        Validates an artifact's frontmatter against its schema.

        Args:
            file_path: The path to the artifact file.

        Returns:
            True if valid, raises an exception otherwise.
        """
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract frontmatter
        parts = content.split('---')
        if len(parts) < 3:
            raise ValueError("Invalid frontmatter format.")

        frontmatter_str = parts[1]
        frontmatter = yaml.safe_load(frontmatter_str)

        # Determine artifact type from path
        artifact_type_name = Path(file_path).parent.name

        artifact_meta = None
        for atype in self.manifest['artifact_types']:
            if atype['location'].strip('/').endswith(artifact_type_name):
                artifact_meta = atype
                break

        if not artifact_meta:
            raise ValueError(f"Could not determine artifact type for {file_path}")

        # Load schema
        schema_path = self.root_path / artifact_meta['schema']
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Validate
        from jsonschema import validate
        validate(instance=frontmatter, schema=schema)

        return True
