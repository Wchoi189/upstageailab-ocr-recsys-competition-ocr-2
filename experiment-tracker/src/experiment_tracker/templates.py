from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssessmentTemplate:
    """Simple container holding metadata for an assessment template."""

    id: str
    title: str
    description: str
    path: Path


class TemplateRegistry:
    """Loads and resolves template metadata stored under .templates."""

    def __init__(self, root_dir: Path, config: dict):
        self.root_dir = Path(root_dir)
        self.config = config or {}
        self.templates_dir = self.root_dir / self.config.get("templates_path", ".templates")
        self.assessment_index_path = self.templates_dir / "assessment_templates.json"
        self._assessment_templates: dict[str, AssessmentTemplate] = {}
        self._load_assessment_templates()

    def list_assessment_templates(self) -> list[AssessmentTemplate]:
        """Return all registered assessment templates."""
        return list(self._assessment_templates.values())

    def get_assessment_template(self, template_key: str) -> AssessmentTemplate | None:
        """Resolve a template either by id or slugified title."""
        if not template_key:
            return None

        normalized = self._normalize(template_key)
        return self._assessment_templates.get(normalized)

    def _load_assessment_templates(self):
        """Load template metadata from assessment index file."""
        if not self.assessment_index_path.exists():
            return

        data = json.loads(self.assessment_index_path.read_text())
        for entry in data.get("assessment_templates", []):
            template_id = entry.get("id")
            title = entry.get("title")
            filename = entry.get("filename")
            description = entry.get("description", "")

            if not template_id or not title or not filename:
                continue

            template_path = self.templates_dir / filename
            if not template_path.exists():
                continue

            normalized_key = self._normalize(template_id)
            template = AssessmentTemplate(
                id=template_id,
                title=title,
                description=description,
                path=template_path,
            )
            self._assessment_templates[normalized_key] = template

            # Also allow title lookup
            title_key = self._normalize(title)
            self._assessment_templates.setdefault(title_key, template)

    @staticmethod
    def _normalize(value: str) -> str:
        """Lowercase slug-style normalization to make lookups forgiving."""
        slug = value.strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        return slug.strip("-")
