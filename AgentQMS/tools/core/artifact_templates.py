#!/usr/bin/env python3
"""
Artifact Template System for AI Agents

This module provides templates and utilities for creating properly formatted
artifacts that follow the project's naming conventions and structure.

Supports extension via plugin system - see .agentqms/plugins/artifact_types/

Usage:
    from artifact_templates import create_artifact, get_template

    # Create a new implementation plan
    create_artifact('implementation_plan', 'my-feature', 'docs/artifacts/')

    # Get template content (including plugin-registered types)
    template = get_template('assessment')
    template = get_template('change_request')  # Plugin-registered type
"""

from datetime import datetime, timedelta
import re
from pathlib import Path
from typing import Any

from AgentQMS.tools.utils.config_loader import ConfigLoader


DEFAULT_CONFIG: dict[str, Any] = {
    "frontmatter_defaults": {
        "type": "{artifact_type}",
        "category": "development",
        "status": "active",
        "version": "1.0",
        "tags": ["{artifact_type}"],
        "ads_version": "1.0",
    },
    "frontmatter_denylist": [
        "output_dir",
        "interactive",
        "steps_to_reproduce",
        "quiet",
    ],
    "default_branch": "main",
    "frontmatter_delimiter": "---",
    "date_formats": {
        "filename_timestamp": "%Y-%m-%d_%H%M",
        "date_only": "%Y-%m-%d",
        "timestamp_with_tz": "%Y-%m-%d %H:%M (KST)",
    },
    "content_defaults": {
        "target_date_offset_days": 7,
    },
    "naming_conventions": {
        "replacements": [[" ", "-"], ["_", "-"], ["--", "-"]],
        "strip_chars": "-_",
        "lowercase": True,
        "type_prefix_separators": ["-", "_"],
    },
    "duplicate_detection": {
        "recent_file_window_seconds": 300,
        "recent_file_window_description": "5 minutes",
        "patterns": {
            "bug_report": "bug_{date}_*_*_{name}.md",
            "default": "{date}_*_{type}_{name}.md",
        },
    },
    "artifact_types": {
        "bug_report": {
            "default_id": "001",
            "separators": ["-", "_"],
        },
    },
}

STANDARDS_CONFIG_DIR = Path(__file__).resolve().parents[2] / "standards" / "tier2-framework" / "config-externalization"

# Try to import plugin registry for extensibility
try:
    from AgentQMS.tools.core.plugins import get_plugin_registry

    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False
# Try to import new utilities for branch and timestamp handling
try:
    from AgentQMS.tools.utils.git import get_current_branch
    from AgentQMS.tools.utils.timestamps import get_kst_timestamp, format_timestamp_for_filename

    UTILITIES_AVAILABLE = True
    _get_current_branch = get_current_branch
    _get_kst_timestamp = get_kst_timestamp
    _format_timestamp_for_filename = format_timestamp_for_filename
except ImportError:
    UTILITIES_AVAILABLE = False
    _get_current_branch = None  # type: ignore[assignment]
    _get_kst_timestamp = None  # type: ignore[assignment]
    _format_timestamp_for_filename = None  # type: ignore[assignment]


class ArtifactTemplates:
    """Templates for creating properly formatted artifacts.

    PURE PLUGIN-BASED SYSTEM

    This class is a pure plugin loader. All artifact types are defined
    exclusively as plugins in .agentqms/plugins/artifact_types/*.yaml

    No hardcoded templates exist. All artifact type definitions must be
    registered through the plugin registry.

    Additional artifact types can be registered by creating plugin YAML files.
    See: AgentQMS/docs/guides/creating-artifact-type-plugins.md
    """

    def __init__(self):
        """
        Initialize artifact templates system.

        Loads all artifact types from plugin registry only.
        No hardcoded templates - all types must be defined as plugins.
        """
        self.templates: dict[str, dict[str, Any]] = {}
        self._config_loader = ConfigLoader(cache_size=5)
        self._config_cache: dict[str, Any] | None = None

        # Load all templates from plugin registry
        self._load_plugin_templates()

        # Validate that plugins are available
        if not self.templates:
            import warnings
            warnings.warn(
                "No artifact type plugins loaded. "
                "Ensure .agentqms/plugins/artifact_types/ contains plugin definitions."
            )

    def _load_plugin_templates(self) -> None:
        """Load artifact templates from plugin registry.

        All artifact types are loaded from plugins. There are no hardcoded
        templates - this method loads exclusively from the plugin system.

        If plugin loading fails, templates dict will be empty and a warning
        will be issued during __init__.
        """
        if not PLUGINS_AVAILABLE:
            return

        try:
            registry = get_plugin_registry()
            artifact_types = registry.get_artifact_types()

            for name, plugin_def in artifact_types.items():
                # Convert plugin schema to template format
                template = self._convert_plugin_to_template(name, plugin_def)
                if template:
                    self.templates[name] = template

        except Exception as e:
            # Plugin loading failure is critical in plugin-only system
            import warnings
            warnings.warn(f"Failed to load artifact type plugins: {e}")

    def _get_config(self) -> dict[str, Any]:
        """Get cached artifact template configuration."""
        if self._config_cache is None:
            self._config_cache = self._load_config()
        return self._config_cache

    def _load_config(self) -> dict[str, Any]:
        """Load artifact_template_config.yaml (all defaults are in the YAML file)."""
        config_path = STANDARDS_CONFIG_DIR / "artifact_template_config.yaml"
        return self._config_loader.get_config(config_path, defaults=DEFAULT_CONFIG)

    def _load_template_defaults(self) -> dict[str, Any]:
        """Load template defaults from external YAML configuration.

        Uses ConfigLoader for consistent configuration management.
        Falls back to minimal defaults if config file not found.
        """
        config_path = STANDARDS_CONFIG_DIR / "template_defaults.yaml"

        defaults = {
            "defaults": {},
            "bug_report": {},
            "frontmatter_denylist": []
        }

        return self._config_loader.get_config(config_path, defaults=defaults)

    def _replace_artifact_type(self, value: Any, artifact_type: str) -> Any:
        """Replace {artifact_type} placeholders in supported types."""
        if isinstance(value, str):
            return value.replace("{artifact_type}", artifact_type)
        if isinstance(value, list):
            return [self._replace_artifact_type(item, artifact_type) for item in value]
        return value

    def _build_default_frontmatter(self, artifact_type: str) -> dict[str, Any]:
        """Build default frontmatter with placeholders resolved and ADS metadata ensured."""
        config = self._get_config()
        defaults = config.get("frontmatter_defaults", DEFAULT_CONFIG["frontmatter_defaults"]).copy()
        resolved = {k: self._replace_artifact_type(v, artifact_type) for k, v in defaults.items()}
        resolved.setdefault("ads_version", "1.0")
        resolved.setdefault("type", artifact_type)
        resolved.setdefault("artifact_type", artifact_type)
        return resolved

    def _merge_frontmatter(self, artifact_type: str, metadata_frontmatter: dict[str, Any] | None) -> dict[str, Any]:
        """Merge plugin frontmatter over defaults while keeping required keys."""
        base = self._build_default_frontmatter(artifact_type)
        if metadata_frontmatter:
            for key, value in metadata_frontmatter.items():
                base[key] = self._replace_artifact_type(value, artifact_type)
        return base

    def _convert_plugin_to_template(self, name: str, plugin_def: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a plugin artifact type definition to template format.

        Plugin schema format:
            metadata:
              filename_pattern: "CR_{date}_{name}.md"
              directory: change_requests/
              frontmatter: {...}
            template: "# Content..."
            template_variables: {...}

        Template format:
            filename_pattern: "YYYY-MM-DD_HHMM_type_{name}.md"
            directory: "directory/"
            frontmatter: {...}
            content_template: "# Content..."
        """
        try:
            metadata = plugin_def.get("metadata", {})

            # Required fields
            filename_pattern = metadata.get("filename_pattern")
            directory = metadata.get("directory")
            template_content = plugin_def.get("template")

            if not all([filename_pattern, directory, template_content]):
                return None

            # Build template dict
            template: dict[str, Any] = {
                "filename_pattern": filename_pattern,
                "directory": directory,
                "frontmatter": self._merge_frontmatter(name, metadata.get("frontmatter")),
                "content_template": template_content,
            }

            # Store template variables for use in create_content
            if "template_variables" in plugin_def:
                template["_plugin_variables"] = plugin_def["template_variables"]

            return template

        except Exception:
            return None

    def get_template(self, template_type: str) -> dict[str, Any] | None:
        """Get template configuration for a specific type."""
        return self.templates.get(template_type)

    def get_available_templates(self) -> list[str]:
        """Get list of available template types."""
        return list(self.templates.keys())

    def _get_available_artifact_types(self) -> dict[str, Any]:
        """
        Get all available artifact types with metadata.

        Returns comprehensive information about all artifact types defined
        in the plugin system.

        Returns:
            dict mapping artifact type name to info dict with keys:
            - name: artifact type name
            - description: human-readable description
            - validation: validation rules from plugin if available
            - template: template configuration

        Example:
            types = artifacts._get_available_artifact_types()
            for name, info in types.items():
                print(f"{name}: {info['description']}")
        """
        artifact_types: dict[str, dict[str, Any]] = {}

        # Load all artifact types from plugin registry
        if PLUGINS_AVAILABLE:
            try:
                registry = get_plugin_registry()
                plugin_types = registry.get_artifact_types()

                for name, plugin_def in plugin_types.items():
                    artifact_types[name] = {
                        "name": name,
                        "description": plugin_def.get("description", f"Artifact type: {name}"),
                        "template": self._convert_plugin_to_template(name, plugin_def),
                        "validation": plugin_def.get("validation"),
                    }

            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load artifact type plugins: {e}")

        return artifact_types

    def get_available_templates_with_metadata(self) -> list[dict[str, Any]]:
        """
        Get list of available templates with validation metadata.

        Returns information about all available artifact types including
        validation rules and descriptions.

        Returns:
            List of dicts with keys:
            - name: template name
            - description: brief description
            - has_validation: bool

        Example:
            templates = artifacts.get_available_templates_with_metadata()
            for t in templates:
                print(f"{t['name']}: {t['description']}")
        """
        types = self._get_available_artifact_types()

        result = []
        for name, info in types.items():
            result.append(
                {
                    "name": name,
                    "description": info.get("description", ""),
                    "has_validation": info.get("validation") is not None,
                }
            )

        return sorted(result, key=lambda x: x["name"])

    def _normalize_name(self, name: str) -> str:
        """Normalize artifact name to lowercase kebab-case."""
        config = self._get_config()
        naming = config.get("naming_conventions", DEFAULT_CONFIG["naming_conventions"])

        normalized = name.lower()
        for old, new in naming["replacements"]:
            normalized = normalized.replace(old, new)
        return normalized.strip(naming["strip_chars"] + "-")

    def _get_timestamp(self) -> str:
        """Get formatted timestamp for filename."""
        if UTILITIES_AVAILABLE and _format_timestamp_for_filename is not None:
            return _format_timestamp_for_filename()

        config = self._get_config()
        return datetime.now().strftime(config["date_formats"]["filename_timestamp"])

    def _strip_duplicate_type_prefix(self, pattern: str, normalized_name: str) -> str:
        """Remove duplicate type prefix from name if it matches pattern."""
        m = re.search(r"\{date\}_(.+?)_\{name\}", pattern)
        if not m:
            return normalized_name

        type_hint = m.group(1).lower()
        variants = {type_hint, type_hint.replace("_", "-")}

        separators = (
            self._get_config()
            .get("naming_conventions", DEFAULT_CONFIG["naming_conventions"])
            .get("type_prefix_separators", ["-", "_"])
        )

        for hint in variants:
            for sep in separators:
                dup = f"{hint}{sep}"
                if normalized_name.startswith(dup):
                    return normalized_name[len(dup):].lstrip("-_")
        return normalized_name

    def _create_bug_report_filename(self, name: str, timestamp: str, pattern: str) -> str:
        """Create filename for bug_report artifact type."""
        bug_config = self._get_config().get("artifact_types", {}).get("bug_report", {})
        default_bug_id = bug_config.get("default_id", DEFAULT_CONFIG["artifact_types"]["bug_report"]["default_id"])
        separators = bug_config.get("separators", DEFAULT_CONFIG["artifact_types"]["bug_report"]["separators"])

        split_parts: list[str] = [name]
        for sep in separators:
            if sep in name:
                split_parts = name.split(sep)
                break

        bug_id = default_bug_id
        descriptive_parts: list[str] = []
        if len(split_parts) >= 2 and split_parts[0].isdigit():
            bug_id = split_parts[0]
            descriptive_parts = split_parts[1:]
        elif len(split_parts) >= 2 and split_parts[1].isdigit():
            bug_id = split_parts[1]
            descriptive_parts = split_parts[2:]
        else:
            descriptive_parts = split_parts[1:] if len(split_parts) > 1 else split_parts

        descriptive_name = "-".join(self._normalize_name(p) for p in descriptive_parts) or self._normalize_name(name)

        context = {"name": descriptive_name, "date": timestamp}
        try:
            filename = pattern.format(**context)
        except KeyError:
            filename = pattern.format(name=descriptive_name)

        return filename.replace("YYYY-MM-DD_HHMM", timestamp).replace("NNN", bug_id)

    def create_filename(self, template_type: str, name: str) -> str:
        """Create a properly formatted filename for an artifact."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        timestamp = self._get_timestamp()
        pattern = template["filename_pattern"]

        # Special handling for bug reports
        if template_type == "bug_report":
            return self._create_bug_report_filename(name, timestamp, pattern)

        # Standard filename generation
        normalized_name = self._normalize_name(name)
        normalized_name = self._strip_duplicate_type_prefix(pattern, normalized_name)

        context = {"name": normalized_name, "date": timestamp}
        try:
            filename = pattern.format(**context)
        except KeyError:
            filename = pattern.format(name=normalized_name)

        return filename.replace("YYYY-MM-DD_HHMM", timestamp)

    def _get_kst_timestamp_str(self) -> str:
        """Get KST timestamp string."""
        if UTILITIES_AVAILABLE and _get_kst_timestamp is not None:
            return _get_kst_timestamp()

        from datetime import timedelta, timezone
        config = self._get_config()
        kst = timezone(timedelta(hours=9))
        return datetime.now(kst).strftime(config["date_formats"]["timestamp_with_tz"])

    def _get_branch_name(self) -> str:
        """Get current git branch name."""
        if UTILITIES_AVAILABLE and _get_current_branch is not None:
            try:
                return _get_current_branch()
            except Exception:
                pass
        return self._get_config()["default_branch"]

    def _format_frontmatter_yaml(self, data: dict[str, Any]) -> str:
        """Format frontmatter dict as YAML."""
        config = self._get_config()
        delimiter = config["frontmatter_delimiter"]

        lines = [delimiter]
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        lines.append(delimiter)
        return "\n".join(lines)

    def create_frontmatter(self, template_type: str, title: str, **kwargs) -> str:
        """Create frontmatter for an artifact."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        frontmatter = template["frontmatter"].copy()
        frontmatter["title"] = title
        frontmatter["date"] = self._get_kst_timestamp_str()
        frontmatter.setdefault("ads_version", "1.0")
        frontmatter.setdefault("artifact_type", template_type)

        if "branch" not in kwargs:
            frontmatter["branch"] = self._get_branch_name()

        # Merge kwargs, excluding denylist
        config = self._get_config()
        denylist = set(config["frontmatter_denylist"])
        for key, value in kwargs.items():
            if key not in denylist:
                frontmatter[key] = value

        return self._format_frontmatter_yaml(frontmatter)

    def _build_content_context(self, template_type: str, title: str, template: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Build context dict for template content formatting."""
        config = self._get_config()
        date_fmt = config.get("date_formats", DEFAULT_CONFIG["date_formats"])["date_only"]
        offset_days = config.get("content_defaults", {}).get(
            "target_date_offset_days",
            DEFAULT_CONFIG["content_defaults"]["target_date_offset_days"],
        )

        now = datetime.now()
        context = {
            "title": title,
            "start_date": now.strftime(date_fmt),
            "target_date": (now + timedelta(days=offset_days)).strftime(date_fmt),
            "assessment_date": now.strftime(date_fmt),
        }

        # Merge legacy config defaults
        legacy_config = self._load_template_defaults()
        context.update(legacy_config.get("defaults", {}))
        if template_type in legacy_config:
            context.update(legacy_config[template_type])

        # Add plugin variables
        if "_plugin_variables" in template:
            context.update(template["_plugin_variables"])

        # User kwargs override all
        context.update(kwargs)
        return context

    def create_content(self, template_type: str, title: str, **kwargs) -> str:
        """Create content for an artifact using the template."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        context = self._build_content_context(template_type, title, template, **kwargs)
        return str(template["content_template"].format(**context))

    def _check_for_duplicate(self, output_path: Path, template_type: str, name: str, quiet: bool) -> str | None:
        """Check if a recently created duplicate exists. Returns path if found, None otherwise."""
        normalized_name = self._normalize_name(name)
        config = self._get_config()
        dup_config = config.get("duplicate_detection", DEFAULT_CONFIG["duplicate_detection"])

        now = datetime.now()
        current_date = now.strftime(config.get("date_formats", DEFAULT_CONFIG["date_formats"])["date_only"])

        # Get pattern from config
        if template_type == "bug_report" and "bug_report" in dup_config["patterns"]:
            pattern_template = dup_config["patterns"]["bug_report"]
        else:
            pattern_template = dup_config["patterns"]["default"]

        # Build glob pattern
        pattern_base = pattern_template.format(
            artifact_type=template_type,
            type=template_type,
            date=current_date,
            time="*",
            name=normalized_name
        )

        existing_files = list(output_path.glob(pattern_base))
        if not existing_files:
            return None

        # Check recent files within window
        window_seconds = dup_config["recent_file_window_seconds"]
        for existing_file in sorted(existing_files, key=lambda p: p.stat().st_mtime, reverse=True):
            file_mtime = datetime.fromtimestamp(existing_file.stat().st_mtime)
            time_diff = (now - file_mtime).total_seconds()

            if time_diff < window_seconds:
                if not quiet:
                    print(f"⚠️  Found recently created file: {existing_file.name}")
                    print("   Reusing existing file instead of creating duplicate.")
                return str(existing_file)

        return None

    def create_artifact(
        self,
        template_type: str,
        name: str,
        title: str,
        output_dir: str = "docs/artifacts/",
        quiet: bool = False,
        **kwargs,
    ) -> str:
        """Create a complete artifact file."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        # Create output directory
        output_path = Path(output_dir) / template["directory"]
        output_path.mkdir(parents=True, exist_ok=True)

        # Check for recent duplicates
        duplicate_path = self._check_for_duplicate(output_path, template_type, name, quiet)
        if duplicate_path:
            return duplicate_path

        # Create new artifact
        filename = self.create_filename(template_type, name)
        file_path = output_path / filename

        frontmatter = self.create_frontmatter(template_type, title, **kwargs)
        content = self.create_content(template_type, title, **kwargs)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(frontmatter + "\n\n" + content)

        return str(file_path)


# Convenience functions
def get_template(template_type: str) -> dict[str, Any] | None:
    """Get template configuration for a specific type."""
    templates = ArtifactTemplates()
    return templates.get_template(template_type)


def create_artifact(
    template_type: str,
    name: str,
    title: str,
    output_dir: str = "docs/artifacts/",
    quiet: bool = False,
    **kwargs,
) -> str:
    """Create a complete artifact file."""
    templates = ArtifactTemplates()
    return templates.create_artifact(template_type, name, title, output_dir, quiet=quiet, **kwargs)


def get_available_templates() -> list[str]:
    """Get list of available template types."""
    templates = ArtifactTemplates()
    return templates.get_available_templates()


if __name__ == "__main__":
    # Example usage
    templates = ArtifactTemplates()

    print("Available templates:")
    for template_type in templates.get_available_templates():
        print(f"  - {template_type}")

    # Example: Create an implementation plan
    try:
        file_path = create_artifact(
            "implementation_plan",
            "my-feature",
            "My Feature Implementation Plan",
            subject="feature implementation",
            methodology="agile development",
        )
        print(f"\nCreated artifact: {file_path}")
    except Exception as e:
        print(f"Error creating artifact: {e}")
