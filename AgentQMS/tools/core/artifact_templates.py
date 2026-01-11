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

    PLUGIN-BASED SYSTEM (Phase 4 - Post Hardcoded Template Removal)

    This class is now a pure plugin loader and wrapper. All artifact types
    are defined as plugins in .agentqms/plugins/artifact_types/*.yaml

    Hardcoded templates have been removed. See archived code for legacy system:
    - Archive: AgentQMS/tools/archive/artifact_templates_legacy.py
    - Migration Guide: docs/artifacts/implementation_plans/phase4_hardcoded_removal_migration.md

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

    def _load_template_defaults(self) -> dict[str, Any]:
        """Load template defaults from external YAML configuration.

        Uses ConfigLoader for consistent configuration management.
        Falls back to minimal defaults if config file not found.
        """
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "template_defaults.yaml"

        defaults = {
            "defaults": {},
            "bug_report": {},
            "frontmatter_denylist": []
        }

        return self._config_loader.get_config(config_path, defaults=defaults)

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
                "frontmatter": metadata.get(
                    "frontmatter",
                    {
                        "type": name,
                        "category": "development",
                        "status": "active",
                        "version": "1.0",
                        "tags": [name],
                    },
                ),
                "content_template": template_content,
            }

            # Store template variables for use in create_content
            if "template_variables" in plugin_def:
                template["_plugin_variables"] = plugin_def["template_variables"]

            return template

        except Exception:
            return None

    def get_template(self, template_type: str) -> dict | None:
        """Get template configuration for a specific type."""
        return self.templates.get(template_type)

    def get_available_templates(self) -> list:
        """Get list of available template types."""
        return list(self.templates.keys())

    def _get_available_artifact_types(self) -> dict[str, Any]:
        """
        Get all available artifact types with source metadata.

        Returns comprehensive information about all artifact types including:
        - Type name and description
        - Source (hardcoded, plugin, or standards)
        - Validation rules if available
        - Template metadata

        Returns:
            dict mapping artifact type name to info dict with keys:
            - name: artifact type name
            - source: "hardcoded", "plugin", or "standards"
            - description: human-readable description
            - validation: validation rules from plugin/standards if available
            - template: template configuration
            - conflict: bool, True if defined in multiple sources

        Example:
            types = artifacts._get_available_artifact_types()
            for name, info in types.items():
                print(f"{name}: {info['source']}")
        """
        artifact_types: dict[str, dict[str, Any]] = {}
        sources_seen: dict[str, list[str]] = {}  # Track which sources define each type

        # 1. Collect hardcoded types (base layer)
        hardcoded_names = {
            "implementation_plan": "Implementation plan for features and changes",
            "walkthrough": "Code walkthrough and explanation",
            "assessment": "Technical assessment and analysis",
            "design": "Design document for architecture",
            "research": "Research findings and documentation",
            "template": "Template for standardized processes",
            "bug_report": "Bug report with reproduction steps",
            "vlm_report": "VLM analysis and evaluation report",
        }

        for name, description in hardcoded_names.items():
            if name not in artifact_types:
                artifact_types[name] = {
                    "name": name,
                    "source": "hardcoded",
                    "description": description,
                    "template": self.templates.get(name),
                    "validation": None,
                    "conflict": False,
                }
                sources_seen[name] = ["hardcoded"]
            else:
                sources_seen[name].append("hardcoded")

        # 2. Collect plugin types (adds or overrides)
        if PLUGINS_AVAILABLE:
            try:
                registry = get_plugin_registry()
                plugin_types = registry.get_artifact_types()

                for name, plugin_def in plugin_types.items():
                    if name not in artifact_types:
                        artifact_types[name] = {
                            "name": name,
                            "source": "plugin",
                            "description": plugin_def.get("description", f"Plugin artifact type: {name}"),
                            "template": self._convert_plugin_to_template(name, plugin_def),
                            "validation": plugin_def.get("validation"),
                            "conflict": False,
                        }
                        sources_seen[name] = ["plugin"]
                    else:
                        # Type defined in multiple sources
                        sources_seen[name].append("plugin")
                        if artifact_types[name]["source"] == "hardcoded":
                            artifact_types[name]["source"] = "hardcoded (plugin available)"
                        artifact_types[name]["conflict"] = True

            except Exception:
                # Plugin loading failed - continue with hardcoded only
                pass

        # 3. Mark naming conflicts and inconsistencies
        # Known duplicates in our system: "assessment" in hardcoded and potentially plugins
        # Note: These are documented in the assessment artifact
        conflict_groups = {
            "assessment": ["hardcoded", "potential plugin"],
            "design": ["hardcoded", "possible design_document variant"],
            "research": ["hardcoded", "potential duplicate in standards"],
        }

        for type_name in conflict_groups:
            if type_name in artifact_types:
                artifact_types[type_name]["_conflict_note"] = conflict_groups[type_name]

        return artifact_types

    def get_available_templates_with_metadata(self) -> list[dict[str, Any]]:
        """
        Get list of available templates with source and validation metadata.

        Enhanced version of get_available_templates() that includes information
        about artifact source (hardcoded vs plugin), validation rules, and conflicts.

        Returns:
            List of dicts with keys:
            - name: template name
            - source: "hardcoded" or "plugin"
            - description: brief description
            - has_validation: bool
            - has_conflict: bool

        Example:
            templates = artifacts.get_available_templates_with_metadata()
            for t in templates:
                print(f"{t['name']}: {t['source']}")
        """
        types = self._get_available_artifact_types()

        result = []
        for name, info in types.items():
            result.append(
                {
                    "name": name,
                    "source": info["source"],
                    "description": info.get("description", ""),
                    "has_validation": info.get("validation") is not None,
                    "has_conflict": info.get("conflict", False),
                }
            )

        return sorted(result, key=lambda x: x["name"])

    def create_filename(self, template_type: str, name: str) -> str:
        """Create a properly formatted filename for an artifact."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        # Normalize name to lowercase kebab-case (artifacts must be lowercase)
        # Convert to lowercase and replace spaces/underscores with hyphens
        normalized_name = name.lower().replace(" ", "-").replace("_", "-").replace("--", "-").strip("-")

        # Generate timestamp using utility if available, fallback to old method
        if UTILITIES_AVAILABLE and _format_timestamp_for_filename is not None:
            timestamp = _format_timestamp_for_filename()
        else:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H%M")

        # Handle special case for bug reports (need bug ID)
        if template_type == "bug_report":
            # Extract bug ID from name or generate one
            if "_" in name:
                # Extract bug ID from original name (e.g., "BUG_001_description" -> "001")
                parts = name.split("_")
                bug_id = parts[0] if parts[0].upper() == "BUG" and len(parts) > 1 else parts[0]
                # If pattern is "BUG_NNN_description", extract NNN
                if len(parts) >= 2 and parts[1].isdigit():
                    bug_id = parts[1]
                    # Descriptive part starts from index 2
                    descriptive_parts = parts[2:]
                else:
                    # Otherwise assume first part is bug ID
                    bug_id = parts[0]
                    descriptive_parts = parts[1:]
                # Normalize descriptive parts: convert underscores to hyphens
                descriptive_name = "-".join(p.lower().replace(" ", "-") for p in descriptive_parts)
            else:
                bug_id = "001"  # Default bug ID
                descriptive_name = normalized_name

            # Build context for format strings
            filename_context = {
                "name": descriptive_name,
                "date": timestamp,
            }

            # Format filename with context, then replace legacy placeholders
            filename = template["filename_pattern"]
            try:
                filename = filename.format(**filename_context)
            except KeyError:
                # Fallback for old-style patterns without {date}
                filename = filename.format(name=descriptive_name)

            return str(filename.replace("YYYY-MM-DD_HHMM", timestamp).replace("NNN", bug_id))
        else:
            # For plugin-based templates, use .format() with available variables
            # Build context with all possible filename variables
            filename_context = {
                "name": normalized_name,
                "date": timestamp,  # Plugin {date} gets full timestamp
            }

            # Prevent duplicate type tokens: if pattern embeds a type prefix
            # (e.g., {date}_implementation_plan_{name}.md) and the slug already
            # starts with the same token, strip it from the slug.
            pattern = template["filename_pattern"]
            m = re.search(r"\{date\}_(.+?)_\{name\}", pattern)
            if m:
                type_hint = m.group(1).lower()
                variants = {type_hint}
                variants.add(type_hint.replace("_", "-"))

                for hint in variants:
                    for sep in ("-", "_"):
                        dup = f"{hint}{sep}"
                        if normalized_name.startswith(dup):
                            normalized_name = normalized_name[len(dup):].lstrip("-_")
                            break
                    # Rebuild context if modified
                    filename_context["name"] = normalized_name

            filename = template["filename_pattern"]

            # Try to format with context (for plugin templates)
            try:
                filename = filename.format(**filename_context)
            except KeyError:
                # Fallback: format with just name
                filename = filename.format(name=normalized_name)

            # Replace builtin pattern for legacy compatibility
            filename = filename.replace("YYYY-MM-DD_HHMM", timestamp)

            return str(filename)

    def create_frontmatter(self, template_type: str, title: str, **kwargs) -> str:
        """Create frontmatter for an artifact."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        frontmatter = template["frontmatter"].copy()
        frontmatter["title"] = title

        # Add timestamp using new utility if available, fallback to old method
        if UTILITIES_AVAILABLE and _get_kst_timestamp is not None:
            frontmatter["date"] = _get_kst_timestamp()
        else:
            from datetime import timedelta, timezone

            kst = timezone(timedelta(hours=9))  # KST is UTC+9
            frontmatter["date"] = datetime.now(kst).strftime("%Y-%m-%d %H:%M (KST)")

        # Add branch name if not explicitly provided in kwargs
        if "branch" not in kwargs:
            if UTILITIES_AVAILABLE and _get_current_branch is not None:
                try:
                    frontmatter["branch"] = _get_current_branch()
                except Exception:
                    frontmatter["branch"] = "main"  # Fallback
            else:
                frontmatter["branch"] = "main"  # Fallback

        # Add any additional frontmatter fields (may override defaults including branch)
        # Load denylist from config to avoid hardcoding
        config = self._load_template_defaults()
        denylist_from_config = set(config.get("frontmatter_denylist", []))

        # Always exclude system args
        system_args = {"output_dir", "interactive", "steps_to_reproduce"}
        denylist = denylist_from_config | system_args

        for key, value in kwargs.items():
            if key not in denylist:
                frontmatter[key] = value

        # Convert to YAML-like format
        lines = ["---"]
        for key, value in frontmatter.items():
            if isinstance(value, list):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f'{key}: "{value}"')
        lines.append("---")

        return "\n".join(lines)

    def create_content(self, template_type: str, title: str, **kwargs) -> str:
        """Create content for an artifact using the template."""
        template = self.get_template(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        content_template = template["content_template"]

        # Load defaults from external config
        config = self._load_template_defaults()

        # Build defaults with computed date values
        now = datetime.now()
        defaults = {
            "title": title,
            "start_date": now.strftime("%Y-%m-%d"),
            "target_date": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
            "assessment_date": now.strftime("%Y-%m-%d"),
        }

        # Merge general defaults from config
        defaults.update(config.get("defaults", {}))

        # Merge artifact-type-specific defaults (e.g., bug_report)
        if template_type in config:
            defaults.update(config[template_type])

        # Add plugin-defined template variables if present
        if "_plugin_variables" in template:
            defaults.update(template["_plugin_variables"])

        # Merge with provided kwargs (user values override defaults)
        context = {**defaults, **kwargs}

        return str(content_template.format(**context))

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

        # Check for recently created files with the same base name to prevent duplicates
        # Look for files created within the last 5 minutes with matching type and name
        normalized_name = name.lower().replace(" ", "-").replace("_", "-").replace("--", "-").strip("-")

        # Build pattern to match files based on artifact type
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")

        # Search for files with same date and base name pattern
        # Different patterns for different artifact types
        if template_type == "bug_report":
            # Bug reports: BUG_YYYY-MM-DD_HHMM_NNN_{name}.md
            # Extract bug ID if present in name (format: "NNN_descriptive-name")
            if "_" in name:
                bug_id = name.split("_")[0]
                descriptive_name = normalized_name.replace(bug_id + "-", "").strip("-")
            else:
                descriptive_name = normalized_name
            # Match files with same descriptive name (bug ID may differ)
            # Pattern needs to account for: BUG_DATE_TIME_BUGID_NAME.md
            pattern_base = f"BUG_{current_date}_*_*_{descriptive_name}.md"
        else:
            # Other types: YYYY-MM-DD_HHMM_{template_type}_{normalized-name}.md
            pattern_base = f"{current_date}_*_{template_type}_{normalized_name}.md"

        existing_files = list(output_path.glob(pattern_base))

        # Check if any existing file was created recently (within 5 minutes)
        if existing_files:
            for existing_file in sorted(existing_files, key=lambda p: p.stat().st_mtime, reverse=True):
                file_mtime = datetime.fromtimestamp(existing_file.stat().st_mtime)
                time_diff = (now - file_mtime).total_seconds()

                # If file was created within the last 5 minutes, reuse it
                if time_diff < 300:  # 5 minutes = 300 seconds
                    if not quiet:
                        print(f"⚠️  Found recently created file: {existing_file.name}")
                        print("   Reusing existing file instead of creating duplicate.")
                    return str(existing_file)

        # Create filename
        filename = self.create_filename(template_type, name)
        file_path = output_path / filename

        # Create content
        frontmatter = self.create_frontmatter(template_type, title, **kwargs)
        content = self.create_content(template_type, title, **kwargs)

        # Write file
        full_content = frontmatter + "\n\n" + content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return str(file_path)


# Convenience functions
def get_template(template_type: str) -> dict | None:
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


def get_available_templates() -> list:
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
