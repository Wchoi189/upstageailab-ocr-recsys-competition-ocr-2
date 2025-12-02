"""Path Utilities for VLM Module.

This module provides centralized path resolution utilities for the VLM module.
It handles common path operations and ensures consistent path resolution.
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Try to use existing project path utilities if available
try:
    from ocr.utils.path_utils import PROJECT_ROOT, get_path_resolver
    _PROJECT_ROOT = PROJECT_ROOT
except ImportError:
    # Fallback if ocr module not available
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _detect_project_root() -> Path:
    """Detect project root using multiple strategies.

    Detection order:
    1. Environment variable VLM_PROJECT_ROOT (explicit override)
    2. From __file__ location (works in packages)
    3. Walk up from CWD looking for project markers
    4. Fallback to CWD (with warning)

    Returns:
        Path to project root directory
    """
    # Strategy 1: Environment variable
    if env_root := os.getenv("VLM_PROJECT_ROOT"):
        root = Path(env_root).resolve()
        if root.exists():
            markers = ["pyproject.toml", ".git"]
            if any((root / marker).exists() for marker in markers):
                return root
            warnings.warn(
                f"VLM_PROJECT_ROOT={env_root} does not appear to be project root "
                "(missing pyproject.toml or .git). Using anyway.",
                UserWarning,
                stacklevel=2,
            )
            return root

    # Strategy 2: From __file__ location
    try:
        file_based = Path(__file__).resolve().parent.parent.parent.parent
        markers = ["pyproject.toml", ".git"]
        if any((file_based / marker).exists() for marker in markers):
            return file_based
    except (AttributeError, OSError):
        pass

    # Strategy 3: Walk up from CWD
    current_path = Path.cwd()
    project_markers = ["pyproject.toml", "requirements.txt", "setup.py", ".git"]

    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in project_markers):
            return parent

    # Strategy 4: Fallback
    warnings.warn(
        f"Could not detect project root. Using current working directory: {current_path}",
        UserWarning,
        stacklevel=2,
    )
    return current_path


# Global PROJECT_ROOT
PROJECT_ROOT = _detect_project_root()


@dataclass
class VLMPathConfig:
    """Configuration class for VLM module paths."""

    # Base directories
    project_root: Path
    vlm_module_dir: Path

    # VLM-specific directories
    cache_dir: Path
    via_annotations_dir: Path
    prompt_templates_dir: Path
    analysis_output_dir: Path

    # Integration paths
    experiment_tracker_dir: Path | None = None
    artifacts_dir: Path | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "VLMPathConfig":
        """Create VLMPathConfig from dictionary configuration."""
        project_root = Path(config.get("project_root", PROJECT_ROOT))
        vlm_module_dir = project_root / "AgentQMS" / "vlm"

        return cls(
            project_root=project_root,
            vlm_module_dir=vlm_module_dir,
            cache_dir=Path(config.get("cache_dir", project_root / ".vlm_cache")),
            via_annotations_dir=Path(
                config.get("via_annotations_dir", vlm_module_dir / "via" / "annotations")
            ),
            prompt_templates_dir=Path(
                config.get("prompt_templates_dir", vlm_module_dir / "prompts")
            ),
            analysis_output_dir=Path(
                config.get("analysis_output_dir", project_root / "artifacts" / "vlm_analysis")
            ),
            experiment_tracker_dir=(
                Path(config["experiment_tracker_dir"])
                if config.get("experiment_tracker_dir")
                else (project_root / "experiment-tracker")
            ),
            artifacts_dir=(
                Path(config["artifacts_dir"]) if config.get("artifacts_dir") else (project_root / "artifacts")
            ),
        )

    def resolve_path(self, path: str | Path, base: Path | None = None) -> Path:
        """Resolve a path relative to a base directory or project root."""
        path = Path(path)

        if path.is_absolute():
            return path

        if base is None:
            base = self.project_root

        return base / path

    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.cache_dir,
            self.via_annotations_dir,
            self.prompt_templates_dir,
            self.analysis_output_dir,
        ]

        if self.experiment_tracker_dir:
            directories.append(self.experiment_tracker_dir)
        if self.artifacts_dir:
            directories.append(self.artifacts_dir)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class VLMPathResolver:
    """Central path resolution manager for VLM module."""

    def __init__(self, config: VLMPathConfig | None = None):
        self.config = config or self._create_default_config()

    def _create_default_config(self) -> VLMPathConfig:
        """Create default path configuration for VLM module."""
        project_root = PROJECT_ROOT
        vlm_module_dir = project_root / "AgentQMS" / "vlm"

        return VLMPathConfig(
            project_root=project_root,
            vlm_module_dir=vlm_module_dir,
            cache_dir=project_root / ".vlm_cache",
            via_annotations_dir=vlm_module_dir / "via" / "annotations",
            prompt_templates_dir=vlm_module_dir / "prompts",
            analysis_output_dir=project_root / "artifacts" / "vlm_analysis",
            experiment_tracker_dir=project_root / "experiment-tracker",
            artifacts_dir=project_root / "artifacts",
        )

    def get_cache_path(self, filename: str = "") -> Path:
        """Get path to VLM cache directory."""
        return self.config.cache_dir / filename if filename else self.config.cache_dir

    def get_via_annotations_path(self, filename: str = "") -> Path:
        """Get path to VIA annotations directory."""
        return (
            self.config.via_annotations_dir / filename
            if filename
            else self.config.via_annotations_dir
        )

    def get_prompt_templates_path(self, filename: str = "") -> Path:
        """Get path to prompt templates directory."""
        return (
            self.config.prompt_templates_dir / filename
            if filename
            else self.config.prompt_templates_dir
        )

    def get_analysis_output_path(self, filename: str = "") -> Path:
        """Get path to analysis output directory."""
        return (
            self.config.analysis_output_dir / filename
            if filename
            else self.config.analysis_output_dir
        )

    def get_experiment_tracker_path(self, *parts: str) -> Path:
        """Get path relative to experiment-tracker directory."""
        if not self.config.experiment_tracker_dir:
            raise ValueError("Experiment tracker directory not configured")
        if parts:
            return self.config.experiment_tracker_dir / Path(*parts)
        return self.config.experiment_tracker_dir

    def resolve_relative_path(self, path: str | Path, base: str | None = None) -> Path:
        """Resolve a path that might be relative to different bases."""
        path = Path(path)

        if path.is_absolute():
            return path

        # Handle common relative path patterns
        if base == "project":
            return self.config.project_root / path
        elif base == "vlm":
            return self.config.vlm_module_dir / path
        elif base == "cache":
            return self.config.cache_dir / path
        elif base == "artifacts":
            if not self.config.artifacts_dir:
                raise ValueError("Artifacts directory not configured")
            return self.config.artifacts_dir / path
        else:
            # Default to project root
            return self.config.project_root / path

    @classmethod
    def from_environment(cls) -> "VLMPathResolver":
        """Create VLMPathResolver from environment variables."""
        config_dict = {}

        # Check for environment variables
        env_mappings = {
            "VLM_PROJECT_ROOT": "project_root",
            "VLM_CACHE_DIR": "cache_dir",
            "VLM_VIA_ANNOTATIONS_DIR": "via_annotations_dir",
            "VLM_PROMPT_TEMPLATES_DIR": "prompt_templates_dir",
            "VLM_ANALYSIS_OUTPUT_DIR": "analysis_output_dir",
            "VLM_EXPERIMENT_TRACKER_DIR": "experiment_tracker_dir",
            "VLM_ARTIFACTS_DIR": "artifacts_dir",
        }

        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                config_dict[config_key] = os.environ[env_var]

        if config_dict:
            path_config = VLMPathConfig.from_dict(config_dict)
            return cls(path_config)
        else:
            return cls()


# Global path resolver instance
_vlm_path_resolver = VLMPathResolver()


def get_path_resolver() -> VLMPathResolver:
    """Get the global VLM path resolver instance."""
    return _vlm_path_resolver


def setup_vlm_paths(config: dict[str, Any] | None = None) -> VLMPathResolver:
    """Setup VLM paths and return resolver.

    Args:
        config: Optional configuration dictionary with path settings.
               If None, reads from environment variables or uses auto-detection.

    Returns:
        Configured VLMPathResolver instance
    """
    global _vlm_path_resolver

    if config:
        path_config = VLMPathConfig.from_dict(config)
        _vlm_path_resolver = VLMPathResolver(path_config)
    else:
        _vlm_path_resolver = VLMPathResolver.from_environment()

    # Ensure all directories exist
    _vlm_path_resolver.config.ensure_directories()

    return _vlm_path_resolver
