#!/usr/bin/env python3
"""
Path Utilities for OCR Project

This module provides centralized path resolution utilities for the OCR project.
It handles common path operations and ensures consistent path resolution across all scripts.
Enhanced with modular path configuration for better reusability.

Modern API:
    from ocr.core.utils.path_utils import setup_project_paths, get_path_resolver

    # Initialize project paths
    setup_project_paths()

    # Access path configuration
    resolver = get_path_resolver()
    data_dir = resolver.config.data_dir
"""

import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ocr.core.utils.experiment_index import get_next_experiment_index

_HYDRA_RESOLVERS_REGISTERED = False


def _register_hydra_resolvers() -> None:
    """Register global Hydra resolvers used across CLI entrypoints."""
    global _HYDRA_RESOLVERS_REGISTERED

    if _HYDRA_RESOLVERS_REGISTERED:
        return

    try:
        OmegaConf.register_new_resolver("exp_index", lambda: get_next_experiment_index())
    except ValueError as exc:
        # Resolver may already be registered if multiple modules import this helper.
        if "exp_index" not in str(exc):
            raise
    finally:
        _HYDRA_RESOLVERS_REGISTERED = True


_register_hydra_resolvers()


@dataclass
class OCRPathConfig:
    """Configuration class for OCR project paths."""

    # Base directories
    project_root: Path
    data_dir: Path
    config_dir: Path
    output_dir: Path

    # Data subdirectories
    images_dir: Path
    annotations_dir: Path
    pseudo_labels_dir: Path

    # Output subdirectories
    logs_dir: Path
    checkpoints_dir: Path
    submissions_dir: Path

    # Model and config paths
    models_dir: Path | None = None
    pretrained_models_dir: Path | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "OCRPathConfig":
        """Create OCRPathConfig from dictionary configuration."""
        return cls(
            project_root=Path(config.get("project_root", ".")),
            data_dir=Path(config.get("data_dir", "data")),
            config_dir=Path(config.get("config_dir", "configs")),
            output_dir=Path(config.get("output_dir", "outputs")),
            images_dir=Path(config.get("images_dir", "data/datasets/images")),
            annotations_dir=Path(config.get("annotations_dir", "data/datasets/jsons")),
            pseudo_labels_dir=Path(config.get("pseudo_labels_dir", "data/pseudo_label")),
            logs_dir=Path(config.get("logs_dir", "outputs/logs")),
            checkpoints_dir=Path(config.get("checkpoints_dir", "outputs/experiments/train/ocr")),
            submissions_dir=Path(config.get("submissions_dir", "outputs/submissions")),
            models_dir=(Path(config.get("models_dir", "models")) if config.get("models_dir") else None),
            pretrained_models_dir=(
                Path(config.get("pretrained_models_dir", "pretrained")) if config.get("pretrained_models_dir") else None
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
            self.data_dir,
            self.config_dir,
            self.output_dir,
            self.images_dir,
            self.annotations_dir,
            self.pseudo_labels_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.submissions_dir,
        ]

        if self.models_dir:
            directories.append(self.models_dir)
        if self.pretrained_models_dir:
            directories.append(self.pretrained_models_dir)

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def _detect_project_root() -> Path:
    """Detect project root using multiple strategies (stable, works from any location).

    Detection order:
    1. Environment variable OCR_PROJECT_ROOT (explicit override)
    2. From __file__ location (works in packages, most reliable)
    3. Walk up from CWD looking for project markers
    4. Fallback to CWD (with warning)

    Returns:
        Path to project root directory
    """
    # Strategy 1: Environment variable (explicit override, highest priority)
    if env_root := os.getenv("OCR_PROJECT_ROOT"):
        root = Path(env_root).resolve()
        if root.exists():
            # Validate it's actually the project root
            markers = ["pyproject.toml", ".git"]
            if any((root / marker).exists() for marker in markers):
                return root
            # Warn if marker not found but path exists
            warnings.warn(
                f"OCR_PROJECT_ROOT={env_root} does not appear to be project root (missing pyproject.toml or .git). Using anyway.",
                UserWarning,
                stacklevel=2,
            )
            return root

    # Strategy 2: From __file__ location (works in packages)
    # This file is at ocr.core.utils/path_utils.py, so go up 2 levels to get project root
    try:
        file_based = Path(__file__).resolve().parent.parent.parent
        # Validate with project markers
        markers = ["pyproject.toml", ".git"]
        if any((file_based / marker).exists() for marker in markers):
            return file_based
    except (AttributeError, OSError):
        # __file__ might not be available in some contexts (e.g., frozen executables)
        pass

    # Strategy 3: Walk up from CWD looking for project markers
    current_path = Path.cwd()
    project_markers = ["pyproject.toml", "requirements.txt", "setup.py", ".git"]

    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in project_markers):
            return parent

    # Strategy 4: Fallback to CWD (with warning)
    warnings.warn(
        f"Could not detect project root. Using current working directory: {current_path}",
        UserWarning,
        stacklevel=2,
    )
    return current_path


# Global PROJECT_ROOT - stable, works from any location
PROJECT_ROOT = _detect_project_root()


class OCRPathResolver:
    """Central path resolution manager for OCR project."""

    def __init__(self, config: OCRPathConfig | None = None):
        self.config = config or self._create_default_config()

    def _create_default_config(self) -> OCRPathConfig:
        """Create default path configuration for OCR project."""
        # Use the global PROJECT_ROOT for consistency
        project_root = PROJECT_ROOT

        return OCRPathConfig(
            project_root=project_root,
            data_dir=project_root / "data",
            config_dir=project_root / "configs",
            output_dir=project_root / "outputs",
            images_dir=project_root / "data" / "datasets" / "images",
            annotations_dir=project_root / "data" / "datasets" / "jsons",
            pseudo_labels_dir=project_root / "data" / "pseudo_label",
            logs_dir=project_root / "outputs" / "logs",
            checkpoints_dir=project_root / "outputs" / "checkpoints",
            submissions_dir=project_root / "outputs" / "submissions",
        )

    def get_data_path(self, dataset: str, split: str = "train") -> Path:
        """Get path to dataset images."""
        return self.config.images_dir / dataset / split

    def get_annotation_path(self, dataset: str, split: str = "train") -> Path:
        """Get path to dataset annotations."""
        return self.config.annotations_dir / f"{split}.json"

    def get_checkpoint_path(self, experiment_name: str, version: str = "v1.0") -> Path:
        """Get path to experiment checkpoints."""
        return self.config.checkpoints_dir / experiment_name / version

    def get_log_path(self, experiment_name: str, version: str = "v1.0") -> Path:
        """Get path to experiment logs."""
        return self.config.logs_dir / experiment_name / version

    def get_submission_path(self, experiment_name: str) -> Path:
        """Get path to experiment submissions."""
        return self.config.submissions_dir / experiment_name

    def resolve_relative_path(self, path: str | Path, base: str | None = None) -> Path:
        """Resolve a path that might be relative to different bases."""
        path = Path(path)

        if path.is_absolute():
            return path

        # Handle common relative path patterns
        if base == "project":
            return self.config.project_root / path
        elif base == "data":
            return self.config.data_dir / path
        elif base == "config":
            return self.config.config_dir / path
        elif base == "output":
            return self.config.output_dir / path
        else:
            # Default to project root
            return self.config.project_root / path

    @classmethod
    def from_environment(cls) -> "OCRPathResolver":
        """Create OCRPathResolver from environment variables."""
        config_dict = {}

        # Check for environment variables
        env_mappings = {
            "OCR_PROJECT_ROOT": "project_root",
            "OCR_DATA_DIR": "data_dir",
            "OCR_CONFIG_DIR": "config_dir",
            "OCR_OUTPUT_DIR": "output_dir",
            "OCR_IMAGES_DIR": "images_dir",
            "OCR_ANNOTATIONS_DIR": "annotations_dir",
            "OCR_LOGS_DIR": "logs_dir",
            "OCR_CHECKPOINTS_DIR": "checkpoints_dir",
            "OCR_SUBMISSIONS_DIR": "submissions_dir",
        }

        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                config_dict[config_key] = os.environ[env_var]

        if config_dict:
            path_config = OCRPathConfig.from_dict(config_dict)
            return cls(path_config)
        else:
            return cls()


# Global path resolver instance
_ocr_path_resolver = OCRPathResolver()


def get_path_resolver() -> OCRPathResolver:
    """Get the global OCR path resolver instance."""
    return _ocr_path_resolver


def setup_project_paths(config: dict[str, Any] | None = None) -> OCRPathResolver:
    """Setup project paths and return resolver.

    This function initializes path configuration from:
    1. Explicit config dict (if provided)
    2. Environment variables (OCR_* vars)
    3. Auto-detection (default)

    It also ensures all required directories exist.

    Args:
        config: Optional configuration dictionary with path settings.
               If None, reads from environment variables or uses auto-detection.

    Returns:
        Configured OCRPathResolver instance

    Environment Variables:
        OCR_PROJECT_ROOT: Override project root directory
        OCR_CONFIG_DIR: Override config directory
        OCR_OUTPUT_DIR: Override output directory
        OCR_DATA_DIR: Override data directory
        OCR_IMAGES_DIR: Override images directory
        OCR_ANNOTATIONS_DIR: Override annotations directory
        OCR_LOGS_DIR: Override logs directory
        OCR_CHECKPOINTS_DIR: Override checkpoints directory
        OCR_SUBMISSIONS_DIR: Override submissions directory

    Example:
        ```python
        # Use environment variables
        import os
        os.environ["OCR_OUTPUT_DIR"] = "/custom/outputs"
        resolver = setup_project_paths()

        # Or use explicit config
        resolver = setup_project_paths({"output_dir": "/custom/outputs"})
        ```
    """
    global _ocr_path_resolver

    if config:
        path_config = OCRPathConfig.from_dict(config)
        _ocr_path_resolver = OCRPathResolver(path_config)
    else:
        # Try environment variables first, fall back to auto-detection
        _ocr_path_resolver = OCRPathResolver.from_environment()

    # Ensure all directories exist (creates them if they don't)
    _ocr_path_resolver.config.ensure_directories()

    return _ocr_path_resolver


# Note: __all__ is defined later in the file to include all exports


def ensure_output_dirs(paths: Iterable[str | os.PathLike[str] | Path]) -> list[Path]:
    """Ensure each provided directory exists.

    Creates the directories with parents and returns the resolved paths.
    Raises if a provided path resolves to a file or is None to fail fast on misconfiguration.
    """
    resolved_paths: list[Path] = []

    for raw_path in paths:
        if raw_path is None:
            raise ValueError("Received None in ensure_output_dirs paths")

        path = Path(raw_path)
        resolved_paths.append(path)

        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"Expected directory but found file at {path}")

        path.mkdir(parents=True, exist_ok=True)

    return resolved_paths


# Convenience functions for backward compatibility and easy importing
def get_project_root() -> Path:
    """Get project root directory.

    Returns the stable PROJECT_ROOT that was detected at module import time.
    This works from any location and uses multiple detection strategies.

    Returns:
        Path to project root directory

    Note:
        For new code, you can directly import PROJECT_ROOT instead:
        ```python
                ```
    """
    return PROJECT_ROOT
