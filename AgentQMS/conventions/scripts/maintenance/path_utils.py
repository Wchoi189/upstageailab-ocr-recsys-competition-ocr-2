"""
Path utilities for the Korean Grammar Correction project.

This module provides centralized path resolution and project setup functionality.
"""

import os
import sys
from pathlib import Path


class PathResolver:
    """Centralized path resolver for the project."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the path resolver with project root."""
        self.project_root = project_root or self._find_project_root()
        self._setup_paths()

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        # Check if PROJECT_ROOT is set in environment
        project_root_env = os.getenv("PROJECT_ROOT")
        if project_root_env:
            return Path(project_root_env).resolve()

        current = Path(__file__).resolve()
        while current.parent != current:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        raise RuntimeError("Could not find project root (pyproject.toml)")

    def _setup_paths(self) -> None:
        """Setup common project paths."""
        self.config = self._PathConfig(self.project_root)

    class _PathConfig:
        """Configuration for common project paths."""

        def __init__(self, project_root: Path):
            self.project_root = project_root

            # Helper to resolve paths (relative to project_root or absolute)
            def resolve_path(env_var: str, default: Path) -> Path:
                """Resolve path from environment variable or use default."""
                env_path = os.getenv(env_var)
                if env_path:
                    path = Path(env_path)
                    if path.is_absolute():
                        return path
                    # Relative path - resolve from project root
                    return project_root / path
                return default

            self.data_dir = resolve_path(
                "GRAMMAR_CORRECTION_DATA_DIR", project_root / "data"
            )

            # Config directory can be overridden
            config_dir_env = os.getenv("STREAMLIT_CONFIG_DIR")
            if config_dir_env:
                config_path = Path(config_dir_env)
                if config_path.is_absolute():
                    self.config_dir = config_path
                else:
                    self.config_dir = project_root / config_path
            else:
                self.config_dir = project_root / "streamlit_app" / "config"

            self.output_dir = resolve_path(
                "GRAMMAR_CORRECTION_OUTPUT_DIR", project_root / "outputs"
            )

            cache_dir_env = os.getenv("GRAMMAR_CORRECTION_CACHE_DIR")
            if cache_dir_env:
                cache_path = Path(cache_dir_env)
                if cache_path.is_absolute():
                    self.cache_dir = cache_path
                else:
                    self.cache_dir = project_root / cache_path
            else:
                self.cache_dir = project_root / ".cache"

            # Subdirectories use resolved output_dir
            self.experiments_dir = self.output_dir / "experiments"
            self.results_dir = project_root / "results"
            self.logs_dir = project_root / "logs"
            self.streamlit_dir = project_root / "streamlit_app"
            self.scripts_dir = project_root / "scripts"
            self.docs_dir = project_root / "docs"

    def get_data_file(self, filename: str) -> Path:
        """Get path to a data file."""
        return self.config.data_dir / filename

    def get_config_file(self, filename: str) -> Path:
        """Get path to a config file."""
        return self.config.config_dir / filename

    def get_output_file(self, filename: str) -> Path:
        """Get path to an output file."""
        return self.config.output_dir / filename

    def get_log_file(self, filename: str) -> Path:
        """Get path to a log file."""
        return self.config.logs_dir / filename


# Global path resolver instance
_path_resolver: PathResolver | None = None


def get_path_resolver() -> PathResolver:
    """Get the global path resolver instance."""
    global _path_resolver
    if _path_resolver is None:
        _path_resolver = PathResolver()
    return _path_resolver


def setup_project_paths() -> None:
    """Setup project paths and add to sys.path."""
    resolver = get_path_resolver()

    # Add project root to Python path
    project_root_str = str(resolver.project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Ensure common directories exist
    for path in [
        resolver.config.data_dir,
        resolver.config.output_dir,
        resolver.config.experiments_dir,
        resolver.config.results_dir,
        resolver.config.logs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory."""
    return get_path_resolver().project_root


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_path_resolver().config.data_dir


def get_config_dir() -> Path:
    """Get the config directory."""
    return get_path_resolver().config.config_dir


def get_output_dir() -> Path:
    """Get the output directory."""
    return get_path_resolver().config.output_dir


def get_experiments_dir() -> Path:
    """Get the experiments directory."""
    return get_path_resolver().config.experiments_dir


def get_results_dir() -> Path:
    """Get the results directory."""
    return get_path_resolver().config.results_dir


def get_streamlit_config_dir() -> Path:
    """Get the Streamlit app config directory."""
    config_dir_env = os.getenv("STREAMLIT_CONFIG_DIR")
    if config_dir_env:
        return Path(config_dir_env)
    return get_path_resolver().config.config_dir


def get_model_config_path() -> Path:
    """
    Get the path to model_config.yaml file.

    Checks environment variables in this order:
    1. MODEL_CONFIG_PATH - Full path to config file
    2. STREAMLIT_CONFIG_DIR - Config directory (uses model_config.yaml)
    3. Default: streamlit_app/config/model_config.yaml

    Returns:
        Path to model_config.yaml file
    """
    # Check for explicit MODEL_CONFIG_PATH
    model_config_path = os.getenv("MODEL_CONFIG_PATH")
    if model_config_path:
        path = Path(model_config_path)
        if path.is_absolute():
            return path
        # Relative path - resolve from project root
        return get_path_resolver().project_root / path

    # Use config directory
    config_dir = get_streamlit_config_dir()
    return config_dir / "model_config.yaml"
