#!/usr/bin/env python3
"""
Configuration Loader Utility for AgentQMS

Framework-agnostic configuration loading with caching, validation, and graceful fallbacks.
Supports YAML format with extensibility for other formats (JSON, TOML).

Design principles:
- Framework-agnostic naming and structure
- Graceful degradation (works without PyYAML)
- Per-path caching for performance
- Type-safe with clear defaults
- Minimal dependencies

Usage:
    from AgentQMS.tools.utils.config_loader import ConfigLoader

    # Simple YAML loading
    config = ConfigLoader.load_yaml("path/to/config.yaml")

    # With fallback defaults
    config = ConfigLoader.load_yaml(
        "path/to/config.yaml",
        defaults={"enabled": True, "timeout": 30}
    )

    # With caching (for repeated access)
    loader = ConfigLoader()
    config1 = loader.get_config("path/to/config.yaml")
    config2 = loader.get_config("path/to/config.yaml")  # From cache

    # Extract nested value
    value = loader.get_config("path/to/config.yaml", key="section.subsection")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigLoader:
    """
    Framework-agnostic configuration loader with caching and fallbacks.

    Features:
    - YAML file loading with graceful fallbacks
    - Per-path caching to avoid repeated I/O
    - Configurable cache size
    - Nested key extraction (dot notation)
    - Type validation and defaults
    - Error handling with informative messages

    Design:
    - Stateless static methods for simple use cases
    - Stateful instance methods with caching for complex scenarios
    - Graceful degradation if PyYAML unavailable
    """

    def __init__(self, cache_size: int = 10):
        """
        Initialize ConfigLoader with optional caching.

        Args:
            cache_size: Maximum number of configs to keep in cache (default: 10).
                       Set to 0 to disable caching.
        """
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_size = cache_size
        self._cache_order: list[str] = []  # Track LRU order

    @staticmethod
    def load_yaml(config_path: Path | str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Load YAML configuration file with fallback defaults.

        Args:
            config_path: Path to YAML file (Path or str)
            defaults: Default dict to return if file not found or YAML unavailable.
                     If not provided, returns empty dict on failure.

        Returns:
            Loaded YAML content as dict, or defaults, or empty dict.

        Example:
            config = ConfigLoader.load_yaml(
                "config.yaml",
                defaults={"enabled": True}
            )
        """
        if defaults is None:
            defaults = {}

        config_path = Path(config_path) if isinstance(config_path, str) else config_path

        # Check if file exists
        if not config_path.exists():
            return defaults

        # Check if YAML is available
        if not YAML_AVAILABLE:
            return defaults

        # Load YAML file
        try:
            with config_path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                # yaml.safe_load returns None for empty files
                return loaded if loaded is not None else defaults
        except Exception:
            # Any error (parse error, permission, etc.) returns defaults
            return defaults

    def get_config(
        self,
        config_path: Path | str,
        key: str | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Any:
        """
        Load configuration with caching and optional nested key extraction.

        Args:
            config_path: Path to YAML file (Path or str)
            key: Optional nested key to extract (dot notation: "section.subsection")
            defaults: Default value if key not found or file missing.
                     If not provided, uses empty dict.

        Returns:
            Full config dict (if key=None), or value at key, or defaults.

        Example:
            # Load full config
            loader = ConfigLoader()
            config = loader.get_config("config.yaml")

            # Extract nested value
            timeout = loader.get_config("config.yaml", key="server.timeout", defaults=30)
        """
        if defaults is None:
            defaults = {}

        config_path_str = str(config_path)

        # Try cache first
        if config_path_str in self._cache:
            config = self._cache[config_path_str]
            # Update LRU order
            self._cache_order.remove(config_path_str)
            self._cache_order.append(config_path_str)
        else:
            # Load from file
            config = self.load_yaml(config_path, defaults={})

            # Add to cache
            if self._cache_size > 0:
                self._cache[config_path_str] = config
                self._cache_order.append(config_path_str)

                # Evict oldest if cache full
                if len(self._cache) > self._cache_size:
                    oldest = self._cache_order.pop(0)
                    del self._cache[oldest]

        # Extract nested key if requested
        if key:
            return self._extract_nested(config, key, defaults)
        else:
            return config if config else defaults

    @staticmethod
    def _extract_nested(data: dict[str, Any], key: str, defaults: Any = None) -> Any:
        """
        Extract nested value from dict using dot notation.

        Args:
            data: Dict to extract from
            key: Dot-notation path (e.g., "server.host" or "db.pool.size")
            defaults: Value to return if key not found

        Returns:
            Value at nested key, or defaults if not found.

        Example:
            data = {"server": {"host": "localhost", "port": 8080}}
            host = ConfigLoader._extract_nested(data, "server.host")  # "localhost"
            port = ConfigLoader._extract_nested(data, "server.port")  # 8080
            timeout = ConfigLoader._extract_nested(data, "server.timeout", defaults=30)  # 30
        """
        if not key or not data:
            return defaults

        keys = key.split(".")
        current = data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return defaults

        return current

    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._cache.clear()
        self._cache_order.clear()

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics for debugging."""
        return {
            "cached_paths": len(self._cache),
            "max_size": self._cache_size,
            "cached_files": list(self._cache_order),
        }


# Module-level convenience instance for basic use cases
_default_loader = ConfigLoader(cache_size=10)


def load_config(config_path: Path | str, key: str | None = None) -> dict[str, Any] | Any:
    """
    Convenience function for loading config with default loader instance.

    Uses module-level cached loader for consistency across imports.

    Args:
        config_path: Path to YAML file
        key: Optional nested key to extract

    Returns:
        Loaded config or nested value
    """
    return _default_loader.get_config(config_path, key=key)


if __name__ == "__main__":
    # Example usage and testing
    from tempfile import NamedTemporaryFile

    print("ConfigLoader Examples:")
    print("=" * 50)

    # Example 1: Load with defaults
    print("\n1. Load non-existent file with defaults:")
    config = ConfigLoader.load_yaml("nonexistent.yaml", defaults={"status": "default"})
    print(f"   Result: {config}")

    # Example 2: Create temp YAML file
    print("\n2. Load from temp YAML file:")
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("server:\n  host: localhost\n  port: 8080\n")
        f.flush()
        temp_path = Path(f.name)

        config = ConfigLoader.load_yaml(temp_path)
        print(f"   Loaded: {config}")

        # Example 3: Nested key extraction
        print("\n3. Extract nested key:")
        loader = ConfigLoader()
        host = loader.get_config(temp_path, key="server.host")
        print(f"   server.host = {host}")

        # Example 4: Caching
        print("\n4. Caching stats:")
        _ = loader.get_config(temp_path)
        _ = loader.get_config(temp_path)  # From cache
        cache_info = loader.get_cache_info()
        print(f"   {cache_info}")

        # Cleanup
        temp_path.unlink()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
