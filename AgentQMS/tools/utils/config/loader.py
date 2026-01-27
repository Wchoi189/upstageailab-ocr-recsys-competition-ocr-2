#!/usr/bin/env python3
"""
Configuration Loader Utility for AgentQMS

Framework-agnostic configuration loading with caching, validation, and graceful fallbacks.
Supports YAML format with extensibility for other formats (JSON, TOML).

Design principles:
- Framework-agnostic naming and structure
- Graceful degradation (works without PyYAML)
- Distributed caching via Redis (optional)
- Per-path caching for performance
- Type-safe with clear defaults
- Minimal dependencies

Usage:
    from AgentQMS.tools.utils.config.loader import ConfigLoader

    # Simple YAML loading
    config = ConfigLoader.load_yaml("path/to/config.yaml")

    # With fallback defaults
    config = ConfigLoader.load_yaml(
        "path/to/config.yaml",
        defaults={"enabled": True, "timeout": 30}
    )

    # With caching (Redis -> Memory -> Disk)
    loader = ConfigLoader()
    config1 = loader.get_config("path/to/config.yaml")
    config2 = loader.get_config("path/to/config.yaml")  # From cache

    # Extract nested value
    value = loader.get_config("path/to/config.yaml", key="section.subsection")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class ConfigLoader:
    """
    Framework-agnostic configuration loader with caching and fallbacks.

    Features:
    - YAML file loading with graceful fallbacks
    - Distributed caching via Redis (priority)
    - Local memory caching (fallback)
    - Per-path caching to avoid repeated I/O
    - Configurable cache size
    - Nested key extraction (dot notation)
    - Type validation and defaults
    - Error handling with informative messages

    Design:
    - Stateless static methods for simple use cases
    - Stateful instance methods with caching for complex scenarios
    - Graceful degradation if PyYAML/Redis unavailable
    - Virtual configuration support (memory-only)
    """

    def __init__(self, cache_size: int = 10, redis_ttl: int = 3600):
        """
        Initialize ConfigLoader with optional caching.

        Args:
            cache_size: Maximum number of configs to keep in local memory cache (default: 10).
            redis_ttl: Time-to-live for Redis cache in seconds (default: 3600).
        """
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_size = cache_size
        self._cache_order: list[str] = []  # Track LRU order
        self._redis_ttl = redis_ttl
        self._redis_client = None

        if REDIS_AVAILABLE:
            try:
                # Use environment variables for configuration
                host = os.getenv("REDIS_HOST", "redis")
                port = int(os.getenv("REDIS_PORT", 6379))
                self._redis_client = redis.Redis(
                    host=host, 
                    port=port, 
                    decode_responses=True, 
                    socket_connect_timeout=1
                )
                # Quick health check (optional, but good for fast failover logic)
                # self._redis_client.ping() 
            except Exception as e:
                logger.warning(f"Failed to initialize Redis client: {e}. Falling back to memory cache.")
                self._redis_client = None

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
        
        Priority:
        1. Local Memory Cache
        2. Redis Cache
        3. File System

        Args:
            config_path: Path to YAML file (Path or str)
            key: Optional nested key to extract (dot notation: "section.subsection")
            defaults: Default value if key not found or file missing.
                     If not provided, uses empty dict.

        Returns:
            Full config dict (if key=None), or value at key, or defaults.
        """
        if defaults is None:
            defaults = {}

        config_path_str = str(config_path)

        # 1. Try Local Memory Cache first (fastest)
        if config_path_str in self._cache:
            config = self._cache[config_path_str]
            # Update LRU order
            self._cache_order.remove(config_path_str)
            self._cache_order.append(config_path_str)
        else:
            config = None
            
            # 2. Try Redis Cache
            if self._redis_client:
                try:
                    redis_key = f"config:{config_path_str}"
                    cached_data = self._redis_client.get(redis_key)
                    if cached_data:
                        config = json.loads(cached_data)
                except Exception as e:
                    logger.warning(f"Redis get failed for {config_path_str}: {e}")

            # 3. Load from File System if not in cache
            if config is None:
                config = self.load_yaml(config_path, defaults={})
                
                # Update Redis
                if self._redis_client and config:
                    try:
                        redis_key = f"config:{config_path_str}"
                        self._redis_client.setex(redis_key, self._redis_ttl, json.dumps(config))
                    except Exception as e:
                        logger.warning(f"Redis set failed for {config_path_str}: {e}")

            # Update Local Memory Cache
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
        """Clear all cached configurations (Memory and Redis)."""
        self._cache.clear()
        self._cache_order.clear()
        
        if self._redis_client:
            try:
                # We only clear keys prefixed with config:
                # Use scan_iter for safe iteration over keys
                for key in self._redis_client.scan_iter("config:*"):
                    self._redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache statistics for debugging."""
        info = {
            "memory_cached_paths": len(self._cache),
            "memory_max_size": self._cache_size,
            "memory_cached_files": list(self._cache_order),
            "redis_connected": False
        }
        
        if self._redis_client:
            try:
                self._redis_client.ping()
                info["redis_connected"] = True
            except Exception:
                pass
                
        return info

    def resolve_active_standards(
        self,
        current_path: Path | str | None = None,
        registry_path: Path | str = "AgentQMS/standards/registry.yaml",
    ) -> list[str]:
        """
        Resolve active standards based on current working directory.

        This implements path-aware discovery by checking the current path against
        path_patterns defined in the registry and returning matching standards.

        Args:
            current_path: Current working directory or file path. If None, uses os.getcwd()
            registry_path: Path to the registry.yaml file

        Returns:
            List of standard file paths that match the current path
        """
        import fnmatch

        if current_path is None:
            current_path = Path.cwd()
        else:
            current_path = Path(current_path)

        # Load registry
        registry = self.get_config(registry_path)
        if not registry or "task_mappings" not in registry:
            return []

        active_standards = []
        seen = set()  # Avoid duplicates

        # Check each task mapping for path pattern matches
        for task_name, task_config in registry["task_mappings"].items():
            triggers = task_config.get("triggers", {})
            path_patterns = triggers.get("path_patterns", [])

            for pattern in path_patterns:
                # Convert current_path to relative path for matching
                try:
                    rel_path = current_path.relative_to(Path.cwd())
                except ValueError:
                    rel_path = current_path

                # Check if path matches pattern (using fnmatch for glob patterns)
                if fnmatch.fnmatch(str(rel_path), pattern) or fnmatch.fnmatch(str(current_path), pattern):
                    # Add standards to active list
                    standards = task_config.get("standards", [])
                    for std in standards:
                        if std not in seen:
                            active_standards.append(std)
                            seen.add(std)

        return active_standards

    def generate_virtual_config(
        self,
        current_path: Path | str | None = None,
        registry_path: Path | str = "AgentQMS/standards/registry.yaml",
        settings_path: Path | str = "AgentQMS/.agentqms/settings.yaml",
    ) -> dict[str, Any]:
        """
        Generate the effective config and return it as a dict.
        Does NOT touch the disk unless physical logging is enabled.
        """
        from datetime import datetime, timezone
        import os

        # Load base settings
        settings = self.get_config(settings_path, defaults={})
        
        # Load registry checks
        active_standards = self.resolve_active_standards(current_path, registry_path)

        # Build the 'Resolved' object (The Virtual State)
        effective = {
            "metadata": {
                "session_id": os.getenv("SESSION_ID", "local"),
                "path": str(current_path) if current_path else str(Path.cwd()),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "virtual": True
            },
            "resolved": {
                **settings.get("base", {}),
                "active_standards": active_standards,
            }
        }
        
        # Merge existing resolved settings if present
        if "resolved" in settings:
             effective["resolved"].update(settings["resolved"])
        
        # Ensure context_integration exists
        if "context_integration" not in effective["resolved"]:
            effective["resolved"]["context_integration"] = {}

        effective["resolved"]["context_integration"]["active_standards"] = active_standards
        
        return effective


    def generate_effective_config(
        self,
        settings_path: Path | str = "AgentQMS/.agentqms/settings.yaml",
        registry_path: Path | str = "AgentQMS/standards/registry.yaml",
        current_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Generate effective.yaml with dynamic context injection.

        Combines base settings with path-aware standard discovery to create
        a context-aware configuration.

        Args:
            settings_path: Path to settings.yaml
            registry_path: Path to registry.yaml
            current_path: Current working directory for path-aware discovery

        Returns:
            Complete effective configuration dict with active standards injected
        """
        from datetime import datetime, timezone

        # Load base settings
        settings = self.get_config(settings_path, defaults={})

        # Resolve active standards based on current path
        active_standards = self.resolve_active_standards(current_path, registry_path)

        # Build effective config
        effective = {
            "layers": {"settings": str(settings_path)},
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator": "AgentQMS ConfigLoader v0.3 (Path-Aware)",
                "schema_version": "0.3",
            },
            "resolved": settings.get("resolved", {}),
        }

        # Inject active standards into context_integration
        if "context_integration" not in effective["resolved"]:
            effective["resolved"]["context_integration"] = {}

        effective["resolved"]["context_integration"]["active_standards"] = active_standards
        effective["resolved"]["context_integration"]["discovery_method"] = "path_aware"

        if current_path:
            effective["resolved"]["context_integration"]["current_path"] = str(current_path)

        return effective


# Module-level convenience instance for basic use cases
_default_loader = ConfigLoader(cache_size=10)


def load_config(config_path: Path | str, key: str | None = None) -> dict[str, Any] | Any:
    """
    Convenience function for loading config with default loader instance.
    Uses module-level cached loader for consistency across imports.
    """
    return _default_loader.get_config(config_path, key=key)


if __name__ == "__main__":
    # Example usage and testing
    from tempfile import NamedTemporaryFile

    print("ConfigLoader Examples (with Redis Support):")
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

        loader = ConfigLoader() # Will try to connect to Redis
        
        # First load (miss)
        print("   First load (IO):")
        config = loader.get_config(temp_path)
        print(f"   Loaded: {config}")

        # Second load (Cache)
        print("   Second load (Memory Cache):")
        config2 = loader.get_config(temp_path)
        print(f"   Loaded: {config2}")

        cache_info = loader.get_cache_info()
        print(f"   {cache_info}")

        # Cleanup
        try:
           temp_path.unlink()
        except:
           pass

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
