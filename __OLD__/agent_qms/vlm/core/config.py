"""Central configuration loader for VLM tools."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field

from agent_qms.vlm.utils.paths import PROJECT_ROOT

VLM_DIR = PROJECT_ROOT / "agent_qms" / "vlm"
CONFIG_PATH = VLM_DIR / "config.yaml"


class OpenRouterSettings(BaseModel):
    base_url: str
    default_model: str
    api_key_env: str = "OPENROUTER_API_KEY"


class SolarPro2Settings(BaseModel):
    endpoint: str
    default_model: str
    api_key_env: str = "SOLAR_PRO2_API_KEY"


class CLISettings(BaseModel):
    default_command: str = "qwen-vl"
    command_env: str = "QWEN_VLM_COMMAND"


class BackendSettings(BaseModel):
    default: str = "openrouter"
    priority: List[str] = Field(default_factory=lambda: ["openrouter", "solar_pro2", "cli"])
    openrouter: OpenRouterSettings
    solar_pro2: SolarPro2Settings
    cli: CLISettings


class ImageSettings(BaseModel):
    max_resolution: int = 2048
    max_dimension: int = 8192
    default_quality: int = 95
    supported_formats: List[str] = Field(default_factory=lambda: ["JPEG", "PNG", "WEBP"])


class BackendDefaults(BaseModel):
    timeout_seconds: int = 60
    max_retries: int = 3
    max_resolution: int = 2048


class EnvSettings(BaseModel):
    search_paths: List[str] = Field(default_factory=lambda: ["agent_qms/vlm", "."])
    files: List[str] = Field(default_factory=lambda: [".env", ".env.local"])
    priority: str = "local"


class VLMConfig(BaseModel):
    backends: BackendSettings
    image: ImageSettings
    backend_defaults: BackendDefaults
    env: EnvSettings


def _load_yaml_config() -> Dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"VLM configuration file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _resolve_search_paths(paths: List[str]) -> List[Path]:
    resolved = []
    for entry in paths:
        candidate = (PROJECT_ROOT / entry).resolve()
        if candidate not in resolved:
            resolved.append(candidate)
    return resolved


def _load_env_values(env_settings: EnvSettings) -> Dict[str, str]:
    values: Dict[str, str] = {}
    search_paths = _resolve_search_paths(env_settings.search_paths)

    for env_file in env_settings.files:
        for base in search_paths:
            candidate = (base / env_file).resolve()
            if candidate.exists():
                data = dotenv_values(candidate)
                values.update({k: v for k, v in data.items() if v is not None})

    # Environment variables override everything else
    values.update({k: v for k, v in os.environ.items()})
    return values


@lru_cache(maxsize=1)
def get_config() -> VLMConfig:
    """Load and cache the VLM configuration."""
    data = _load_yaml_config()
    return VLMConfig(**data)


@lru_cache(maxsize=1)
def get_env_values() -> Dict[str, str]:
    """Load and cache environment values from .env files and OS environment."""
    config = get_config()
    return _load_env_values(config.env)


def resolve_env_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """Lookup a configuration value from env files or OS environment."""
    env_values = get_env_values()
    return env_values.get(key) or os.getenv(key) or default


def refresh_config_cache() -> None:
    """Clear cached configuration and env values (useful for tests)."""
    get_config.cache_clear()  # type: ignore[attr-defined]
    get_env_values.cache_clear()  # type: ignore[attr-defined]
