"""VLM backend implementations."""

from agent_qms.vlm.backends.base import BaseVLMBackend
from agent_qms.vlm.backends.cli_qwen import CLIQwenBackend
from agent_qms.vlm.backends.openrouter import OpenRouterBackend
from agent_qms.vlm.backends.solar_pro2 import SolarPro2Backend

__all__ = [
    "BaseVLMBackend",
    "OpenRouterBackend",
    "SolarPro2Backend",
    "CLIQwenBackend",
    "create_backend",
]


def create_backend(backend_type: str, config: dict | None = None) -> BaseVLMBackend:
    """Create a VLM backend instance.

    Args:
        backend_type: Type of backend ('openrouter', 'solar_pro2', 'cli')
        config: Optional backend configuration dict

    Returns:
        Backend instance

    Raises:
        ValueError: If backend type is invalid
        BackendError: If backend cannot be created
    """
    from agent_qms.vlm.core.contracts import BackendConfig

    if config is None:
        config = {}

    backend_config = BackendConfig(
        backend_type=backend_type,
        **config,
    )

    if backend_type == "openrouter":
        return OpenRouterBackend(backend_config)
    elif backend_type == "solar_pro2":
        return SolarPro2Backend(backend_config)
    elif backend_type == "cli":
        return CLIQwenBackend(backend_config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
