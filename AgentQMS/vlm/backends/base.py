"""Base Backend Implementation.

Provides common functionality for all VLM backends.
"""

from AgentQMS.vlm.core.contracts import BackendConfig
from AgentQMS.vlm.core.interfaces import VLMBackend


class BaseVLMBackend(VLMBackend):
    """Base implementation with common functionality."""

    def __init__(self, config: BackendConfig):
        """Initialize base backend.

        Args:
            config: Backend configuration
        """
        super().__init__(config)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate backend configuration."""
        if self.config.backend_type not in ("openrouter", "solar_pro2", "cli"):
            raise ValueError(f"Invalid backend type: {self.config.backend_type}")

    @property
    def model_name(self) -> str:
        """Return a human-readable model identifier for this backend.

        Subclasses can override to provide a richer identifier. By default this
        falls back to the configured model string when available, otherwise the
        backend_type.
        """
        # Some backends (e.g. CLI) may not have a model configured; in that case
        # the backend_type still provides a stable identifier.
        return self.config.model or self.config.backend_type

    def get_max_resolution(self) -> int:
        """Get maximum supported image resolution.

        Returns:
            Maximum resolution from config
        """
        return self.config.max_resolution

    def supports_batch(self) -> bool:
        """Check if backend supports batch processing.

        Default implementation returns False.
        Subclasses can override.
        """
        return False
