"""CLI Qwen VLM Backend Implementation."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent_qms.vlm.backends.base import BaseVLMBackend
from agent_qms.vlm.core.config import get_config, resolve_env_value
from agent_qms.vlm.core.contracts import AnalysisMode, BackendConfig, ProcessedImage
from agent_qms.vlm.core.interfaces import BackendError


class CLIQwenBackend(BaseVLMBackend):
    """CLI Qwen VLM backend for local execution."""

    def __init__(self, config: BackendConfig):
        """Initialize CLI Qwen backend.

        Args:
            config: Backend configuration
        """
        super().__init__(config)
        settings = get_config().backends.cli
        command_override = resolve_env_value(settings.command_env)
        self.qwen_command = command_override or settings.default_command
        self._check_qwen_available()

    @property
    def model_name(self) -> str:
        """Human-readable identifier for the CLI Qwen backend.

        Since the actual model may be configured inside the CLI tool, we
        surface the command used as a stable identifier.
        """
        return f"cli-qwen:{self.qwen_command}"

    def _check_qwen_available(self) -> None:
        """Check if Qwen VLM CLI is available."""
        try:
            result = subprocess.run(
                [self.qwen_command, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise BackendError(f"Qwen VLM CLI not available. Command '{self.qwen_command}' failed.")
        except FileNotFoundError:
            raise BackendError(
                f"Qwen VLM CLI not found. Install Qwen VLM or set QWEN_VLM_COMMAND environment variable."
            )
        except subprocess.TimeoutExpired:
            raise BackendError("Qwen VLM CLI command timed out.")

    def analyze_image(
        self,
        image_data: ProcessedImage,
        prompt: str,
        mode: AnalysisMode,
        **kwargs: Any,
    ) -> str:
        """Analyze an image using CLI Qwen VLM.

        Args:
            image_data: Preprocessed image data
            prompt: Analysis prompt
            mode: Analysis mode
            **kwargs: Additional parameters

        Returns:
            Analysis text from VLM

        Raises:
            BackendError: If analysis fails
        """
        # Use processed image path if available, otherwise original
        image_path = image_data.processed_path or image_data.original_path

        if not image_path or not image_path.exists():
            raise BackendError(f"Image path does not exist: {image_path}")

        # Create temporary prompt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as prompt_file:
            prompt_file.write(prompt)
            prompt_file_path = Path(prompt_file.name)

        try:
            # Build command
            cmd = [
                self.qwen_command,
                "analyze",
                "--image", str(image_path),
                "--prompt-file", str(prompt_file_path),
                "--mode", mode.value,
            ]

            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )

            if result.returncode != 0:
                raise BackendError(
                    f"Qwen VLM CLI failed with return code {result.returncode}.\n"
                    f"Error: {result.stderr}"
                )

            # Parse output (assuming JSON or plain text)
            output = result.stdout.strip()
            try:
                # Try to parse as JSON first
                parsed = json.loads(output)
                if "analysis" in parsed:
                    return parsed["analysis"]
                elif "text" in parsed:
                    return parsed["text"]
            except json.JSONDecodeError:
                # Not JSON, return as-is
                pass

            return output

        except subprocess.TimeoutExpired:
            raise BackendError(f"Qwen VLM CLI timed out after {self.config.timeout_seconds} seconds")
        except Exception as e:
            raise BackendError(f"Failed to analyze image with Qwen VLM CLI: {e}") from e
        finally:
            # Clean up temporary file
            if prompt_file_path.exists():
                prompt_file_path.unlink()

    def supports_batch(self) -> bool:
        """CLI Qwen supports batch processing via multiple command invocations."""
        return True

    def is_available(self) -> bool:
        """Check if CLI Qwen backend is available."""
        try:
            self._check_qwen_available()
            return True
        except BackendError:
            return False
