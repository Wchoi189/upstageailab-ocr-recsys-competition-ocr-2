"""
Compass Integration Module for ETK CLI.

Provides environment validation and session management by reading/writing
Project Compass configuration files with schema validation and atomic writes.

PRINCIPLE: "If it's in the Compass, it's a rule. If it's a rule, ETK must enforce it."
"""

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Optional jsonschema for validation
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class CompassPaths:
    """Resolve paths to Project Compass directories and files."""

    def __init__(self, project_root: Path | None = None):
        if project_root:
            self.project_root = project_root
        else:
            # Auto-detect project root by finding project_compass/
            current = Path.cwd()
            while current != current.parent:
                if (current / "project_compass").exists():
                    self.project_root = current
                    break
                current = current.parent
            else:
                self.project_root = Path.cwd()

        self.compass_dir = self.project_root / "project_compass"
        self.config_dir = self.compass_dir / ".config"
        self.schemas_dir = self.config_dir / "schemas"
        self.environments_dir = self.compass_dir / "environments"
        self.active_context_dir = self.compass_dir / "active_context"

    @property
    def uv_lock_state(self) -> Path:
        return self.environments_dir / "uv_lock_state.yml"

    @property
    def current_session(self) -> Path:
        return self.active_context_dir / "current_session.yml"

    @property
    def env_lock_schema(self) -> Path:
        return self.schemas_dir / "env_lock.schema.json"

    @property
    def session_schema(self) -> Path:
        return self.schemas_dir / "session.schema.json"


def validate_against_schema(data: dict[str, Any], schema_path: Path) -> tuple[bool, str]:
    """
    Validate data against a JSON schema.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if not HAS_JSONSCHEMA:
        # Skip validation if jsonschema not installed
        return True, ""

    if not schema_path.exists():
        return False, f"Schema file not found: {schema_path}"

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=schema)
        return True, ""
    except jsonschema.ValidationError as e:
        return False, f"Schema validation failed: {e.message}"
    except json.JSONDecodeError as e:
        return False, f"Invalid schema JSON: {e}"


def atomic_yaml_write(data: dict[str, Any], target_path: Path, header_comment: str = "") -> None:
    """
    Atomically write YAML data using tempfile + os.replace pattern.

    Args:
        data: Dictionary to write as YAML
        target_path: Destination file path
        header_comment: Optional comment to add at top of file
    """
    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize to YAML
    yaml_content = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    if header_comment:
        yaml_content = header_comment + "\n" + yaml_content

    # Atomic write: write to temp file, then replace
    fd, temp_path = tempfile.mkstemp(
        suffix=".yml.tmp",
        dir=target_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        os.replace(temp_path, target_path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def atomic_json_write(data: dict[str, Any], target_path: Path) -> None:
    """
    Atomically write JSON data using tempfile + os.replace pattern.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    json_content = json.dumps(data, indent=2, ensure_ascii=False)

    fd, temp_path = tempfile.mkstemp(
        suffix=".json.tmp",
        dir=target_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json_content)
        os.replace(temp_path, target_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


class EnvironmentChecker:
    """
    Environment Guard: Validates current environment against uv_lock_state.yml.

    Checks:
    1. `which uv` matches expected_path
    2. `python --version` matches requirement
    3. `torch.cuda.is_available()` and `torch.__version__` match CUDA lock
    """

    def __init__(self, paths: CompassPaths | None = None):
        self.paths = paths or CompassPaths()
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check_all(self) -> tuple[bool, list[str], list[str]]:
        """
        Run all environment checks.

        Returns:
            Tuple of (all_passed, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        lock_state = self._load_lock_state()
        if lock_state is None:
            return False, self.errors, self.warnings

        self._check_uv_binary(lock_state.get("uv_binary", {}))
        self._check_python(lock_state.get("python", {}))
        self._check_cuda(lock_state.get("cuda", {}))

        return len(self.errors) == 0, self.errors, self.warnings

    def _load_lock_state(self) -> dict[str, Any] | None:
        """Load uv_lock_state.yml."""
        if not self.paths.uv_lock_state.exists():
            self.errors.append(f"Lock state file not found: {self.paths.uv_lock_state}")
            return None

        try:
            content = self.paths.uv_lock_state.read_text(encoding="utf-8")
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            self.errors.append(f"Failed to parse lock state: {e}")
            return None

    def _check_uv_binary(self, config: dict[str, Any]) -> None:
        """Verify UV binary path matches expected."""
        expected_path = config.get("expected_path")
        if not expected_path:
            self.warnings.append("No expected_path defined for uv_binary")
            return

        try:
            result = subprocess.run(
                ["which", "uv"],
                capture_output=True,
                text=True,
                timeout=5
            )
            actual_path = result.stdout.strip()

            if result.returncode != 0:
                self.errors.append(f"UV not found in PATH (expected: {expected_path})")
            elif actual_path != expected_path:
                self.errors.append(
                    f"UV path mismatch:\n"
                    f"  Expected: {expected_path}\n"
                    f"  Actual:   {actual_path}\n"
                    f"  Fix: export PATH=\"{Path(expected_path).parent}:$PATH\""
                )
        except subprocess.TimeoutExpired:
            self.errors.append("Timeout checking UV binary")
        except FileNotFoundError:
            self.errors.append("'which' command not available")

    def _check_python(self, config: dict[str, Any]) -> None:
        """Verify Python version matches requirement."""
        expected_version = config.get("version")
        if not expected_version:
            self.warnings.append("No version defined for python")
            return

        try:
            result = subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Output: "Python X.Y.Z"
            actual_version = result.stdout.strip().replace("Python ", "")

            if not actual_version.startswith(expected_version.rsplit(".", 1)[0]):
                # Compare major.minor at minimum
                self.errors.append(
                    f"Python version mismatch:\n"
                    f"  Expected: {expected_version}\n"
                    f"  Actual:   {actual_version}"
                )
        except subprocess.TimeoutExpired:
            self.errors.append("Timeout checking Python version")
        except FileNotFoundError:
            self.errors.append("Python not found in PATH")

    def _check_cuda(self, config: dict[str, Any]) -> None:
        """Verify CUDA/PyTorch configuration."""
        if not config.get("enabled", True):
            return  # CUDA check disabled

        expected_torch = config.get("torch_version")
        if not expected_torch:
            self.warnings.append("No torch_version defined for cuda")
            return

        try:
            import torch

            # Check CUDA availability
            if not torch.cuda.is_available():
                self.errors.append(
                    "CUDA not available but required by lock state.\n"
                    "  This may indicate:\n"
                    "  - CPU-only PyTorch installed\n"
                    "  - Missing NVIDIA drivers\n"
                    "  - Running in CPU-only container"
                )
                return

            # Check torch version
            actual_torch = torch.__version__
            if actual_torch != expected_torch:
                self.errors.append(
                    f"PyTorch version mismatch (CUDA lock):\n"
                    f"  Expected: {expected_torch}\n"
                    f"  Actual:   {actual_torch}\n"
                    f"  Fix: uv add torch=={expected_torch}"
                )

        except ImportError:
            self.errors.append("PyTorch not installed (required for CUDA check)")


class SessionManager:
    """
    Session Management: Handle current_session.yml updates with schema validation.
    """

    def __init__(self, paths: CompassPaths | None = None):
        self.paths = paths or CompassPaths()

    def init_session(
        self,
        objective: str,
        active_pipeline: str = "kie"
    ) -> tuple[bool, str]:
        """
        Initialize or update the current session.

        Args:
            objective: Primary goal for this session
            active_pipeline: One of text_detection, text_recognition, layout_analysis, kie

        Returns:
            Tuple of (success, message)
        """
        # Load current session if exists
        current_data = self._load_current_session()

        # Generate new session ID
        today = datetime.now().strftime("%Y%m%d")
        session_num = self._get_next_session_number(today, current_data)
        new_session_id = f"{today}_session_{session_num:02d}"

        # Get current environment info for env_lock
        env_lock = self._build_env_lock()

        # Build new session data
        new_session = {
            "session_id": new_session_id,
            "objective": {
                "primary_goal": objective,
                "active_pipeline": active_pipeline,
                "success_criteria": "",
            },
            "env_lock": env_lock,
            "active_blockers": current_data.get("active_blockers", []),
            "working_references": current_data.get("working_references", {}),
        }

        # Validate against schema
        is_valid, error_msg = validate_against_schema(
            new_session,
            self.paths.session_schema
        )
        if not is_valid:
            return False, f"Session data failed schema validation: {error_msg}"

        # Atomic write
        header = (
            "# ACTIVE SESSION CONTEXT\n"
            "# MANDATORY: Verify env_lock before execution.\n"
            "# SCHEMA: ../.config/schemas/session.schema.json"
        )
        try:
            atomic_yaml_write(new_session, self.paths.current_session, header)
            return True, f"Session initialized: {new_session_id}"
        except Exception as e:
            return False, f"Failed to write session file: {e}"

    def _load_current_session(self) -> dict[str, Any]:
        """Load current session data if exists."""
        if not self.paths.current_session.exists():
            return {}

        try:
            content = self.paths.current_session.read_text(encoding="utf-8")
            return yaml.safe_load(content) or {}
        except yaml.YAMLError:
            return {}

    def _get_next_session_number(self, today: str, current_data: dict[str, Any]) -> int:
        """Determine next session number for today."""
        current_id = current_data.get("session_id", "")
        if current_id.startswith(today):
            # Same day, increment
            try:
                current_num = int(current_id.split("_")[-1])
                return current_num + 1
            except (ValueError, IndexError):
                return 1
        return 1

    def _build_env_lock(self) -> dict[str, Any]:
        """Build env_lock from current environment state."""
        # Get UV path
        try:
            result = subprocess.run(["which", "uv"], capture_output=True, text=True, timeout=5)
            uv_path = result.stdout.strip() if result.returncode == 0 else "/opt/uv/bin/uv"
        except Exception:
            uv_path = "/opt/uv/bin/uv"

        # Get Python version
        try:
            result = subprocess.run(["python", "--version"], capture_output=True, text=True, timeout=5)
            python_version = result.stdout.strip().replace("Python ", "")
        except Exception:
            python_version = "3.11"

        # Check GPU requirement
        try:
            import torch
            is_gpu_required = torch.cuda.is_available()
        except ImportError:
            is_gpu_required = False

        # Timestamp in KST format
        verified_at = datetime.now().strftime("%Y-%m-%d %H:%M") + " (KST)"

        return {
            "uv_path": uv_path,
            "python_version": python_version,
            "is_gpu_required": is_gpu_required,
            "verified_at": verified_at,
        }


def format_uv_command(command: str) -> str:
    """
    Prefix a command with 'uv run' for UV-managed environments.

    Example:
        format_uv_command("python train.py") -> "uv run python train.py"
    """
    if command.startswith("uv "):
        return command
    return f"uv run {command}"
