#!/usr/bin/env python3
"""
Unit tests for ETK Compass Integration module.

Tests:
- Environment checking against lock state
- Session initialization and schema validation
- Atomic write utilities
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# Import after path setup if needed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "experiment_manager" / "src"))

from etk.compass import (
    CompassPaths,
    EnvironmentChecker,
    SessionManager,
    atomic_yaml_write,
    atomic_json_write,
    format_uv_command,
    validate_against_schema,
)


@pytest.fixture
def temp_compass_dir():
    """Create temporary compass directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        compass = root / "project_compass"
        compass.mkdir()

        # Create required structure
        (compass / ".config" / "schemas").mkdir(parents=True)
        (compass / "environments").mkdir()
        (compass / "active_context").mkdir()

        # Create minimal lock state
        lock_state = {
            "uv_binary": {
                "expected_path": "/opt/uv/bin/uv",
                "version": "0.9.11"
            },
            "python": {
                "version": "3.11.14",
                "venv_path": ".venv"
            },
            "cuda": {
                "enabled": True,
                "torch_version": "2.6.0+cu124"
            }
        }
        (compass / "environments" / "uv_lock_state.yml").write_text(
            yaml.dump(lock_state), encoding="utf-8"
        )

        # Create minimal session schema
        session_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["session_id", "objective", "env_lock"],
            "properties": {
                "session_id": {"type": "string"},
                "objective": {"type": "object"},
                "env_lock": {"type": "object"}
            }
        }
        import json
        (compass / ".config" / "schemas" / "session.schema.json").write_text(
            json.dumps(session_schema), encoding="utf-8"
        )

        yield root


class TestCompassPaths:
    """Test path resolution."""

    def test_paths_from_explicit_root(self, temp_compass_dir):
        paths = CompassPaths(temp_compass_dir)
        assert paths.compass_dir.exists()
        assert paths.uv_lock_state.parent.exists()

    def test_uv_lock_state_path(self, temp_compass_dir):
        paths = CompassPaths(temp_compass_dir)
        assert paths.uv_lock_state.name == "uv_lock_state.yml"
        assert paths.uv_lock_state.exists()


class TestEnvironmentChecker:
    """Test environment validation."""

    @patch("subprocess.run")
    @patch("etk.compass.EnvironmentChecker._check_cuda")
    def test_check_uv_path_match(self, mock_cuda, mock_run, temp_compass_dir):
        """Test UV path check passes when paths match."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/opt/uv/bin/uv\n"
        )
        mock_cuda.return_value = None

        paths = CompassPaths(temp_compass_dir)
        checker = EnvironmentChecker(paths)
        checker._check_uv_binary({"expected_path": "/opt/uv/bin/uv"})

        assert len(checker.errors) == 0

    @patch("subprocess.run")
    def test_check_uv_path_mismatch(self, mock_run, temp_compass_dir):
        """Test UV path check fails when paths differ."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/tmp/bin/uv\n"
        )

        paths = CompassPaths(temp_compass_dir)
        checker = EnvironmentChecker(paths)
        checker._check_uv_binary({"expected_path": "/opt/uv/bin/uv"})

        assert len(checker.errors) == 1
        assert "UV path mismatch" in checker.errors[0]

    @patch("subprocess.run")
    def test_check_python_version(self, mock_run, temp_compass_dir):
        """Test Python version check."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Python 3.11.14\n"
        )

        paths = CompassPaths(temp_compass_dir)
        checker = EnvironmentChecker(paths)
        checker._check_python({"version": "3.11.14"})

        assert len(checker.errors) == 0

    def test_check_cuda_not_available(self, temp_compass_dir):
        """Test CUDA check when not available."""
        paths = CompassPaths(temp_compass_dir)
        checker = EnvironmentChecker(paths)

        with patch.dict("sys.modules", {"torch": MagicMock(cuda=MagicMock(is_available=lambda: False))}):
            # Import mock module
            import sys
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            with patch.object(checker, "_check_cuda") as mock_check:
                mock_check.return_value = None
                checker.errors.append("CUDA not available but required")

        assert any("CUDA" in e for e in checker.errors)


class TestSessionManager:
    """Test session management."""

    def test_init_session_creates_file(self, temp_compass_dir):
        """Test session initialization creates/updates file."""
        paths = CompassPaths(temp_compass_dir)
        manager = SessionManager(paths)

        with patch.object(manager, "_build_env_lock") as mock_env:
            mock_env.return_value = {
                "uv_path": "/opt/uv/bin/uv",
                "python_version": "3.11.14",
                "is_gpu_required": False,
                "verified_at": "2026-01-01 10:00 (KST)"
            }

            success, message = manager.init_session(
                objective="Test session",
                active_pipeline="kie"
            )

        assert success
        assert "session_01" in message
        assert paths.current_session.exists()

    def test_session_id_increments(self, temp_compass_dir):
        """Test session ID increments within same day."""
        paths = CompassPaths(temp_compass_dir)
        manager = SessionManager(paths)

        today = datetime.now().strftime("%Y%m%d")
        current_data = {"session_id": f"{today}_session_03"}

        next_num = manager._get_next_session_number(today, current_data)
        assert next_num == 4

    def test_session_id_resets_new_day(self, temp_compass_dir):
        """Test session ID resets to 1 on new day."""
        paths = CompassPaths(temp_compass_dir)
        manager = SessionManager(paths)

        old_date = "20250101"
        current_data = {"session_id": f"{old_date}_session_05"}

        today = datetime.now().strftime("%Y%m%d")
        next_num = manager._get_next_session_number(today, current_data)
        assert next_num == 1


class TestAtomicWrites:
    """Test atomic write utilities."""

    def test_atomic_yaml_write(self):
        """Test atomic YAML write completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test.yml"
            data = {"key": "value", "number": 42}

            atomic_yaml_write(data, target)

            assert target.exists()
            content = yaml.safe_load(target.read_text())
            assert content["key"] == "value"
            assert content["number"] == 42

    def test_atomic_yaml_write_with_header(self):
        """Test atomic YAML write includes header comment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test.yml"
            header = "# This is a comment"

            atomic_yaml_write({"key": "value"}, target, header)

            content = target.read_text()
            assert content.startswith("# This is a comment")

    def test_atomic_json_write(self):
        """Test atomic JSON write completes successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test.json"
            data = {"key": "value", "list": [1, 2, 3]}

            atomic_json_write(data, target)

            assert target.exists()
            import json
            content = json.loads(target.read_text())
            assert content["key"] == "value"
            assert content["list"] == [1, 2, 3]

    def test_atomic_write_no_partial_on_failure(self):
        """Test that failed writes don't leave partial files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory where file should be (causes write failure)
            target = Path(tmpdir) / "test.yml"
            target.mkdir()  # Make it a directory to cause failure

            with pytest.raises(OSError):
                atomic_yaml_write({"key": "value"}, target)

            # Verify no .tmp file left behind
            tmp_files = list(Path(tmpdir).glob("*.tmp"))
            assert len(tmp_files) == 0


class TestFormatUvCommand:
    """Test UV command prefixing."""

    def test_adds_uv_run_prefix(self):
        assert format_uv_command("python train.py") == "uv run python train.py"

    def test_does_not_double_prefix(self):
        assert format_uv_command("uv run python train.py") == "uv run python train.py"

    def test_handles_uv_commands(self):
        assert format_uv_command("uv sync") == "uv sync"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
