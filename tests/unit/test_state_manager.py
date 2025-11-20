"""
Unit tests for the StateManager class.

Tests cover:
- State initialization and loading
- State saving and backup
- Context management
- Artifact tracking
- State validation and health checks
- Error handling
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from agent_qms.toolbelt.state import StateError, StateManager


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create .agentqms directory structure
        agentqms_dir = tmpdir_path / ".agentqms"
        agentqms_dir.mkdir()
        (agentqms_dir / "sessions").mkdir()

        # Create config.yaml
        config = {
            "framework": {
                "name": "test-framework",
                "version": "1.0.0",
                "state_schema_version": "1.0.0"
            },
            "paths": {
                "state_file": ".agentqms/state.json",
                "sessions_dir": ".agentqms/sessions",
                "artifacts_dir": "artifacts"
            },
            "settings": {
                "max_sessions": 100,
                "max_session_age_days": 90,
                "auto_backup": True,
                "backup_dir": ".agentqms/backups",
                "max_backups": 10
            },
            "tracking": {
                "enable_session_tracking": True,
                "enable_artifact_tracking": True,
                "enable_context_preservation": True,
                "auto_index_artifacts": True
            }
        }

        config_path = agentqms_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Create artifacts directory
        (tmpdir_path / "artifacts").mkdir()

        yield tmpdir_path


@pytest.fixture
def state_manager(temp_state_dir):
    """Create a StateManager instance with temporary directory."""
    config_path = temp_state_dir / ".agentqms" / "config.yaml"
    return StateManager(config_path=str(config_path))


class TestStateManagerInitialization:
    """Tests for StateManager initialization."""

    def test_init_creates_default_state(self, state_manager, temp_state_dir):
        """Test that initialization creates default state."""
        state_file = temp_state_dir / ".agentqms" / "state.json"
        assert state_file.exists()

        with open(state_file, 'r') as f:
            state = json.load(f)

        assert state['schema_version'] == "1.0.0"
        assert state['framework']['name'] == "test-framework"
        assert state['current_context']['active_session_id'] is None
        assert state['artifacts']['total_count'] == 0

    def test_init_missing_config_raises_error(self, temp_state_dir):
        """Test that missing config file raises StateError."""
        with pytest.raises(StateError, match="Configuration file not found"):
            StateManager(config_path="nonexistent/config.yaml")

    def test_init_loads_existing_state(self, temp_state_dir):
        """Test that existing state is loaded correctly."""
        state_file = temp_state_dir / ".agentqms" / "state.json"

        # Create an existing state
        existing_state = {
            "schema_version": "1.0.0",
            "last_updated": "2025-11-20T12:00:00+09:00",
            "framework": {
                "name": "test-framework",
                "version": "1.0.0"
            },
            "current_context": {
                "active_session_id": "test-session",
                "current_branch": "main",
                "current_phase": "phase1",
                "active_artifacts": [],
                "pending_tasks": []
            },
            "sessions": {
                "total_count": 5,
                "active_session": "test-session",
                "session_history": []
            },
            "artifacts": {
                "total_count": 10,
                "by_type": {"plan": 5, "assessment": 5},
                "by_status": {"draft": 8, "validated": 2},
                "index": []
            },
            "relationships": {
                "artifact_dependencies": {},
                "session_artifacts": {}
            },
            "statistics": {
                "total_sessions": 5,
                "total_artifacts_created": 10,
                "total_artifacts_validated": 2,
                "total_artifacts_deployed": 0,
                "last_session_timestamp": "2025-11-20T11:00:00+09:00"
            }
        }

        with open(state_file, 'w') as f:
            json.dump(existing_state, f)

        # Create state manager - should load existing state
        config_path = temp_state_dir / ".agentqms" / "config.yaml"
        manager = StateManager(config_path=str(config_path))

        assert manager.state['current_context']['active_session_id'] == "test-session"
        assert manager.state['artifacts']['total_count'] == 10
        assert manager.state['sessions']['total_count'] == 5


class TestStateManagerContextManagement:
    """Tests for context management methods."""

    def test_get_current_context(self, state_manager):
        """Test getting current context."""
        context = state_manager.get_current_context()
        assert 'active_session_id' in context
        assert 'current_branch' in context
        assert 'active_artifacts' in context

    def test_update_current_context(self, state_manager):
        """Test updating context fields."""
        state_manager.update_current_context(
            current_branch="feature-branch",
            current_phase="testing"
        )

        context = state_manager.get_current_context()
        assert context['current_branch'] == "feature-branch"
        assert context['current_phase'] == "testing"

    def test_set_active_session(self, state_manager):
        """Test setting active session."""
        state_manager.set_active_session("session-123")

        assert state_manager.state['current_context']['active_session_id'] == "session-123"
        assert state_manager.state['sessions']['active_session'] == "session-123"

    def test_clear_active_session(self, state_manager):
        """Test clearing active session."""
        state_manager.set_active_session("session-123")
        state_manager.clear_active_session()

        assert state_manager.state['current_context']['active_session_id'] is None
        assert state_manager.state['sessions']['active_session'] is None

    def test_set_current_branch(self, state_manager):
        """Test setting current branch."""
        state_manager.set_current_branch("main")
        assert state_manager.state['current_context']['current_branch'] == "main"

    def test_set_current_phase(self, state_manager):
        """Test setting current phase."""
        state_manager.set_current_phase("phase-2")
        assert state_manager.state['current_context']['current_phase'] == "phase-2"

    def test_add_active_artifact(self, state_manager):
        """Test adding active artifact."""
        state_manager.add_active_artifact("artifacts/plan1.md")
        assert "artifacts/plan1.md" in state_manager.state['current_context']['active_artifacts']

    def test_add_active_artifact_no_duplicates(self, state_manager):
        """Test that adding same artifact twice doesn't create duplicates."""
        state_manager.add_active_artifact("artifacts/plan1.md")
        state_manager.add_active_artifact("artifacts/plan1.md")
        assert state_manager.state['current_context']['active_artifacts'].count("artifacts/plan1.md") == 1

    def test_remove_active_artifact(self, state_manager):
        """Test removing active artifact."""
        state_manager.add_active_artifact("artifacts/plan1.md")
        state_manager.remove_active_artifact("artifacts/plan1.md")
        assert "artifacts/plan1.md" not in state_manager.state['current_context']['active_artifacts']

    def test_get_active_artifacts(self, state_manager):
        """Test getting active artifacts list."""
        state_manager.add_active_artifact("artifacts/plan1.md")
        state_manager.add_active_artifact("artifacts/plan2.md")

        artifacts = state_manager.get_active_artifacts()
        assert len(artifacts) == 2
        assert "artifacts/plan1.md" in artifacts
        assert "artifacts/plan2.md" in artifacts


class TestStateManagerArtifactTracking:
    """Tests for artifact tracking methods."""

    def test_add_artifact(self, state_manager):
        """Test adding a new artifact."""
        state_manager.add_artifact(
            artifact_path="artifacts/test_plan.md",
            artifact_type="implementation_plan",
            status="draft",
            metadata={"author": "test-agent"}
        )

        artifact = state_manager.get_artifact("artifacts/test_plan.md")
        assert artifact is not None
        assert artifact['type'] == "implementation_plan"
        assert artifact['status'] == "draft"
        assert artifact['metadata']['author'] == "test-agent"
        assert state_manager.state['artifacts']['total_count'] == 1

    def test_add_artifact_updates_counts(self, state_manager):
        """Test that adding artifacts updates type and status counts."""
        state_manager.add_artifact("artifacts/plan1.md", "plan", "draft")
        state_manager.add_artifact("artifacts/plan2.md", "plan", "validated")

        assert state_manager.state['artifacts']['by_type']['plan'] == 2
        assert state_manager.state['artifacts']['by_status']['draft'] == 1
        assert state_manager.state['artifacts']['by_status']['validated'] == 1

    def test_update_existing_artifact(self, state_manager):
        """Test updating an existing artifact."""
        state_manager.add_artifact("artifacts/plan1.md", "plan", "draft")
        state_manager.add_artifact("artifacts/plan1.md", "plan", "validated")

        artifact = state_manager.get_artifact("artifacts/plan1.md")
        assert artifact['status'] == "validated"
        # Should not increase total count
        assert state_manager.state['artifacts']['total_count'] == 1

    def test_get_artifact_not_found(self, state_manager):
        """Test getting non-existent artifact returns None."""
        artifact = state_manager.get_artifact("nonexistent.md")
        assert artifact is None

    def test_update_artifact_status(self, state_manager):
        """Test updating artifact status."""
        state_manager.add_artifact("artifacts/plan1.md", "plan", "draft")
        state_manager.update_artifact_status("artifacts/plan1.md", "validated")

        artifact = state_manager.get_artifact("artifacts/plan1.md")
        assert artifact['status'] == "validated"
        assert state_manager.state['statistics']['total_artifacts_validated'] == 1

    def test_update_artifact_status_not_found_raises_error(self, state_manager):
        """Test updating non-existent artifact raises error."""
        with pytest.raises(StateError, match="Artifact not found"):
            state_manager.update_artifact_status("nonexistent.md", "validated")

    def test_get_artifacts_by_type(self, state_manager):
        """Test getting artifacts by type."""
        state_manager.add_artifact("artifacts/plan1.md", "plan", "draft")
        state_manager.add_artifact("artifacts/plan2.md", "plan", "validated")
        state_manager.add_artifact("artifacts/assessment1.md", "assessment", "draft")

        plans = state_manager.get_artifacts_by_type("plan")
        assert len(plans) == 2
        assert all(a['type'] == "plan" for a in plans)

    def test_get_artifacts_by_status(self, state_manager):
        """Test getting artifacts by status."""
        state_manager.add_artifact("artifacts/plan1.md", "plan", "draft")
        state_manager.add_artifact("artifacts/plan2.md", "plan", "draft")
        state_manager.add_artifact("artifacts/assessment1.md", "assessment", "validated")

        drafts = state_manager.get_artifacts_by_status("draft")
        assert len(drafts) == 2
        assert all(a['status'] == "draft" for a in drafts)

    def test_get_all_artifacts(self, state_manager):
        """Test getting all artifacts."""
        state_manager.add_artifact("artifacts/plan1.md", "plan", "draft")
        state_manager.add_artifact("artifacts/plan2.md", "plan", "validated")

        all_artifacts = state_manager.get_all_artifacts()
        assert len(all_artifacts) == 2


class TestStateManagerValidationAndHealth:
    """Tests for state validation and health checks."""

    def test_validate_state(self, state_manager):
        """Test state validation."""
        assert state_manager.validate_state() is True

    def test_validate_state_invalid(self, state_manager):
        """Test validation fails for invalid state."""
        # Remove required key
        del state_manager.state['artifacts']
        assert state_manager.validate_state() is False

    def test_get_state_health(self, state_manager, temp_state_dir):
        """Test getting state health information."""
        health = state_manager.get_state_health()

        assert health['is_valid'] is True
        assert health['schema_version'] == "1.0.0"
        assert 'last_updated' in health
        assert health['total_artifacts'] == 0
        assert health['total_sessions'] == 0
        assert health['state_file_exists'] is True
        assert health['state_file_size_bytes'] > 0

    def test_get_statistics(self, state_manager):
        """Test getting statistics."""
        state_manager.add_artifact("artifacts/plan1.md", "plan", "draft")
        state_manager.update_artifact_status("artifacts/plan1.md", "validated")

        stats = state_manager.get_statistics()
        assert stats['total_artifacts_created'] == 1
        assert stats['total_artifacts_validated'] == 1


class TestStateManagerBackupAndRecovery:
    """Tests for backup and recovery functionality."""

    def test_corrupted_state_file_recovery(self, temp_state_dir):
        """Test recovery from corrupted state file."""
        state_file = temp_state_dir / ".agentqms" / "state.json"

        # Write corrupted JSON
        with open(state_file, 'w') as f:
            f.write("{invalid json content")

        # Should recover by creating default state
        config_path = temp_state_dir / ".agentqms" / "config.yaml"
        manager = StateManager(config_path=str(config_path))

        assert manager.state is not None
        assert manager.validate_state() is True

    def test_auto_backup_creation(self, state_manager, temp_state_dir):
        """Test that backups are created when saving state."""
        backup_dir = temp_state_dir / ".agentqms" / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Trigger state save multiple times
        for i in range(3):
            state_manager.add_artifact(f"artifacts/plan{i}.md", "plan", "draft")

        # Check that backups were created
        backups = list(backup_dir.glob("state_backup_*.json"))
        assert len(backups) > 0
