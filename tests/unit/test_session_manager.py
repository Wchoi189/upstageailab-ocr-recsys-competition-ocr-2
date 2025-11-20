"""
Unit tests for the SessionManager class.

Tests cover:
- Session lifecycle (start/end)
- Session context management (goals, outcomes, challenges, decisions)
- Artifact tracking
- Session retrieval and search
- Context restoration
- Session cleanup
"""

import datetime
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from agent_qms.toolbelt.session import SessionError, SessionManager
from agent_qms.toolbelt.state import StateManager


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
                "auto_backup": False,
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


@pytest.fixture
def session_manager(state_manager):
    """Create a SessionManager instance."""
    return SessionManager(state_manager)


class TestSessionManagerInitialization:
    """Tests for SessionManager initialization."""

    def test_init_creates_sessions_dir(self, session_manager, temp_state_dir):
        """Test that initialization ensures sessions directory exists."""
        sessions_dir = temp_state_dir / ".agentqms" / "sessions"
        assert sessions_dir.exists()
        assert sessions_dir.is_dir()

    def test_init_with_state_manager(self, state_manager):
        """Test initialization with StateManager."""
        session_mgr = SessionManager(state_manager)
        assert session_mgr.state_manager is state_manager
        assert session_mgr.sessions_dir == state_manager.sessions_dir


class TestSessionLifecycle:
    """Tests for session lifecycle management."""

    def test_start_session_basic(self, session_manager):
        """Test starting a basic session."""
        session_id = session_manager.start_session()

        assert session_id is not None
        assert session_id.startswith("session-")
        assert session_manager.state_manager.state['sessions']['active_session'] == session_id
        assert session_id in session_manager.state_manager.state['sessions']['session_history']
        assert session_manager.state_manager.state['sessions']['total_count'] == 1

    def test_start_session_with_params(self, session_manager):
        """Test starting a session with parameters."""
        goals = ["Goal 1", "Goal 2"]
        session_id = session_manager.start_session(
            branch="feature/test",
            phase="phase-1",
            goals=goals
        )

        session_data = session_manager.get_session(session_id)
        assert session_data is not None
        assert session_data['branch'] == "feature/test"
        assert session_data['phase'] == "phase-1"
        assert session_data['context']['goals'] == goals

    def test_start_session_creates_snapshot(self, session_manager, temp_state_dir):
        """Test that starting a session creates a snapshot file."""
        session_id = session_manager.start_session()

        session_file = temp_state_dir / ".agentqms" / "sessions" / f"{session_id}.json"
        assert session_file.exists()

        with open(session_file, 'r') as f:
            session_data = json.load(f)
            assert session_data['session_id'] == session_id
            assert session_data['started_at'] is not None
            assert session_data['ended_at'] is None

    def test_start_session_fails_if_active(self, session_manager):
        """Test that starting a session fails if one is already active."""
        session_manager.start_session()

        with pytest.raises(SessionError, match="already active"):
            session_manager.start_session()

    def test_end_session_basic(self, session_manager):
        """Test ending a session."""
        session_id = session_manager.start_session()
        session_manager.end_session(summary="Test session completed")

        assert session_manager.state_manager.state['sessions']['active_session'] is None

        session_data = session_manager.get_session(session_id)
        assert session_data['ended_at'] is not None
        assert session_data['summary'] == "Test session completed"
        assert session_data['statistics']['duration_minutes'] is not None

    def test_end_session_with_outcomes(self, session_manager):
        """Test ending a session with outcomes and challenges."""
        session_id = session_manager.start_session()
        outcomes = ["Outcome 1", "Outcome 2"]
        challenges = ["Challenge 1"]

        session_manager.end_session(
            summary="Test completed",
            outcomes=outcomes,
            challenges=challenges
        )

        session_data = session_manager.get_session(session_id)
        assert session_data['context']['outcomes'] == outcomes
        assert session_data['context']['challenges'] == challenges

    def test_end_session_fails_if_none_active(self, session_manager):
        """Test that ending a session fails if none is active."""
        with pytest.raises(SessionError, match="No active session"):
            session_manager.end_session()


class TestSessionContextManagement:
    """Tests for session context management."""

    def test_add_goal(self, session_manager):
        """Test adding a goal to active session."""
        session_id = session_manager.start_session()
        session_manager.add_goal("New goal")

        session_data = session_manager.get_session(session_id)
        assert "New goal" in session_data['context']['goals']

    def test_add_outcome(self, session_manager):
        """Test adding an outcome to active session."""
        session_id = session_manager.start_session()
        session_manager.add_outcome("New outcome")

        session_data = session_manager.get_session(session_id)
        assert "New outcome" in session_data['context']['outcomes']

    def test_add_challenge(self, session_manager):
        """Test adding a challenge to active session."""
        session_id = session_manager.start_session()
        session_manager.add_challenge("New challenge")

        session_data = session_manager.get_session(session_id)
        assert "New challenge" in session_data['context']['challenges']

    def test_add_decision(self, session_manager):
        """Test adding a decision to active session."""
        session_id = session_manager.start_session()
        session_manager.add_decision("Use JSON storage", "Simpler and more portable")

        session_data = session_manager.get_session(session_id)
        decisions = session_data['context']['decisions']
        assert len(decisions) == 1
        assert decisions[0]['decision'] == "Use JSON storage"
        assert decisions[0]['rationale'] == "Simpler and more portable"

    def test_context_methods_fail_without_active_session(self, session_manager):
        """Test that context methods fail if no active session."""
        with pytest.raises(SessionError, match="No active session"):
            session_manager.add_goal("Goal")

        with pytest.raises(SessionError, match="No active session"):
            session_manager.add_outcome("Outcome")

        with pytest.raises(SessionError, match="No active session"):
            session_manager.add_challenge("Challenge")

        with pytest.raises(SessionError, match="No active session"):
            session_manager.add_decision("Decision", "Rationale")


class TestArtifactTracking:
    """Tests for artifact tracking in sessions."""

    def test_track_artifact_creation(self, session_manager):
        """Test tracking artifact creation."""
        session_id = session_manager.start_session()
        session_manager.track_artifact_creation("artifacts/plan1.md")

        artifacts = session_manager.state_manager.state['relationships']['session_artifacts'][session_id]
        assert "artifacts/plan1.md" in artifacts

    def test_track_multiple_artifacts(self, session_manager):
        """Test tracking multiple artifacts."""
        session_id = session_manager.start_session()
        session_manager.track_artifact_creation("artifacts/plan1.md")
        session_manager.track_artifact_creation("artifacts/plan2.md")

        artifacts = session_manager.state_manager.state['relationships']['session_artifacts'][session_id]
        assert len(artifacts) == 2
        assert "artifacts/plan1.md" in artifacts
        assert "artifacts/plan2.md" in artifacts

    def test_track_artifact_no_duplicates(self, session_manager):
        """Test that tracking same artifact twice doesn't create duplicates."""
        session_id = session_manager.start_session()
        session_manager.track_artifact_creation("artifacts/plan1.md")
        session_manager.track_artifact_creation("artifacts/plan1.md")

        artifacts = session_manager.state_manager.state['relationships']['session_artifacts'][session_id]
        assert artifacts.count("artifacts/plan1.md") == 1

    def test_track_artifact_modification(self, session_manager):
        """Test tracking artifact modification."""
        session_id = session_manager.start_session()
        session_manager.track_artifact_modification("artifacts/existing.md")

        session_data = session_manager.get_session(session_id)
        assert "artifacts/existing.md" in session_data['artifacts_modified']
        assert session_data['statistics']['artifacts_modified_count'] == 1

    def test_track_artifact_without_active_session(self, session_manager):
        """Test that tracking without active session doesn't fail."""
        # Should not raise error, just silently ignore
        session_manager.track_artifact_creation("artifacts/plan1.md")
        session_manager.track_artifact_modification("artifacts/existing.md")


class TestSessionRetrieval:
    """Tests for session retrieval and search."""

    def test_get_active_session(self, session_manager):
        """Test getting active session."""
        session_id = session_manager.start_session()
        active_session = session_manager.get_active_session()

        assert active_session is not None
        assert active_session['session_id'] == session_id

    def test_get_active_session_when_none(self, session_manager):
        """Test getting active session when none exists."""
        active_session = session_manager.get_active_session()
        assert active_session is None

    def test_get_session_by_id(self, session_manager):
        """Test getting a specific session by ID."""
        session_id = session_manager.start_session()
        session_manager.end_session()

        session_data = session_manager.get_session(session_id)
        assert session_data is not None
        assert session_data['session_id'] == session_id

    def test_get_nonexistent_session(self, session_manager):
        """Test getting a non-existent session."""
        session_data = session_manager.get_session("nonexistent")
        assert session_data is None

    def test_list_sessions(self, session_manager):
        """Test listing recent sessions."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            sid = session_manager.start_session()
            session_ids.append(sid)
            session_manager.end_session()

        sessions = session_manager.list_sessions(limit=10)
        assert len(sessions) == 3
        # Should be in reverse chronological order
        assert sessions[0]['session_id'] == session_ids[2]

    def test_list_sessions_with_limit(self, session_manager):
        """Test listing sessions with limit."""
        # Create 5 sessions
        for i in range(5):
            session_manager.start_session()
            session_manager.end_session()

        sessions = session_manager.list_sessions(limit=2)
        assert len(sessions) == 2

    def test_search_sessions_by_branch(self, session_manager):
        """Test searching sessions by branch."""
        session_manager.start_session(branch="feature/a")
        session_manager.end_session()

        session_manager.start_session(branch="feature/b")
        session_manager.end_session()

        results = session_manager.search_sessions(branch="feature/a")
        assert len(results) == 1
        assert results[0]['branch'] == "feature/a"

    def test_search_sessions_by_phase(self, session_manager):
        """Test searching sessions by phase."""
        session_manager.start_session(phase="phase-1")
        session_manager.end_session()

        session_manager.start_session(phase="phase-2")
        session_manager.end_session()

        results = session_manager.search_sessions(phase="phase-1")
        assert len(results) == 1
        assert results[0]['phase'] == "phase-1"

    def test_search_sessions_multiple_criteria(self, session_manager):
        """Test searching sessions with multiple criteria."""
        session_manager.start_session(branch="feature/a", phase="phase-1")
        session_manager.end_session()

        session_manager.start_session(branch="feature/a", phase="phase-2")
        session_manager.end_session()

        results = session_manager.search_sessions(branch="feature/a", phase="phase-1")
        assert len(results) == 1
        assert results[0]['branch'] == "feature/a"
        assert results[0]['phase'] == "phase-1"


class TestContextRestoration:
    """Tests for session context restoration."""

    def test_restore_session_context(self, session_manager):
        """Test restoring context from a session."""
        session_id = session_manager.start_session(branch="main", phase="phase-1")
        session_manager.add_goal("Goal 1")
        session_manager.add_outcome("Outcome 1")
        session_manager.add_decision("Decision 1", "Rationale 1")
        session_manager.track_artifact_creation("artifacts/plan1.md")
        session_manager.end_session()

        context = session_manager.restore_session_context(session_id)

        assert context['session_id'] == session_id
        assert context['branch'] == "main"
        assert context['phase'] == "phase-1"
        assert "Goal 1" in context['goals']
        assert "Outcome 1" in context['outcomes']
        assert len(context['decisions']) == 1
        assert "artifacts/plan1.md" in context['artifacts_created']

    def test_restore_nonexistent_session(self, session_manager):
        """Test restoring context from non-existent session."""
        with pytest.raises(SessionError, match="Session not found"):
            session_manager.restore_session_context("nonexistent")


class TestSessionCleanup:
    """Tests for session cleanup."""

    def test_cleanup_old_sessions(self, session_manager):
        """Test cleanup of old sessions."""
        # Create an old session (mock old timestamp)
        session_id = session_manager.start_session()
        session_data = session_manager.get_session(session_id)

        # Make it 100 days old
        old_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=100)
        session_data['started_at'] = old_date.isoformat()
        session_manager._save_session(session_id, session_data)

        session_manager.end_session()

        # Cleanup sessions older than 90 days
        deleted = session_manager.cleanup_old_sessions(max_age_days=90)

        assert deleted == 1
        assert session_manager.get_session(session_id) is None

    def test_cleanup_keeps_recent_sessions(self, session_manager):
        """Test that cleanup keeps recent sessions."""
        session_id = session_manager.start_session()
        session_manager.end_session()

        # Cleanup sessions older than 90 days (this one is recent)
        deleted = session_manager.cleanup_old_sessions(max_age_days=90)

        assert deleted == 0
        assert session_manager.get_session(session_id) is not None

    def test_cleanup_with_zero_age(self, session_manager):
        """Test that cleanup with zero age doesn't delete anything."""
        session_manager.start_session()
        session_manager.end_session()

        deleted = session_manager.cleanup_old_sessions(max_age_days=0)
        assert deleted == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_session_id_format(self, session_manager):
        """Test that session IDs have correct format."""
        session_id = session_manager.start_session()

        # Format: session-YYYYMMDD-HHMMSS-UUID
        parts = session_id.split('-')
        assert parts[0] == "session"
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 8  # UUID (8 chars)

    def test_session_duration_calculation(self, session_manager):
        """Test that session duration is calculated correctly."""
        session_id = session_manager.start_session()

        # Simulate some work
        import time
        time.sleep(0.1)

        session_manager.end_session()

        session_data = session_manager.get_session(session_id)
        duration = session_data['statistics']['duration_minutes']

        assert duration is not None
        assert duration >= 0
        assert duration < 1  # Should be less than 1 minute

    def test_session_statistics_updated(self, session_manager):
        """Test that session statistics are updated correctly."""
        session_id = session_manager.start_session()
        session_manager.track_artifact_creation("artifacts/plan1.md")
        session_manager.track_artifact_creation("artifacts/plan2.md")
        session_manager.end_session()

        session_data = session_manager.get_session(session_id)
        assert session_data['statistics']['artifacts_created_count'] == 2
