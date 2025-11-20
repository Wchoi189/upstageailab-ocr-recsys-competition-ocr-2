"""
Session Tracking Module for Agent Framework

This module provides session lifecycle management, snapshot creation, and context restoration
capabilities for the agent framework.
"""

import datetime
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from .state import StateManager, StateError


class SessionError(Exception):
    """Raised when session operations fail."""
    pass


class SessionManager:
    """
    Manages agent session lifecycle, snapshots, and context restoration.

    The SessionManager handles:
    - Starting and ending sessions
    - Creating session snapshots
    - Restoring session context
    - Tracking artifacts created/modified in sessions
    - Session search and retrieval
    """

    def __init__(self, state_manager: StateManager):
        """
        Initialize the SessionManager.

        Args:
            state_manager: StateManager instance for state persistence
        """
        self.state_manager = state_manager
        self.sessions_dir = state_manager.sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format with KST timezone."""
        return datetime.datetime.now(ZoneInfo("Asia/Seoul")).isoformat()

    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID.

        Returns:
            Unique session ID in format: session-YYYYMMDD-HHMMSS-UUID
        """
        timestamp = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"session-{timestamp}-{unique_id}"

    def start_session(
        self,
        branch: Optional[str] = None,
        phase: Optional[str] = None,
        goals: Optional[List[str]] = None
    ) -> str:
        """
        Start a new session.

        Args:
            branch: Git branch for this session
            phase: Project phase for this session
            goals: List of goals for this session

        Returns:
            Session ID

        Raises:
            SessionError: If a session is already active
        """
        # Check if there's already an active session
        active_session = self.state_manager.state['sessions']['active_session']
        if active_session:
            raise SessionError(
                f"Session '{active_session}' is already active. "
                "End the current session before starting a new one."
            )

        # Generate new session ID
        session_id = self._generate_session_id()

        # Create session snapshot
        session_data = {
            "session_id": session_id,
            "started_at": self._get_timestamp(),
            "ended_at": None,
            "branch": branch,
            "phase": phase,
            "artifacts_created": [],
            "artifacts_modified": [],
            "context": {
                "goals": goals or [],
                "outcomes": [],
                "challenges": [],
                "decisions": []
            },
            "summary": "",
            "statistics": {
                "duration_minutes": None,
                "artifacts_created_count": 0,
                "artifacts_modified_count": 0,
                "files_changed": 0
            },
            "related_sessions": []
        }

        # Save session snapshot
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        # Update state
        self.state_manager.set_active_session(session_id)
        if branch:
            self.state_manager.set_current_branch(branch)
        if phase:
            self.state_manager.set_current_phase(phase)

        # Update session history
        self.state_manager.state['sessions']['session_history'].insert(0, session_id)
        self.state_manager.state['sessions']['total_count'] += 1
        self.state_manager.state['statistics']['total_sessions'] += 1
        self.state_manager.state['statistics']['last_session_timestamp'] = self._get_timestamp()

        # Initialize relationship tracking
        self.state_manager.state['relationships']['session_artifacts'][session_id] = []

        self.state_manager._save_state()

        return session_id

    def end_session(
        self,
        summary: Optional[str] = None,
        outcomes: Optional[List[str]] = None,
        challenges: Optional[List[str]] = None
    ) -> None:
        """
        End the current active session.

        Args:
            summary: Brief summary of the session
            outcomes: List of session outcomes
            challenges: List of challenges encountered

        Raises:
            SessionError: If no session is active
        """
        session_id = self.state_manager.state['sessions']['active_session']
        if not session_id:
            raise SessionError("No active session to end")

        # Load session snapshot
        session_data = self._load_session(session_id)
        if not session_data:
            raise SessionError(f"Session snapshot not found: {session_id}")

        # Update session data
        session_data['ended_at'] = self._get_timestamp()
        if summary:
            session_data['summary'] = summary
        if outcomes:
            session_data['context']['outcomes'].extend(outcomes)
        if challenges:
            session_data['context']['challenges'].extend(challenges)

        # Calculate duration
        started_at = datetime.datetime.fromisoformat(session_data['started_at'])
        ended_at = datetime.datetime.fromisoformat(session_data['ended_at'])
        duration_minutes = (ended_at - started_at).total_seconds() / 60
        session_data['statistics']['duration_minutes'] = round(duration_minutes, 2)

        # Update statistics from state
        session_artifacts = self.state_manager.state['relationships']['session_artifacts'].get(session_id, [])
        session_data['statistics']['artifacts_created_count'] = len(session_artifacts)
        session_data['artifacts_created'] = session_artifacts

        # Save updated session snapshot
        self._save_session(session_id, session_data)

        # Clear active session
        self.state_manager.clear_active_session()

    def add_goal(self, goal: str) -> None:
        """
        Add a goal to the current active session.

        Args:
            goal: Goal description

        Raises:
            SessionError: If no session is active
        """
        session_id = self.state_manager.state['sessions']['active_session']
        if not session_id:
            raise SessionError("No active session")

        session_data = self._load_session(session_id)
        if not session_data:
            raise SessionError(f"Session snapshot not found: {session_id}")

        session_data['context']['goals'].append(goal)
        self._save_session(session_id, session_data)

    def add_outcome(self, outcome: str) -> None:
        """
        Add an outcome to the current active session.

        Args:
            outcome: Outcome description

        Raises:
            SessionError: If no session is active
        """
        session_id = self.state_manager.state['sessions']['active_session']
        if not session_id:
            raise SessionError("No active session")

        session_data = self._load_session(session_id)
        if not session_data:
            raise SessionError(f"Session snapshot not found: {session_id}")

        session_data['context']['outcomes'].append(outcome)
        self._save_session(session_id, session_data)

    def add_challenge(self, challenge: str) -> None:
        """
        Add a challenge to the current active session.

        Args:
            challenge: Challenge description

        Raises:
            SessionError: If no session is active
        """
        session_id = self.state_manager.state['sessions']['active_session']
        if not session_id:
            raise SessionError("No active session")

        session_data = self._load_session(session_id)
        if not session_data:
            raise SessionError(f"Session snapshot not found: {session_id}")

        session_data['context']['challenges'].append(challenge)
        self._save_session(session_id, session_data)

    def add_decision(self, decision: str, rationale: str) -> None:
        """
        Add a decision to the current active session.

        Args:
            decision: Decision made
            rationale: Rationale for the decision

        Raises:
            SessionError: If no session is active
        """
        session_id = self.state_manager.state['sessions']['active_session']
        if not session_id:
            raise SessionError("No active session")

        session_data = self._load_session(session_id)
        if not session_data:
            raise SessionError(f"Session snapshot not found: {session_id}")

        session_data['context']['decisions'].append({
            "decision": decision,
            "rationale": rationale
        })
        self._save_session(session_id, session_data)

    def track_artifact_creation(self, artifact_path: str) -> None:
        """
        Track an artifact created in the current session.

        Args:
            artifact_path: Path to the created artifact

        Raises:
            SessionError: If no session is active
        """
        session_id = self.state_manager.state['sessions']['active_session']
        if not session_id:
            return  # Silently ignore if no active session

        # Add to session artifacts in state
        if session_id not in self.state_manager.state['relationships']['session_artifacts']:
            self.state_manager.state['relationships']['session_artifacts'][session_id] = []

        if artifact_path not in self.state_manager.state['relationships']['session_artifacts'][session_id]:
            self.state_manager.state['relationships']['session_artifacts'][session_id].append(artifact_path)
            self.state_manager._save_state()

    def track_artifact_modification(self, artifact_path: str) -> None:
        """
        Track an artifact modified in the current session.

        Args:
            artifact_path: Path to the modified artifact

        Raises:
            SessionError: If no session is active
        """
        session_id = self.state_manager.state['sessions']['active_session']
        if not session_id:
            return  # Silently ignore if no active session

        session_data = self._load_session(session_id)
        if not session_data:
            return

        if artifact_path not in session_data['artifacts_modified']:
            session_data['artifacts_modified'].append(artifact_path)
            session_data['statistics']['artifacts_modified_count'] += 1
            self._save_session(session_id, session_data)

    def get_active_session(self) -> Optional[Dict[str, Any]]:
        """
        Get the current active session data.

        Returns:
            Session data dictionary or None if no active session
        """
        session_id = self.state_manager.state['sessions']['active_session']
        if not session_id:
            return None

        return self._load_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            Session data dictionary or None if not found
        """
        return self._load_session(session_id)

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session data dictionaries
        """
        session_history = self.state_manager.state['sessions']['session_history']
        sessions = []

        for session_id in session_history[:limit]:
            session_data = self._load_session(session_id)
            if session_data:
                sessions.append(session_data)

        return sessions

    def search_sessions(
        self,
        branch: Optional[str] = None,
        phase: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search sessions by criteria.

        Args:
            branch: Filter by branch
            phase: Filter by phase
            date_from: Filter by start date (ISO format)
            date_to: Filter by end date (ISO format)

        Returns:
            List of matching session data dictionaries
        """
        matching_sessions = []
        session_history = self.state_manager.state['sessions']['session_history']

        for session_id in session_history:
            session_data = self._load_session(session_id)
            if not session_data:
                continue

            # Apply filters
            if branch and session_data.get('branch') != branch:
                continue
            if phase and session_data.get('phase') != phase:
                continue
            if date_from and session_data['started_at'] < date_from:
                continue
            if date_to and session_data['started_at'] > date_to:
                continue

            matching_sessions.append(session_data)

        return matching_sessions

    def restore_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Restore context from a previous session.

        Args:
            session_id: Session ID to restore context from

        Returns:
            Dictionary containing restored context

        Raises:
            SessionError: If session not found
        """
        session_data = self._load_session(session_id)
        if not session_data:
            raise SessionError(f"Session not found: {session_id}")

        # Build context dictionary
        context = {
            "session_id": session_id,
            "branch": session_data.get('branch'),
            "phase": session_data.get('phase'),
            "goals": session_data['context']['goals'],
            "outcomes": session_data['context']['outcomes'],
            "challenges": session_data['context']['challenges'],
            "decisions": session_data['context']['decisions'],
            "artifacts_created": session_data['artifacts_created'],
            "artifacts_modified": session_data['artifacts_modified'],
            "summary": session_data.get('summary', ''),
            "duration_minutes": session_data['statistics'].get('duration_minutes')
        }

        return context

    def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from snapshot file.

        Args:
            session_id: Session ID

        Returns:
            Session data dictionary or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load session {session_id}: {e}")
            return None

    def _save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Save session data to snapshot file.

        Args:
            session_id: Session ID
            session_data: Session data to save
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        # Atomic write using temporary file
        temp_file = session_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        temp_file.replace(session_file)

    def cleanup_old_sessions(self, max_age_days: int) -> int:
        """
        Clean up old session snapshots beyond the max age.

        Args:
            max_age_days: Maximum age of sessions in days

        Returns:
            Number of sessions deleted
        """
        if max_age_days <= 0:
            return 0

        cutoff_date = datetime.datetime.now(ZoneInfo("Asia/Seoul")) - datetime.timedelta(days=max_age_days)
        cutoff_iso = cutoff_date.isoformat()

        deleted_count = 0
        session_history = self.state_manager.state['sessions']['session_history'].copy()

        for session_id in session_history:
            session_data = self._load_session(session_id)
            if not session_data:
                continue

            # Check if session is older than cutoff
            if session_data['started_at'] < cutoff_iso:
                # Delete session snapshot
                session_file = self.sessions_dir / f"{session_id}.json"
                session_file.unlink(missing_ok=True)

                # Remove from history
                self.state_manager.state['sessions']['session_history'].remove(session_id)
                deleted_count += 1

        if deleted_count > 0:
            self.state_manager._save_state()

        return deleted_count
