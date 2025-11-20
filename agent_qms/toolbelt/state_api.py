"""
State Access API for Agent Framework

This module provides a simple, convenient API for accessing state information
without needing to directly interact with StateManager or SessionManager.

Usage:
    from agent_qms.toolbelt.state_api import get_current_context, get_recent_artifacts

    # Get current context
    context = get_current_context()
    print(f"Active session: {context['active_session_id']}")

    # Get recent artifacts
    artifacts = get_recent_artifacts(limit=10)
    for artifact in artifacts:
        print(f"{artifact['path']}: {artifact['status']}")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .session import SessionManager
from .state import StateManager


# Global state manager instances (initialized lazily)
_state_manager: Optional[StateManager] = None
_session_manager: Optional[SessionManager] = None


def _get_state_manager() -> StateManager:
    """Get or initialize the global StateManager instance."""
    global _state_manager

    if _state_manager is None:
        config_path = Path(".agentqms/config.yaml")
        if not config_path.exists():
            raise RuntimeError(
                "State tracking not initialized. "
                "Run initialization or create .agentqms/config.yaml"
            )
        _state_manager = StateManager(config_path=str(config_path))

    return _state_manager


def _get_session_manager() -> SessionManager:
    """Get or initialize the global SessionManager instance."""
    global _session_manager

    if _session_manager is None:
        state_mgr = _get_state_manager()
        _session_manager = SessionManager(state_mgr)

    return _session_manager


# Context API

def get_current_context() -> Dict[str, Any]:
    """
    Get the current agent context.

    Returns:
        Dictionary containing active session, branch, phase, and active artifacts
    """
    return _get_state_manager().get_current_context()


def get_active_session_id() -> Optional[str]:
    """
    Get the ID of the currently active session.

    Returns:
        Session ID or None if no active session
    """
    return _get_state_manager().state['sessions']['active_session']


def get_current_branch() -> Optional[str]:
    """
    Get the current git branch.

    Returns:
        Branch name or None
    """
    return _get_state_manager().state['current_context']['current_branch']


def get_current_phase() -> Optional[str]:
    """
    Get the current project phase.

    Returns:
        Phase name or None
    """
    return _get_state_manager().state['current_context']['current_phase']


# Artifact API

def get_artifact(artifact_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific artifact.

    Args:
        artifact_path: Path to the artifact

    Returns:
        Artifact dictionary or None if not found
    """
    return _get_state_manager().get_artifact(artifact_path)


def get_recent_artifacts(limit: int = 10, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get recently created artifacts.

    Args:
        limit: Maximum number of artifacts to return
        artifact_type: Optional filter by artifact type

    Returns:
        List of artifact dictionaries, sorted by creation date (newest first)
    """
    if artifact_type:
        artifacts = _get_state_manager().get_artifacts_by_type(artifact_type)
    else:
        artifacts = _get_state_manager().get_all_artifacts()

    # Sort by created_at (newest first)
    artifacts_sorted = sorted(
        artifacts,
        key=lambda a: a.get('created_at', ''),
        reverse=True
    )

    return artifacts_sorted[:limit]


def get_artifacts_by_status(status: str) -> List[Dict[str, Any]]:
    """
    Get all artifacts with a specific status.

    Args:
        status: Status to filter by (e.g., 'draft', 'validated', 'deployed')

    Returns:
        List of artifact dictionaries
    """
    return _get_state_manager().get_artifacts_by_status(status)


def get_artifacts_by_type(artifact_type: str) -> List[Dict[str, Any]]:
    """
    Get all artifacts of a specific type.

    Args:
        artifact_type: Type of artifacts (e.g., 'implementation_plan', 'assessment')

    Returns:
        List of artifact dictionaries
    """
    return _get_state_manager().get_artifacts_by_type(artifact_type)


def get_artifact_dependencies(artifact_path: str) -> List[str]:
    """
    Get all dependencies for an artifact.

    Args:
        artifact_path: Path to the artifact

    Returns:
        List of dependency artifact paths
    """
    return _get_state_manager().get_artifact_dependencies(artifact_path)


def get_dependency_tree(artifact_path: str) -> Dict[str, Any]:
    """
    Get the full dependency tree for an artifact.

    Args:
        artifact_path: Path to the artifact

    Returns:
        Dictionary representing the dependency tree
    """
    return _get_state_manager().get_dependency_tree(artifact_path)


# Session API

def get_active_session() -> Optional[Dict[str, Any]]:
    """
    Get the currently active session.

    Returns:
        Session data dictionary or None if no active session
    """
    return _get_session_manager().get_active_session()


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific session by ID.

    Args:
        session_id: Session ID

    Returns:
        Session data dictionary or None if not found
    """
    return _get_session_manager().get_session(session_id)


def get_recent_sessions(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recently completed sessions.

    Args:
        limit: Maximum number of sessions to return

    Returns:
        List of session data dictionaries
    """
    return _get_session_manager().list_sessions(limit=limit)


def search_sessions(
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
    return _get_session_manager().search_sessions(
        branch=branch,
        phase=phase,
        date_from=date_from,
        date_to=date_to
    )


# Statistics API

def get_statistics() -> Dict[str, Any]:
    """
    Get aggregate statistics.

    Returns:
        Dictionary containing framework statistics
    """
    return _get_state_manager().get_statistics()


def get_state_health() -> Dict[str, Any]:
    """
    Get state health information.

    Returns:
        Dictionary containing health metrics
    """
    return _get_state_manager().get_state_health()


def get_artifact_count() -> int:
    """
    Get total number of artifacts.

    Returns:
        Total artifact count
    """
    return _get_state_manager().state['artifacts']['total_count']


def get_session_count() -> int:
    """
    Get total number of sessions.

    Returns:
        Total session count
    """
    return _get_state_manager().state['sessions']['total_count']


# Utility API

def is_state_tracking_available() -> bool:
    """
    Check if state tracking is available and initialized.

    Returns:
        True if state tracking is available, False otherwise
    """
    try:
        config_path = Path(".agentqms/config.yaml")
        return config_path.exists()
    except Exception:
        return False


def get_framework_info() -> Dict[str, Any]:
    """
    Get framework information.

    Returns:
        Dictionary containing framework name, version, and schema version
    """
    return _get_state_manager().state['framework'].copy()


def reset_managers() -> None:
    """
    Reset the global manager instances.

    This is mainly useful for testing or when reinitializing state.
    """
    global _state_manager, _session_manager
    _state_manager = None
    _session_manager = None
