"""
State Management Module for Agent Framework

This module provides state tracking and persistence capabilities for the agent framework,
including session tracking, artifact indexing, and context preservation across conversations.
"""

import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml


class StateError(Exception):
    """Raised when state operations fail."""
    pass


class StateManager:
    """
    Manages agent framework state persistence and retrieval.

    The StateManager handles:
    - Loading and saving state to JSON files
    - Managing current context (active session, branch, artifacts)
    - Tracking artifacts and their relationships
    - Creating and restoring session snapshots
    - State validation and error handling
    """

    def __init__(self, config_path: str = ".agentqms/config.yaml"):
        """
        Initialize the StateManager.

        Args:
            config_path: Path to the configuration YAML file

        Raises:
            StateError: If configuration file is not found or invalid
        """
        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent.parent

        if not self.config_path.exists():
            raise StateError(f"Configuration file not found: {config_path}")

        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up paths
        self.state_file = self.project_root / self.config['paths']['state_file']
        self.sessions_dir = self.project_root / self.config['paths']['sessions_dir']
        self.artifacts_dir = self.project_root / self.config['paths']['artifacts_dir']

        # Create backup directory if needed
        if self.config['settings'].get('auto_backup', False):
            self.backup_dir = self.project_root / self.config['settings']['backup_dir']
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.state: Optional[Dict[str, Any]] = None
        self._load_state()

    def _load_state(self) -> None:
        """
        Load state from the state file.

        If the state file doesn't exist or is corrupted, initialize with default state.

        Raises:
            StateError: If state file exists but cannot be parsed
        """
        if not self.state_file.exists():
            self._initialize_default_state()
            return

        try:
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)

            # Validate state schema version
            if self.state.get('schema_version') != self.config['framework']['state_schema_version']:
                print(f"Warning: State schema version mismatch. "
                      f"Expected {self.config['framework']['state_schema_version']}, "
                      f"found {self.state.get('schema_version')}")

        except json.JSONDecodeError as e:
            # State file is corrupted - backup and reinitialize
            backup_path = self._backup_corrupted_state()
            print(f"Warning: Corrupted state file backed up to {backup_path}")
            self._initialize_default_state()

        except Exception as e:
            raise StateError(f"Failed to load state: {e}")

    def _initialize_default_state(self) -> None:
        """Initialize state with default schema."""
        self.state = {
            "schema_version": self.config['framework']['state_schema_version'],
            "last_updated": self._get_timestamp(),
            "framework": {
                "name": self.config['framework']['name'],
                "version": self.config['framework']['version']
            },
            "current_context": {
                "active_session_id": None,
                "current_branch": None,
                "current_phase": None,
                "active_artifacts": [],
                "pending_tasks": []
            },
            "sessions": {
                "total_count": 0,
                "active_session": None,
                "session_history": []
            },
            "artifacts": {
                "total_count": 0,
                "by_type": {},
                "by_status": {},
                "index": []
            },
            "relationships": {
                "artifact_dependencies": {},
                "session_artifacts": {}
            },
            "statistics": {
                "total_sessions": 0,
                "total_artifacts_created": 0,
                "total_artifacts_validated": 0,
                "total_artifacts_deployed": 0,
                "last_session_timestamp": None
            }
        }
        self._save_state()

    def _save_state(self) -> None:
        """
        Save current state to the state file.

        Raises:
            StateError: If state cannot be saved
        """
        if self.state is None:
            raise StateError("No state to save")

        try:
            # Update last_updated timestamp
            self.state['last_updated'] = self._get_timestamp()

            # Create backup if enabled
            if self.config['settings'].get('auto_backup', False) and self.state_file.exists():
                self._create_backup()

            # Write state file atomically
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.state, f, indent=2)

            # Atomic rename
            temp_file.replace(self.state_file)

        except Exception as e:
            raise StateError(f"Failed to save state: {e}")

    def _create_backup(self) -> None:
        """Create a backup of the current state file."""
        if not self.state_file.exists():
            return

        timestamp = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"state_backup_{timestamp}.json"

        shutil.copy2(self.state_file, backup_file)

        # Cleanup old backups
        self._cleanup_old_backups()

    def _cleanup_old_backups(self) -> None:
        """Remove old backups beyond max_backups limit."""
        max_backups = self.config['settings'].get('max_backups', 10)

        backups = sorted(self.backup_dir.glob("state_backup_*.json"), reverse=True)

        for old_backup in backups[max_backups:]:
            old_backup.unlink()

    def _backup_corrupted_state(self) -> Path:
        """
        Backup corrupted state file.

        Returns:
            Path to the backup file
        """
        timestamp = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
        backup_file = self.state_file.parent / f"state_corrupted_{timestamp}.json"

        shutil.copy2(self.state_file, backup_file)

        return backup_file

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format with KST timezone."""
        return datetime.datetime.now(ZoneInfo("Asia/Seoul")).isoformat()

    # Context Management Methods

    def get_current_context(self) -> Dict[str, Any]:
        """
        Get the current context.

        Returns:
            Dictionary containing current context information
        """
        return self.state['current_context'].copy()

    def update_current_context(self, **kwargs) -> None:
        """
        Update current context fields.

        Args:
            **kwargs: Fields to update in current_context
        """
        for key, value in kwargs.items():
            if key in self.state['current_context']:
                self.state['current_context'][key] = value

        self._save_state()

    def set_active_session(self, session_id: str) -> None:
        """
        Set the active session ID.

        Args:
            session_id: The session ID to set as active
        """
        self.state['current_context']['active_session_id'] = session_id
        self.state['sessions']['active_session'] = session_id
        self._save_state()

    def clear_active_session(self) -> None:
        """Clear the active session."""
        self.state['current_context']['active_session_id'] = None
        self.state['sessions']['active_session'] = None
        self._save_state()

    def set_current_branch(self, branch: str) -> None:
        """
        Set the current git branch.

        Args:
            branch: Branch name
        """
        self.state['current_context']['current_branch'] = branch
        self._save_state()

    def set_current_phase(self, phase: str) -> None:
        """
        Set the current project phase.

        Args:
            phase: Phase name/identifier
        """
        self.state['current_context']['current_phase'] = phase
        self._save_state()

    def add_active_artifact(self, artifact_path: str) -> None:
        """
        Add an artifact to the active artifacts list.

        Args:
            artifact_path: Path to the artifact
        """
        if artifact_path not in self.state['current_context']['active_artifacts']:
            self.state['current_context']['active_artifacts'].append(artifact_path)
            self._save_state()

    def remove_active_artifact(self, artifact_path: str) -> None:
        """
        Remove an artifact from the active artifacts list.

        Args:
            artifact_path: Path to the artifact
        """
        if artifact_path in self.state['current_context']['active_artifacts']:
            self.state['current_context']['active_artifacts'].remove(artifact_path)
            self._save_state()

    def get_active_artifacts(self) -> List[str]:
        """
        Get list of active artifacts.

        Returns:
            List of active artifact paths
        """
        return self.state['current_context']['active_artifacts'].copy()

    # Artifact Tracking Methods

    def add_artifact(
        self,
        artifact_path: str,
        artifact_type: str,
        status: str = "draft",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an artifact to the index.

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (e.g., 'implementation_plan', 'assessment')
            status: Artifact status (default: 'draft')
            metadata: Additional metadata for the artifact
        """
        # Check if artifact already exists
        existing = self.get_artifact(artifact_path)
        if existing:
            # Update existing artifact in the index
            for artifact in self.state['artifacts']['index']:
                if artifact['path'] == artifact_path:
                    old_status = artifact['status']
                    artifact['status'] = status
                    artifact['last_updated'] = self._get_timestamp()
                    if metadata:
                        artifact['metadata'].update(metadata)

                    # Update status counts if status changed
                    if old_status != status:
                        if old_status in self.state['artifacts']['by_status']:
                            self.state['artifacts']['by_status'][old_status] -= 1
                            if self.state['artifacts']['by_status'][old_status] <= 0:
                                del self.state['artifacts']['by_status'][old_status]

                        if status not in self.state['artifacts']['by_status']:
                            self.state['artifacts']['by_status'][status] = 0
                        self.state['artifacts']['by_status'][status] += 1
                    break
        else:
            # Add new artifact
            artifact_entry = {
                "path": artifact_path,
                "type": artifact_type,
                "status": status,
                "created_at": self._get_timestamp(),
                "last_updated": self._get_timestamp(),
                "metadata": metadata or {}
            }

            self.state['artifacts']['index'].append(artifact_entry)

            # Update counts
            self.state['artifacts']['total_count'] += 1
            self.state['statistics']['total_artifacts_created'] += 1

            # Update type counts
            if artifact_type not in self.state['artifacts']['by_type']:
                self.state['artifacts']['by_type'][artifact_type] = 0
            self.state['artifacts']['by_type'][artifact_type] += 1

            # Update status counts
            if status not in self.state['artifacts']['by_status']:
                self.state['artifacts']['by_status'][status] = 0
            self.state['artifacts']['by_status'][status] += 1

        self._save_state()

    def get_artifact(self, artifact_path: str) -> Optional[Dict[str, Any]]:
        """
        Get artifact information by path.

        Args:
            artifact_path: Path to the artifact

        Returns:
            Artifact dictionary or None if not found
        """
        for artifact in self.state['artifacts']['index']:
            if artifact['path'] == artifact_path:
                return artifact.copy()
        return None

    def update_artifact_status(self, artifact_path: str, new_status: str) -> None:
        """
        Update an artifact's status.

        Args:
            artifact_path: Path to the artifact
            new_status: New status value
        """
        for artifact in self.state['artifacts']['index']:
            if artifact['path'] == artifact_path:
                old_status = artifact['status']
                artifact['status'] = new_status
                artifact['last_updated'] = self._get_timestamp()

                # Update status counts
                if old_status in self.state['artifacts']['by_status']:
                    self.state['artifacts']['by_status'][old_status] -= 1
                    if self.state['artifacts']['by_status'][old_status] <= 0:
                        del self.state['artifacts']['by_status'][old_status]

                if new_status not in self.state['artifacts']['by_status']:
                    self.state['artifacts']['by_status'][new_status] = 0
                self.state['artifacts']['by_status'][new_status] += 1

                # Update statistics
                if new_status == 'validated':
                    self.state['statistics']['total_artifacts_validated'] += 1
                elif new_status == 'deployed':
                    self.state['statistics']['total_artifacts_deployed'] += 1

                self._save_state()
                return

        raise StateError(f"Artifact not found: {artifact_path}")

    def get_artifacts_by_type(self, artifact_type: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts of a specific type.

        Args:
            artifact_type: Type of artifacts to retrieve

        Returns:
            List of artifact dictionaries
        """
        return [
            artifact.copy()
            for artifact in self.state['artifacts']['index']
            if artifact['type'] == artifact_type
        ]

    def get_artifacts_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all artifacts with a specific status.

        Args:
            status: Status to filter by

        Returns:
            List of artifact dictionaries
        """
        return [
            artifact.copy()
            for artifact in self.state['artifacts']['index']
            if artifact['status'] == status
        ]

    def get_all_artifacts(self) -> List[Dict[str, Any]]:
        """
        Get all artifacts in the index.

        Returns:
            List of all artifact dictionaries
        """
        return [artifact.copy() for artifact in self.state['artifacts']['index']]

    # Artifact Relationship Management

    def add_artifact_dependency(self, artifact_path: str, dependency_path: str) -> None:
        """
        Add a dependency relationship between artifacts.

        Args:
            artifact_path: Path to the artifact that depends on another
            dependency_path: Path to the dependency artifact

        Raises:
            StateError: If either artifact doesn't exist or circular dependency detected
        """
        # Verify both artifacts exist
        if not self.get_artifact(artifact_path):
            raise StateError(f"Artifact not found: {artifact_path}")
        if not self.get_artifact(dependency_path):
            raise StateError(f"Dependency artifact not found: {dependency_path}")

        # Check for circular dependencies
        if self._would_create_circular_dependency(artifact_path, dependency_path):
            raise StateError(
                f"Adding dependency would create circular dependency: "
                f"{artifact_path} -> {dependency_path}"
            )

        # Add dependency
        if artifact_path not in self.state['relationships']['artifact_dependencies']:
            self.state['relationships']['artifact_dependencies'][artifact_path] = []

        if dependency_path not in self.state['relationships']['artifact_dependencies'][artifact_path]:
            self.state['relationships']['artifact_dependencies'][artifact_path].append(dependency_path)
            self._save_state()

    def remove_artifact_dependency(self, artifact_path: str, dependency_path: str) -> None:
        """
        Remove a dependency relationship between artifacts.

        Args:
            artifact_path: Path to the artifact
            dependency_path: Path to the dependency to remove
        """
        if artifact_path in self.state['relationships']['artifact_dependencies']:
            dependencies = self.state['relationships']['artifact_dependencies'][artifact_path]
            if dependency_path in dependencies:
                dependencies.remove(dependency_path)
                # Clean up empty dependency list
                if not dependencies:
                    del self.state['relationships']['artifact_dependencies'][artifact_path]
                self._save_state()

    def get_artifact_dependencies(self, artifact_path: str) -> List[str]:
        """
        Get all dependencies for an artifact.

        Args:
            artifact_path: Path to the artifact

        Returns:
            List of dependency artifact paths
        """
        return self.state['relationships']['artifact_dependencies'].get(artifact_path, []).copy()

    def get_artifacts_depending_on(self, dependency_path: str) -> List[str]:
        """
        Get all artifacts that depend on a given artifact (reverse lookup).

        Args:
            dependency_path: Path to the dependency artifact

        Returns:
            List of artifact paths that depend on this artifact
        """
        depending_artifacts = []

        for artifact_path, deps in self.state['relationships']['artifact_dependencies'].items():
            if dependency_path in deps:
                depending_artifacts.append(artifact_path)

        return depending_artifacts

    def get_dependency_tree(self, artifact_path: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Get the full dependency tree for an artifact.

        Args:
            artifact_path: Path to the artifact
            max_depth: Maximum depth to traverse (prevents infinite loops)

        Returns:
            Dictionary representing the dependency tree
        """
        def build_tree(path: str, depth: int = 0) -> Dict[str, Any]:
            if depth >= max_depth:
                return {"path": path, "dependencies": [], "max_depth_reached": True}

            dependencies = self.get_artifact_dependencies(path)
            return {
                "path": path,
                "dependencies": [build_tree(dep, depth + 1) for dep in dependencies],
                "max_depth_reached": False
            }

        return build_tree(artifact_path)

    def _would_create_circular_dependency(self, artifact_path: str, new_dependency: str) -> bool:
        """
        Check if adding a dependency would create a circular dependency.

        Args:
            artifact_path: Path to the artifact
            new_dependency: Path to the potential new dependency

        Returns:
            True if circular dependency would be created
        """
        # If new_dependency depends on artifact_path (directly or indirectly),
        # adding artifact_path -> new_dependency would create a cycle

        visited = set()

        def has_path_to(from_path: str, to_path: str) -> bool:
            if from_path == to_path:
                return True
            if from_path in visited:
                return False

            visited.add(from_path)

            dependencies = self.get_artifact_dependencies(from_path)
            for dep in dependencies:
                if has_path_to(dep, to_path):
                    return True

            return False

        return has_path_to(new_dependency, artifact_path)

    def propagate_status_update(self, artifact_path: str, new_status: str) -> List[str]:
        """
        Update an artifact's status and optionally propagate to dependent artifacts.

        Args:
            artifact_path: Path to the artifact
            new_status: New status to apply

        Returns:
            List of artifact paths that were updated

        Note:
            This is a basic implementation. Status propagation rules can be customized
            based on specific requirements (e.g., deprecating an artifact might deprecate
            all artifacts that depend on it).
        """
        updated = []

        # Update the artifact itself
        self.update_artifact_status(artifact_path, new_status)
        updated.append(artifact_path)

        # For now, only propagate "deprecated" status to dependent artifacts
        if new_status == "deprecated":
            depending_artifacts = self.get_artifacts_depending_on(artifact_path)
            for dep_artifact in depending_artifacts:
                current_artifact = self.get_artifact(dep_artifact)
                if current_artifact and current_artifact['status'] != "deprecated":
                    self.update_artifact_status(dep_artifact, "deprecated")
                    updated.append(dep_artifact)

        return updated

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get state statistics.

        Returns:
            Dictionary containing statistics
        """
        return self.state['statistics'].copy()

    # State Health and Validation

    def validate_state(self) -> bool:
        """
        Validate the current state structure.

        Returns:
            True if state is valid, False otherwise
        """
        required_keys = [
            'schema_version', 'last_updated', 'framework',
            'current_context', 'sessions', 'artifacts',
            'relationships', 'statistics'
        ]

        for key in required_keys:
            if key not in self.state:
                return False

        return True

    def get_state_health(self) -> Dict[str, Any]:
        """
        Get state health information.

        Returns:
            Dictionary containing health metrics
        """
        return {
            "is_valid": self.validate_state(),
            "schema_version": self.state.get('schema_version'),
            "last_updated": self.state.get('last_updated'),
            "total_artifacts": self.state['artifacts']['total_count'],
            "total_sessions": self.state['sessions']['total_count'],
            "active_session": self.state['current_context']['active_session_id'],
            "state_file_exists": self.state_file.exists(),
            "state_file_size_bytes": self.state_file.stat().st_size if self.state_file.exists() else 0
        }
