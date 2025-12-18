"""
Metadata synchronization utilities for keeping state.json and .metadata/ files in sync.
"""

import json
from pathlib import Path
from typing import Any

import yaml

from experiment_tracker.utils.path_utils import ExperimentPaths


class MetadataSync:
    """Handles synchronization between state.json and .metadata/ files."""

    def __init__(self, experiment_id: str, tracker_root: Path):
        self.experiment_id = experiment_id
        self.paths = ExperimentPaths(experiment_id, tracker_root)
        self.state_file = self.paths.base_path / "state.yml"  # Changed to YAML

    def sync_state_to_metadata(self) -> bool:
        """
        Update .metadata/ files from state.yml.
        This ensures .metadata/ reflects the current state.
        """
        if not self.state_file.exists():
            return False

        try:
            with open(self.state_file) as f:
                state = yaml.safe_load(f) or {}

            # Sync to .metadata/state.yml
            self._sync_state_yml(state)

            return True
        except Exception as e:
            print(f"Error syncing state to metadata: {e}")
            return False

    def sync_metadata_to_state(self) -> bool:
        """
        Update state.json from .metadata/ files.
        This ensures state.json reflects metadata changes.
        """
        if not self.state_file.exists():
            return False

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            # Sync tasks from .metadata/tasks.yml
            self._sync_tasks_to_state(state)

            # Sync decisions from .metadata/decisions.yml
            self._sync_decisions_to_state(state)

            # Save updated state
            with open(self.state_file, "w") as f:
                yaml.dump(state, f, default_flow_style=False, allow_unicode=True)

            return True
        except Exception as e:
            print(f"Error syncing metadata to state: {e}")
            return False

    def validate_consistency(self) -> list[str]:
        """
        Check for inconsistencies between state.json and .metadata/ files.
        Returns list of issues found.
        """
        issues = []

        if not self.state_file.exists():
            issues.append("state.yml does not exist")
            return issues

        try:
            with open(self.state_file) as f:
                state = yaml.safe_load(f) or {}

            # Check tasks consistency
            tasks_file = self.paths.get_context_file("tasks")
            if tasks_file.exists():
                with open(tasks_file) as f:
                    tasks_data = yaml.safe_load(f) or {}
                    tasks = tasks_data.get("tasks", [])
                    # Basic validation - could be more thorough
                    if len(tasks) > 0 and "tasks" not in state:
                        issues.append("Tasks exist in .metadata/tasks.yml but not in state.json")

            # Check decisions consistency
            decisions_file = self.paths.get_context_file("decisions")
            if decisions_file.exists():
                with open(decisions_file) as f:
                    decisions_data = yaml.safe_load(f) or {}
                    decisions = decisions_data.get("decisions", [])
                    if len(decisions) > 0 and "decisions" not in state:
                        issues.append("Decisions exist in .metadata/decisions.yml but not in state.json")

        except Exception as e:
            issues.append(f"Error validating consistency: {e}")

        return issues

    def add_cross_references(self) -> bool:
        """
        Add cross-references between tasks, decisions, and assessments.
        This helps maintain relationships between related items.
        """
        try:
            # Load all metadata files
            tasks_file = self.paths.get_context_file("tasks")
            decisions_file = self.paths.get_context_file("decisions")

            if tasks_file.exists():
                with open(tasks_file) as f:
                    yaml.safe_load(f) or {"tasks": []}

            if decisions_file.exists():
                with open(decisions_file) as f:
                    yaml.safe_load(f) or {"decisions": []}

            # Note: Cross-referencing logic would be more sophisticated
            # For now, this is a placeholder that can be extended
            # based on naming patterns or explicit references

            return True
        except Exception as e:
            print(f"Error adding cross-references: {e}")
            return False

    def _sync_state_yml(self, state: dict[str, Any]):
        """Sync state.json to .metadata/state.yml"""
        state_yml_file = self.paths.get_context_file("state")

        # Load existing state.yml or create new
        if state_yml_file.exists():
            with open(state_yml_file) as f:
                state_yml = yaml.safe_load(f) or {}
        else:
            state_yml = {}

        # Update key fields from state.json
        state_yml["current_state"] = state.get("status", "UNKNOWN")
        if "intention" in state:
            state_yml["experiment_goal"] = state["intention"]
            state_yml["initial_assessment"] = state["intention"]

        # Preserve other fields that might have been manually added
        # (like current_phase, success_rate, etc.)

        with open(state_yml_file, "w") as f:
            yaml.dump(state_yml, f, default_flow_style=False, sort_keys=False)

    def _sync_tasks_to_state(self, state: dict[str, Any]):
        """Sync tasks from .metadata/tasks.yml to state.json"""
        tasks_file = self.paths.get_context_file("tasks")
        if not tasks_file.exists():
            return

        with open(tasks_file) as f:
            tasks_data = yaml.safe_load(f) or {}
            tasks_data.get("tasks", [])

        # Update state with task count/summary
        # Note: Full task details remain in .metadata/tasks.yml
        # state.json can have a summary or reference
        if "tasks" not in state:
            state["tasks"] = []
        # Could add task summary here if needed

    def _sync_decisions_to_state(self, state: dict[str, Any]):
        """Sync decisions from .metadata/decisions.yml to state.json"""
        decisions_file = self.paths.get_context_file("decisions")
        if not decisions_file.exists():
            return

        with open(decisions_file) as f:
            decisions_data = yaml.safe_load(f) or {}
            decisions_data.get("decisions", [])

        # Update state with decision count/summary
        if "decisions" not in state:
            state["decisions"] = []
        # Could add decision summary here if needed


def sync_experiment_metadata(experiment_id: str, tracker_root: Path, direction: str = "both") -> bool:
    """
    Convenience function to sync metadata for an experiment.

    Args:
        experiment_id: Experiment ID
        tracker_root: Tracker root directory
        direction: "both", "to_metadata", or "to_state"

    Returns:
        True if successful
    """
    sync = MetadataSync(experiment_id, tracker_root)

    if direction == "both":
        result1 = sync.sync_state_to_metadata()
        result2 = sync.sync_metadata_to_state()
        return result1 and result2
    elif direction == "to_metadata":
        return sync.sync_state_to_metadata()
    elif direction == "to_state":
        return sync.sync_metadata_to_state()
    else:
        print(f"Unknown direction: {direction}")
        return False
