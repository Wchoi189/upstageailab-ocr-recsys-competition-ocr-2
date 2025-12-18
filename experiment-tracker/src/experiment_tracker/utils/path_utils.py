from pathlib import Path


class ExperimentPaths:
    def __init__(self, experiment_id: str, tracker_root: Path | None = None):
        if tracker_root is None:
            # Assuming this file is in src/experiment_tracker/utils/path_utils.py
            # Root is ../../../
            tracker_root = Path(__file__).parent.parent.parent.parent.resolve()
        self.tracker_root = tracker_root
        self.experiment_id = experiment_id
        self.base_path = tracker_root / "experiments" / experiment_id
        # print(f"DEBUG: ExperimentPaths base_path: {self.base_path}")

    def get_artifacts_path(self) -> Path:
        return self.base_path / "artifacts"

    def get_logs_path(self) -> Path:
        return self.base_path / "logs"

    def get_metadata_path(self) -> Path:
        return self.base_path / ".metadata"

    def get_incident_reports_path(self) -> Path:
        return self.base_path / "incident_reports"

    def ensure_structure(self) -> None:
        """Create all required experiment subdirectories"""
        for path in [
            self.get_artifacts_path(),
            self.get_logs_path(),
            self.get_metadata_path(),
            self.base_path / "scripts",
            self.base_path / "assessments",
            self.get_incident_reports_path(),
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def get_context_file(self, context_type: str) -> Path:
        """Get path to specific context file (state, tasks, decisions, components)"""
        return self.get_metadata_path() / f"{context_type}.yml"

    def resolve_artifact_path(self, path: str, allow_placeholder: bool = True) -> Path | None:
        """
        Resolve artifact path with {timestamp} placeholder support.

        Args:
            path: Path string, may contain {timestamp} placeholder
            allow_placeholder: If True, resolve {timestamp} to latest matching artifact

        Returns:
            Resolved Path or None if not found
        """
        # If path contains {timestamp}, try to resolve it
        if allow_placeholder and "{timestamp}" in path:
            # Extract pattern (everything before and after {timestamp})
            pattern_parts = path.split("{timestamp}")
            if len(pattern_parts) == 2:
                prefix = pattern_parts[0]
                suffix = pattern_parts[1]
                # Find latest artifact matching this pattern
                return self.find_latest_artifact(f"{prefix}*{suffix}")

        # Try as relative path first (relative to experiment directory)
        resolved = self.base_path / path
        if resolved.exists():
            return resolved

        # Try as absolute path
        resolved = Path(path)
        if resolved.exists():
            return resolved

        # Try relative to artifacts directory
        resolved = self.get_artifacts_path() / path
        if resolved.exists():
            return resolved

        return None

    def find_latest_artifact(self, pattern: str) -> Path | None:
        """
        Find latest artifact matching pattern.
        Uses modification time to determine "latest".

        Args:
            pattern: Glob pattern (e.g., "*_worst_performers_test/results.json")

        Returns:
            Path to latest matching artifact or None
        """
        artifacts_dir = self.get_artifacts_path()
        if not artifacts_dir.exists():
            return None

        # Search recursively
        matches = list(artifacts_dir.rglob(pattern))
        if not matches:
            return None

        # Sort by modification time, newest first
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def interactive_select_artifact(self, candidates: list[Path]) -> Path | None:
        """
        Interactive selection if multiple artifact candidates found.

        Args:
            candidates: List of candidate paths

        Returns:
            Selected path or None if cancelled
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        print("\nMultiple artifacts found:")
        for i, path in enumerate(candidates, 1):
            rel_path = path.relative_to(self.base_path)
            print(f"  {i}. {rel_path}")

        try:
            choice = input(f"\nSelect artifact (1-{len(candidates)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except (ValueError, KeyboardInterrupt):
            pass

        return None

    @staticmethod
    def detect_experiment_id(current_path: Path | None = None) -> str | None:
        """
        Attempts to detect the experiment ID based on the current working directory or provided path.
        Assumes structure: .../experiments/<experiment_id>/...
        """
        if current_path is None:
            current_path = Path.cwd()

        path = current_path.resolve()

        # Traverse up until we find 'experiments'
        for parent in path.parents:
            if parent.name == "experiments":
                # The folder immediately inside 'experiments' is the ID
                # We need to find which part of 'path' is the child of 'parent'
                relative = path.relative_to(parent)
                return relative.parts[0]

        return None


def get_tracker_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.resolve()


def setup_script_paths(script_path: Path | None = None) -> tuple[Path, str | None, ExperimentPaths | None]:
    """
    Helper function to set up path resolution for experiment scripts.

    This function:
    1. Gets the tracker root
    2. Auto-detects the experiment ID from the script location
    3. Creates an ExperimentPaths instance if experiment ID is found

    Args:
        script_path: Path to the script file (defaults to __file__ if called from script)

    Returns:
        Tuple of (tracker_root, experiment_id, experiment_paths)
        experiment_id and experiment_paths will be None if detection fails
    """
    tracker_root = get_tracker_root()

    if script_path is None:
        # Try to detect from current working directory
        experiment_id = ExperimentPaths.detect_experiment_id()
    else:
        experiment_id = ExperimentPaths.detect_experiment_id(script_path)

    if experiment_id:
        experiment_paths = ExperimentPaths(experiment_id, tracker_root)
    else:
        experiment_paths = None

    return tracker_root, experiment_id, experiment_paths
