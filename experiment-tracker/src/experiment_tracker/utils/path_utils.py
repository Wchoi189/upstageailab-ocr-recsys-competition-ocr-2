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

    def ensure_structure(self) -> None:
        """Create all required experiment subdirectories"""
        for path in [
            self.get_artifacts_path(),
            self.get_logs_path(),
            self.get_metadata_path(),
            self.base_path / "scripts",
            self.base_path / "assessments",
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def get_context_file(self, context_type: str) -> Path:
        """Get path to specific context file (state, tasks, decisions, components)"""
        return self.get_metadata_path() / f"{context_type}.yml"

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
