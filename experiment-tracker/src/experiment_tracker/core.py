import datetime
import json
import shutil
from pathlib import Path

import yaml

from experiment_tracker.utils.path_utils import ExperimentPaths


class ExperimentTracker:
    def _format_timestamp(self, dt: datetime.datetime = None) -> str:
        """Format timestamp in human-readable format: 2025-11-23 00:52 (KST)"""
        if dt is None:
            dt = datetime.datetime.now()
        format_str = self.config.get("timestamp_format", "%Y-%m-%d %H:%M (KST)")
        return dt.strftime(format_str)
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            # Default to the parent of the src directory if not provided
            # Assuming src/experiment_tracker/core.py
            self.root_dir = Path(__file__).parent.parent.parent.resolve()
        else:
            self.root_dir = Path(root_dir).resolve()

        self.config = self._load_config()
        self.experiments_dir = self.root_dir / self.config.get("base_path", "experiments")
        self.current_experiment_file = self.experiments_dir / ".current"

        # Ensure directories exist
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        config_path = self.root_dir / ".config" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _get_current_experiment_id(self) -> str | None:
        if self.current_experiment_file.exists():
            return self.current_experiment_file.read_text().strip()
        return None

    def _set_current_experiment_id(self, experiment_id: str):
        self.current_experiment_file.write_text(experiment_id)

    def _clear_current_experiment_id(self):
        if self.current_experiment_file.exists():
            self.current_experiment_file.unlink()

    def _load_state(self, experiment_id: str) -> dict:
        state_path = self.experiments_dir / experiment_id / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                return json.load(f)
        raise FileNotFoundError(f"State file for experiment {experiment_id} not found.")

    def _save_state(self, experiment_id: str, state: dict):
        state_path = self.experiments_dir / experiment_id / "state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def start_experiment(self, intention: str, experiment_type: str) -> str:
        # Stash current if exists
        current_id = self._get_current_experiment_id()
        if current_id:
            self.stash_incomplete(current_id)

        # Create new experiment
        timestamp = datetime.datetime.now().strftime(self.config.get("date_format", "%Y%m%d_%H%M%S"))
        experiment_id = f"{timestamp}_{experiment_type}"

        # Use ExperimentPaths to create structure
        paths = self._get_paths(experiment_id)
        paths.ensure_structure()

        state = {
            "id": experiment_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "type": experiment_type,
            "intention": intention,
            "status": "ACTIVE",
            "artifacts": [],
            "assessments": [],
            "incident_reports": [],
            "open_questions": [],
            "roadmap": [],
        }

        self._save_state(experiment_id, state)
        self._init_metadata(paths, state)
        self._set_current_experiment_id(experiment_id)

        print(f"Started experiment: {experiment_id}")
        return experiment_id

    def stash_incomplete(self, experiment_id: str) -> bool:
        try:
            state = self._load_state(experiment_id)
            if state["status"] == "ACTIVE":
                state["status"] = "INCOMPLETE"
                self._save_state(experiment_id, state)

                # Update state.yml
                paths = self._get_paths(experiment_id)
                state_file = paths.get_context_file("state")
                if state_file.exists():
                    with open(state_file) as f:
                        meta_state = yaml.safe_load(f) or {}
                    meta_state["current_state"] = "INCOMPLETE"
                    meta_state["last_updated"] = datetime.datetime.now().isoformat()
                    with open(state_file, "w") as f:
                        yaml.dump(meta_state, f)

                print(f"Stashed incomplete experiment: {experiment_id}")
            self._clear_current_experiment_id()
            return True
        except Exception as e:
            print(f"Error stashing experiment {experiment_id}: {e}")
            return False

    def record_artifact(self, artifact_path: str, metadata: dict = None, experiment_id: str = None) -> bool:
        if experiment_id is None:
            experiment_id = self._get_current_experiment_id()
            if not experiment_id:
                print("No active experiment found.")
                return False

        try:
            state = self._load_state(experiment_id)
            paths = self._get_paths(experiment_id)

            # Copy artifact to experiment folder
            src_path = Path(artifact_path)
            if not src_path.exists():
                print(f"Artifact not found: {src_path}")
                return False

            dest_dir = paths.get_artifacts_path()
            dest_path = dest_dir / src_path.name
            shutil.copy2(src_path, dest_path)

            artifact_record = {
                "path": f"artifacts/{src_path.name}",
                "type": metadata.get("type", "unknown") if metadata else "unknown",
                "timestamp": datetime.datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            state["artifacts"].append(artifact_record)
            self._save_state(experiment_id, state)
            print(f"Recorded artifact: {src_path.name} to {experiment_id}")
            return True
        except Exception as e:
            print(f"Error recording artifact: {e}")
            return False

    def get_context(self, experiment_id: str = None) -> dict:
        if experiment_id is None:
            experiment_id = self._get_current_experiment_id()

        if not experiment_id:
            return {}

        # Load core state
        context = self._load_state(experiment_id)

        # Load extended context
        paths = self._get_paths(experiment_id)

        for meta_type in ["tasks", "decisions", "components", "state"]:
            meta_file = paths.get_context_file(meta_type)
            if meta_file.exists():
                with open(meta_file) as f:
                    context[meta_type] = yaml.safe_load(f)

        return context

    def provide_feedback(self, feedback: str, experiment_id: str = None) -> bool:
        # This could append to a feedback log or assessment
        # For now, let's just print it, or maybe add to open_questions/roadmap if structured
        # Or create a feedback artifact
        if experiment_id is None:
            experiment_id = self._get_current_experiment_id()

        if not experiment_id:
            return False

        feedback_file = self.experiments_dir / experiment_id / "assessments" / "ai_feedback.md"
        with open(feedback_file, "a") as f:
            f.write(f"\n## Feedback {datetime.datetime.now().isoformat()}\n{feedback}\n")

        return True

    def export_experiment(self, experiment_id: str, format: str = "archive", destination: str = "./exports") -> str:
        experiment_dir = self.experiments_dir / experiment_id
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")

        dest_path = Path(destination)
        dest_path.mkdir(parents=True, exist_ok=True)

        if format == "archive":
            archive_name = dest_path / experiment_id
            shutil.make_archive(str(archive_name), "zip", experiment_dir)
            print(f"Exported experiment to {archive_name}.zip")
            return str(archive_name) + ".zip"

        return ""

    def list_experiments(self) -> list[dict]:
        experiments = []
        for d in self.experiments_dir.iterdir():
            if d.is_dir() and (d / "state.json").exists():
                try:
                    state = self._load_state(d.name)
                    experiments.append(state)
                except:
                    pass
        return sorted(experiments, key=lambda x: x["timestamp"], reverse=True)

    def _get_paths(self, experiment_id: str) -> ExperimentPaths:
        return ExperimentPaths(experiment_id, self.root_dir)

    def _init_metadata(self, paths: ExperimentPaths, state: dict):
        # state.yml
        # Convert ISO timestamp to readable format for metadata files
        try:
            # Handle both ISO format and already-formatted timestamps
            if 'T' in state["timestamp"]:
                created_dt = datetime.datetime.fromisoformat(state["timestamp"].replace('Z', '+00:00'))
                readable_timestamp = self._format_timestamp(created_dt)
            else:
                # Already in readable format
                readable_timestamp = state["timestamp"]
        except (ValueError, AttributeError):
            # Fallback to current time if parsing fails
            readable_timestamp = self._format_timestamp()

        with open(paths.get_context_file("state"), "w") as f:
            yaml.dump(
                {
                    "current_state": state["status"],
                    "initial_assessment": state["intention"],
                    "experiment_goal": state["intention"],
                    "created_at": readable_timestamp,
                    "last_updated": readable_timestamp,
                },
                f,
            )

        # tasks.yml
        with open(paths.get_context_file("tasks"), "w") as f:
            yaml.dump({"tasks": []}, f)

        # decisions.yml
        with open(paths.get_context_file("decisions"), "w") as f:
            yaml.dump({"decisions": []}, f)

        # components.yml
        with open(paths.get_context_file("components"), "w") as f:
            yaml.dump({"components": {}}, f)

    def add_task(self, description: str, experiment_id: str = None) -> str | None:
        if experiment_id is None:
            experiment_id = self._get_current_experiment_id()
        if not experiment_id:
            print("No active experiment found.")
            return None

        paths = self._get_paths(experiment_id)
        task_file = paths.get_context_file("tasks")

        if task_file.exists():
            with open(task_file) as f:
                data = yaml.safe_load(f) or {"tasks": []}
        else:
            data = {"tasks": []}

        task_id = f"task_{len(data.get('tasks', [])) + 1:03d}"
        new_task = {"id": task_id, "description": description, "status": "in_progress", "created_at": self._format_timestamp()}
        if "tasks" not in data:
            data["tasks"] = []
        data["tasks"].append(new_task)

        with open(task_file, "w") as f:
            yaml.dump(data, f)

        print(f"Added task: {task_id} to {task_file}")
        return task_id

    def complete_task(self, task_id: str, notes: str = None, experiment_id: str = None) -> bool:
        if experiment_id is None:
            experiment_id = self._get_current_experiment_id()
        if not experiment_id:
            print("No active experiment found.")
            return False

        paths = self._get_paths(experiment_id)
        task_file = paths.get_context_file("tasks")

        if not task_file.exists():
            print("No tasks file found.")
            return False

        with open(task_file) as f:
            data = yaml.safe_load(f) or {"tasks": []}

        found = False
        for task in data.get("tasks", []):
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed_at"] = self._format_timestamp()
                if notes:
                    task["notes"] = notes
                found = True
                break

        if found:
            with open(task_file, "w") as f:
                yaml.dump(data, f)
            print(f"Completed task: {task_id}")
            return True
        else:
            print(f"Task {task_id} not found.")
            return False

    def record_decision(self, decision: str, rationale: str, alternatives: list[str] = None, experiment_id: str = None) -> bool:
        if experiment_id is None:
            experiment_id = self._get_current_experiment_id()
        if not experiment_id:
            print("No active experiment found.")
            return False

        paths = self._get_paths(experiment_id)
        decision_file = paths.get_context_file("decisions")

        if decision_file.exists():
            with open(decision_file) as f:
                data = yaml.safe_load(f) or {"decisions": []}
        else:
            data = {"decisions": []}

        decision_id = f"dec_{len(data.get('decisions', [])) + 1:03d}"
        new_decision = {
            "id": decision_id,
            "timestamp": self._format_timestamp(),
            "decision": decision,
            "rationale": rationale,
            "alternatives_considered": alternatives or [],
        }

        if "decisions" not in data:
            data["decisions"] = []
        data["decisions"].append(new_decision)

        with open(decision_file, "w") as f:
            yaml.dump(data, f)

        print(f"Recorded decision: {decision_id}")
        return True

    def log_insight(self, insight: str, category: str = "general", experiment_id: str = None) -> bool:
        if experiment_id is None:
            experiment_id = self._get_current_experiment_id()
        if not experiment_id:
            print("No active experiment found.")
            return False

        paths = self._get_paths(experiment_id)
        logs_dir = paths.get_logs_path()
        logs_dir.mkdir(exist_ok=True)

        insight_file = logs_dir / "insights.md"

        timestamp = datetime.datetime.now().isoformat()
        entry = f"\n### Insight [{timestamp}] ({category})\n{insight}\n"

        with open(insight_file, "a") as f:
            f.write(entry)

        print("Logged insight.")
        return True

    def record_incident_report(self, report_path: str, experiment_id: str = None) -> bool:
        """Record an incident report in the experiment state"""
        if experiment_id is None:
            experiment_id = self._get_current_experiment_id()
        if not experiment_id:
            print("No active experiment found.")
            return False

        try:
            state = self._load_state(experiment_id)
            # Ensure incident_reports array exists
            if "incident_reports" not in state:
                state["incident_reports"] = []

            # Add report path if not already present
            if report_path not in state["incident_reports"]:
                state["incident_reports"].append(report_path)
                self._save_state(experiment_id, state)
                print(f"Recorded incident report: {report_path}")
            else:
                print(f"Incident report already tracked: {report_path}")
            return True
        except Exception as e:
            print(f"Error recording incident report: {e}")
            return False
