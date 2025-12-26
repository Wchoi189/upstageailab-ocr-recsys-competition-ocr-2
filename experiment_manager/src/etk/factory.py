import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Union, List

from .schemas import ExperimentManifest, Task, TaskStatus, Insight, InsightType, Artifact, ExperimentStatus

class ExperimentFactory:
    def __init__(self, base_dir: Union[str, Path] = "experiments"):
        """
        Initialize the ExperimentFactory.

        Args:
            base_dir: The root directory where experiments will be stored.
                      Defaults to "experiments" in the current working directory.
        """
        self.base_dir = Path(base_dir).resolve()

    def _get_experiment_dir(self, experiment_id: str) -> Path:
        return self.base_dir / experiment_id

    def _get_manifest_path(self, experiment_id: str) -> Path:
        return self._get_experiment_dir(experiment_id) / "manifest.json"

    def _load_manifest(self, experiment_id: str) -> ExperimentManifest:
        manifest_path = self._get_manifest_path(experiment_id)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for experiment {experiment_id}")

        with open(manifest_path, 'r') as f:
            data = json.load(f)

        return ExperimentManifest(**data)

    def _save_manifest(self, manifest: ExperimentManifest):
        """
        Atomically save the manifest using a temporary file and os.replace.
        """
        experiment_dir = self._get_experiment_dir(manifest.experiment_id)
        manifest_path = self._get_manifest_path(manifest.experiment_id)

        # Ensure directory exists (it should, but safety first)
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Serialize with indentation for readability
        json_content = manifest.model_dump_json(indent=2)

        # Atomic write
        temp_path = manifest_path.with_suffix(".tmp")
        try:
            with open(temp_path, 'w') as f:
                f.write(json_content)
            os.replace(temp_path, manifest_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise IOError(f"Failed to save manifest for {manifest.experiment_id}: {e}")

    def init_experiment(self, name: str, type: str, intention: str = "") -> ExperimentManifest:
        """
        Initialize a new experiment.

        Args:
            name: Human readable name of the experiment.
            type: Type of the experiment (currently used for slug generation/logging).
            intention: Optional description of the experiment's goal.

        Returns:
            The created ExperimentManifest.
        """
        # Generate ID: YYYYMMDD_HHMMSS_slug
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = name.lower().replace(" ", "_")[:30] # Limit slug length
        # Sanitize slug
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        experiment_id = f"{timestamp_str}_{slug}"

        # Create directory
        experiment_dir = self._get_experiment_dir(experiment_id)
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create EDS v1.0 Structure
        (experiment_dir / ".metadata" / "assessments").mkdir(parents=True, exist_ok=True)
        (experiment_dir / ".metadata" / "reports").mkdir(parents=True, exist_ok=True)
        (experiment_dir / ".metadata" / "guides").mkdir(parents=True, exist_ok=True)
        (experiment_dir / ".metadata" / "scripts").mkdir(parents=True, exist_ok=True)
        (experiment_dir / "artifacts").mkdir(exist_ok=True)
        (experiment_dir / "outputs").mkdir(exist_ok=True)


        # Create initial manifest
        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            name=name,
            status=ExperimentStatus.ACTIVE,
            created_at=datetime.utcnow(),
            intention=intention
        )

        self._save_manifest(manifest)
        return manifest

    def add_task(self, experiment_id: str, description: str) -> Task:
        """
        Add a task to an existing experiment.
        """
        manifest = self._load_manifest(experiment_id)

        # Generate simple ID for task (e.g., hash or incremental)
        # Using incremental based on list length for simplicity and readability
        task_id = f"task_{len(manifest.tasks) + 1:03d}"

        new_task = Task(
            id=task_id,
            description=description,
            status=TaskStatus.BACKLOG
        )

        manifest.tasks.append(new_task)
        manifest.updated_at = datetime.utcnow()
        self._save_manifest(manifest)

        return new_task

    def record_artifact(self, experiment_id: str, path: str, type: str) -> Artifact:
        """
        Record an artifact metadata entry.
        Notes the file path and type in the manifest.
        """
        manifest = self._load_manifest(experiment_id)

        artifact = Artifact(
            path=path,
            type=type,
            timestamp=datetime.utcnow()
        )

        manifest.artifacts.append(artifact)
        manifest.updated_at = datetime.utcnow()
        self._save_manifest(manifest)

        return artifact
