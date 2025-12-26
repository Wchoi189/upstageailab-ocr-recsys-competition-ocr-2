import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from experiment_manager.core import ExperimentTracker

tracker = ExperimentTracker()
experiment_id = tracker._get_current_experiment_id()
experiment_dir = tracker.experiments_dir / experiment_id

print(f"Reorganizing experiment: {experiment_id}")

# Create scripts directory
scripts_dir = experiment_dir / "scripts"
scripts_dir.mkdir(exist_ok=True)

# Load state
state = tracker._load_state(experiment_id)
new_artifacts = []
new_assessments = []

# Iterate over artifacts and move them
for artifact in state["artifacts"]:
    path = Path(artifact["path"])
    full_path = experiment_dir / path

    if not full_path.exists():
        print(f"Warning: {full_path} does not exist")
        continue

    if full_path.suffix == ".py":
        # Move to scripts
        dest = scripts_dir / full_path.name
        shutil.move(str(full_path), str(dest))
        print(f"Moved {full_path.name} to scripts/")
        # We don't track scripts in 'artifacts' list usually, or we track them with type 'script'
        # But let's keep them in artifacts list but update path?
        # Or maybe we should have a separate 'scripts' list in state?
        # The schema has 'artifacts' and 'assessments'.
        # Let's keep them in artifacts but update path.
        artifact["path"] = f"scripts/{full_path.name}"
        artifact["type"] = "script"
        new_artifacts.append(artifact)

    elif full_path.suffix == ".md":
        # Move to assessments if it looks like an assessment
        # The user said docs/assessments/* are insights.
        dest_dir = experiment_dir / "assessments"
        dest = dest_dir / full_path.name
        shutil.move(str(full_path), str(dest))
        print(f"Moved {full_path.name} to assessments/")

        # Add to assessments list (which is just strings of paths or content?)
        # Schema says: "assessments": { "type": "array", "items": { "type": "string" } }
        # It seems assessments list is just filenames or paths.
        new_assessments.append(f"assessments/{full_path.name}")

        # Also keep in artifacts? Maybe not if it's in assessments.
        # But 'artifacts' is the main record of files.
        # Let's remove from artifacts if it's an assessment to avoid duplication in concept,
        # or keep it as an artifact of type 'assessment'.
        # Let's keep it in artifacts for now to be safe, but update path.
        artifact["path"] = f"assessments/{full_path.name}"
        artifact["type"] = "assessment"
        new_artifacts.append(artifact)
    else:
        new_artifacts.append(artifact)

state["artifacts"] = new_artifacts
# Update assessments list
state["assessments"] = list(set(state["assessments"] + new_assessments))

tracker._save_state(experiment_id, state)
print("Reorganization complete.")
