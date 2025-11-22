import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from experiment_tracker.core import ExperimentTracker

tracker = ExperimentTracker()
experiment_id = tracker._get_current_experiment_id()
print(f"Importing into: {experiment_id}")

files_to_import = [
    # Scripts
    "../scripts/test_perspective_robust.py",
    "../scripts/test_perspective_on_rembg.py",
    "../scripts/test_perspective_doctr_rembg.py",
    "../scripts/optimized_rembg.py",
    "../scripts/QUICK_START_OPTIMIZED.md",
    "../scripts/test_pipeline_rembg_perspective.py",
    "../scripts/test_scantailor_integration.py",
    "../scripts/verify_gpu_usage.py",
    # Assessments
    "../docs/assessments/perspective_correction_failures_analysis.md",
    "../docs/assessments/rembg_gpu_setup_status.md",
    "../docs/assessments/rembg_gpu_verification.md",
    "../docs/assessments/rembg_optimization_implementation_summary.md",
    "../docs/assessments/rembg_optimization_test_plan.md",
    "../docs/assessments/rembg_performance_optimization.md",
    "../docs/assessments/rembg_pipeline_summary.md",
    "../docs/assessments/rembg_test_results_summary.md",
]

for file_path in files_to_import:
    path = Path(file_path)
    if path.exists():
        metadata = {"original_path": str(path)}
        if path.suffix == ".py":
            metadata["type"] = "script"
        elif path.suffix == ".md":
            metadata["type"] = "assessment"
        else:
            metadata["type"] = "unknown"

        print(f"Importing {path.name}...")
        tracker.record_artifact(str(path), metadata)
    else:
        print(f"File not found: {path}")

print("Import complete.")
