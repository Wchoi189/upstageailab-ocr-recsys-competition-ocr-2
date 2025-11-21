from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ocr.utils.experiment_name import find_run_dirs_for_exp_name, resolve_run_directory_experiment_name


@dataclass(frozen=True)
class SubmissionRun:
    run_dir: Path
    exp_name: str | None
    submissions_dir: Path
    modified_time: float


def _safe_stat_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def discover_submission_runs(outputs_root: Path = Path("outputs")) -> list[SubmissionRun]:
    """List submission-capable runs sorted by most recent activity."""
    outputs_root = Path(outputs_root)
    if not outputs_root.exists():
        return []

    runs: list[SubmissionRun] = []
    for candidate in outputs_root.iterdir():
        if not candidate.is_dir():
            continue
        submissions_dir = candidate / "submissions"
        if not submissions_dir.exists():
            continue

        modified = _safe_stat_mtime(submissions_dir) or _safe_stat_mtime(candidate)
        runs.append(
            SubmissionRun(
                run_dir=candidate,
                exp_name=resolve_run_directory_experiment_name(candidate),
                submissions_dir=submissions_dir,
                modified_time=modified,
            )
        )

    runs.sort(key=lambda run: run.modified_time, reverse=True)
    return runs


def gather_submission_files_for_exp(
    exp_name: str | None,
    *,
    outputs_root: Path = Path("outputs"),
) -> list[Path]:
    """Collect submission files for the given experiment across matching run directories."""
    if not exp_name:
        return []

    outputs_root = Path(outputs_root)
    if not outputs_root.exists():
        return []

    files: list[tuple[float, Path]] = []
    seen: set[Path] = set()

    run_dirs = find_run_dirs_for_exp_name(exp_name, outputs_root)
    if not run_dirs:
        # Allow users to pass the raw directory name if metadata is missing.
        direct_dir = outputs_root / exp_name
        if direct_dir.exists():
            run_dirs = [direct_dir]

    for run_dir in run_dirs:
        submissions_dir = run_dir / "submissions"
        if not submissions_dir.exists():
            continue
        for json_file in submissions_dir.glob("*.json"):
            resolved = json_file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append((_safe_stat_mtime(json_file), resolved))

    files.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in files]


def find_latest_submission_json(exp_name: str | None, *, outputs_root: Path = Path("outputs")) -> Path | None:
    """Return the newest submission JSON path for an experiment, if available."""
    files = gather_submission_files_for_exp(exp_name, outputs_root=outputs_root)
    return files[0] if files else None
