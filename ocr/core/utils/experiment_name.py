from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ocr.core.utils.config_utils import is_config

LOGGER = logging.getLogger(__name__)

_METADATA_SUFFIXES = (".metadata.yaml", ".metadata.yml", ".metadata.json")
_CONFIG_SIDECARE_SUFFIXES = (".config.json", ".config.yaml", ".config.yml", ".resolved.config.json")
_CONFIG_FILENAMES = ("config.yaml", "hparams.yaml")


def resolve_experiment_name(
    checkpoint_path: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    config_sources: Iterable[Mapping[str, Any] | None] = (),
) -> str | None:
    """Resolve experiment name for a checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file.
        metadata: Optional metadata mapping already loaded for this checkpoint.
        config_sources: Optional iterable of config mappings to inspect before reading from disk.

    Returns:
        Experiment name string if discovered, otherwise None.
    """
    checkpoint_path = Path(checkpoint_path).resolve()

    # 1. Prefer provided metadata/config sources to avoid redundant IO.
    first_pass_sources: list[Mapping[str, Any] | None] = [metadata]
    first_pass_sources.extend(list(config_sources or ()))

    for source in first_pass_sources:
        exp_name = _extract_exp_name(source)
        if exp_name:
            return exp_name

    # 2. Check on-disk metadata sidecars (.metadata.yaml/.json).
    metadata_from_disk = _load_metadata_sidecar(checkpoint_path)
    exp_name = _extract_exp_name(metadata_from_disk)
    if exp_name:
        return exp_name

    # 3. Inspect Hydra/config files located near the checkpoint.
    for config_data in _load_config_candidates(checkpoint_path):
        exp_name = _extract_exp_name(config_data)
        if exp_name:
            return exp_name

    # 4. Fallback to directory name (index-based directory if nothing else exists).
    return _exp_name_from_path(checkpoint_path)


def _extract_exp_name(source: Mapping[str, Any] | None) -> str | None:
    if source is None:
        return None

    try:
        value = source.get("exp_name")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive
        return None

    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def _load_metadata_sidecar(checkpoint_path: Path) -> dict[str, Any] | None:
    for suffix in _METADATA_SUFFIXES:
        candidate = checkpoint_path.with_suffix(suffix)
        data = _read_structured_file(candidate)
        if is_config(data):
            return data
    return None


def _load_config_candidates(checkpoint_path: Path) -> list[dict[str, Any]]:
    """Load config files associated with a checkpoint (without duplication)."""
    candidates: list[Path] = []
    seen: set[Path] = set()

    # Sidecar configs written alongside checkpoint files.
    for suffix in _CONFIG_SIDECARE_SUFFIXES:
        candidates.append(checkpoint_path.with_suffix(suffix))

    # Hydra configs stored in experiment directories (walk up to two parents).
    parents = list(checkpoint_path.parents)
    hydra_roots = [
        checkpoint_path.parent.parent,
        checkpoint_path.parent.parent.parent if len(parents) > 2 else None,
    ]
    for root in hydra_roots:
        if root is None:
            continue
        candidates.append(root / ".hydra" / "config.yaml")

    # Common config filenames inside checkpoint directories.
    for directory in (checkpoint_path.parent, checkpoint_path.parent.parent):
        for filename in _CONFIG_FILENAMES:
            candidates.append(directory / filename)

    loaded_configs: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)

        data = _read_structured_file(candidate)
        if is_config(data):
            loaded_configs.append(data)

    return loaded_configs


def _exp_name_from_path(checkpoint_path: Path) -> str | None:
    """Fallback: derive exp name from directory structure."""
    for parent in checkpoint_path.parents:
        if parent.name == "checkpoints":
            grandparent = parent.parent
            if grandparent.name:
                return grandparent.name
    # If we couldn't find a checkpoints parent, use immediate parent.
    if len(checkpoint_path.parents) >= 2:
        return checkpoint_path.parent.parent.name
    return checkpoint_path.parent.name


def _read_structured_file(path: Path) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return _cached_read_structured_file(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("Failed to read %s: %s", path, exc)
        return None


@lru_cache(maxsize=1024)
def _cached_read_structured_file(path_str: str) -> Any:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return json.load(handle)


def resolve_run_directory_experiment_name(run_dir: Path) -> str | None:
    """Resolve experiment name associated with a Hydra outputs directory."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        return None
    return _resolve_run_directory_experiment_name_cached(str(run_dir))


@lru_cache(maxsize=1024)
def _resolve_run_directory_experiment_name_cached(run_dir_str: str) -> str | None:
    run_dir = Path(run_dir_str)
    if not run_dir.exists():
        return None

    # 1. Check Hydra/config files within the run directory.
    config_candidates = [
        run_dir / ".hydra" / "config.yaml",
        run_dir / "config.yaml",
        run_dir / "hparams.yaml",
    ]
    for config_path in config_candidates:
        data = _read_structured_file(config_path)
        exp_name = _extract_exp_name(data if isinstance(data, Mapping) else None)
        if exp_name:
            return exp_name

    # 2. Fall back to scanning checkpoint metadata within the run directory.
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        for ckpt_path in sorted(checkpoints_dir.rglob("*.ckpt"), reverse=True):
            exp_name = resolve_experiment_name(ckpt_path)
            if exp_name:
                return exp_name

    return None


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def find_run_dirs_for_exp_name(exp_name: str | None, outputs_root: Path | str = Path("outputs")) -> list[Path]:
    """Locate Hydra output directories that correspond to an experiment name."""
    if not exp_name:
        return []

    outputs_root = Path(outputs_root).resolve()
    if not outputs_root.exists():
        return []

    candidate_dirs: list[Path] = []
    direct_path = outputs_root / exp_name
    if direct_path.exists():
        candidate_dirs.append(direct_path)

    for entry in outputs_root.iterdir():
        if entry.is_dir():
            candidate_dirs.append(entry)

    matches: list[tuple[float, Path]] = []
    seen: set[Path] = set()
    for run_dir in candidate_dirs:
        run_dir = run_dir.resolve()
        if run_dir in seen or not run_dir.exists():
            continue
        seen.add(run_dir)

        detected_name = resolve_run_directory_experiment_name(run_dir)
        if detected_name == exp_name or run_dir.name == exp_name:
            matches.append((_safe_mtime(run_dir), run_dir))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in matches]


def list_output_runs(outputs_root: Path | str = Path("outputs")) -> list[tuple[Path, str | None]]:
    """Enumerate output directories with their resolved experiment names."""
    outputs_root = Path(outputs_root).resolve()
    if not outputs_root.exists():
        return []

    entries: list[tuple[float, Path, str | None]] = []
    for entry in outputs_root.iterdir():
        if not entry.is_dir():
            continue
        exp_name = resolve_run_directory_experiment_name(entry)
        entries.append((_safe_mtime(entry), entry, exp_name))

    entries.sort(key=lambda item: item[0], reverse=True)
    return [(path, exp_name) for _, path, exp_name in entries]
