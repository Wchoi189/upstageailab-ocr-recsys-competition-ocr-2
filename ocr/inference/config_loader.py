from __future__ import annotations

"""Configuration helpers for OCR inference."""

import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dependencies import OCR_MODULES_AVAILABLE, PROJECT_ROOT, DictConfig, yaml

LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = (640, 640)
DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]
DEFAULT_BINARIZATION_THRESH = 0.3
DEFAULT_BOX_THRESH = 0.4
DEFAULT_MAX_CANDIDATES = 300
DEFAULT_MIN_DETECTION_SIZE = 5


@dataclass(slots=True)
class NormalizationSettings:
    mean: list[float]
    std: list[float]


@dataclass(slots=True)
class PreprocessSettings:
    image_size: tuple[int, int]
    normalization: NormalizationSettings
    enable_background_normalization: bool = False


@dataclass(slots=True)
class PostprocessSettings:
    binarization_thresh: float
    box_thresh: float
    max_candidates: int
    min_detection_size: int


@dataclass(slots=True)
class ModelConfigBundle:
    raw_config: Any
    preprocess: PreprocessSettings
    postprocess: PostprocessSettings


def resolve_config_path(checkpoint_path: str | Path, explicit_config: str | Path | None, search_dirs: Iterable[Path]) -> Path | None:
    """Return the configuration file path for a checkpoint."""

    if explicit_config is not None:
        config_path = Path(explicit_config)
        if config_path.exists():
            return config_path
        LOGGER.warning("Explicit config %s does not exist.", config_path)

    checkpoint_path = Path(checkpoint_path).resolve()

    # First, check for .config.json file alongside the checkpoint (new filesystem refactoring)
    config_json_path = checkpoint_path.with_suffix(".config.json")
    if config_json_path.exists():
        return config_json_path

    checkpoint_parent = checkpoint_path.parent
    candidates = list(search_dirs)
    candidates.extend([checkpoint_parent, checkpoint_parent.parent])

    for directory in candidates:
        for pattern in ("config.yaml", "hparams.yaml", "train.yaml", "predict.yaml"):
            candidate = directory / pattern
            if candidate.exists():
                return candidate

    # Check .hydra directories at different levels
    hydra_candidates = [
        checkpoint_parent.parent / ".hydra" / "config.yaml",  # checkpoints/.hydra/config.yaml
        checkpoint_parent.parent.parent / ".hydra" / "config.yaml",  # experiment/.hydra/config.yaml
    ]
    for candidate in hydra_candidates:
        if candidate.exists():
            return candidate

    return None


def load_model_config(config_path: str | Path) -> ModelConfigBundle:
    """Load and parse a model configuration from disk."""

    path = Path(config_path)
    LOGGER.info("Using config file: %s", path)

    def load_config_from_path(config_path: str | Path):
        """
        Load Hydra config from full path by extracting config name.
        Uses ocr/utils/config_utils.py::load_config() which properly
        resolves defaults and @package directives.
        """
        path = Path(config_path).resolve()
        config_name = path.stem
        try:
            from ocr.utils.config_utils import load_config as load_hydra_config

            cfg = load_hydra_config(config_name=config_name)
            return cfg
        except Exception as e:
            LOGGER.error(f"Failed to load config from {config_path}: {e}")
            raise

    # Check if this is a .config.json file (pre-generated checkpoint config)
    # These should be loaded directly as JSON, not through Hydra
    if path.suffix == ".json" and ".config.json" in path.name:
        LOGGER.info("Loading pre-generated checkpoint config from JSON file")
        with path.open("r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        config_container = DictConfig(config_dict)
    elif OCR_MODULES_AVAILABLE:
        config_container = load_config_from_path(path)
    else:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix in {".yaml", ".yml"}:
                if yaml is None:
                    raise RuntimeError("PyYAML is not available to parse configuration files.")
                config_dict = yaml.safe_load(handle)
            else:
                config_dict = json.load(handle)
        config_container = DictConfig(config_dict)

    preprocess_settings = _extract_preprocess_settings(config_container)
    postprocess_settings = _extract_postprocess_settings(config_container)

    return ModelConfigBundle(
        raw_config=config_container,
        preprocess=preprocess_settings,
        postprocess=postprocess_settings,
    )


def _extract_preprocess_settings(config: Any) -> PreprocessSettings:
    image_size = DEFAULT_IMAGE_SIZE
    mean = DEFAULT_NORMALIZE_MEAN.copy()
    std = DEFAULT_NORMALIZE_STD.copy()
    enable_background_normalization = False

    preprocessing = _get_attr(config, "preprocessing")
    if preprocessing:
        if target_size := _coerce_tuple(_get_attr(preprocessing, "target_size")):
            image_size = target_size
        # Read background normalization flag from config
        bg_norm_value = _get_attr(preprocessing, "enable_background_normalization")
        if bg_norm_value is not None:
            enable_background_normalization = bool(bg_norm_value)
    
    if not preprocessing or not _coerce_tuple(_get_attr(preprocessing, "target_size")):
        if transforms_section := _get_attr(config, "transforms"):
            transform_key = "predict_transform" if _has_attr(transforms_section, "predict_transform") else "test_transform"
            if transform_config := _get_attr(transforms_section, transform_key):
                transforms_list = _get_attr(transform_config, "transforms") or []
                for transform in transforms_list:
                    max_size = _get_attr(transform, "max_size")
                    min_width = _get_attr(transform, "min_width")
                    min_height = _get_attr(transform, "min_height")
                    if max_size:
                        image_size = (int(max_size), int(max_size))
                        break
                    if min_width and min_height:
                        image_size = (int(min_width), int(min_height))
                        break

                for transform in transforms_list:
                    mean_candidate = _get_attr(transform, "mean")
                    std_candidate = _get_attr(transform, "std")
                    if mean_candidate and std_candidate:
                        mean = [float(value) for value in _as_sequence(mean_candidate)]
                        std = [float(value) for value in _as_sequence(std_candidate)]
                        break

    normalization = NormalizationSettings(mean=mean, std=std)
    return PreprocessSettings(
        image_size=image_size,
        normalization=normalization,
        enable_background_normalization=enable_background_normalization,
    )


def _extract_postprocess_settings(config: Any) -> PostprocessSettings:
    binarization = DEFAULT_BINARIZATION_THRESH
    box_thresh = DEFAULT_BOX_THRESH
    max_candidates = DEFAULT_MAX_CANDIDATES

    head_config = _get_attr(_get_attr(config, "model"), "head")
    postprocess = _get_attr(head_config, "postprocess") if head_config else None
    if postprocess:
        thresh = _get_attr(postprocess, "thresh")
        box = _get_attr(postprocess, "box_thresh")
        max_cands = _get_attr(postprocess, "max_candidates")
        if thresh is not None:
            binarization = float(thresh)
        if box is not None:
            box_thresh = float(box)
        if max_cands is not None:
            max_candidates = int(max_cands)

    return PostprocessSettings(
        binarization_thresh=binarization,
        box_thresh=box_thresh,
        max_candidates=max_candidates,
        min_detection_size=DEFAULT_MIN_DETECTION_SIZE,
    )


def _has_attr(obj: Any, attr: str) -> bool:
    return hasattr(obj, attr) or (isinstance(obj, dict) and attr in obj)


def _get_attr(obj: Any, attr: str, default: Any | None = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, attr):
        return getattr(obj, attr)
    return obj.get(attr, default) if isinstance(obj, dict) else default


def _coerce_tuple(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    sequence = _as_sequence(value)
    return (int(sequence[0]), int(sequence[1])) if len(sequence) >= 2 else None


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, list | tuple):
        return value
    if hasattr(value, "__iter__") and not isinstance(value, str | bytes):
        return list(value)
    return [value]


def _determine_hydra_location(config_file: Path) -> tuple[Path, str]:
    """Return the Hydra search directory and config name for the given config file."""
    config_file = config_file.resolve()
    project_configs_dir = (PROJECT_ROOT / "configs").resolve()

    if project_configs_dir in config_file.parents:
        relative_path = config_file.relative_to(project_configs_dir)
        config_dir = project_configs_dir
        config_name = relative_path.with_suffix("")
    else:
        config_dir = config_file.parent
        config_name = config_file.with_suffix("").name

    return config_dir, _normalize_hydra_config_name(config_name)


def _normalize_hydra_config_name(config_name: str | Path) -> str:
    """Ensure Hydra config names always use forward slashes."""
    path_obj = Path(config_name)
    return path_obj.as_posix()


def _compute_relative_hydra_path(target_dir: Path) -> str:
    """Compute a relative path acceptable for hydra.initialize()."""
    try:
        relative = target_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        relative = Path(os.path.relpath(target_dir, Path.cwd()))
    normalized = relative.as_posix()
    return normalized or "."
