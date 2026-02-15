"""OCR datasets package.

This package is imported by many modules (including lightweight schema shims).
To keep import-time dependencies minimal, heavy submodules are loaded lazily.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from ocr.core.utils.config_utils import ensure_dict

__all__ = [
    "ValidatedOCRDataset",
    "LensStylePreprocessorAlbumentations",
    "get_datasets_by_cfg",
]

logger = logging.getLogger(__name__)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ValidatedOCRDataset": ("ocr.domains.detection.data.dataset", "ValidatedOCRDataset"),
    "LensStylePreprocessorAlbumentations": ("ocr.domains.detection.data.preprocessing", "LensStylePreprocessorAlbumentations"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


def get_datasets_by_cfg(datasets_config, data_config=None, full_config=None, splits=None):
    """
    Create datasets from Hydra configuration.

    Args:
        datasets_config: Hydra config containing dataset definitions
        data_config: Optional data configuration for dataset limiting
        full_config: Optional full configuration for custom paths
        splits: Optional list of dataset splits to create. If None, creates all.
                Valid values: ["train", "val", "test", "predict"]

    Returns:
        Dictionary mapping split names to Dataset instances (or None for unused splits)
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import Subset

    # Default: create all splits (backward compatible)
    if splits is None:
        splits = ["train", "val", "test", "predict"]

    logger.info(f"Creating datasets for splits: {splits}")

    # Pre-instantiate tokenizer once using get_or_create() to leverage caching
    # This avoids redundant tokenizer loading for each dataset split
    tokenizer_instance = None
    if hasattr(datasets_config, "tokenizer") and datasets_config.tokenizer is not None:
        from ocr.domains.recognition.data.tokenizer import KoreanOCRTokenizer
        tokenizer_cfg = datasets_config.tokenizer
        tokenizer_instance = KoreanOCRTokenizer.get_or_create(
            charset_path=tokenizer_cfg.charset_path,
            max_len=tokenizer_cfg.max_len
        )

    # Helper to instantiate dataset with cached tokenizer
    def instantiate_dataset(dataset_cfg):
        if tokenizer_instance is not None:
            # Use Hydra's instantiate with tokenizer override
            # This bypasses OmegaConf's restriction on non-primitive types
            return instantiate(dataset_cfg, tokenizer=tokenizer_instance)
        return instantiate(dataset_cfg)

    # Lazy instantiation: only create requested splits
    datasets = {}

    if "train" in splits:
        datasets["train"] = instantiate_dataset(datasets_config.train_dataset)
    else:
        datasets["train"] = None

    if "val" in splits:
        datasets["val"] = instantiate_dataset(datasets_config.val_dataset)
    else:
        datasets["val"] = None

    if "test" in splits:
        datasets["test"] = instantiate_dataset(datasets_config.test_dataset)
    else:
        datasets["test"] = None

    # Handle custom image directory for prediction
    if "predict" in splits:
        if full_config is not None and hasattr(full_config, "image_dir") and full_config.image_dir is not None:
            # Override predict_dataset to use custom image directory
            predict_config = OmegaConf.create(ensure_dict(datasets_config.predict_dataset, resolve=True))
            predict_config.config.image_path = full_config.image_dir
            if tokenizer_instance is not None:
                datasets["predict"] = instantiate(predict_config, tokenizer=tokenizer_instance)
            else:
                datasets["predict"] = instantiate(predict_config)
        else:
            datasets["predict"] = instantiate_dataset(datasets_config.predict_dataset)
    else:
        datasets["predict"] = None

    # Apply dataset limiting if configured
    if data_config is not None:
        try:
            if datasets["train"] is not None and getattr(data_config, "train_num_samples", None) is not None:
                train_limit = min(data_config.train_num_samples, len(datasets["train"]))
                datasets["train"] = Subset(datasets["train"], range(train_limit))

            if datasets["val"] is not None and getattr(data_config, "val_num_samples", None) is not None:
                val_limit = min(data_config.val_num_samples, len(datasets["val"]))
                datasets["val"] = Subset(datasets["val"], range(val_limit))

            if datasets["test"] is not None and getattr(data_config, "test_num_samples", None) is not None:
                test_limit = min(data_config.test_num_samples, len(datasets["test"]))
                datasets["test"] = Subset(datasets["test"], range(test_limit))
        except (AttributeError, KeyError):
            # If config access fails, use full datasets
            pass

    return datasets
