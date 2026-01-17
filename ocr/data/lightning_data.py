# ocr/data/lightning_data.py
# Domain-agnostic PyTorch Lightning DataModule for OCR
# Extracted from ocr/core/lightning/ocr_pl.py during Phase 2 surgical refactor

from typing import Any

import lightning.pytorch as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader


class OCRDataPLModule(pl.LightningDataModule):
    """Domain-agnostic Lightning DataModule for OCR datasets.

    Handles train/val/test/predict dataloaders with configurable collate functions.
    """

    def __init__(self, dataset, config):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.dataloaders_cfg = self.config.dataloaders
        self.collate_cfg = self.config.collate_fn

    def _build_collate_fn(self, *, inference_mode: bool) -> Any:
        # Create collate function (no longer using cache - using pre-processed maps instead)
        collate_fn = instantiate(self.collate_cfg)
        if hasattr(collate_fn, "inference_mode"):
            collate_fn.inference_mode = inference_mode
        return collate_fn

    def train_dataloader(self):
        train_loader_config = self.dataloaders_cfg.train_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if train_loader_config.get("num_workers", 0) == 0:
            train_loader_config = {k: v for k, v in train_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=False)
        return DataLoader(self.dataset["train"], collate_fn=collate_fn, **train_loader_config)

    def val_dataloader(self):
        val_loader_config = self.dataloaders_cfg.val_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if val_loader_config.get("num_workers", 0) == 0:
            val_loader_config = {k: v for k, v in val_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=False)
        return DataLoader(self.dataset["val"], collate_fn=collate_fn, **val_loader_config)

    def test_dataloader(self):
        test_loader_config = self.dataloaders_cfg.test_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if test_loader_config.get("num_workers", 0) == 0:
            test_loader_config = {k: v for k, v in test_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=False)
        return DataLoader(self.dataset["test"], collate_fn=collate_fn, **test_loader_config)

    def predict_dataloader(self):
        predict_loader_config = self.dataloaders_cfg.predict_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if predict_loader_config.get("num_workers", 0) == 0:
            predict_loader_config = {k: v for k, v in predict_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=True)
        return DataLoader(self.dataset["predict"], collate_fn=collate_fn, **predict_loader_config)
