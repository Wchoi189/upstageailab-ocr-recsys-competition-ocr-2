#!/usr/bin/env python
"""CLI for priming dataset caches and running preprocessing transforms.

This script replaces the legacy OCRDataset-based preprocessing entrypoint with a
Hydra-driven workflow. It instantiates a ``ValidatedOCRDataset`` directly from a
``DatasetConfig`` structure, ensuring parity with the runtime configuration used
by the training and inference pipelines.

Usage example (processing the training split):

    uv run python scripts/data/preprocess_data.py dataset_key=train_dataset limit=100

The script will iterate through the requested dataset, materialising any cache
artifacts configured in ``DatasetConfig.cache_config`` (e.g., tensor cache,
preloaded images) while exercising the full transform stack.
"""

from scripts.data.preprocess import main  # re-export entrypoint for backward compatibility

if __name__ == "__main__":
    main()
