"""Utility for managing sequential experiment indices for output directory naming.

This module provides thread-safe, file-based index management to ensure
unique, sequential experiment numbers for better organization and sorting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock

LOGGER = logging.getLogger(__name__)

# Global lock for thread-safe index management
_index_lock = Lock()

# Default path for storing the index counter
_DEFAULT_INDEX_FILE = Path("outputs/.experiment_index.json")


def get_next_experiment_index(index_file: Path | None = None) -> int:
    """Get the next sequential experiment index.

    This function is thread-safe and uses file-based locking to ensure
    unique indices across multiple processes.

    Args:
        index_file: Path to the index file. If None, uses default location.

    Returns:
        The next available experiment index (1-indexed).
    """
    if index_file is None:
        index_file = _DEFAULT_INDEX_FILE

    index_file = Path(index_file)
    index_file.parent.mkdir(parents=True, exist_ok=True)

    with _index_lock:
        # Read current index
        current_index = 0
        if index_file.exists():
            try:
                with open(index_file) as f:
                    data = json.load(f)
                    current_index = data.get("last_index", 0)
            except (json.JSONDecodeError, KeyError, OSError) as e:
                LOGGER.warning("Failed to read index file %s: %s. Starting from 1.", index_file, e)
                current_index = 0

        # Increment and save
        next_index = current_index + 1
        try:
            with open(index_file, "w") as f:
                json.dump({"last_index": next_index}, f, indent=2)
        except OSError as e:
            LOGGER.warning("Failed to write index file %s: %s. Using index %d.", index_file, e, next_index)

        return next_index


def get_current_experiment_index(index_file: Path | None = None) -> int:
    """Get the current experiment index without incrementing.

    Args:
        index_file: Path to the index file. If None, uses default location.

    Returns:
        The current experiment index (0 if no index file exists).
    """
    if index_file is None:
        index_file = _DEFAULT_INDEX_FILE

    index_file = Path(index_file)
    if not index_file.exists():
        return 0

    try:
        with open(index_file) as f:
            data = json.load(f)
            return data.get("last_index", 0)
    except (json.JSONDecodeError, KeyError, OSError):
        return 0


def reset_experiment_index(index_file: Path | None = None) -> None:
    """Reset the experiment index counter to 0.

    Args:
        index_file: Path to the index file. If None, uses default location.
    """
    if index_file is None:
        index_file = _DEFAULT_INDEX_FILE

    index_file = Path(index_file)
    index_file.parent.mkdir(parents=True, exist_ok=True)

    with _index_lock:
        try:
            with open(index_file, "w") as f:
                json.dump({"last_index": 0}, f, indent=2)
        except OSError as e:
            LOGGER.warning("Failed to reset index file %s: %s", index_file, e)
