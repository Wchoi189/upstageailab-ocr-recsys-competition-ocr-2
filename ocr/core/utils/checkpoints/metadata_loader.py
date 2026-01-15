from __future__ import annotations
import yaml
from pathlib import Path

from .types import CheckpointMetadataV1

def save_metadata(metadata: CheckpointMetadataV1, checkpoint_path: Path) -> Path:
    """
    Save checkpoint metadata to a YAML file.

    Args:
        metadata: The metadata object to save.
        checkpoint_path: The corresponding checkpoint path.

    Returns:
        The path to the saved metadata file.
    """
    metadata_path = checkpoint_path.with_suffix('.metadata.yaml')

    # Convert pydantic model to dict
    data = metadata.model_dump(mode='json')

    with open(metadata_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    return metadata_path
