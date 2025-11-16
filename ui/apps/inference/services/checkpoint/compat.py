"""Compatibility layer for migrating from legacy checkpoint_catalog to V2.

This module provides compatibility functions that maintain the legacy API
while using the V2 catalog system internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...models.checkpoint import CheckpointInfo
from .catalog import CheckpointCatalogBuilder, build_catalog
from .types import CheckpointCatalogEntry


@dataclass(slots=True)
class CatalogOptions:
    """Legacy CatalogOptions for backward compatibility.

    This is a compatibility shim that matches the legacy API.
    New code should use build_catalog() directly with parameters.
    """

    outputs_dir: Path
    hydra_config_filenames: tuple[str, ...]

    @classmethod
    def from_paths(cls, paths) -> CatalogOptions:
        """Create CatalogOptions from PathConfig (legacy compatibility)."""
        outputs_dir = paths.outputs_dir
        if not outputs_dir.is_absolute():
            # Try to discover outputs path relative to project root
            from ocr.utils.path_utils import get_path_resolver

            project_root = get_path_resolver().config.project_root
            outputs_dir = project_root / outputs_dir

        return cls(
            outputs_dir=outputs_dir,
            hydra_config_filenames=tuple(paths.hydra_config_filenames),
        )


def build_lightweight_catalog(options: CatalogOptions) -> list[CheckpointInfo]:
    """Build lightweight checkpoint catalog (legacy API compatibility).

    This function maintains the legacy API while using V2 catalog system internally.
    It converts V2 CheckpointCatalogEntry objects to legacy CheckpointInfo objects.

    Args:
        options: Catalog building options (legacy format)

    Returns:
        List of CheckpointInfo objects compatible with UI
    """
    # Use V2 catalog builder
    builder = CheckpointCatalogBuilder(
        outputs_dir=options.outputs_dir,
        use_cache=True,
        use_wandb_fallback=True,
        config_filenames=options.hydra_config_filenames,
    )

    catalog = builder.build_catalog()

    # Convert V2 entries to legacy CheckpointInfo
    infos = [_convert_v2_entry_to_checkpoint_info(entry) for entry in catalog.entries]

    return infos


def _convert_v2_entry_to_checkpoint_info(entry: CheckpointCatalogEntry) -> CheckpointInfo:
    """Convert V2 CheckpointCatalogEntry to legacy CheckpointInfo.

    This adapter ensures backward compatibility with the UI while using
    the new V2 catalog system internally.

    Args:
        entry: V2 catalog entry

    Returns:
        Legacy CheckpointInfo object
    """
    return CheckpointInfo(
        checkpoint_path=entry.checkpoint_path,
        config_path=entry.config_path,
        display_name=entry.display_name,
        exp_name=entry.exp_name,
        epochs=entry.epochs,
        created_timestamp=entry.created_timestamp,
        hmean=entry.hmean,
        architecture=entry.architecture,
        backbone=entry.backbone,
        monitor=entry.monitor,
        monitor_mode=entry.monitor_mode,
    )

