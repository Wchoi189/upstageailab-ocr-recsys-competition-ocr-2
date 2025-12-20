"""Checkpoint discovery and management service."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Checkpoint(BaseModel):
    """Checkpoint metadata for UI selection."""

    checkpoint_path: str
    display_name: str
    size_mb: float
    modified_at: str
    epoch: int | None = None
    global_step: int | None = None
    precision: float | None = None
    recall: float | None = None
    hmean: float | None = None


class CheckpointService:
    """Service for discovering and caching checkpoint metadata.

    Provides TTL-based caching to prevent redundant disk I/O.
    """

    def __init__(self, checkpoint_root: Path, cache_ttl: float = 5.0):
        """Initialize checkpoint service.

        Args:
            checkpoint_root: Root directory for checkpoint discovery
            cache_ttl: Time-to-live for cache in seconds (default: 5.0)
        """
        self.checkpoint_root = checkpoint_root
        self.cache_ttl = cache_ttl
        self._cache: list[Checkpoint] | None = None
        self._last_update: datetime | None = None

    async def list_checkpoints(self, limit: int = 100) -> list[Checkpoint]:
        """List available checkpoints with TTL caching.

        Args:
            limit: Maximum number of checkpoints to return

        Returns:
            List of discovered checkpoints
        """
        current_time = datetime.utcnow()

        # Check cache validity
        if (self._cache is not None and
            self._last_update is not None and
            (current_time - self._last_update).total_seconds() < self.cache_ttl):
            return self._cache[:limit]

        # Cache miss - rediscover
        checkpoints = self._discover_sync(limit=limit)
        self._cache = checkpoints
        self._last_update = current_time

        return checkpoints

    def get_latest(self) -> Checkpoint | None:
        """Get the most recent checkpoint.

        Returns:
            Latest checkpoint or None if no checkpoints found
        """
        if self._cache:
            return self._cache[0] if self._cache else None

        # Synchronous discovery if cache empty
        checkpoints = self._discover_sync(limit=1)
        return checkpoints[0] if checkpoints else None

    async def preload_checkpoints(self, limit: int = 100) -> None:
        """Preload checkpoint metadata cache in the background.

        This method discovers and caches checkpoint metadata asynchronously
        to warm up the cache before the first API request.

        Args:
            limit: Maximum number of checkpoints to discover
        """
        logger.info("🔄 Preloading checkpoint metadata cache...")
        checkpoints = self._discover_sync(limit=limit)
        self._cache = checkpoints
        self._last_update = datetime.utcnow()
        logger.info("✅ Checkpoint cache preloaded | count=%d", len(checkpoints))

    def _discover_sync(self, limit: int) -> list[Checkpoint]:
        """Synchronously discover checkpoints using pregenerated metadata YAML files.

        Args:
            limit: Maximum number of checkpoints to discover

        Returns:
            List of discovered checkpoints sorted by modification time (newest first)
        """
        import yaml  # Lazy import

        if not self.checkpoint_root.exists():
            logger.warning("Checkpoint root missing: %s", self.checkpoint_root)
            return []

        # Find all pregenerated metadata files
        metadata_files = sorted(
            self.checkpoint_root.rglob("*.ckpt.metadata.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        start_time = datetime.utcnow()
        metadata_count = len(metadata_files)
        results: list[Checkpoint] = []

        for meta_path in metadata_files[:limit]:
            try:
                # Parse YAML metadata (fast - no state dict loading)
                with open(meta_path) as f:
                    meta = yaml.safe_load(f)

                # Reconstruct checkpoint path from metadata file path
                ckpt_path = meta_path.with_suffix("").with_suffix("")  # Remove .metadata.yaml

                # Get file stats
                stat = meta_path.stat()
                ckpt_stat = ckpt_path.stat() if ckpt_path.exists() else stat

                display_name = str(ckpt_path.relative_to(self.checkpoint_root))

                results.append(
                    Checkpoint(
                        checkpoint_path=str(ckpt_path),
                        display_name=display_name,
                        size_mb=round(ckpt_stat.st_size / (1024 * 1024), 2) if ckpt_path.exists() else 0.0,
                        modified_at=meta.get("created_at", datetime.fromtimestamp(stat.st_mtime).isoformat()),
                        epoch=meta.get("training", {}).get("epoch"),
                        global_step=meta.get("training", {}).get("global_step"),
                        precision=meta.get("metrics", {}).get("precision"),
                        recall=meta.get("metrics", {}).get("recall"),
                        hmean=meta.get("metrics", {}).get("hmean"),
                    )
                )
            except Exception as e:
                logger.warning("Failed to parse metadata file %s: %s", meta_path, e)
                continue

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.debug(
            "Checkpoint discovery complete | metadata_found=%d | metadata_files_scanned=%d | limit=%d | elapsed=%.3fs | root=%s",
            len(results),
            metadata_count,
            limit,
            elapsed,
            self.checkpoint_root,
        )

        if not metadata_files:
            logger.warning(
                "No checkpoint metadata files were found. Run 'make checkpoint-metadata' to generate .ckpt.metadata.yaml files for fast discovery."
            )

        return results
