"""Callback helpers for runner entrypoints.

Centralizes Hydra-driven callback instantiation and resolved-config attachment
for callbacks that expect `_resolved_config`.
"""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


def _is_enabled(cb_conf: DictConfig) -> bool:
    """Return True if callback config is enabled (default True)."""
    return cb_conf.get("enabled", True) not in (False, "false", "False")


def build_callbacks(config: DictConfig) -> list[Any]:
    """Instantiate callbacks defined under `config.callbacks`.

    - Skips entries without `_target_` or disabled callbacks.
    - Attaches resolved config to callbacks exposing `_resolved_config` for
      checkpoint metadata compatibility.
    """
    callbacks: list[Any] = []

    callbacks_conf = config.get("callbacks")
    if not callbacks_conf:
        return callbacks

    for _, cb_conf in callbacks_conf.items():
        if not isinstance(cb_conf, DictConfig):
            continue
        if "_target_" not in cb_conf:
            continue
        if not _is_enabled(cb_conf):
            continue

        callback = hydra.utils.instantiate(cb_conf)

        if hasattr(callback, "_resolved_config"):
            resolved_config = OmegaConf.to_container(config, resolve=True)
            callback._resolved_config = resolved_config

        callbacks.append(callback)

    return callbacks
