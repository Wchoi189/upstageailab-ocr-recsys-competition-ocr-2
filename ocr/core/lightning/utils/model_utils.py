from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def load_state_dict_with_fallback(
    model: torch.nn.Module, 
    state_dict: Mapping[str, Any], 
    strict: bool = True
) -> tuple[list[str], list[str]]:
    """Load state dict with torch.compile prefix handling ONLY.
    
    Supports:
    1. Standard loading (strict=True)
    2. torch.compile "._orig_mod." prefix removal
    
    NO LEGACY CHECKPOINT FORMATS. Migrate old checkpoints explicitly.
    Use scripts/checkpoints/convert_legacy_checkpoints.py for old formats.
    
    Args:
        model: The model to load state into
        state_dict: The state dictionary to load
        strict: Whether to strictly enforce key matching
    
    Returns:
        Tuple of (missing_keys, unexpected_keys)
        
    Raises:
        RuntimeError: If checkpoint is incompatible with model architecture
    """
    # Try 1: Direct load
    try:
        result = model.load_state_dict(state_dict, strict=strict)
        return result.missing_keys, result.unexpected_keys
    except RuntimeError as e:
        error_str = str(e)
        # Check if this is a torch.compile prefix issue
        has_orig_mod = any("._orig_mod." in key for key in state_dict.keys())
        if not has_orig_mod and "_orig_mod" not in error_str:
            # Not a torch.compile issue, fail fast
            raise RuntimeError(
                f"Checkpoint incompatible with current model architecture.\\n"
                f"Original error: {e}\\n"
                f"For legacy checkpoints, use: scripts/checkpoints/convert_legacy_checkpoints.py"
            ) from e
    
    # Try 2: torch.compile prefix handling ONLY
    modified_state_dict = {
        key.replace("._orig_mod.", "."): value 
        for key, value in state_dict.items()
    }
    
    result = model.load_state_dict(modified_state_dict, strict=strict)
    return result.missing_keys, result.unexpected_keys
