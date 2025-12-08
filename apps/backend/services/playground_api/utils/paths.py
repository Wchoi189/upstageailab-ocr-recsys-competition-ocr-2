"""Path utilities shared by the playground API modules.

This module provides a stable PROJECT_ROOT import that works from any location.
All path resolution should use this centralized definition.
"""

from __future__ import annotations

# Import PROJECT_ROOT from central path utility (stable, works from any location)

# Re-export for backward compatibility with existing imports
__all__ = ["PROJECT_ROOT"]
