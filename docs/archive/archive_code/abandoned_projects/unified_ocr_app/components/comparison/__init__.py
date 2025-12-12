"""Comparison mode components for the unified OCR app.

This module provides UI components for A/B testing and parameter sweep functionality.
Components support comparing preprocessing configurations, inference settings, and
end-to-end pipeline performance.
"""

from .metrics_display import render_metrics_display
from .parameter_sweep import render_parameter_sweep
from .results_comparison import render_results_comparison

__all__ = [
    "render_parameter_sweep",
    "render_results_comparison",
    "render_metrics_display",
]
