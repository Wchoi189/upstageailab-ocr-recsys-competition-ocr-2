"""Unified OCR Development Studio.

A single Streamlit app combining preprocessing tuning, model inference,
and A/B comparison modes.

Architecture:
    - components/: UI components (rendering only)
    - services/: Business logic (preprocessing, inference, config)
    - models/: Data contracts (Pydantic models)
    - utils/: Shared utilities

Configuration:
    - All UI behavior driven by YAML files in configs/ui/
    - Schema validation via docs/schemas/
    - Mode-specific configs in configs/ui/modes/
"""

__version__ = "1.0.0"
