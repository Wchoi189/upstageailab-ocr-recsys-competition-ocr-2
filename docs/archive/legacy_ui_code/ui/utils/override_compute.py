"""
Pure functions for computing Hydra overrides from schema values.

This module is Streamlit-free and can be used by both the FastAPI API
and the Streamlit UI without triggering Streamlit initialization.
"""

from __future__ import annotations

from typing import Any


def _to_override(k: str, v: Any) -> str | None:
    """Convert a key-value pair to a Hydra override string.

    Args:
        k: The override key
        v: The override value

    Returns:
        Hydra override string like "key=value" or "key=\"value\"", or None if value is empty
    """
    if v is None or v == "":
        return None

    v_str = str(v).lower() if isinstance(v, bool) else str(v)

    # Quote values containing special characters that confuse Hydra parser
    # Characters that need quoting: , = : { } [ ] space tab newline return ' "
    # According to Hydra docs, these chars need escaping or the value needs to be quoted
    special_chars = ",=:{}[] \t\n\r'\""
    needs_quotes = any(char in v_str for char in special_chars)

    # For interpolations or special chars, quote the VALUE only (not the entire override)
    # Hydra expects: key="value" (where value is quoted)
    # The shell quoting 'key="value"' is handled separately when displaying the command
    contains_interpolation = "${" in v_str

    if needs_quotes or contains_interpolation:
        # Escape any double quotes in the value for proper nesting
        escaped_value = v_str.replace('"', '\\"')
        # Use double quotes around the value: key="value"
        return f'{k}="{escaped_value}"'
    else:
        return f"{k}={v_str}"


def compute_overrides(schema: dict[str, Any], values: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Compute hydra overrides from schema and collected values (pure function).

    This function is Streamlit-free and can be used by both the FastAPI API
    and the Streamlit UI without triggering Streamlit initialization.

    Args:
        schema: Schema dictionary with ui_elements and constant_overrides
        values: Dictionary of form values collected from UI

    Returns:
        Tuple of (overrides, constant_overrides) lists
    """
    elements: list[dict[str, Any]] = schema.get("ui_elements", [])
    constant_overrides: list[str] = schema.get("constant_overrides", [])
    overrides: list[str] = []

    # Check if preprocessing profile is active
    preprocessing_profile = values.get("preprocessing_profile", "none")
    preprocessing_active = preprocessing_profile and preprocessing_profile != "none"

    for element in elements:
        key = element.get("key")
        if not isinstance(key, str) or not key:
            continue
        override_key = element.get("hydra_override")
        if not override_key:
            continue

        value = values.get(key)

        # Skip dataset override if preprocessing is active
        # (preprocessing profiles will set data=preprocessing automatically)
        if key == "dataset" and preprocessing_active:
            continue

        if isinstance(override_key, list):
            for k in override_key:
                if ov := _to_override(k, value):
                    overrides.append(ov)
        elif ov := _to_override(override_key, value):
            overrides.append(ov)

    return overrides, constant_overrides
