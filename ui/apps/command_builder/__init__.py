"""Command builder Streamlit app package."""

# Lazy imports to avoid circular import deadlocks when imported from FastAPI router
# These are only needed when running the Streamlit app, not when importing services

__all__ = ["main", "run"]


def __getattr__(name: str):
    """Lazy import for app functions to avoid deadlocks."""
    if name == "main":
        from .app import main

        return main
    if name == "run":
        from .app import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
