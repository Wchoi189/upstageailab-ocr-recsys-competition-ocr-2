"""Services for command builder app."""

# Lazy imports to avoid circular import deadlocks when imported from FastAPI router
# These are imported directly from their modules when needed, not from this __init__.py

__all__ = [
    "UseCaseRecommendationService",
    "build_additional_overrides",
]


def __getattr__(name: str):
    """Lazy import for services to avoid deadlocks."""
    if name == "UseCaseRecommendationService":
        from .recommendations import UseCaseRecommendationService

        return UseCaseRecommendationService
    if name == "build_additional_overrides":
        from .overrides import build_additional_overrides

        return build_additional_overrides
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
