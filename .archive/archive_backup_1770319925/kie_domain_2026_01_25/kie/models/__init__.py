"""KIE model definitions."""

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "LayoutLMv3Wrapper":
        from ocr.domains.kie.models.model import LayoutLMv3Wrapper
        return LayoutLMv3Wrapper
    elif name == "LiLTWrapper":
        from ocr.domains.kie.models.model import LiLTWrapper
        return LiLTWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["LayoutLMv3Wrapper", "LiLTWrapper"]
