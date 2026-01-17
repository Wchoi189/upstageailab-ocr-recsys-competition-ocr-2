"""KIE data handling."""

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "KIEDataset":
        from ocr.features.kie.data.dataset import KIEDataset
        return KIEDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["KIEDataset"]
