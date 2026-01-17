"""Recognition-specific model components."""

# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "PARSeq":
        from ocr.features.recognition.models.architecture import PARSeq
        return PARSeq
    elif name == "register_parseq_components":
        from ocr.features.recognition.models.architecture import register_parseq_components
        return register_parseq_components
    elif name == "PARSeqDecoder":
        from ocr.features.recognition.models.decoder import PARSeqDecoder
        return PARSeqDecoder
    elif name == "PARSeqHead":
        from ocr.features.recognition.models.head import PARSeqHead
        return PARSeqHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PARSeq",
    "PARSeqDecoder",
    "PARSeqHead",
    "register_parseq_components",
]
