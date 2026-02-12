"""Recognition-specific model components."""

# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "PARSeq":
        from .architecture import PARSeq
        return PARSeq
    elif name == "register_parseq_components":
        from .architecture import register_parseq_components
        return register_parseq_components
    elif name == "PARSeqDecoder":
        from .decoder import PARSeqDecoder
        return PARSeqDecoder
    elif name == "PARSeqHead":
        from .head import PARSeqHead
        return PARSeqHead
    elif name == "PermutationLanguageModeling":
        from .plm import PermutationLanguageModeling
        return PermutationLanguageModeling
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PARSeq",
    "PARSeqDecoder",
    "PARSeqHead",
    "PermutationLanguageModeling",
    "register_parseq_components",
]
