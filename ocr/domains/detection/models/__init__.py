"""Detection model components."""

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    # Architectures - Note: These don't exist as classes, only as registration functions
    # The actual classes are accessed via registry, not direct imports
    if name == "CRAFT":
        raise AttributeError("CRAFT architecture is accessed via registry, not direct import. Use registry.get_architecture('craft')")
    elif name == "DBNet":
        raise AttributeError("DBNet architecture is accessed via registry, not direct import. Use registry.get_architecture('dbnet')")
    elif name == "DBNetPP":
        raise AttributeError("DBNetPP architecture is accessed via registry, not direct import. Use registry.get_architecture('dbnetpp')")
    # Heads
    elif name == "CRAFTHead":
        from ocr.domains.detection.models.heads.craft_head import CraftHead
        return CraftHead
    elif name == "DBHead":
        from ocr.domains.detection.models.heads.db_head import DBHead
        return DBHead
    # Postprocess
    elif name == "CRAFTPostProcessor":
        from ocr.domains.detection.models.postprocess.craft_postprocess import CraftPostProcessor
        return CraftPostProcessor
    elif name == "DBPostProcessor":
        from ocr.domains.detection.models.postprocess.db_postprocess import DBPostProcessor
        return DBPostProcessor
    # Decoders
    elif name == "CRAFTDecoder":
        from ocr.domains.detection.models.decoders.craft_decoder import CraftDecoder
        return CraftDecoder
    elif name == "DBPPDecoder":
        from ocr.domains.detection.models.decoders.dbpp_decoder import DBPPDecoder
        return DBPPDecoder
    elif name == "FPNDecoder":
        from ocr.domains.detection.models.decoders.fpn_decoder import FPNDecoder
        return FPNDecoder
    # Encoders
    elif name == "CRAFTVGG":
        from ocr.domains.detection.models.encoders.craft_vgg import CraftVGGEncoder
        return CraftVGGEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CRAFT", "DBNet", "DBNetPP",
    "CRAFTHead", "DBHead",
    "CRAFTPostProcessor", "DBPostProcessor",
    "CRAFTDecoder", "DBPPDecoder", "FPNDecoder",
    "CRAFTVGG"
]
