"""Detection model components."""

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    # Architectures
    if name == "CRAFT":
        from ocr.features.detection.models.architectures.craft import CRAFT
        return CRAFT
    elif name == "DBNet":
        from ocr.features.detection.models.architectures.dbnet import DBNet
        return DBNet
    elif name == "DBNetPP":
        from ocr.features.detection.models.architectures.dbnetpp import DBNetPP
        return DBNetPP
    # Heads
    elif name == "CRAFTHead":
        from ocr.features.detection.models.heads.craft_head import CRAFTHead
        return CRAFTHead
    elif name == "DBHead":
        from ocr.features.detection.models.heads.db_head import DBHead
        return DBHead
    # Postprocess
    elif name == "CRAFTPostProcessor":
        from ocr.features.detection.models.postprocess.craft_postprocess import CRAFTPostProcessor
        return CRAFTPostProcessor
    elif name == "DBPostProcessor":
        from ocr.features.detection.models.postprocess.db_postprocess import DBPostProcessor
        return DBPostProcessor
    # Decoders
    elif name == "CRAFTDecoder":
        from ocr.features.detection.models.decoders.craft_decoder import CRAFTDecoder
        return CRAFTDecoder
    elif name == "DBPPDecoder":
        from ocr.features.detection.models.decoders.dbpp_decoder import DBPPDecoder
        return DBPPDecoder
    elif name == "FPNDecoder":
        from ocr.features.detection.models.decoders.fpn_decoder import FPNDecoder
        return FPNDecoder
    # Encoders
    elif name == "CRAFTVGG":
        from ocr.features.detection.models.encoders.craft_vgg import CRAFTVGG
        return CRAFTVGG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CRAFT", "DBNet", "DBNetPP",
    "CRAFTHead", "DBHead",
    "CRAFTPostProcessor", "DBPostProcessor",
    "CRAFTDecoder", "DBPPDecoder", "FPNDecoder",
    "CRAFTVGG"
]
