"""Registration of DBNet++ architecture components."""

from __future__ import annotations

from ocr.core import registry
from ocr.domains.detection.models.decoders.dbpp_decoder import DBPPDecoder
from ocr.core.models.encoder.timm_backbone import TimmBackbone
from ocr.domains.detection.models.heads.db_head import DBHead
from ocr.domains.detection.models.loss.db_loss import DBLoss


def register_dbnetpp_components() -> None:
    registry.register_encoder("dbnetpp_backbone", TimmBackbone)
    registry.register_decoder("dbnetpp_decoder", DBPPDecoder)
    registry.register_head("dbnetpp_head", DBHead)
    registry.register_loss("dbnetpp_loss", DBLoss)

    registry.register_architecture(
        name="dbnetpp",
        encoder="dbnetpp_backbone",
        decoder="dbnetpp_decoder",
        head="dbnetpp_head",
        loss="dbnetpp_loss",
    )


register_dbnetpp_components()
