from . import architectures as _architectures  # noqa: F401
from .architecture import OCRModel


def get_model_by_cfg(config):
    arch_name = getattr(config, "architecture_name", None) or getattr(config, "architectures", None)
    if arch_name == "parseq":
        from .architectures.parseq import PARSeq
        return PARSeq(config)
    return OCRModel(config)
