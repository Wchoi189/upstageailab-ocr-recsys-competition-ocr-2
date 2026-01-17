from . import architectures as _architectures  # noqa: F401


def get_model_by_cfg(config):
    from ocr.core.models.architecture import OCRModel

    arch_name = getattr(config, "architecture_name", None) or getattr(config, "architectures", None)
    if arch_name == "parseq":
        from ocr.features.recognition.models import PARSeq
        return PARSeq(config)

    if "architecture" in config and "_target_" in config.architecture:
        import hydra
        return hydra.utils.instantiate(config.architecture)

    return OCRModel(config)
