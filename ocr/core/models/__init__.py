from . import architectures as _architectures  # noqa: F401


def get_model_by_cfg(config):
    from ocr.core.models.architecture import OCRModel

    # V5.0: Check for atomic architecture with _target_
    architectures = getattr(config, "architectures", None)
    if architectures and "_target_" in architectures:
        import hydra
        return hydra.utils.instantiate(architectures)

    # Legacy: Check for singular architecture with _target_
    if "architecture" in config and "_target_" in config.architecture:
        import hydra
        return hydra.utils.instantiate(config.architecture)

    # Legacy: Check for string name
    arch_name = getattr(config, "architecture_name", None) or getattr(config, "architectures", None)
    if arch_name == "parseq":
        from ocr.domains.recognition.models import PARSeq
        return PARSeq(config)

    return OCRModel(config)
