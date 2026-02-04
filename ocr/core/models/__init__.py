


def get_model_by_cfg(config):
    from ocr.core.models.architecture import OCRModel

    # V5.0: Check for atomic architecture with _target_
    architectures = getattr(config, "architectures", None)
    if architectures and "_target_" in architectures:
        import hydra
        return hydra.utils.instantiate(architectures, cfg=config)

    # Legacy: Check for singular architecture with _target_
    if "architecture" in config and "_target_" in config.architecture:
        import hydra
        return hydra.utils.instantiate(config.architecture)

    # Legacy: Check for string name
    arch_name = getattr(config, "architecture_name", None) or getattr(config, "architectures", None)
    if arch_name == "parseq":
        import importlib
        # Lazy load to avoid circular import/layering violation
        module = importlib.import_module("ocr.domains.recognition.models")
        PARSeq = getattr(module, "PARSeq")
        return PARSeq(config)

    return OCRModel(config)
