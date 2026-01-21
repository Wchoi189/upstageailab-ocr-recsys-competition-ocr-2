


def get_model_by_cfg(config):
    from ocr.core.models.architecture import OCRModel

    # V5.0: Check for atomic architecture with _target_
    architectures = getattr(config, "architectures", None)
    if architectures and "_target_" in architectures:
        import hydra
        # Disable recursive instantiation to prevent Hydra from trying to instantiate
        # the 'optimizer' inside cfg (which fails due to missing params)
        if hasattr(architectures, "_recursive_"):
             architectures._recursive_ = False
        else:
             from omegaconf import OmegaConf
             OmegaConf.set_struct(architectures, False) 
             architectures["_recursive_"] = False
             OmegaConf.set_struct(architectures, True)

        return hydra.utils.instantiate(architectures, cfg=config)

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
