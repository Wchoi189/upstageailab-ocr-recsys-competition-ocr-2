from ocr.data.datasets import get_datasets_by_cfg
from ocr.core.models import get_model_by_cfg
from ocr.data.lightning_data import OCRDataPLModule


def get_pl_modules_by_cfg(config):
    """Legacy factory for Lightning modules. Deprecated in favor of OCRProjectOrchestrator.

    .. deprecated:: 2026-01-19
        Use `ocr.pipelines.orchestrator.OCRProjectOrchestrator` instead.
        This function will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "get_pl_modules_by_cfg() is deprecated. Use OCRProjectOrchestrator instead. "
        "See ocr/pipelines/orchestrator.py for the new entry point.",
        DeprecationWarning,
        stacklevel=2
    )
    import hydra
    # Inject vocab size into model config if needed
    if "tokenizer" in config.data:
        tokenizer = hydra.utils.instantiate(config.data.tokenizer)
        if "component_overrides" in config.model and config.model.component_overrides:
            if "head" in config.model.component_overrides:
                if "params" not in config.model.component_overrides.head:
                    config.model.component_overrides.head.params = {}
                config.model.component_overrides.head.params.out_channels = tokenizer.vocab_size

            # Inject vocab_size into Decoder (specifically for PARSeq/Transformers with Embeddings)
            if "decoder" in config.model.component_overrides:
                if "params" not in config.model.component_overrides.decoder:
                    config.model.component_overrides.decoder.params = {}
                config.model.component_overrides.decoder.params.vocab_size = tokenizer.vocab_size

    model = get_model_by_cfg(config.model)
    data_config = getattr(config, "data", None)
    dataset = get_datasets_by_cfg(config.datasets, data_config, config)
    metric_cfg = None
    if "metrics" in config and "eval" in config.metrics:
        metric_cfg = config.metrics.eval

    # Select domain-specific Lightning module
    domain = getattr(config, "domain", None)
    if domain == "detection":
        from ocr.domains.detection.module import DetectionPLModule
        modules = DetectionPLModule(model=model, dataset=dataset, config=config, metric_cfg=metric_cfg)
    elif domain == "recognition":
        from ocr.domains.recognition.module import RecognitionPLModule
        modules = RecognitionPLModule(model=model, dataset=dataset, config=config, metric_cfg=metric_cfg)
    else:
        raise ValueError(f"Unknown domain: {domain}. Must be 'detection' or 'recognition'.")

    data_modules = OCRDataPLModule(dataset=dataset, config=config)
    return modules, data_modules
